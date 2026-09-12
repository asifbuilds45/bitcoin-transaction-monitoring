"""
IP-TXID-Wallet Correlation Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.correlation.correlator import DualLayerCorrelator, calculate_record_correlation_confidence


class EntityCorrelator:
    """
    Correlates network-layer entities (IPs, Ports, ASNs) with
    blockchain-layer entities (TXIDs, Wallets, UTXOs).
    Backward-compatible wrapper supporting both combined and dual-layer data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)
        if 'correlation_confidence' not in self.df.columns:
            # Precompute correlation confidence if missing
            conf_scores = []
            conf_pcts = []
            for _, row in self.df.iterrows():
                conf = calculate_record_correlation_confidence(
                    net_ts=row.get('network_timestamp', row.get('timestamp')),
                    block_ts=row.get('timestamp'),
                    has_exact_txid=bool(row.get('txid') and str(row.get('txid')) != 'UNKNOWN'),
                    src_port=int(row.get('src_port', 8333)),
                    protocol=str(row.get('protocol', 'TCP'))
                )
                conf_scores.append(round(conf, 4))
                conf_pcts.append(f"{int(round(conf * 100))}%")
            self.df['correlation_confidence'] = conf_scores
            self.df['correlation_confidence_pct'] = conf_pcts

        self._build_indexes()

    def _build_indexes(self) -> None:
        """Precompute lookup indexes and aggregation metrics for fast O(1) correlation queries."""
        self.ip_to_txids = self.df.groupby('src_ip')['txid'].apply(list).to_dict() if 'src_ip' in self.df.columns else {}
        self.ip_to_wallets = self.df.groupby('src_ip')['source_wallet'].unique().apply(list).to_dict() if 'src_ip' in self.df.columns and 'source_wallet' in self.df.columns else {}
        self.wallet_to_ips = self.df.groupby('source_wallet')['src_ip'].unique().apply(list).to_dict() if 'src_ip' in self.df.columns and 'source_wallet' in self.df.columns else {}
        
        src_map = self.df.groupby('source_wallet')['txid'].apply(list).to_dict() if 'source_wallet' in self.df.columns else {}
        dst_map = self.df.groupby('destination_wallet')['txid'].apply(list).to_dict() if 'destination_wallet' in self.df.columns else {}
        self.wallet_to_txids = {}
        all_wallets = set(src_map.keys()).union(set(dst_map.keys()))
        for w in all_wallets:
            txs = list(set(src_map.get(w, []) + dst_map.get(w, [])))
            self.wallet_to_txids[w] = txs

        if 'source_wallet' in self.df.columns and 'destination_wallet' in self.df.columns:
            pair_counts = (
                self.df.groupby(['source_wallet', 'destination_wallet'])
                .agg(
                    tx_count=('txid', 'count'),
                    total_btc=('total_output_amount_btc', 'sum'),
                    unique_ips=('src_ip', 'nunique')
                )
                .reset_index()
            )
            self.wallet_pairs = pair_counts.sort_values(by='tx_count', ascending=False)
        else:
            self.wallet_pairs = pd.DataFrame()

        self.txid_map = self.df.set_index('txid').to_dict(orient='index') if 'txid' in self.df.columns else {}

    def get_tx_details(self, txid: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete network-blockchain correlation details for a specific TXID."""
        return self.txid_map.get(txid)

    def get_ip_correlations(self, ip: str) -> Dict[str, Any]:
        """Retrieve all blockchain entities and activity associated with an IP address."""
        associated_txids = self.ip_to_txids.get(ip, [])
        associated_wallets = self.ip_to_wallets.get(ip, [])
        tx_subset = self.df[self.df['src_ip'] == ip] if 'src_ip' in self.df.columns else pd.DataFrame()

        total_btc = float(tx_subset['total_output_amount_btc'].sum()) if not tx_subset.empty and 'total_output_amount_btc' in tx_subset.columns else 0.0
        unique_dest_wallets = tx_subset['destination_wallet'].nunique() if not tx_subset.empty and 'destination_wallet' in tx_subset.columns else 0
        countries = tx_subset['src_country'].unique().tolist() if not tx_subset.empty and 'src_country' in tx_subset.columns else []
        asns = tx_subset['src_asn'].unique().tolist() if not tx_subset.empty and 'src_asn' in tx_subset.columns else []
        avg_conf = float(tx_subset['correlation_confidence'].mean()) if not tx_subset.empty and 'correlation_confidence' in tx_subset.columns else 0.85

        return {
            "ip": ip,
            "transaction_count": len(associated_txids),
            "associated_txids": associated_txids,
            "associated_wallets": associated_wallets,
            "unique_wallets_count": len(associated_wallets),
            "unique_dest_wallets_count": unique_dest_wallets,
            "total_btc_volume": round(total_btc, 4),
            "countries": countries,
            "asns": asns,
            "avg_correlation_confidence": round(avg_conf, 4),
            "correlation_descriptor": f"Network observation associated with transaction/wallet activity ({int(round(avg_conf*100))}% confidence)"
        }

    def get_wallet_correlations(self, wallet: str) -> Dict[str, Any]:
        """Retrieve all network vantage points and transactions associated with a wallet entity."""
        associated_ips = self.wallet_to_ips.get(wallet, [])
        associated_txids = self.wallet_to_txids.get(wallet, [])
        
        src_subset = self.df[self.df['source_wallet'] == wallet] if 'source_wallet' in self.df.columns else pd.DataFrame()
        dst_subset = self.df[self.df['destination_wallet'] == wallet] if 'destination_wallet' in self.df.columns else pd.DataFrame()
        
        total_sent_btc = float(src_subset['total_output_amount_btc'].sum()) if not src_subset.empty and 'total_output_amount_btc' in src_subset.columns else 0.0
        total_received_btc = float(dst_subset['total_output_amount_btc'].sum()) if not dst_subset.empty and 'total_output_amount_btc' in dst_subset.columns else 0.0
        
        counterparty_wallets = list(set(
            (src_subset['destination_wallet'].unique().tolist() if not src_subset.empty and 'destination_wallet' in src_subset.columns else []) + 
            (dst_subset['source_wallet'].unique().tolist() if not dst_subset.empty and 'source_wallet' in dst_subset.columns else [])
        ))
        if wallet in counterparty_wallets:
            counterparty_wallets.remove(wallet)

        return {
            "wallet": wallet,
            "associated_ips": associated_ips,
            "unique_ip_count": len(associated_ips),
            "total_transactions": len(associated_txids),
            "sent_transactions_count": len(src_subset),
            "received_transactions_count": len(dst_subset),
            "total_sent_btc": round(total_sent_btc, 4),
            "total_received_btc": round(total_received_btc, 4),
            "counterparty_wallets": counterparty_wallets,
            "counterparty_count": len(counterparty_wallets)
        }

    def get_top_reused_ips(self, top_n: int = 10) -> pd.DataFrame:
        """Find IPs broadcasting on behalf of the highest number of distinct wallets."""
        if 'src_ip' not in self.df.columns:
            return pd.DataFrame()
        ip_agg = (
            self.df.groupby('src_ip')
            .agg(
                wallet_count=('source_wallet', 'nunique'),
                tx_count=('txid', 'count'),
                total_btc=('total_output_amount_btc', 'sum'),
                country=('src_country', 'first'),
                asn=('src_asn', 'first'),
                avg_confidence=('correlation_confidence', 'mean')
            )
            .reset_index()
            .sort_values(by='wallet_count', ascending=False)
            .head(top_n)
        )
        return ip_agg

    def get_top_dispersed_wallets(self, top_n: int = 10) -> pd.DataFrame:
        """Find wallets originating from the highest number of distinct IPs (IP dispersion)."""
        if 'source_wallet' not in self.df.columns:
            return pd.DataFrame()
        wallet_agg = (
            self.df.groupby('source_wallet')
            .agg(
                ip_count=('src_ip', 'nunique'),
                tx_count=('txid', 'count'),
                total_btc=('total_output_amount_btc', 'sum'),
                country_count=('src_country', 'nunique'),
                asn_count=('src_asn', 'nunique')
            )
            .reset_index()
            .sort_values(by='ip_count', ascending=False)
            .head(top_n)
        )
        return wallet_agg


def load_correlation_edges(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Load pre-supplied correlation edge CSV files if present."""
    edge_files = {
        "ip_txid_wallet": os.path.join(data_dir, "ip_txid_wallet_correlations.csv"),
        "transaction_wallet": os.path.join(data_dir, "transaction_wallet_edges.csv"),
        "wallet_wallet": os.path.join(data_dir, "wallet_wallet_edges.csv"),
        "ip_wallet": os.path.join(data_dir, "ip_wallet_edges.csv")
    }
    loaded = {}
    for name, path in edge_files.items():
        if os.path.exists(path):
            loaded[name] = pd.read_csv(path)
    return loaded
