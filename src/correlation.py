"""
IP-TXID-Wallet Correlation Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class EntityCorrelator:
    """
    Correlates network-layer entities (IPs, Ports, ASNs) with
    blockchain-layer entities (TXIDs, Wallets, UTXOs).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Precompute lookup indexes and aggregation metrics for fast O(1) correlation queries."""
        # 1. IP to Transactions mapping
        self.ip_to_txids = self.df.groupby('src_ip')['txid'].apply(list).to_dict()
        
        # 2. IP to Wallets mapping
        self.ip_to_wallets = self.df.groupby('src_ip')['source_wallet'].unique().apply(list).to_dict()
        
        # 3. Wallet to IPs mapping
        self.wallet_to_ips = self.df.groupby('source_wallet')['src_ip'].unique().apply(list).to_dict()
        
        # 4. Wallet to TXIDs mapping (as source or destination)
        src_map = self.df.groupby('source_wallet')['txid'].apply(list).to_dict()
        dst_map = self.df.groupby('destination_wallet')['txid'].apply(list).to_dict()
        self.wallet_to_txids = {}
        all_wallets = set(src_map.keys()).union(set(dst_map.keys()))
        for w in all_wallets:
            txs = list(set(src_map.get(w, []) + dst_map.get(w, [])))
            self.wallet_to_txids[w] = txs

        # 5. Repeated Wallet Pairs (Wallet -> Wallet aggregation)
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

        # 6. TXID to full record lookup
        self.txid_map = self.df.set_index('txid').to_dict(orient='index')

    def get_tx_details(self, txid: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete network-blockchain correlation details for a specific TXID."""
        return self.txid_map.get(txid)

    def get_ip_correlations(self, ip: str) -> Dict[str, Any]:
        """
        Retrieve all blockchain entities and activity associated with an IP address.
        """
        associated_txids = self.ip_to_txids.get(ip, [])
        associated_wallets = self.ip_to_wallets.get(ip, [])
        tx_subset = self.df[self.df['src_ip'] == ip]

        total_btc = float(tx_subset['total_output_amount_btc'].sum()) if not tx_subset.empty else 0.0
        unique_dest_wallets = tx_subset['destination_wallet'].nunique() if not tx_subset.empty else 0
        countries = tx_subset['src_country'].unique().tolist() if not tx_subset.empty else []
        asns = tx_subset['src_asn'].unique().tolist() if not tx_subset.empty else []

        return {
            "ip": ip,
            "transaction_count": len(associated_txids),
            "associated_txids": associated_txids,
            "associated_wallets": associated_wallets,
            "unique_wallets_count": len(associated_wallets),
            "unique_dest_wallets_count": unique_dest_wallets,
            "total_btc_volume": round(total_btc, 4),
            "countries": countries,
            "asns": asns
        }

    def get_wallet_correlations(self, wallet: str) -> Dict[str, Any]:
        """
        Retrieve all network vantage points and transactions associated with a wallet entity.
        """
        associated_ips = self.wallet_to_ips.get(wallet, [])
        associated_txids = self.wallet_to_txids.get(wallet, [])
        
        src_subset = self.df[self.df['source_wallet'] == wallet]
        dst_subset = self.df[self.df['destination_wallet'] == wallet]
        
        total_sent_btc = float(src_subset['total_output_amount_btc'].sum()) if not src_subset.empty else 0.0
        total_received_btc = float(dst_subset['total_output_amount_btc'].sum()) if not dst_subset.empty else 0.0
        
        counterparty_wallets = list(set(
            src_subset['destination_wallet'].unique().tolist() + 
            dst_subset['source_wallet'].unique().tolist()
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
        ip_agg = (
            self.df.groupby('src_ip')
            .agg(
                wallet_count=('source_wallet', 'nunique'),
                tx_count=('txid', 'count'),
                total_btc=('total_output_amount_btc', 'sum'),
                country=('src_country', 'first'),
                asn=('src_asn', 'first')
            )
            .reset_index()
            .sort_values(by='wallet_count', ascending=False)
            .head(top_n)
        )
        return ip_agg

    def get_top_dispersed_wallets(self, top_n: int = 10) -> pd.DataFrame:
        """Find wallets originating from the highest number of distinct IPs (IP dispersion)."""
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
    """
    Load pre-supplied correlation edge CSV files if present.
    """
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
