"""
Layer Correlation & Correlation Confidence Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Correlates independent Network-Layer observations with Blockchain-Layer transactions
and calculates a separate Correlation Confidence score (0-100%).
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def calculate_record_correlation_confidence(
    net_ts: Any,
    block_ts: Any,
    has_exact_txid: bool = True,
    src_port: int = 8333,
    protocol: str = "TCP",
    repeat_count: int = 1
) -> float:
    """
    Calculate Correlation Confidence (0.0 to 1.0) indicating how strongly
    a network-layer observation correlates with a blockchain-layer transaction.
    
    This is NOT an anomaly or risk score.
    """
    confidence = 0.0
    
    # 1. Exact TXID match (base weight = 0.45)
    if has_exact_txid:
        confidence += 0.45
    else:
        confidence += 0.10

    # 2. Timestamp alignment (base weight = 0.30)
    try:
        ts1 = pd.to_datetime(net_ts)
        ts2 = pd.to_datetime(block_ts)
        delta_sec = abs((ts1 - ts2).total_seconds())
        
        if delta_sec <= 60:
            confidence += 0.30
        elif delta_sec <= 300:
            confidence += 0.25
        elif delta_sec <= 1800:
            confidence += 0.15
        elif delta_sec <= 7200:
            confidence += 0.08
        else:
            confidence += 0.02
    except Exception:
        confidence += 0.10

    # 3. Network P2P Protocol Consistency (standard Bitcoin port 8333 or TCP) (weight = 0.15)
    if src_port in [8333, 18333, 38333] or str(protocol).upper() in ['TCP', 'BITCOIN']:
        confidence += 0.15
    else:
        confidence += 0.05

    # 4. Observation Repeatability (weight = 0.10)
    if repeat_count >= 3:
        confidence += 0.10
    elif repeat_count >= 2:
        confidence += 0.06
    else:
        confidence += 0.03

    return float(np.clip(confidence, 0.0, 1.0))


class DualLayerCorrelator:
    """
    Correlates independent Network DataFrames and Blockchain DataFrames.
    Builds IP -> TXID -> Wallet and Wallet -> TXID -> Wallet relationships
    and computes Correlation Confidence.
    """

    def __init__(self, df_network: pd.DataFrame, df_blockchain: pd.DataFrame):
        self.df_network = df_network.copy(deep=True)
        self.df_blockchain = df_blockchain.copy(deep=True)
        self.correlated_df = pd.DataFrame()
        self._correlate_layers()
        self._build_indexes()

    def _correlate_layers(self) -> None:
        """
        Merge network observations and blockchain transactions on 'txid' if available,
        or perform temporal proximity alignment.
        """
        # Outer join or inner join on TXID
        if 'txid' in self.df_network.columns and 'txid' in self.df_blockchain.columns:
            merged = pd.merge(
                self.df_network,
                self.df_blockchain,
                on='txid',
                how='inner',
                suffixes=('_net', '_chain')
            )
            
            # If timestamps were duplicated in columns, resolve them
            if 'timestamp_chain' in merged.columns:
                merged['timestamp'] = merged['timestamp_chain']
            elif 'timestamp_net' in merged.columns:
                merged['timestamp'] = merged['timestamp_net']
                
            if 'network_timestamp' not in merged.columns and 'timestamp_net' in merged.columns:
                merged['network_timestamp'] = merged['timestamp_net']
        else:
            # Fallback if txid column is missing in one dataset
            merged = self.df_blockchain.copy(deep=True)
            for col in self.df_network.columns:
                if col not in merged.columns:
                    merged[col] = self.df_network[col].values[:len(merged)]

        # Calculate Correlation Confidence for each correlated record
        conf_scores = []
        conf_pcts = []
        
        # Calculate IP observation repeat counts
        ip_counts = merged['src_ip'].value_counts().to_dict() if 'src_ip' in merged.columns else {}

        for idx, row in merged.iterrows():
            net_ts = row.get('network_timestamp', row.get('timestamp'))
            block_ts = row.get('timestamp')
            has_txid = bool(row.get('txid') and str(row.get('txid')) != 'UNKNOWN')
            port = int(row.get('src_port', 8333))
            proto = str(row.get('protocol', 'TCP'))
            src_ip = str(row.get('src_ip', ''))
            rep = ip_counts.get(src_ip, 1)

            conf = calculate_record_correlation_confidence(
                net_ts=net_ts,
                block_ts=block_ts,
                has_exact_txid=has_txid,
                src_port=port,
                protocol=proto,
                repeat_count=rep
            )
            conf_scores.append(round(conf, 4))
            conf_pcts.append(f"{int(round(conf * 100))}%")

        merged['correlation_confidence'] = conf_scores
        merged['correlation_confidence_pct'] = conf_pcts
        self.correlated_df = merged

    def _build_indexes(self) -> None:
        """Precompute lookup dicts for fast entity queries."""
        df = self.correlated_df
        self.ip_to_txids = df.groupby('src_ip')['txid'].apply(list).to_dict() if 'src_ip' in df.columns else {}
        self.ip_to_wallets = df.groupby('src_ip')['source_wallet'].unique().apply(list).to_dict() if 'src_ip' in df.columns and 'source_wallet' in df.columns else {}
        self.wallet_to_ips = df.groupby('source_wallet')['src_ip'].unique().apply(list).to_dict() if 'src_ip' in df.columns and 'source_wallet' in df.columns else {}

        src_map = df.groupby('source_wallet')['txid'].apply(list).to_dict() if 'source_wallet' in df.columns else {}
        dst_map = df.groupby('destination_wallet')['txid'].apply(list).to_dict() if 'destination_wallet' in df.columns else {}
        self.wallet_to_txids = {}
        all_wallets = set(src_map.keys()).union(set(dst_map.keys()))
        for w in all_wallets:
            self.wallet_to_txids[w] = list(set(src_map.get(w, []) + dst_map.get(w, [])))

        self.txid_map = df.set_index('txid').to_dict(orient='index') if 'txid' in df.columns else {}

    def get_correlated_dataframe(self) -> pd.DataFrame:
        """Return the fully correlated dataset with correlation confidence scores."""
        return self.correlated_df

    def get_ip_correlations(self, ip: str) -> Dict[str, Any]:
        """Retrieve associated entities for a given IP."""
        txids = self.ip_to_txids.get(ip, [])
        wallets = self.ip_to_wallets.get(ip, [])
        subset = self.correlated_df[self.correlated_df['src_ip'] == ip] if 'src_ip' in self.correlated_df.columns else pd.DataFrame()

        avg_conf = float(subset['correlation_confidence'].mean()) if not subset.empty else 0.0
        btc_vol = float(subset['total_output_amount_btc'].sum()) if not subset.empty and 'total_output_amount_btc' in subset.columns else 0.0

        return {
            "ip": ip,
            "associated_txids": txids,
            "associated_wallets": wallets,
            "transaction_count": len(txids),
            "total_btc_volume": round(btc_vol, 4),
            "avg_correlation_confidence": round(avg_conf, 4),
            "correlation_descriptor": f"Network observation associated with transaction/wallet activity ({int(round(avg_conf*100))}% mean confidence)"
        }
