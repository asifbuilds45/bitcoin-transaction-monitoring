"""
Behavioural Fingerprinting Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Builds behavioural profiles for IPs, Wallets, and Transactions combining
network flow characteristics and on-chain UTXO shape metrics.
Strictly excludes ground-truth and scenario labels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class BehaviouralProfiler:
    """
    Constructs entity-level behavioural fingerprints and computes
    normalized Behavioural Evidence Scores.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)

    def compute_ip_fingerprint(self, ip: str) -> Dict[str, Any]:
        """Generate behavioural profile for an IP address."""
        sub = self.df[self.df['src_ip'] == ip] if 'src_ip' in self.df.columns else pd.DataFrame()
        if sub.empty:
            return {"ip": ip, "tx_count": 0, "profile": "Inactive / Unknown"}

        tx_count = len(sub)
        unique_wallets = sub['source_wallet'].nunique() if 'source_wallet' in sub.columns else 0
        total_btc = float(sub['total_output_amount_btc'].sum()) if 'total_output_amount_btc' in sub.columns else 0.0
        avg_packets = float(sub['packet_count'].mean()) if 'packet_count' in sub.columns else 0.0
        countries = sub['src_country'].unique().tolist() if 'src_country' in sub.columns else []

        # Behavioural classification
        if unique_wallets >= 5:
            pattern = "Proxy / Multi-Wallet Relay Hub"
        elif tx_count >= 50:
            pattern = "Automated High-Frequency Broadcaster"
        elif len(countries) > 1:
            pattern = "Multi-Geographic Vantage Point"
        else:
            pattern = "Standard User Vantage Point"

        return {
            "ip": ip,
            "transaction_count": tx_count,
            "unique_wallets_broadcasted": unique_wallets,
            "total_volume_btc": round(total_btc, 4),
            "avg_packets_per_tx": round(avg_packets, 1),
            "associated_countries": countries,
            "behavioural_pattern": pattern
        }

    def compute_wallet_fingerprint(self, wallet: str) -> Dict[str, Any]:
        """Generate behavioural profile for a wallet address."""
        sub = self.df[(self.df['source_wallet'] == wallet) | (self.df['destination_wallet'] == wallet)] if 'source_wallet' in self.df.columns else pd.DataFrame()
        if sub.empty:
            return {"wallet": wallet, "tx_count": 0, "profile": "Inactive / Unknown"}

        tx_count = len(sub)
        unique_ips = sub['src_ip'].nunique() if 'src_ip' in sub.columns else 0
        max_outputs = int(sub['num_outputs'].max()) if 'num_outputs' in sub.columns else 1
        max_inputs = int(sub['num_inputs'].max()) if 'num_inputs' in sub.columns else 1
        max_freq = int(sub['transaction_frequency_24h'].max()) if 'transaction_frequency_24h' in sub.columns else 1

        if max_outputs >= 8:
            pattern = "Fan-Out Mixing / Distribution Wallet"
        elif max_inputs >= 8:
            pattern = "Fan-In Consolidation / Aggregator Wallet"
        elif unique_ips >= 5:
            pattern = "Multi-Proxy IP Dispersed Wallet"
        elif max_freq >= 30:
            pattern = "High-Velocity Botnet/Service Wallet"
        else:
            pattern = "Standard Wallet Entity"

        return {
            "wallet": wallet,
            "transaction_count": tx_count,
            "unique_ip_vantage_points": unique_ips,
            "max_fan_out_outputs": max_outputs,
            "max_fan_in_inputs": max_inputs,
            "max_24h_velocity": max_freq,
            "behavioural_pattern": pattern
        }

    def compute_transaction_behavioural_evidence(self, row: pd.Series) -> Tuple[float, List[str]]:
        """
        Compute normalized Behavioural Evidence Score (0.0 to 1.0) and reasons for a transaction.
        """
        reasons = []
        points = 0.0

        # IP Dispersion
        unique_ips = float(row.get('unique_ip_count', 1))
        if unique_ips >= 10:
            points += 0.35
            reasons.append(f"Proxy/Botnet dispersion: Wallet observed broadcasting across {int(unique_ips)} distinct IPs.")
        elif unique_ips >= 4:
            points += 0.18
            reasons.append(f"Multi-IP activity: Associated with {int(unique_ips)} IPs.")

        # Wallet Hub Degree
        degree = float(row.get('wallet_degree', 1))
        if degree >= 50:
            points += 0.30
            reasons.append(f"High graph hub degree: Connected to {int(degree)} network entities.")
        elif degree >= 20:
            points += 0.15
            reasons.append(f"Elevated connectivity: {int(degree)} graph links.")

        # UTXO Shape Anomaly (Fan-Out / Fan-In)
        n_in = int(row.get('num_inputs', 1))
        n_out = int(row.get('num_outputs', 1))
        if n_out >= 8:
            points += 0.25
            reasons.append(f"Fan-out distribution shape: {n_out} output addresses.")
        elif n_in >= 8:
            points += 0.25
            reasons.append(f"Fan-in consolidation shape: {n_in} input addresses.")

        # Miner Fee Ratio Anomaly
        fee = float(row.get('fee_btc', 0.0))
        amt = float(row.get('total_output_amount_btc', 0.0001))
        ratio = fee / max(amt, 0.0001)
        if ratio >= 0.10: # Fee is >10% of total transfer volume
            points += 0.20
            reasons.append(f"Unusual miner fee ratio: Fee is {ratio*100:.1f}% of transaction value.")

        final_score = float(np.clip(points, 0.0, 1.0))
        if not reasons:
            reasons.append("Behavioural fingerprint matches baseline profile.")

        return final_score, reasons


def extract_behavioural_evidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich dataframe with 'behavioural_evidence_score' and 'behavioural_reasons'.
    """
    profiler = BehaviouralProfiler(df)
    df_out = df.copy(deep=True)

    scores = []
    reasons_list = []
    for _, row in df_out.iterrows():
        score, reasons = profiler.compute_transaction_behavioural_evidence(row)
        scores.append(round(score, 4))
        reasons_list.append("; ".join(reasons))

    df_out['behavioural_evidence_score'] = scores
    df_out['behavioural_reasons'] = reasons_list
    return df_out
