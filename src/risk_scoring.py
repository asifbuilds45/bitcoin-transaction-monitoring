"""
Risk Scoring & Explainability Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class RiskScoringEngine:
    """
    Computes holistic 0-100 cybersecurity risk scores by combining unsupervised
    Isolation Forest anomaly scores with multi-factor domain behavioral heuristics.
    Generates explainable natural-language investigative evidence.
    """

    def __init__(
        self,
        ml_weight: float = 0.45,
        heuristic_weight: float = 0.55
    ):
        self.ml_weight = ml_weight
        self.heuristic_weight = heuristic_weight

    def evaluate_transaction_risk(
        self,
        row: pd.Series,
        normalized_anomaly_score: float
    ) -> Tuple[int, str, List[str]]:
        """
        Evaluate risk score, risk level, and human-readable reasons for a single transaction.
        
        Args:
            row: Transaction record Series.
            normalized_anomaly_score: Isolation Forest score normalized to [0, 1].
            
        Returns:
            Tuple of (risk_score_0_to_100, risk_level, list_of_reasons)
        """
        reasons = []
        heuristic_points = 0.0
        max_possible_points = 100.0

        # 1. Transaction Frequency Heuristic (Rapid Burst)
        tx_freq = float(row.get('transaction_frequency_24h', 0))
        if tx_freq >= 80:
            heuristic_points += 18
            reasons.append(f"Extreme transaction velocity: {int(tx_freq)} transactions in 24h.")
        elif tx_freq >= 25:
            heuristic_points += 10
            reasons.append(f"Elevated transaction frequency: {int(tx_freq)} transactions in 24h.")

        # 2. Timing Gap Heuristic (Rapid Chain / Micro-timing)
        time_gap = float(row.get('avg_time_gap_min', 999))
        if time_gap <= 5.0:
            heuristic_points += 15
            reasons.append(f"Ultra-rapid succession: Average time gap is only {time_gap:.1f} minutes.")
        elif time_gap <= 20.0:
            heuristic_points += 8
            reasons.append(f"Short inter-transaction gap: Average {time_gap:.1f} minutes between transactions.")

        # 3. Wallet Graph Connectivity & Hub Degree
        wallet_deg = float(row.get('wallet_degree', 0))
        if wallet_deg >= 50:
            heuristic_points += 15
            reasons.append(f"High-degree hub wallet: Connected to {int(wallet_deg)} distinct entities in graph.")
        elif wallet_deg >= 20:
            heuristic_points += 8
            reasons.append(f"Moderate hub connectivity: {int(wallet_deg)} graph links.")

        # 4. Multi-IP Vantage Point & Entity Dispersion
        unique_ips = float(row.get('unique_ip_count', 1))
        if unique_ips >= 10:
            heuristic_points += 15
            reasons.append(f"Multi-IP proxy/botnet pattern: Wallet broadcast from {int(unique_ips)} distinct IPs.")
        elif unique_ips >= 4:
            heuristic_points += 8
            reasons.append(f"IP address dispersion: Associated with {int(unique_ips)} distinct IPs.")

        # 5. Cross-Border & Geographic Diversity
        country_cnt = float(row.get('country_count', 1))
        asn_cnt = float(row.get('asn_count', 1))
        src_country = str(row.get('src_country', ''))
        dst_country = str(row.get('dst_country', ''))
        
        if country_cnt >= 4 or asn_cnt >= 4:
            heuristic_points += 12
            reasons.append(f"Broad international infrastructure: Spanning {int(country_cnt)} countries and {int(asn_cnt)} ASNs.")
        elif src_country != dst_country and src_country != 'UNKNOWN' and dst_country != 'UNKNOWN':
            heuristic_points += 5
            reasons.append(f"Cross-border transaction broadcast ({src_country} -> {dst_country}).")

        # 6. Transaction UTXO Topology (Fan-Out / Fan-In)
        num_inputs = int(row.get('num_inputs', 1))
        num_outputs = int(row.get('num_outputs', 1))
        if num_outputs >= 8:
            heuristic_points += 12
            reasons.append(f"Fan-out distribution pattern: Funds split into {num_outputs} output addresses.")
        elif num_inputs >= 8:
            heuristic_points += 12
            reasons.append(f"Fan-in aggregation pattern: Consolidating funds from {num_inputs} input addresses.")

        # 7. Network Telemetry Bursts
        packet_count = float(row.get('packet_count', 0))
        bytes_tx = float(row.get('bytes_transferred', 0))
        if packet_count >= 1500 or bytes_tx >= 200_000:
            heuristic_points += 10
            reasons.append(f"High network traffic volume: {int(packet_count):,} packets ({int(bytes_tx / 1024):,} KB).")

        # 8. High Value Transfer Volume
        btc_vol = float(row.get('total_output_amount_btc', 0))
        if btc_vol >= 5.0:
            heuristic_points += 8
            reasons.append(f"Substantial value transfer: {btc_vol:.3f} BTC.")

        # If Isolation Forest identified strong anomaly
        if normalized_anomaly_score >= 0.70:
            reasons.append(f"AI Model detected high-dimensional behavioral divergence (Anomaly Score: {normalized_anomaly_score:.3f}).")

        # Combine ML Score and Heuristic Points
        # ML contribution: normalized_anomaly_score * 100 * ml_weight
        # Heuristic contribution: (min(heuristic_points, 100)) * heuristic_weight
        clamped_heuristic = min(heuristic_points, 100.0)
        composite_score = (normalized_anomaly_score * 100.0 * self.ml_weight) + (clamped_heuristic * self.heuristic_weight)
        
        final_score = int(np.clip(np.round(composite_score), 0, 100))

        # Determine Risk Tier
        if final_score >= 75:
            risk_level = "CRITICAL"
        elif final_score >= 50:
            risk_level = "HIGH"
        elif final_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if not reasons:
            reasons.append("Normal transaction parameters within expected baseline thresholds.")

        return final_score, risk_level, reasons

    def compute_risk_scores(
        self,
        df: pd.DataFrame,
        normalized_anomaly_scores: np.ndarray,
        raw_anomaly_scores: np.ndarray,
        is_anomaly_preds: np.ndarray
    ) -> pd.DataFrame:
        """
        Enrich dataframe with risk scores, risk levels, and formatted explanations.
        """
        df_scored = df.copy(deep=True)
        
        df_scored['raw_anomaly_score'] = raw_anomaly_scores
        df_scored['anomaly_score'] = normalized_anomaly_scores
        df_scored['is_anomaly'] = is_anomaly_preds

        risk_scores = []
        risk_levels = []
        reasons_list = []

        for idx, row in df_scored.iterrows():
            norm_score = float(normalized_anomaly_scores[idx])
            score, level, reasons = self.evaluate_transaction_risk(row, norm_score)
            risk_scores.append(score)
            risk_levels.append(level)
            reasons_list.append("; ".join(reasons))

        df_scored['risk_score'] = risk_scores
        df_scored['risk_level'] = risk_levels
        df_scored['risk_reasons'] = reasons_list

        return df_scored


def generate_ranked_alerts(df_scored: pd.DataFrame, min_risk: str = "MEDIUM") -> pd.DataFrame:
    """
    Generate ranked investigation alert feed sorted by risk score descending.
    """
    level_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_rank = level_order.get(min_risk.upper(), 1)

    filtered = df_scored[df_scored['risk_level'].map(lambda x: level_order.get(x, 0)) >= min_rank].copy()
    sorted_df = filtered.sort_values(by=['risk_score', 'anomaly_score'], ascending=[False, False]).reset_index(drop=True)
    sorted_df['Rank'] = sorted_df.index + 1

    alert_columns = [
        'Rank', 'txid', 'risk_score', 'risk_level', 'anomaly_score',
        'src_ip', 'dst_ip', 'source_wallet', 'destination_wallet',
        'src_country', 'dst_country', 'total_output_amount_btc', 'risk_reasons'
    ]
    present_cols = [c for c in alert_columns if c in sorted_df.columns]
    return sorted_df[present_cols]
