"""
Temporal Correlation & Evidence Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Analyzes timing gaps, transaction bursts, and temporal alignment between
network telemetry and blockchain blocks to generate normalized Temporal Evidence Scores.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


def compute_temporal_evidence(
    row: pd.Series
) -> Tuple[float, List[str]]:
    """
    Compute normalized Temporal Evidence Score (0.0 to 1.0) and explainable reasons
    for a single transaction record.
    
    Args:
        row: Data row containing temporal features.
        
    Returns:
        Tuple of (temporal_evidence_score_0_to_1, list_of_temporal_reasons)
    """
    reasons = []
    score_points = 0.0

    # 1. 24-hour Transaction Velocity
    tx_freq = float(row.get('transaction_frequency_24h', 0))
    if tx_freq >= 80:
        score_points += 0.35
        reasons.append(f"Extreme transaction velocity: {int(tx_freq)} transactions within 24 hours.")
    elif tx_freq >= 25:
        score_points += 0.20
        reasons.append(f"High transaction frequency: {int(tx_freq)} transactions within 24 hours.")
    elif tx_freq >= 10:
        score_points += 0.10

    # 2. Inter-transaction Time Gap (Rapid Burst / Micro-timing)
    time_gap = float(row.get('avg_time_gap_min', 999.0))
    if time_gap <= 2.0:
        score_points += 0.35
        reasons.append(f"Ultra-rapid succession: Average inter-transaction gap is only {time_gap:.1f} minutes.")
    elif time_gap <= 10.0:
        score_points += 0.20
        reasons.append(f"Short time gap: Average {time_gap:.1f} minutes between transactions.")
    elif time_gap <= 30.0:
        score_points += 0.08

    # 3. Network Timestamp vs Blockchain Timestamp Delta Alignment
    try:
        net_ts = pd.to_datetime(row.get('network_timestamp', row.get('timestamp')))
        block_ts = pd.to_datetime(row.get('timestamp'))
        delta_sec = abs((net_ts - block_ts).total_seconds())

        if delta_sec <= 30:
            reasons.append(f"Tight temporal alignment: Network broadcast aligned within {int(delta_sec)}s of block timestamp.")
            score_points += 0.15
        elif delta_sec > 14400: # >4 hours mismatch
            reasons.append(f"Significant temporal skew: Network broadcast and block timestamp differ by {int(delta_sec / 60)} minutes.")
            score_points += 0.15
    except Exception:
        pass

    # 4. Connection Duration vs Traffic Rate
    duration = float(row.get('connection_duration_sec', 0.0))
    packets = float(row.get('packet_count', 0.0))
    if duration > 0 and packets > 0:
        rate = packets / duration
        if rate >= 100.0:
            score_points += 0.15
            reasons.append(f"High network burst rate: {rate:.1f} packets/second during connection.")

    final_score = float(np.clip(score_points, 0.0, 1.0))
    if not reasons:
        reasons.append("Temporal metrics within baseline expected distribution.")

    return final_score, reasons


def analyze_temporal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich dataframe with 'temporal_evidence_score' and 'temporal_reasons'.
    """
    df_temp = df.copy(deep=True)
    temp_scores = []
    temp_reasons_list = []

    for _, row in df_temp.iterrows():
        score, reasons = compute_temporal_evidence(row)
        temp_scores.append(round(score, 4))
        temp_reasons_list.append("; ".join(reasons))

    df_temp['temporal_evidence_score'] = temp_scores
    df_temp['temporal_reasons'] = temp_reasons_list

    return df_temp
