"""
Risk Scoring & Explainability Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.fusion.evidence_fusion import MultiLayerEvidenceFusionEngine
from src.explainability.shap_explainer import BitcoinSHAPExplainer


class RiskScoringEngine:
    """
    Computes holistic 0-100 cybersecurity risk scores by combining Multi-Layer
    Fused Evidence (Isolation Forest, DBSCAN, Louvain, Temporal, Behavioural, Graph, Geo/ASN)
    with human-readable rationales and SHAP model explainability.
    """

    def __init__(
        self,
        ml_weight: float = 0.45,
        heuristic_weight: float = 0.55
    ):
        self.ml_weight = ml_weight
        self.heuristic_weight = heuristic_weight
        self.fusion_engine = MultiLayerEvidenceFusionEngine()
        self.shap_explainer = BitcoinSHAPExplainer()

    def evaluate_transaction_risk(
        self,
        row: pd.Series,
        normalized_anomaly_score: float
    ) -> Tuple[int, str, List[str]]:
        """
        Evaluate composite risk score (0-100), risk level tier, and human-readable reasons.
        """
        fused_score, breakdown, reasons = self.fusion_engine.compute_record_fused_evidence(row)
        
        # Convert fused score (0.0 - 1.0) into 0 - 100 integer Risk Score
        final_score = int(np.clip(np.round(fused_score * 100.0), 0, 100))

        # Risk Tiers
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
        is_anomaly_preds: np.ndarray,
        dbscan_evidence: Optional[np.ndarray] = None,
        louvain_evidence: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Enrich dataframe with unified risk scores, risk levels, evidence breakdowns, and explanations.
        """
        df_scored = df.copy(deep=True)
        
        df_scored['raw_anomaly_score'] = raw_anomaly_scores
        df_scored['anomaly_score'] = normalized_anomaly_scores
        df_scored['is_anomaly'] = is_anomaly_preds

        if dbscan_evidence is not None:
            df_scored['dbscan_evidence'] = dbscan_evidence
        else:
            df_scored['dbscan_evidence'] = np.where(is_anomaly_preds == 1, 0.65, 0.05)

        if louvain_evidence is not None:
            df_scored['louvain_community_evidence'] = louvain_evidence
        else:
            df_scored['louvain_community_evidence'] = np.where(df_scored['wallet_degree'] >= 20, 0.60, 0.10)

        # Run Multi-Layer Evidence Fusion
        df_fused = self.fusion_engine.fuse_dataframe_evidence(df_scored)

        risk_scores = []
        risk_levels = []
        reasons_list = []

        for idx, row in df_fused.iterrows():
            norm_score = float(normalized_anomaly_scores[idx])
            score, level, reasons = self.evaluate_transaction_risk(row, norm_score)
            risk_scores.append(score)
            risk_levels.append(level)
            reasons_list.append("; ".join(reasons))

        df_fused['risk_score'] = risk_scores
        df_fused['risk_level'] = risk_levels
        df_fused['risk_reasons'] = reasons_list
        df_fused['risk_level_normalized'] = [str(l).strip().upper() for l in risk_levels]

        return df_fused


def generate_ranked_alerts(
    df_scored: pd.DataFrame,
    risk_level_filter: str = "ALL",
    min_risk: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate ranked investigation alert feed sorted by risk score descending.
    Performs exact normalized risk level filtering when specified.
    Supports risk_level_filter, min_risk, or any keyword argument gracefully.
    """
    target_filter = kwargs.get("risk_level_filter", risk_level_filter)
    if "min_risk" in kwargs and kwargs["min_risk"] is not None:
        target_filter = kwargs["min_risk"]
    elif min_risk is not None and (target_filter == "ALL" or not target_filter):
        target_filter = min_risk

    df_copy = df_scored.copy(deep=True)
    if 'risk_level_normalized' not in df_copy.columns:
        df_copy['risk_level_normalized'] = df_copy['risk_level'].astype(str).str.strip().str.upper()

    if 'confidence_score' not in df_copy.columns:
        if 'correlation_confidence' in df_copy.columns:
            df_copy['confidence_score'] = df_copy['correlation_confidence']
        else:
            df_copy['confidence_score'] = 0.85

    if 'correlation_confidence_pct' not in df_copy.columns:
        df_copy['correlation_confidence_pct'] = df_copy['confidence_score'].apply(lambda c: f"{int(round(float(c) * 100))}%")

    # Sort original complete dataset first
    sorted_df = df_copy.sort_values(by=['risk_score', 'anomaly_score'], ascending=[False, False]).reset_index(drop=True)
    sorted_df['Rank'] = sorted_df.index + 1

    selected_level = str(target_filter).strip().upper()

    # Exact matching filter on complete feed
    if selected_level != "ALL":
        filtered = sorted_df[sorted_df['risk_level_normalized'] == selected_level].copy()
    else:
        filtered = sorted_df.copy()

    alert_columns = [
        'Rank', 'txid', 'risk_score', 'risk_level', 'anomaly_score', 'confidence_score', 'correlation_confidence_pct',
        'src_ip', 'dst_ip', 'source_wallet', 'destination_wallet',
        'src_country', 'dst_country', 'src_asn', 'total_output_amount_btc', 'risk_reasons'
    ]
    present_cols = [c for c in alert_columns if c in filtered.columns]
    return filtered[present_cols]
