"""
Multi-Layer Evidence Fusion Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Fuses 9 independent evidence channels normalized to [0.0, 1.0] using explicit,
configurable weights that sum to 1.0.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Default Evidence Weights (Must sum to 1.0)
DEFAULT_EVIDENCE_WEIGHTS = {
    "isolation_forest_evidence": 0.20,  # AI Unsupervised Tree Isolation
    "dbscan_evidence": 0.10,            # Behavioural Clustering & Noise
    "louvain_community_evidence": 0.10, # Graph Community Modularity
    "temporal_evidence": 0.12,          # Time Gap & 24h Velocity
    "behavioural_evidence": 0.13,       # Wallet Degree & Proxy Dispersion
    "graph_evidence": 0.10,             # Network Topology & Centrality
    "geo_asn_evidence": 0.08,           # Geographic/ASN Diversity
    "blockchain_evidence": 0.09,        # Volume & Miner Fee Ratio
    "network_evidence": 0.08            # Packet Burst & Duration
}


class MultiLayerEvidenceFusionEngine:
    """
    Normalizes and fuses heterogeneous evidence channels into a unified evidence score.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights is None:
            self.weights = DEFAULT_EVIDENCE_WEIGHTS.copy()
        else:
            self.weights = weights.copy()
            
        # Normalize weights to ensure sum = 1.0
        total_w = sum(self.weights.values())
        if total_w > 0 and abs(total_w - 1.0) > 1e-4:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute_record_fused_evidence(
        self,
        row: pd.Series
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Compute fused evidence score (0.0 to 1.0), breakdown by channel, and top contributing reasons.
        """
        breakdown = {}
        top_reasons = []

        # 1. Isolation Forest Evidence
        if_score = float(row.get('anomaly_score', row.get('isolation_forest_evidence', 0.0)))
        breakdown['isolation_forest_evidence'] = float(np.clip(if_score, 0.0, 1.0))
        if if_score >= 0.65:
            top_reasons.append(f"AI Isolation Forest score: {if_score:.3f}")

        # 2. DBSCAN Evidence
        db_score = float(row.get('dbscan_evidence', 0.0))
        breakdown['dbscan_evidence'] = float(np.clip(db_score, 0.0, 1.0))
        if db_score >= 0.70:
            top_reasons.append("DBSCAN flagged unclustered behavioural noise/outlier")

        # 3. Louvain Community Evidence
        louv_score = float(row.get('louvain_community_evidence', 0.0))
        breakdown['louvain_community_evidence'] = float(np.clip(louv_score, 0.0, 1.0))
        if louv_score >= 0.50:
            top_reasons.append("Member of elevated-risk Louvain graph community")

        # 4. Temporal Evidence
        temp_score = float(row.get('temporal_evidence_score', row.get('temporal_evidence', 0.0)))
        breakdown['temporal_evidence'] = float(np.clip(temp_score, 0.0, 1.0))
        if temp_score >= 0.50:
            top_reasons.append(row.get('temporal_reasons', 'Temporal burst anomaly').split(';')[0])

        # 5. Behavioural Evidence
        beh_score = float(row.get('behavioural_evidence_score', row.get('behavioural_evidence', 0.0)))
        breakdown['behavioural_evidence'] = float(np.clip(beh_score, 0.0, 1.0))
        if beh_score >= 0.50:
            top_reasons.append(row.get('behavioural_reasons', 'High behavioural dispersion').split(';')[0])

        # 6. Graph Evidence
        degree = float(row.get('wallet_degree', 1))
        graph_score = min(degree / 50.0, 1.0)
        breakdown['graph_evidence'] = float(np.clip(graph_score, 0.0, 1.0))
        if degree >= 20:
            top_reasons.append(f"High graph degree centrality: {int(degree)} links")

        # 7. Geo/ASN Evidence
        src_country = str(row.get('src_country', ''))
        dst_country = str(row.get('dst_country', ''))
        country_cnt = float(row.get('country_count', 1))
        geo_score = min((country_cnt - 1) / 4.0 + (0.2 if src_country != dst_country else 0.0), 1.0)
        breakdown['geo_asn_evidence'] = float(np.clip(geo_score, 0.0, 1.0))
        if country_cnt >= 3:
            top_reasons.append(f"International infrastructure across {int(country_cnt)} countries")

        # 8. Blockchain Evidence
        btc_vol = float(row.get('total_output_amount_btc', 0.0))
        fee = float(row.get('fee_btc', 0.0))
        chain_score = min(btc_vol / 10.0 + (fee / max(btc_vol, 0.001) * 2.0), 1.0)
        breakdown['blockchain_evidence'] = float(np.clip(chain_score, 0.0, 1.0))
        if btc_vol >= 5.0:
            top_reasons.append(f"Large transfer volume: {btc_vol:.3f} BTC")

        # 9. Network Evidence
        pkts = float(row.get('packet_count', 0))
        bytes_tx = float(row.get('bytes_transferred', 0))
        net_score = min((pkts / 2000.0) + (bytes_tx / 200000.0), 1.0)
        breakdown['network_evidence'] = float(np.clip(net_score, 0.0, 1.0))
        if pkts >= 1500:
            top_reasons.append(f"High network packet count: {int(pkts):,} packets")

        # Compute weighted sum
        fused_score = sum(breakdown[k] * self.weights.get(k, 0.0) for k in breakdown)
        clamped_fused = float(np.clip(fused_score, 0.0, 1.0))

        return clamped_fused, breakdown, top_reasons

    def fuse_dataframe_evidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataframe with 'fused_evidence_score', 'evidence_breakdown', and 'top_fused_reasons'.
        """
        df_fused = df.copy(deep=True)
        fused_scores = []
        top_reasons_list = []

        for _, row in df_fused.iterrows():
            score, breakdown, reasons = self.compute_record_fused_evidence(row)
            fused_scores.append(round(score, 4))
            top_reasons_list.append("; ".join(reasons) if reasons else "Baseline evidence thresholds.")

        df_fused['fused_evidence_score'] = fused_scores
        df_fused['fused_reasons'] = top_reasons_list
        return df_fused
