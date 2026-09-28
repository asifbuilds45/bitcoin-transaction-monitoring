"""
Multi-Layer Evidence Fusion Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Fuses 11 independent evidence channels normalized to [0.0, 1.0] using explicit,
configurable weights that sum to 1.0.
Channels 10-11 added: peeling_chain_evidence, coinjoin_evidence (Behavioural Pattern Detection).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Default Evidence Weights (Must sum to 1.0)
# Original 9 channels renormalized by factor 0.91; 0.09 freed for 2 new pattern channels.
DEFAULT_EVIDENCE_WEIGHTS = {
    "isolation_forest_evidence": 0.182,  # AI Unsupervised Tree Isolation
    "dbscan_evidence": 0.091,            # Behavioural Clustering & Noise
    "louvain_community_evidence": 0.091, # Graph Community Modularity
    "temporal_evidence": 0.109,          # Time Gap & 24h Velocity
    "behavioural_evidence": 0.118,       # Wallet Degree & Proxy Dispersion
    "graph_evidence": 0.091,             # Network Topology & Centrality
    "geo_asn_evidence": 0.073,           # Geographic/ASN Diversity
    "blockchain_evidence": 0.082,        # Volume & Miner Fee Ratio
    "network_evidence": 0.073,           # Packet Burst & Duration
    "peeling_chain_evidence": 0.050,     # Peeling-chain pattern behavioural indicator
    "coinjoin_evidence": 0.040,          # CoinJoin-like / mixing pattern indicator
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

        # 2. HDBSCAN Behavioural Evidence
        db_score = float(row.get('hdbscan_evidence', row.get('dbscan_evidence', 0.0)))
        breakdown['dbscan_evidence'] = float(np.clip(db_score, 0.0, 1.0))
        breakdown['hdbscan_evidence'] = breakdown['dbscan_evidence']
        if db_score >= 0.70:
            top_reasons.append("HDBSCAN flagged unclustered behavioural noise/outlier")

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
        raw_cc = row.get('country_count', 1)
        country_cnt = 1.0 if pd.isna(raw_cc) else float(raw_cc)
        geo_score = min((country_cnt - 1) / 4.0 + (0.2 if src_country != dst_country else 0.0), 1.0)
        if pd.isna(geo_score):
            geo_score = 0.0
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

        # 10. Peeling-Chain Pattern Evidence
        peel_score = float(row.get('peeling_chain_evidence', 0.0))
        breakdown['peeling_chain_evidence'] = float(np.clip(peel_score, 0.0, 1.0))
        if peel_score >= 0.30:
            peel_chain_id = str(row.get('peeling_chain_id', ''))
            peel_pos = int(row.get('peeling_chain_position', 0))
            peel_len = int(row.get('peeling_chain_length', 0))
            top_reasons.append(
                f"Peeling-chain pattern detected ({peel_chain_id}, hop {peel_pos}/{peel_len}) — "
                "Investigation evidence. Requires further investigation."
            )

        # 11. CoinJoin-Like / Mixing Pattern Evidence
        cj_score = float(row.get('coinjoin_evidence', 0.0))
        breakdown['coinjoin_evidence'] = float(np.clip(cj_score, 0.0, 1.0))
        if cj_score >= 0.30:
            cj_eq_count = int(row.get('coinjoin_equal_output_count', 0))
            cj_eq_ratio = float(row.get('coinjoin_equal_output_ratio', 0.0))
            top_reasons.append(
                f"CoinJoin-like / mixing pattern detected ({cj_eq_count} equal outputs, "
                f"ratio={cj_eq_ratio:.2f}) — Behavioural indicator. Requires further investigation."
            )


        fused_score = sum(breakdown[k] * self.weights.get(k, 0.0) for k in breakdown)
        if pd.isna(fused_score):
            fused_score = 0.0
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
