"""
Louvain Community Detection Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

IMPORTANT ARCHITECTURAL RULE:
Louvain is a GRAPH COMMUNITY DETECTION algorithm, NOT an anomaly detector or binary classifier.
It identifies tightly connected clusters of IP, TXID, and Wallet entities.
Community statistics and connectivity indicators are passed into the Multi-Layer Evidence Fusion engine.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple


class LouvainCommunityDetector:
    """
    Graph community detection using Louvain modularity optimization.
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.UG = G.to_undirected()
        self.node_community_map: Dict[str, int] = {}
        self.community_stats: List[Dict[str, Any]] = []
        self._detect_communities()

    def _detect_communities(self) -> None:
        """Run Louvain community partitioning on the graph."""
        if self.UG.number_of_nodes() == 0:
            return

        try:
            # NetworkX built-in louvain communities
            communities = nx.community.louvain_communities(self.UG, seed=42)
        except Exception:
            # Fallback to connected components if louvain fails
            communities = list(nx.connected_components(self.UG))

        node_types = nx.get_node_attributes(self.G, 'node_type')

        for comm_id, node_set in enumerate(communities):
            ip_count = sum(1 for n in node_set if node_types.get(n) == 'IP')
            tx_count = sum(1 for n in node_set if node_types.get(n) == 'TXID')
            wallet_count = sum(1 for n in node_set if node_types.get(n) == 'WALLET')

            # Extract subgraph for internal connectivity
            sub = self.UG.subgraph(node_set)
            internal_edges = sub.number_of_edges()
            node_cnt = len(node_set)
            
            # Density calculation
            max_possible_edges = (node_cnt * (node_cnt - 1)) / 2 if node_cnt > 1 else 1
            density = round(internal_edges / max_possible_edges, 4) if max_possible_edges > 0 else 0.0

            # Descriptor assignment
            if wallet_count >= 10 or ip_count >= 10:
                descriptor = "High-risk behavioural community requiring investigation"
                risk_indicator = "HIGH"
            elif density >= 0.15 and node_cnt >= 5:
                descriptor = "Dense connected transaction community"
                risk_indicator = "MEDIUM"
            else:
                descriptor = "Standard connected community baseline"
                risk_indicator = "LOW"

            comm_info = {
                "community_id": comm_id,
                "node_count": node_cnt,
                "ip_count": ip_count,
                "tx_count": tx_count,
                "wallet_count": wallet_count,
                "internal_edges": internal_edges,
                "density": density,
                "risk_indicator": risk_indicator,
                "descriptor": descriptor
            }

            self.community_stats.append(comm_info)

            for node in node_set:
                self.node_community_map[node] = comm_id

    def get_community_for_node(self, node_id: str) -> Dict[str, Any]:
        """Retrieve community details for a specific node."""
        comm_id = self.node_community_map.get(node_id, -1)
        for c in self.community_stats:
            if c["community_id"] == comm_id:
                return c
        return {
            "community_id": -1,
            "node_count": 0,
            "ip_count": 0,
            "tx_count": 0,
            "wallet_count": 0,
            "descriptor": "Unassigned Node"
        }

    def compute_community_evidence_score(self, txid: str) -> Tuple[float, str]:
        """
        Convert Louvain community properties into a normalized Community Evidence Score (0.0 to 1.0).
        """
        comm = self.get_community_for_node(txid)
        if comm["community_id"] == -1:
            return 0.0, "No community assignment"

        node_cnt = comm["node_count"]
        ip_cnt = comm["ip_count"]
        wallet_cnt = comm["wallet_count"]
        density = comm["density"]

        points = 0.0
        reasons = []

        if wallet_cnt >= 15 or ip_cnt >= 15:
            points += 0.40
            reasons.append(f"Member of large connected community #{comm['community_id']} ({wallet_cnt} wallets, {ip_cnt} IPs).")
        elif wallet_cnt >= 5 or ip_cnt >= 5:
            points += 0.20
            reasons.append(f"Member of multi-entity community #{comm['community_id']} ({wallet_cnt} wallets, {ip_cnt} IPs).")

        if density >= 0.20:
            points += 0.30
            reasons.append(f"High community internal connectivity density ({density:.2f}).")
        elif density >= 0.10:
            points += 0.15

        final_score = float(np.clip(points, 0.0, 1.0))
        reason_str = "; ".join(reasons) if reasons else f"Member of community #{comm['community_id']}."

        return round(final_score, 4), reason_str
