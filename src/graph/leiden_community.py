"""
Leiden Community Detection Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

IMPORTANT ARCHITECTURAL RULE:
The Leiden algorithm is a major upgrade over the Louvain algorithm for community detection.
It guarantees well-connected communities, resolves disconnected sub-communities,
and optimizes modularity much faster on large-scale Bitcoin transaction graphs.
Community statistics and structural indicators feed directly into the Evidence Fusion engine.
"""

import logging
from typing import Dict, Any, List, Set, Tuple
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False


class LeidenCommunityDetector:
    """
    Graph community detection using the Leiden algorithm (guaranteed connected communities,
    improved modularity optimization over Louvain).
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.UG = G.to_undirected()
        self.node_community_map: Dict[str, int] = {}
        self.community_stats: List[Dict[str, Any]] = []
        self.algorithm_used = "Leiden"
        self._detect_communities()

    def _detect_communities(self) -> None:
        """Run Leiden community partitioning on the graph, with fallback to Louvain."""
        if self.UG.number_of_nodes() == 0:
            return

        communities: List[Set[Any]] = []

        if LEIDEN_AVAILABLE:
            try:
                # Map NetworkX nodes to igraph graph
                nodes = list(self.UG.nodes())
                node_idx_map = {n: idx for idx, n in enumerate(nodes)}
                
                edges = [(node_idx_map[u], node_idx_map[v]) for u, v in self.UG.edges()]
                g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)

                partition = leidenalg.find_partition(
                    g_ig,
                    leidenalg.ModularityVertexPartition,
                    seed=42
                )
                
                for cluster in partition:
                    comm_nodes = {nodes[idx] for idx in cluster}
                    communities.append(comm_nodes)
                self.algorithm_used = "Leiden (igraph)"
            except Exception as e:
                logger.warning(f"Leiden detection failed ({e}), falling back to Louvain: {e}")
                communities = []

        if not communities:
            # Fallback to Louvain if Leiden not available or failed
            try:
                communities = list(nx.community.louvain_communities(self.UG, seed=42))
                self.algorithm_used = "Louvain (NetworkX Fallback)"
            except Exception:
                communities = list(nx.connected_components(self.UG))
                self.algorithm_used = "Connected Components (Fallback)"

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
                "descriptor": descriptor,
                "algorithm": self.algorithm_used
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
            "density": 0.0,
            "risk_indicator": "NONE",
            "descriptor": "Unassigned Node",
            "algorithm": self.algorithm_used
        }

    def compute_community_evidence_score(self, txid: str) -> Tuple[float, str]:
        """
        Convert community properties into a normalized Community Evidence Score (0.0 to 1.0).
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
