"""
Graph Analytics & NetworkX / Neo4j Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Constructs and analyzes heterogeneous entity graphs (IPs, TXIDs, Wallets)
using NetworkX, Leiden / Louvain Community Detection, and optional Neo4j graph persistence.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import pandas as pd
import numpy as np

from src.graph.leiden_community import LeidenCommunityDetector
from src.graph.louvain_community import LouvainCommunityDetector
from src.graph.neo4j_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)


class BitcoinGraphEngine:
    """
    Constructs and analyzes heterogeneous entity graphs (IPs, TXIDs, Wallets)
    using NetworkX, Leiden Community Detection, and optional Neo4j persistence.
    """

    def __init__(self, df: pd.DataFrame, use_leiden: bool = True, sync_to_neo4j: bool = False):
        self.df = df.copy(deep=True) if df is not None else pd.DataFrame()
        self.G = nx.DiGraph()
        self._build_graph()

        # Initialize community detector: default to Leiden, fallback to Louvain
        if use_leiden:
            try:
                self.community_detector = LeidenCommunityDetector(self.G)
            except Exception as e:
                logger.warning(f"Leiden init notice: {e}, using Louvain")
                self.community_detector = LouvainCommunityDetector(self.G)
        else:
            self.community_detector = LouvainCommunityDetector(self.G)

        # Backward compatibility aliases
        self.louvain_detector = self.community_detector
        self.community_stats = self.community_detector.community_stats

        # Optional Neo4j synchronization
        self.neo4j_manager = Neo4jGraphManager()
        if sync_to_neo4j and self.neo4j_manager.is_connected:
            self.neo4j_manager.sync_networkx_graph(self.G)

    @property
    def graph(self) -> nx.DiGraph:
        """Alias to self.G for intuitive graph access."""
        return self.G

    def _build_graph(self) -> None:
        """
        Build heterogeneous directed graph:
        - IP -> TXID (BROADCASTS)
        - Source Wallet -> TXID (INPUT_FUNDS)
        - TXID -> Destination Wallet (OUTPUT_FUNDS)
        """
        if self.df.empty:
            return

        for _, row in self.df.iterrows():
            txid = str(row.get('txid', ''))
            src_ip = str(row.get('src_ip', ''))
            src_wallet = str(row.get('source_wallet', ''))
            dst_wallet = str(row.get('destination_wallet', ''))
            btc_amount = float(row.get('total_output_amount_btc', 0.0) or 0.0)
            is_anomaly = bool(row.get('is_anomaly', False))
            risk_level = str(row.get('risk_level', 'UNKNOWN'))

            # Add TXID Node
            if txid:
                self.G.add_node(
                    txid,
                    node_type='TXID',
                    label=txid,
                    amount_btc=btc_amount,
                    is_anomaly=is_anomaly,
                    risk_level=risk_level,
                    timestamp=str(row.get('timestamp', ''))
                )

            # Add IP Node
            if src_ip and src_ip not in self.G:
                self.G.add_node(
                    src_ip,
                    node_type='IP',
                    label=src_ip,
                    country=str(row.get('src_country', 'UNKNOWN')),
                    asn=str(row.get('src_asn', 'UNKNOWN'))
                )

            # Add Wallet Nodes
            if src_wallet and src_wallet not in self.G:
                self.G.add_node(src_wallet, node_type='WALLET', label=src_wallet)
            if dst_wallet and dst_wallet not in self.G:
                self.G.add_node(dst_wallet, node_type='WALLET', label=dst_wallet)

            # Add Directed Edges
            if src_ip and txid:
                self.G.add_edge(src_ip, txid, relationship='BROADCASTS', weight=1.0, btc=btc_amount)
            if src_wallet and txid:
                self.G.add_edge(src_wallet, txid, relationship='INPUT_FUNDS', weight=1.0, btc=btc_amount)
            if txid and dst_wallet:
                self.G.add_edge(txid, dst_wallet, relationship='OUTPUT_FUNDS', weight=1.0, btc=btc_amount)

    def get_graph_summary(self) -> Dict[str, Any]:
        """Compute graph topological statistics."""
        node_types = nx.get_node_attributes(self.G, 'node_type')
        ip_count = sum(1 for t in node_types.values() if t == 'IP')
        tx_count = sum(1 for t in node_types.values() if t == 'TXID')
        wallet_count = sum(1 for t in node_types.values() if t == 'WALLET')

        UG = self.G.to_undirected()
        components = list(nx.connected_components(UG))

        algo_used = getattr(self.community_detector, 'algorithm_used', 'Leiden')

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "ip_nodes": ip_count,
            "tx_nodes": tx_count,
            "wallet_nodes": wallet_count,
            "connected_components_count": len(components),
            "largest_component_size": len(max(components, key=len)) if components else 0,
            "communities_count": len(self.community_stats),
            "louvain_communities_count": len(self.community_stats),
            "community_algorithm": algo_used,
            "neo4j_status": self.neo4j_manager.get_status()
        }

    def get_high_degree_entities(self, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Identify high-degree hub entities across IPs and Wallets."""
        degrees = dict(self.G.degree())
        node_types = nx.get_node_attributes(self.G, 'node_type')

        ip_degrees = [
            {"entity": n, "degree": degrees[n], "type": "IP"}
            for n, t in node_types.items() if t == 'IP'
        ]
        wallet_degrees = [
            {"entity": n, "degree": degrees[n], "type": "WALLET"}
            for n, t in node_types.items() if t == 'WALLET'
        ]

        ip_degrees.sort(key=lambda x: x['degree'], reverse=True)
        wallet_degrees.sort(key=lambda x: x['degree'], reverse=True)

        return {
            "top_ips": ip_degrees[:top_n],
            "top_wallets": wallet_degrees[:top_n]
        }

    def extract_subgraph(
        self,
        center_node: str,
        radius: int = 1,
        max_nodes: int = 60
    ) -> nx.DiGraph:
        """
        Extract cohesive, highly-connected ego subgraph centered around a node.
        Guarantees complete transaction triads (src_wallet -> txid -> dst_wallet + src_ip)
        to prevent dangling leaf nodes, and dynamically expands along fund-flow paths
        to reach the user-requested max_nodes budget.
        """
        if center_node not in self.G:
            return nx.DiGraph()

        import heapq
        sub_nodes = {center_node}
        visited = set()
        queue = [(0, 0, center_node)]

        # Expand along connected paths up to max_nodes
        while queue and len(sub_nodes) < max_nodes:
            dist, _, curr = heapq.heappop(queue)
            if curr in visited:
                continue
            visited.add(curr)

            if dist >= radius and len(sub_nodes) >= max_nodes:
                break

            curr_type = self.G.nodes[curr].get('node_type', '')

            if curr_type == 'WALLET':
                txs = []
                for pred in self.G.predecessors(curr):
                    if self.G.nodes[pred].get('node_type') == 'TXID':
                        txs.append(pred)
                for succ in self.G.successors(curr):
                    if self.G.nodes[succ].get('node_type') == 'TXID':
                        txs.append(succ)

                for tx in txs:
                    if len(sub_nodes) >= max_nodes:
                        break
                    unit_nodes = {tx}
                    for p in self.G.predecessors(tx):
                        unit_nodes.add(p)
                    for s in self.G.successors(tx):
                        unit_nodes.add(s)

                    sub_nodes.update(unit_nodes)
                    for un in unit_nodes:
                        if un not in visited:
                            heapq.heappush(queue, (dist + 1, -self.G.degree(un), un))

            elif curr_type == 'TXID':
                for p in self.G.predecessors(curr):
                    sub_nodes.add(p)
                    if p not in visited:
                        heapq.heappush(queue, (dist + 1, -self.G.degree(p), p))
                for s in self.G.successors(curr):
                    sub_nodes.add(s)
                    if s not in visited:
                        heapq.heappush(queue, (dist + 1, -self.G.degree(s), s))

            elif curr_type == 'IP':
                ip_txs = list(self.G.successors(curr))
                def tx_priority(tx):
                    succs = set(self.G.successors(tx))
                    preds = set(self.G.predecessors(tx))
                    overlap = len((succs | preds) & sub_nodes)
                    return (overlap, self.G.nodes[tx].get('amount_btc', 0))
                ip_txs.sort(key=tx_priority, reverse=True)
                for tx in ip_txs[:6]:
                    if len(sub_nodes) >= max_nodes:
                        break
                    unit_nodes = {tx}
                    for p in self.G.predecessors(tx):
                        unit_nodes.add(p)
                    for s in self.G.successors(tx):
                        unit_nodes.add(s)
                    sub_nodes.update(unit_nodes)
                    for un in unit_nodes:
                        if un not in visited:
                            heapq.heappush(queue, (dist + 1, -self.G.degree(un), un))

        # If sub_nodes exceeds max_nodes, prune lowest-connectivity nodes while strictly preserving center_node
        if len(sub_nodes) > max_nodes:
            cand_g = self.G.subgraph(sub_nodes)
            in_degrees = dict(cand_g.degree())
            in_degrees[center_node] = 999999
            sorted_nodes = sorted(sub_nodes, key=lambda n: in_degrees.get(n, 0), reverse=True)
            sub_nodes = set(sorted_nodes[:max_nodes])

        return self.G.subgraph(sub_nodes).copy()

    def get_node_community_info(self, node_id: str) -> Dict[str, Any]:
        """Retrieve community info for a node."""
        return self.community_detector.get_community_for_node(node_id)
