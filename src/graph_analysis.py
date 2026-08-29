"""
Graph Analytics Engine
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import networkx as nx
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set


class BitcoinGraphEngine:
    """
    Constructs and analyzes multi-modal entity graphs (IPs, TXIDs, Wallets)
    using NetworkX.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.G = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """
        Build a heterogeneous directed graph representing network and on-chain interactions:
        - IP -> TXID (Network Broadcast)
        - Source Wallet -> TXID (Input Funds)
        - TXID -> Destination Wallet (Output Funds)
        """
        # Node and Edge accumulation for fast batch addition
        for _, row in self.df.iterrows():
            txid = str(row['txid'])
            src_ip = str(row['src_ip'])
            src_wallet = str(row['source_wallet'])
            dst_wallet = str(row['destination_wallet'])
            btc_amount = float(row.get('total_output_amount_btc', 0.0))
            is_anomaly = bool(row.get('is_anomaly', False))
            risk_level = str(row.get('risk_level', 'UNKNOWN'))

            # Add TXID node
            self.G.add_node(
                txid,
                node_type='TXID',
                label=txid,
                amount_btc=btc_amount,
                is_anomaly=is_anomaly,
                risk_level=risk_level,
                timestamp=str(row.get('timestamp', ''))
            )

            # Add IP node
            if src_ip not in self.G:
                self.G.add_node(
                    src_ip,
                    node_type='IP',
                    label=src_ip,
                    country=str(row.get('src_country', 'UNKNOWN')),
                    asn=str(row.get('src_asn', 'UNKNOWN'))
                )

            # Add Wallet nodes
            if src_wallet not in self.G:
                self.G.add_node(src_wallet, node_type='WALLET', label=src_wallet)
            if dst_wallet not in self.G:
                self.G.add_node(dst_wallet, node_type='WALLET', label=dst_wallet)

            # Add Directed Edges
            # 1. IP broadcasts TXID
            self.G.add_edge(src_ip, txid, relationship='BROADCASTS', weight=1.0)
            # 2. Source Wallet inputs to TXID
            self.G.add_edge(src_wallet, txid, relationship='INPUT_FUNDS', btc=btc_amount)
            # 3. TXID outputs to Destination Wallet
            self.G.add_edge(txid, dst_wallet, relationship='OUTPUT_FUNDS', btc=btc_amount)

    def get_graph_summary(self) -> Dict[str, Any]:
        """Compute high-level topological statistics of the entity graph."""
        node_types = nx.get_node_attributes(self.G, 'node_type')
        ip_count = sum(1 for t in node_types.values() if t == 'IP')
        tx_count = sum(1 for t in node_types.values() if t == 'TXID')
        wallet_count = sum(1 for t in node_types.values() if t == 'WALLET')

        # Undirected conversion for connected components
        UG = self.G.to_undirected()
        components = list(nx.connected_components(UG))

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "ip_nodes": ip_count,
            "tx_nodes": tx_count,
            "wallet_nodes": wallet_count,
            "connected_components_count": len(components),
            "largest_component_size": len(max(components, key=len)) if components else 0
        }

    def get_high_degree_entities(self, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Identify high-connectivity hubs across IPs and Wallets."""
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
        Extract an ego subgraph centered around an IP, TXID, or Wallet.
        """
        if center_node not in self.G:
            sub = nx.DiGraph()
            return sub

        # Collect neighbors up to radius
        UG = self.G.to_undirected()
        sub_nodes = {center_node}
        current_layer = {center_node}
        
        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                for neighbor in UG.neighbors(node):
                    if neighbor not in sub_nodes:
                        next_layer.add(neighbor)
            sub_nodes.update(next_layer)
            current_layer = next_layer
            if len(sub_nodes) >= max_nodes:
                break

        # Truncate to max_nodes if too large
        sub_nodes = set(list(sub_nodes)[:max_nodes])
        return self.G.subgraph(sub_nodes).copy()

    def filter_subgraph_by_entity(
        self,
        entity_type: str,
        entity_value: str,
        depth: int = 2
    ) -> nx.DiGraph:
        """Filter graph by a specific entity value (e.g. TXID, Wallet ID, or IP)."""
        return self.extract_subgraph(center_node=entity_value, radius=depth)
