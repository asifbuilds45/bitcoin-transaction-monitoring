"""
Neo4j Graph Persistence & Cypher Query Manager
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Provides scalable graph database connectivity via Neo4j Bolt driver with Cypher queries.
Designed for air-gapped forensic environments: if a local Neo4j instance is running,
it synchronizes the entity graph and enables Cypher-based path analysis. If Neo4j is offline,
it gracefully falls back to in-memory NetworkX with zero application degradation.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import networkx as nx

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_LIB_AVAILABLE = True
except ImportError:
    NEO4J_LIB_AVAILABLE = False
    Driver = Any


class Neo4jGraphManager:
    """
    Manages Neo4j Bolt connection, node/edge synchronization, and Cypher queries
    with automatic in-memory NetworkX fallback.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "bitcoin_secure_2025")
        self.driver: Optional[Driver] = None
        self.is_connected = False
        self.connection_msg = ""
        self._initialize_driver()

    def _initialize_driver(self) -> None:
        """Attempt connection to local Neo4j instance with short timeout."""
        if not NEO4J_LIB_AVAILABLE:
            self.connection_msg = "Neo4j Python driver not installed; using NetworkX in-memory fallback."
            logger.info(self.connection_msg)
            return

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=2.0,
                max_connection_lifetime=60
            )
            # Verify connectivity with simple query
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            self.is_connected = True
            self.connection_msg = f"Connected to local Neo4j at {self.uri}."
            logger.info(self.connection_msg)
        except Exception as e:
            self.is_connected = False
            self.driver = None
            self.connection_msg = f"Local Neo4j offline ({e}); running on in-memory NetworkX graph engine."
            logger.info(self.connection_msg)

    def sync_networkx_graph(self, G: nx.DiGraph, batch_size: int = 500) -> bool:
        """
        Synchronize a NetworkX DiGraph into Neo4j using Cypher UNWIND batches.
        Returns True if synced, False if fallback.
        """
        if not self.is_connected or self.driver is None or G.number_of_nodes() == 0:
            return False

        try:
            with self.driver.session() as session:
                # Prepare node batches by type
                nodes = []
                for n, attrs in G.nodes(data=True):
                    nodes.append({
                        "id": str(n),
                        "type": str(attrs.get("node_type", "ENTITY")),
                        "label": str(attrs.get("label", n)),
                        "country": str(attrs.get("country", "")),
                        "asn": str(attrs.get("asn", "")),
                        "risk_level": str(attrs.get("risk_level", "LOW")),
                        "is_anomaly": bool(attrs.get("is_anomaly", False)),
                        "amount_btc": float(attrs.get("amount_btc", 0.0))
                    })

                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i:i + batch_size]
                    session.run("""
                        UNWIND $batch AS item
                        MERGE (e:Entity {id: item.id})
                        SET e.type = item.type,
                            e.label = item.label,
                            e.country = item.country,
                            e.asn = item.asn,
                            e.risk_level = item.risk_level,
                            e.is_anomaly = item.is_anomaly,
                            e.amount_btc = item.amount_btc
                    """, batch=batch)

                # Prepare edge batches
                edges = []
                for u, v, attrs in G.edges(data=True):
                    edges.append({
                        "source": str(u),
                        "target": str(v),
                        "rel": str(attrs.get("relationship", "CONNECTED_TO")),
                        "weight": float(attrs.get("weight", 1.0)),
                        "btc": float(attrs.get("btc", 0.0))
                    })

                for i in range(0, len(edges), batch_size):
                    batch = edges[i:i + batch_size]
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (a:Entity {id: item.source})
                        MATCH (b:Entity {id: item.target})
                        MERGE (a)-[r:RELATION {type: item.rel}]->(b)
                        SET r.weight = item.weight,
                            r.btc = item.btc
                    """, batch=batch)

            logger.info(f"Successfully synced {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to Neo4j.")
            return True
        except Exception as e:
            logger.warning(f"Error syncing to Neo4j: {e}")
            return False

    def query_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read Cypher query and return results as a list of dicts."""
        if not self.is_connected or self.driver is None:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Cypher query error: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Return connectivity status information."""
        return {
            "connected": self.is_connected,
            "uri": self.uri,
            "engine": "Neo4j Bolt Graph" if self.is_connected else "NetworkX (In-Memory Fallback)",
            "message": self.connection_msg
        }

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass
