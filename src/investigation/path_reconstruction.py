"""
Investigation Path Reconstruction Module
SIH Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Reconstructs bounded, prioritized investigation paths connecting network vantage points,
blockchain transactions, and wallet fund-flows with strict non-attribution semantics.
"""

from typing import List, Dict, Any, Optional, Set
import pandas as pd


class InvestigationPathReconstructor:
    """
    Reconstructs and prioritizes multi-hop investigation paths across network observations,
    on-chain transactions, and wallet entities.
    
    Strict Semantics:
    - IP -> TXID: Network observation associated with transaction broadcast.
    - Wallet -> TXID: On-chain input funding relationship.
    - TXID -> Wallet: On-chain output fund-flow relationship.
    - Never asserts that an IP belongs to a wallet owner.
    """

    def __init__(self, df: pd.DataFrame, graph_engine: Optional[Any] = None):
        """
        Args:
            df: Scored/enriched dataframe containing transaction and network attributes.
            graph_engine: Optional BitcoinGraphEngine containing the NetworkX DiGraph.
        """
        self.df = df.copy()
        # Fast indexed lookups
        self.tx_map = {}
        for _, row in self.df.iterrows():
            txid = str(row.get('txid', ''))
            if txid:
                self.tx_map[txid] = row.to_dict()

        # Build wallet to txid mappings
        self.wallet_to_txids: Dict[str, Set[str]] = {}
        for txid, row in self.tx_map.items():
            src_w = str(row.get('source_wallet', ''))
            dst_w = str(row.get('destination_wallet', ''))
            if src_w:
                self.wallet_to_txids.setdefault(src_w, set()).add(txid)
            if dst_w:
                self.wallet_to_txids.setdefault(dst_w, set()).add(txid)

        self.G = graph_engine.G if graph_engine is not None and hasattr(graph_engine, 'G') else None

    def reconstruct_path(
        self,
        txid: str,
        max_depth: int = 3,
        max_paths: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct bounded investigation paths starting from a selected transaction.

        Path Structure:
        Step 1: Network Observation (Source IP vantage point)
        Step 2: Selected Transaction (TXID)
        Step 3: Input/Output Wallets
        Step 4: Related Transactions connected via those wallets (if depth >= 3)
        Step 5: Downstream / Upstream Wallets (if depth >= 4)

        Args:
            txid: Target transaction ID.
            max_depth: Maximum traversal depth (default 3, bounded between 1 and 5).
            max_paths: Maximum number of prioritized paths to return.

        Returns:
            List of structured path dictionaries, sorted by priority score.
        """
        txid = str(txid)
        if txid not in self.tx_map:
            return []

        primary_tx = self.tx_map[txid]
        src_ip = str(primary_tx.get('src_ip', 'Unknown'))
        dst_ip = str(primary_tx.get('dst_ip', 'Unknown'))
        src_wallet = str(primary_tx.get('source_wallet', 'Unknown'))
        dst_wallet = str(primary_tx.get('destination_wallet', 'Unknown'))
        tx_risk = float(primary_tx.get('risk_score', 0.0))
        tx_amount = float(primary_tx.get('total_output_amount_btc', 0.0))

        # Base elements common to all paths for this transaction
        base_net_step = {
            "step": 1,
            "entity_type": "IP",
            "entity_id": src_ip,
            "label": "Network Observation Source",
            "relationship": "Network observation associated with transaction broadcast",
            "metadata": {
                "destination_ip": dst_ip,
                "port": primary_tx.get('src_port', 8333),
                "country": primary_tx.get('src_country', 'Unknown'),
                "asn": primary_tx.get('src_asn', 'Unknown'),
                "timestamp": str(primary_tx.get('network_timestamp', primary_tx.get('timestamp', '')))
            }
        }

        base_tx_step = {
            "step": 2,
            "entity_type": "TXID",
            "entity_id": txid,
            "label": "Observed Transaction",
            "relationship": "Broadcast on Bitcoin network and recorded on-chain",
            "metadata": {
                "amount_btc": tx_amount,
                "risk_score": tx_risk,
                "risk_level": str(primary_tx.get('risk_level', 'UNKNOWN')),
                "anomaly_score": float(primary_tx.get('anomaly_score', 0.0)),
                "timestamp": str(primary_tx.get('timestamp', ''))
            }
        }

        if max_depth <= 1:
            return [{
                "path_id": 1,
                "priority_score": tx_risk,
                "total_hops": 2,
                "steps": [base_net_step, base_tx_step]
            }]

        candidate_paths = []

        # Depth 2: Explore Input & Output Wallets
        wallets_to_explore = []
        if src_wallet and src_wallet != 'Unknown':
            wallets_to_explore.append((src_wallet, "INPUT_FUNDS", "Input Wallet Entity"))
        if dst_wallet and dst_wallet != 'Unknown':
            wallets_to_explore.append((dst_wallet, "OUTPUT_FUNDS", "Destination Wallet Entity"))

        if not wallets_to_explore:
            return [{
                "path_id": 1,
                "priority_score": tx_risk,
                "total_hops": 2,
                "steps": [base_net_step, base_tx_step]
            }]

        path_idx = 1
        for wallet_addr, rel_type, wallet_label in wallets_to_explore:
            wallet_step = {
                "step": 3,
                "entity_type": "WALLET",
                "entity_id": wallet_addr,
                "label": wallet_label,
                "relationship": "On-chain input fund contribution" if rel_type == "INPUT_FUNDS" else "On-chain output fund dispersal",
                "metadata": {
                    "wallet_address": wallet_addr,
                    "direction": "Input" if rel_type == "INPUT_FUNDS" else "Output"
                }
            }

            if max_depth == 2:
                candidate_paths.append({
                    "path_id": path_idx,
                    "priority_score": tx_risk + (5.0 if rel_type == "OUTPUT_FUNDS" else 0.0),
                    "total_hops": 3,
                    "steps": [base_net_step, base_tx_step, wallet_step]
                })
                path_idx += 1
                continue

            # Depth >= 3: Explore connected transactions via this wallet
            related_txs = [
                t for t in self.wallet_to_txids.get(wallet_addr, set())
                if t != txid and t in self.tx_map
            ]

            if not related_txs:
                candidate_paths.append({
                    "path_id": path_idx,
                    "priority_score": tx_risk,
                    "total_hops": 3,
                    "steps": [base_net_step, base_tx_step, wallet_step]
                })
                path_idx += 1
                continue

            # Sort related transactions by risk score (descending) then amount
            related_txs.sort(
                key=lambda t: (
                    float(self.tx_map[t].get('risk_score', 0)),
                    float(self.tx_map[t].get('total_output_amount_btc', 0))
                ),
                reverse=True
            )

            for next_txid in related_txs[:3]:  # Top 3 connected transactions per wallet
                next_tx_data = self.tx_map[next_txid]
                next_risk = float(next_tx_data.get('risk_score', 0))
                next_amount = float(next_tx_data.get('total_output_amount_btc', 0))

                next_tx_step = {
                    "step": 4,
                    "entity_type": "RELATED_TXID",
                    "entity_id": next_txid,
                    "label": "Connected Transaction",
                    "relationship": f"Connected via shared wallet {wallet_addr}",
                    "metadata": {
                        "amount_btc": next_amount,
                        "risk_score": next_risk,
                        "risk_level": str(next_tx_data.get('risk_level', 'UNKNOWN')),
                        "timestamp": str(next_tx_data.get('timestamp', ''))
                    }
                }

                if max_depth == 3:
                    score = tx_risk * 0.6 + next_risk * 0.4 + (10.0 if next_risk >= 70 else 0.0)
                    candidate_paths.append({
                        "path_id": path_idx,
                        "priority_score": round(score, 2),
                        "total_hops": 4,
                        "steps": [base_net_step, base_tx_step, wallet_step, next_tx_step]
                    })
                    path_idx += 1
                    continue

                # Depth >= 4: Explore counterpart wallet on the connected transaction
                downstream_w = str(next_tx_data.get('destination_wallet', ''))
                upstream_w = str(next_tx_data.get('source_wallet', ''))
                counterpart_w = downstream_w if downstream_w != wallet_addr and downstream_w else upstream_w

                if counterpart_w and counterpart_w != 'Unknown' and counterpart_w != wallet_addr:
                    counterpart_step = {
                        "step": 5,
                        "entity_type": "CONNECTED_WALLET",
                        "entity_id": counterpart_w,
                        "label": "Downstream/Connected Wallet",
                        "relationship": "Subsequent fund-flow destination / origin",
                        "metadata": {
                            "wallet_address": counterpart_w,
                            "via_txid": next_txid
                        }
                    }
                    score = tx_risk * 0.5 + next_risk * 0.4 + 15.0
                    candidate_paths.append({
                        "path_id": path_idx,
                        "priority_score": round(score, 2),
                        "total_hops": 5,
                        "steps": [base_net_step, base_tx_step, wallet_step, next_tx_step, counterpart_step]
                    })
                else:
                    score = tx_risk * 0.6 + next_risk * 0.4
                    candidate_paths.append({
                        "path_id": path_idx,
                        "priority_score": round(score, 2),
                        "total_hops": 4,
                        "steps": [base_net_step, base_tx_step, wallet_step, next_tx_step]
                    })
                path_idx += 1

        # Sort candidate paths by priority score descending
        candidate_paths.sort(key=lambda p: p['priority_score'], reverse=True)
        # Re-index path_id
        for i, p in enumerate(candidate_paths):
            p["path_id"] = i + 1

        return candidate_paths[:max_paths]
