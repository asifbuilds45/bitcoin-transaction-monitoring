"""
Evidence Timeline Builder Module
SIH Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Constructs chronological evidence timelines from actual network observations, on-chain
transaction confirmations, wallet fund-flows, and detection events.
Strict Rule: Never assert an unverified 'attack sequence' without conclusive proof.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


class EvidenceTimelineBuilder:
    """
    Builds chronological sequence of forensic events across network observations,
    blockchain ledger activity, behavioural clustering, and multi-layer alert generation.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Enriched/scored dataframe containing network and blockchain telemetry.
        """
        self.df = df.copy()
        self.tx_map = {}
        for _, row in self.df.iterrows():
            txid = str(row.get('txid', ''))
            if txid:
                self.tx_map[txid] = row.to_dict()

    def build_transaction_timeline(self, txid: str) -> List[Dict[str, Any]]:
        """
        Build a chronological evidence timeline for a single transaction.

        Args:
            txid: Transaction identifier.

        Returns:
            Chronologically sorted list of event dictionaries.
        """
        txid = str(txid)
        if txid not in self.tx_map:
            return []

        row = self.tx_map[txid]
        events = []

        # 1. Network Observation Event
        net_ts = row.get('network_timestamp')
        src_ip = str(row.get('src_ip', 'Unknown'))
        dst_ip = str(row.get('dst_ip', 'Unknown'))
        src_port = row.get('src_port', 8333)
        dst_port = row.get('dst_port', 8333)
        country = row.get('src_country', 'Unknown')
        asn = row.get('src_asn', 'Unknown')

        if net_ts is not None and str(net_ts).strip() and str(net_ts) != 'nan':
            events.append({
                "timestamp": str(net_ts),
                "event_type": "NETWORK_OBSERVATION",
                "badge": "🌐 Network Observation",
                "entity": f"{src_ip}:{src_port} → {dst_ip}:{dst_port}",
                "description": (
                    f"P2P network broadcast observed. Origin vantage point: {src_ip} "
                    f"(Country: {country}, ASN: {asn}). Associated with transaction broadcast."
                ),
                "severity": "INFO",
                "raw_timestamp": pd.to_datetime(net_ts, errors='coerce')
            })

        # 2. Blockchain Transaction Confirmation Event
        chain_ts = row.get('timestamp')
        btc_amount = float(row.get('total_output_amount_btc', 0.0))
        fee = float(row.get('fee_btc', row.get('fee', 0.0)))
        block_height = row.get('block_height', 'Pending/Mempool')

        if chain_ts is not None and str(chain_ts).strip() and str(chain_ts) != 'nan':
            events.append({
                "timestamp": str(chain_ts),
                "event_type": "TRANSACTION_OBSERVED",
                "badge": "⛓️ Transaction Broadcast",
                "entity": f"TXID: {txid}",
                "description": (
                    f"Transaction confirmed on-chain (Block {block_height}). "
                    f"Transferred volume: {btc_amount:.6f} BTC (Miner fee: {fee:.6f} BTC)."
                ),
                "severity": "INFO",
                "raw_timestamp": pd.to_datetime(chain_ts, errors='coerce')
            })

            # 3. Input Wallet Contribution
            src_wallet = str(row.get('source_wallet', ''))
            if src_wallet and src_wallet != 'Unknown':
                events.append({
                    "timestamp": str(chain_ts),
                    "event_type": "INPUT_FUNDING",
                    "badge": "📥 Input Fund Flow",
                    "entity": f"Wallet: {src_wallet}",
                    "description": f"Source wallet {src_wallet} provided input UTXOs to transaction {txid}.",
                    "severity": "INFO",
                    "raw_timestamp": pd.to_datetime(chain_ts, errors='coerce')
                })

            # 4. Output Wallet Fund Dispersal
            dst_wallet = str(row.get('destination_wallet', ''))
            if dst_wallet and dst_wallet != 'Unknown':
                events.append({
                    "timestamp": str(chain_ts),
                    "event_type": "OUTPUT_DISPERSAL",
                    "badge": "📤 Output Fund Dispersal",
                    "entity": f"Wallet: {dst_wallet}",
                    "description": f"Transaction dispersed {btc_amount:.6f} BTC to destination wallet {dst_wallet}.",
                    "severity": "INFO",
                    "raw_timestamp": pd.to_datetime(chain_ts, errors='coerce')
                })

            # 5. DBSCAN Behavioural Clustering
            cluster_id = row.get('dbscan_cluster')
            is_noise = bool(row.get('dbscan_is_noise', False))
            if cluster_id is not None:
                if is_noise or cluster_id == -1:
                    cluster_desc = "Unsupervised DBSCAN identified transaction as sparse unclustered noise anomaly point (-1)."
                    c_sev = "WARNING"
                else:
                    cluster_desc = f"Unsupervised DBSCAN assigned transaction to behavioral cluster #{cluster_id}."
                    c_sev = "INFO"

                events.append({
                    "timestamp": str(chain_ts),
                    "event_type": "BEHAVIOURAL_CLUSTERING",
                    "badge": "🔍 Behavioural Cluster",
                    "entity": f"DBSCAN: {'Noise (-1)' if is_noise or cluster_id == -1 else f'Cluster #{cluster_id}'}",
                    "description": cluster_desc,
                    "severity": c_sev,
                    "raw_timestamp": pd.to_datetime(chain_ts, errors='coerce')
                })

            # 6. Multi-Layer Investigation Alert
            risk_score = float(row.get('risk_score', 0.0))
            risk_level = str(row.get('risk_level', 'LOW')).upper()
            anomaly_score = float(row.get('anomaly_score', 0.0))
            reasons = str(row.get('risk_reasons', 'Normal traffic characteristics.'))

            if risk_level in ['HIGH', 'CRITICAL'] or row.get('is_anomaly', False):
                events.append({
                    "timestamp": str(chain_ts),
                    "event_type": "INVESTIGATION_ALERT",
                    "badge": f"🚨 {risk_level} Risk Alert",
                    "entity": f"Risk Score: {int(risk_score)}/100 (AI Score: {anomaly_score:.3f})",
                    "description": f"Multi-layer fusion generated priority investigation alert. Key signals: {reasons}",
                    "severity": "CRITICAL" if risk_level == "CRITICAL" else "HIGH",
                    "raw_timestamp": pd.to_datetime(chain_ts, errors='coerce')
                })

        # Sort strictly chronologically by raw_timestamp
        events.sort(key=lambda x: (
            x["raw_timestamp"] is pd.NaT,
            x["raw_timestamp"],
            0 if x["event_type"] == "NETWORK_OBSERVATION" else (
                1 if x["event_type"] == "TRANSACTION_OBSERVED" else (
                    2 if x["event_type"] == "INPUT_FUNDING" else (
                        3 if x["event_type"] == "OUTPUT_DISPERSAL" else (
                            4 if x["event_type"] == "BEHAVIOURAL_CLUSTERING" else 5
                        )
                    )
                )
            )
        ))

        return events

    def build_case_timeline(self, txids: List[str]) -> List[Dict[str, Any]]:
        """
        Build an aggregated chronological evidence timeline across multiple transactions in a case.

        Args:
            txids: List of transaction IDs comprising the case.

        Returns:
            Chronologically merged list of forensic events.
        """
        all_events = []
        seen_events = set()

        for txid in txids:
            tx_events = self.build_transaction_timeline(txid)
            for ev in tx_events:
                # Deduplicate identical network/wallet events
                ev_key = (ev["event_type"], ev["timestamp"], ev["entity"])
                if ev_key not in seen_events:
                    seen_events.add(ev_key)
                    all_events.append(ev)

        # Sort chronologically across the entire case
        all_events.sort(key=lambda x: (
            x["raw_timestamp"] is pd.NaT,
            x["raw_timestamp"],
            0 if x["event_type"] == "NETWORK_OBSERVATION" else (
                1 if x["event_type"] == "TRANSACTION_OBSERVED" else (
                    2 if x["event_type"] == "INPUT_FUNDING" else (
                        3 if x["event_type"] == "OUTPUT_DISPERSAL" else (
                            4 if x["event_type"] == "BEHAVIOURAL_CLUSTERING" else 5
                        )
                    )
                )
            )
        ))

        return all_events
