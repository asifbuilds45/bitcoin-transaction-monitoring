"""
Investigation Knowledge Store
SIH Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Maintains a comprehensive, structured local knowledge base of current investigation results:
transactions, network telemetry, correlation, AI/ML anomalies, DBSCAN clusters, risk scores,
graph topology, investigation paths, evidence timelines, and deduplicated cases.
Strict Rule: Contains only actual outputs from the current investigation session. No fabricated data.
"""

from typing import Dict, Any, List, Optional, Set
import pandas as pd
import numpy as np


class InvestigationKnowledgeStore:
    """
    Structured in-memory knowledge store indexing all evidence and findings from the current investigation.
    Provides fast O(1) lookups for evidence retrieval and offline assistant synthesis.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge store.
        
        Args:
            context: The investigation context dictionary returned by run_investigation_pipeline.
        """
        self.transactions: Dict[str, Dict[str, Any]] = {}
        self.cases: Dict[str, Dict[str, Any]] = {}
        self.wallets: Dict[str, Dict[str, Any]] = {}
        self.ips: Dict[str, Dict[str, Any]] = {}
        self.global_summary: Dict[str, Any] = {}
        self.is_populated = False

        if context is not None:
            self.populate(context)

    def populate(self, context: Dict[str, Any]) -> None:
        """
        Extract and index all entities and relationships from the investigation context.
        """
        scored_df: pd.DataFrame = context.get("scored_df", pd.DataFrame())
        graph_engine = context.get("graph_engine")
        case_grouper = context.get("case_grouper")
        path_reconstructor = context.get("path_reconstructor")
        timeline_builder = context.get("timeline_builder")
        kpis = context.get("kpis", {})

        if scored_df.empty:
            return

        self.transactions.clear()
        self.cases.clear()
        self.wallets.clear()
        self.ips.clear()

        # 1. Index Cases
        case_list = case_grouper.get_cases() if case_grouper else []
        tx_to_case_id: Dict[str, str] = {}

        for c in case_list:
            c_id = c["case_id"]
            self.cases[c_id] = {
                "case_id": c_id,
                "title": c.get("title", "Coordinated Activity"),
                "severity": c.get("severity", "LOW"),
                "max_risk_score": float(c.get("max_risk_score", 0.0)),
                "transaction_count": int(c.get("transaction_count", 0)),
                "primary_txid": c.get("primary_txid", ""),
                "txids": c.get("txids", []),
                "wallets": c.get("wallets", []),
                "ips": c.get("ips", []),
                "total_btc_volume": float(c.get("total_btc_volume", 0.0)),
                "earliest_timestamp": str(c.get("earliest_timestamp", "")),
                "latest_timestamp": str(c.get("latest_timestamp", "")),
                "grouping_reasons": c.get("grouping_reasons", [])
            }
            for t in c.get("txids", []):
                tx_to_case_id[str(t)] = c_id

        # 2. Index Transactions
        for _, row in scored_df.iterrows():
            txid = str(row.get("txid", "")).strip()
            if not txid:
                continue

            src_wallet = str(row.get("source_wallet", "Unknown")).strip()
            dst_wallet = str(row.get("destination_wallet", "Unknown")).strip()
            src_ip = str(row.get("src_ip", "Unknown")).strip()
            dst_ip = str(row.get("dst_ip", "Unknown")).strip()

            reasons_raw = str(row.get("risk_reasons", ""))
            reasons_list = [r.strip() for r in reasons_raw.split(";") if r.strip()]

            # Determine case mapping
            case_id = tx_to_case_id.get(txid, "N/A (Isolated)")

            tx_record = {
                # Transaction Core
                "txid": txid,
                "case_id": case_id,
                "timestamp": str(row.get("timestamp", "")),
                "block_height": int(row.get("block_height", 0)) if pd.notna(row.get("block_height")) else 0,
                "amount_btc": float(row.get("total_output_amount_btc", 0.0)),
                "fee_btc": float(row.get("fee_btc", row.get("fee", 0.0))),
                "script_type": str(row.get("script_type", "P2PKH")),
                "num_inputs": int(row.get("num_inputs", 1)),
                "num_outputs": int(row.get("num_outputs", 1)),
                "input_addresses": row.get("input_addresses", src_wallet),
                "output_addresses": row.get("output_addresses", dst_wallet),

                # Network & GeoIP
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": int(row.get("src_port", 8333)) if pd.notna(row.get("src_port")) else 8333,
                "dst_port": int(row.get("dst_port", 8333)) if pd.notna(row.get("dst_port")) else 8333,
                "protocol": str(row.get("protocol", "TCP")),
                "network_timestamp": str(row.get("network_timestamp", row.get("timestamp", ""))),
                "packet_count": int(row.get("packet_count", 0)),
                "bytes_transferred": int(row.get("bytes_transferred", 0)),
                "country": str(row.get("geoip_src_country") if pd.notna(row.get("geoip_src_country")) and str(row.get("geoip_src_country")).strip().lower() not in ("unknown", "none", "nan", "") else row.get("src_country", "Unknown")),
                "country_code": str(row.get("geoip_src_country_code") if pd.notna(row.get("geoip_src_country_code")) and str(row.get("geoip_src_country_code")).strip().lower() not in ("unknown", "none", "nan", "") else row.get("src_country", "Unknown")),
                "asn": str(row.get("geoip_src_asn") if pd.notna(row.get("geoip_src_asn")) and str(row.get("geoip_src_asn")).strip().lower() not in ("unknown", "none", "nan", "") else row.get("src_asn", "Unknown")),
                "asn_org": str(row.get("geoip_src_org", "Unknown")) if pd.notna(row.get("geoip_src_org")) else "Unknown",

                # Correlation
                "correlation_confidence": float(row.get("correlation_confidence", 0.85)),
                "correlation_confidence_pct": str(row.get("correlation_confidence_pct", "85%")),

                # Machine Learning & Clustering
                "anomaly_score": float(row.get("anomaly_score", 0.0)),
                "is_anomaly": bool(row.get("is_anomaly", False)),
                "dbscan_cluster": int(row.get("dbscan_cluster", -1)) if pd.notna(row.get("dbscan_cluster")) else -1,
                "dbscan_is_noise": bool(row.get("dbscan_is_noise", False)),

                # Risk & Priority
                "risk_score": float(row.get("risk_score", 0.0)),
                "risk_level": str(row.get("risk_level", "LOW")).upper(),
                "risk_reasons": reasons_list,

                # Graph & Behavioral
                "source_wallet": src_wallet,
                "destination_wallet": dst_wallet,
                "wallet_degree": int(row.get("wallet_degree", 1)) if pd.notna(row.get("wallet_degree")) else 1,
                "community_id": int(row.get("community_id", -1)) if pd.notna(row.get("community_id")) else -1,
                "velocity_24h": float(row.get("transaction_frequency_24h", 1.0)),
                "time_gap_min": float(row.get("avg_time_gap_min", 60.0)),

                # Behavioural Pattern Detection
                "peeling_chain_pattern": {
                    "detected": bool(row.get("peeling_chain_detected", False)),
                    "evidence_score": float(row.get("peeling_chain_evidence", 0.0)),
                    "chain_id": str(row.get("peeling_chain_id", "")),
                    "position": int(row.get("peeling_chain_position", 0)),
                    "chain_length": int(row.get("peeling_chain_length", 0)),
                    "reasons": str(row.get("peeling_chain_reasons", "")),
                },
                "coinjoin_pattern": {
                    "detected": bool(row.get("coinjoin_detected", False)),
                    "evidence_score": float(row.get("coinjoin_evidence", 0.0)),
                    "equal_output_count": int(row.get("coinjoin_equal_output_count", 0)),
                    "equal_output_ratio": float(row.get("coinjoin_equal_output_ratio", 0.0)),
                    "equal_amount_btc": float(row.get("coinjoin_equal_amount_btc", 0.0)),
                    "reasons": str(row.get("coinjoin_reasons", "")),
                },
            }

            self.transactions[txid] = tx_record

            # 3. Index Wallets
            for w, role in [(src_wallet, "source"), (dst_wallet, "destination")]:
                if w and w != "Unknown":
                    if w not in self.wallets:
                        self.wallets[w] = {
                            "wallet": w,
                            "txids": set(),
                            "associated_ips": set(),
                            "counterpart_wallets": set(),
                            "total_btc": 0.0,
                            "max_risk": 0.0,
                            "risk_levels": set()
                        }
                    self.wallets[w]["txids"].add(txid)
                    if src_ip != "Unknown":
                        self.wallets[w]["associated_ips"].add(src_ip)
                    counterpart = dst_wallet if role == "source" else src_wallet
                    if counterpart and counterpart != "Unknown" and counterpart != w:
                        self.wallets[w]["counterpart_wallets"].add(counterpart)
                    self.wallets[w]["total_btc"] += tx_record["amount_btc"]
                    self.wallets[w]["max_risk"] = max(self.wallets[w]["max_risk"], tx_record["risk_score"])
                    self.wallets[w]["risk_levels"].add(tx_record["risk_level"])

            # 4. Index IPs
            if src_ip and src_ip != "Unknown":
                if src_ip not in self.ips:
                    self.ips[src_ip] = {
                        "ip": src_ip,
                        "country": tx_record["country"],
                        "asn": tx_record["asn"],
                        "asn_org": tx_record["asn_org"],
                        "txids": set(),
                        "associated_wallets": set(),
                        "total_packets": 0,
                        "max_risk": 0.0
                    }
                self.ips[src_ip]["txids"].add(txid)
                if src_wallet != "Unknown":
                    self.ips[src_ip]["associated_wallets"].add(src_wallet)
                if dst_wallet != "Unknown":
                    self.ips[src_ip]["associated_wallets"].add(dst_wallet)
                self.ips[src_ip]["total_packets"] += tx_record["packet_count"]
                self.ips[src_ip]["max_risk"] = max(self.ips[src_ip]["max_risk"], tx_record["risk_score"])

        # 5. Global KPIs
        total_tx = len(scored_df)
        detected_anom = int(scored_df["is_anomaly"].sum()) if "is_anomaly" in scored_df.columns else 0
        crit_alerts = int((scored_df["risk_level_normalized"] == "CRITICAL").sum()) if "risk_level_normalized" in scored_df.columns else 0
        high_alerts = int((scored_df["risk_level_normalized"] == "HIGH").sum()) if "risk_level_normalized" in scored_df.columns else 0

        self.global_summary = {
            "total_transactions": total_tx,
            "detected_anomalies": detected_anom,
            "critical_alerts": crit_alerts,
            "high_alerts": high_alerts,
            "total_cases": len(self.cases),
            "unique_wallets": len(self.wallets),
            "unique_ips": len(self.ips),
            "mean_correlation_confidence": kpis.get("mean_conf_pct", 85)
        }

        # Keep references to specialized engines
        self.path_reconstructor = path_reconstructor
        self.timeline_builder = timeline_builder
        self.is_populated = True

    def get_transaction(self, txid: str) -> Optional[Dict[str, Any]]:
        """Lookup transaction by TXID (case-insensitive)."""
        txid_clean = str(txid).strip().upper()
        for k, v in self.transactions.items():
            if k.upper() == txid_clean:
                return v
        return None

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Lookup investigation case by ID (e.g., CASE-001)."""
        c_clean = str(case_id).strip().upper()
        for k, v in self.cases.items():
            if k.upper() == c_clean:
                return v
        return None

    def get_cases(self) -> List[Dict[str, Any]]:
        """Retrieve all cases ordered by maximum risk score."""
        return sorted(self.cases.values(), key=lambda c: (c["max_risk_score"], c["transaction_count"]), reverse=True)

    def get_kpis(self) -> Dict[str, Any]:
        """Return global KPIs dictionary."""
        d = self.global_summary.copy()
        d["total_tx"] = d.get("total_transactions", len(self.transactions))
        return d

    def get_wallet(self, wallet: str) -> Optional[Dict[str, Any]]:
        """Lookup wallet details and associated activity."""
        w_clean = str(wallet).strip().upper()
        for k, v in self.wallets.items():
            if k.upper() == w_clean:
                res = v.copy()
                res["txids"] = sorted(list(res["txids"]))
                res["associated_ips"] = sorted(list(res["associated_ips"]))
                res["counterpart_wallets"] = sorted(list(res["counterpart_wallets"]))
                res["risk_levels"] = sorted(list(res["risk_levels"]))
                return res
        return None

    def get_ip(self, ip: str) -> Optional[Dict[str, Any]]:
        """Lookup IP details and associated activity."""
        ip_clean = str(ip).strip()
        for k, v in self.ips.items():
            if k == ip_clean:
                res = v.copy()
                res["txids"] = sorted(list(res["txids"]))
                res["associated_wallets"] = sorted(list(res["associated_wallets"]))
                return res
        return None

    def get_top_anomalies(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top anomalous transactions ordered by risk score."""
        sorted_txs = sorted(self.transactions.values(), key=lambda t: t["risk_score"], reverse=True)
        return sorted_txs[:top_n]

    def get_top_cases(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top prioritized cases ordered by maximum risk score."""
        sorted_cases = sorted(self.cases.values(), key=lambda c: (c["max_risk_score"], c["transaction_count"]), reverse=True)
        return sorted_cases[:top_n]
