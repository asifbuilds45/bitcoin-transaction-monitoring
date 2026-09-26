"""
Evidence Retrieval Engine
SIH Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Performs targeted, evidence-grounded information retrieval from the InvestigationKnowledgeStore.
Understands investigator questions, extracts entities and intents, resolves conversational context,
and builds small, auditable evidence packages without dumping complete datasets into prompts.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from src.assistant.investigation_knowledge import InvestigationKnowledgeStore


@dataclass
class RetrievedEvidence:
    """Encapsulates retrieved factual evidence for grounding answers and user auditability."""
    intent: str
    target_entity: Optional[str] = None
    target_type: Optional[str] = None  # 'TXID', 'CASE', 'WALLET', 'IP', 'GLOBAL'
    evidence_items: List[str] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict)
    is_supported: bool = True
    unsupported_reason: Optional[str] = None


class EvidenceRetriever:
    """
    Intelligent query analysis and evidence extraction layer.
    Extracts entities, matches investigation intents, and extracts precise evidence.
    """

    def __init__(self, knowledge_store: InvestigationKnowledgeStore):
        self.store = knowledge_store

    def retrieve(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RetrievedEvidence:
        """
        Analyze investigator query and retrieve relevant forensic evidence.
        """
        q = query.strip()
        q_lower = q.lower()
        active_txid = context.get("active_txid") if context else None
        active_case_id = context.get("active_case_id") if context else None

        # 1. Entity Extraction
        txid_match = self._extract_txid(q)
        case_match = self._extract_case_id(q)
        wallet_match = self._extract_wallet(q)
        ip_match = self._extract_ip(q)

        # 2. Check for Unsupported / Out-of-Scope Queries (Anti-Hallucination)
        unsupported_keywords = [
            "who owns", "owner of", "real identity", "person name", "home address",
            "passport", "kyc", "phone number", "ceo of bitcoin", "satoshi identity",
            "weather", "sports", "recipe", "stock price"
        ]
        if any(kw in q_lower for kw in unsupported_keywords):
            return RetrievedEvidence(
                intent="UNSUPPORTED",
                is_supported=False,
                unsupported_reason=(
                    "The current investigation dataset contains cryptographic and network telemetry only. "
                    "It does not contain real-world identity, personal ownership, or off-chain KYC records. "
                    "Per strict non-attribution policy, wallet ownership cannot be fabricated."
                )
            )

        # 3. Resolve Target Entity using query or conversational memory context
        target_txid = txid_match or (active_txid if ("it" in q_lower.split() or "this" in q_lower or "the transaction" in q_lower) else None)
        target_case = case_match or (active_case_id if ("this case" in q_lower or "the case" in q_lower or "these alerts" in q_lower) else None)

        # If question asks about TXID
        if target_txid:
            tx_data = self.store.get_transaction(target_txid)
            if tx_data:
                return self._retrieve_tx_evidence(q_lower, tx_data)
            elif txid_match:
                return RetrievedEvidence(
                    intent="NOT_FOUND",
                    target_entity=target_txid,
                    target_type="TXID",
                    is_supported=False,
                    unsupported_reason=f"Transaction '{target_txid}' was not found in the current monitored dataset."
                )

        # If question asks about Case
        if target_case:
            case_data = self.store.get_case(target_case)
            if case_data:
                return self._retrieve_case_evidence(q_lower, case_data)
            elif case_match:
                return RetrievedEvidence(
                    intent="NOT_FOUND",
                    target_entity=target_case,
                    target_type="CASE",
                    is_supported=False,
                    unsupported_reason=f"Case '{target_case}' was not found in the current investigation."
                )

        # If question asks about Wallet
        if wallet_match:
            wallet_data = self.store.get_wallet(wallet_match)
            if wallet_data:
                return self._retrieve_wallet_evidence(q_lower, wallet_data)
            else:
                return RetrievedEvidence(
                    intent="NOT_FOUND",
                    target_entity=wallet_match,
                    target_type="WALLET",
                    is_supported=False,
                    unsupported_reason=f"Wallet '{wallet_match}' was not found in the current investigation dataset."
                )

        # If question asks about IP
        if ip_match:
            ip_data = self.store.get_ip(ip_match)
            if ip_data:
                return self._retrieve_ip_evidence(q_lower, ip_data)
            else:
                return RetrievedEvidence(
                    intent="NOT_FOUND",
                    target_entity=ip_match,
                    target_type="IP",
                    is_supported=False,
                    unsupported_reason=f"IP address '{ip_match}' was not observed in the current network telemetry dataset."
                )

        # General Investigation Summaries / Top Alerts
        if any(term in q_lower for term in ["top alert", "top anomalies", "highest risk", "summary", "overall", "findings", "status"]):
            return self._retrieve_global_evidence(q_lower)

        # Default fallback: unsupported query within current investigation scope
        return RetrievedEvidence(
            intent="UNKNOWN_SCOPE",
            is_supported=False,
            unsupported_reason=(
                "I don't have sufficient evidence in the current investigation data to answer that specific query. "
                "Please specify a valid TXID (e.g. TX10000342), Case ID (e.g. CASE-001), Wallet ID, or ask for a summary of top alerts."
            )
        )

    # -------------------------------------------------------------------------
    # ENTITY REGEX EXTRACTORS
    # -------------------------------------------------------------------------
    def _extract_txid(self, text: str) -> Optional[str]:
        # Match TX10000137, TX001, etc.
        m = re.search(r'\b(TX\d{4,8})\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Look for partial matches in knowledge store
        for k in self.store.transactions.keys():
            if re.search(r'\b' + re.escape(k) + r'\b', text, re.IGNORECASE):
                return k
        return None

    def _extract_case_id(self, text: str) -> Optional[str]:
        # Match CASE-001, CASE #001, CASE 001, CASE-1
        m = re.search(r'\bCASE[\s\-_#]*0*(\d+)\b', text, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            return f"CASE-{num:03d}"
        return None

    def _extract_wallet(self, text: str) -> Optional[str]:
        m = re.search(r'\b(W\d{4,8})\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        for w in self.store.wallets.keys():
            if re.search(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE):
                return w
        return None

    def _extract_ip(self, text: str) -> Optional[str]:
        m = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', text)
        if m:
            return m.group(1)
        return None

    # -------------------------------------------------------------------------
    # TRANSACTION EVIDENCE BUILDER
    # -------------------------------------------------------------------------
    def _retrieve_tx_evidence(self, q: str, tx: Dict[str, Any]) -> RetrievedEvidence:
        txid = tx["txid"]
        base_items = [
            f"TXID: {txid}",
            f"Unified Risk Score: {int(tx['risk_score'])}/100 ({tx['risk_level']} Risk)",
            f"AI Anomaly Score: {tx['anomaly_score']:.3f} (Flagged: {tx['is_anomaly']})",
            f"Assigned Case: {tx['case_id']}",
            f"Transferred Amount: {tx['amount_btc']:.6f} BTC (Fee: {tx['fee_btc']:.6f} BTC)",
            f"Vantage Point IP: {tx['src_ip']} (Country: {tx['country']}, ASN: {tx['asn']})",
            f"Source Wallet: {tx['source_wallet']}",
            f"Destination Wallet: {tx['destination_wallet']}"
        ]

        # Append behavioural pattern evidence if detected
        peel = tx.get("peeling_chain_pattern", {})
        if peel.get("detected"):
            base_items.append(
                f"Behavioural Pattern — Peeling-Chain Detected: Chain {peel.get('chain_id', '')}, "
                f"Hop {peel.get('position', 0)}/{peel.get('chain_length', 0)}, "
                f"Evidence Score: {peel.get('evidence_score', 0.0):.3f}. "
                "Investigation evidence. Requires further investigation."
            )

        cj = tx.get("coinjoin_pattern", {})
        if cj.get("detected"):
            base_items.append(
                f"Behavioural Pattern — CoinJoin-Like / Mixing Pattern Detected: "
                f"{cj.get('equal_output_count', 0)} equal outputs at ≈{cj.get('equal_amount_btc', 0.0):.6f} BTC "
                f"(ratio={cj.get('equal_output_ratio', 0.0):.2f}), "
                f"Evidence Score: {cj.get('evidence_score', 0.0):.3f}. "
                "Behavioural indicator. Requires further investigation."
            )

        # Intent 1: Risk Score only
        if "risk score" in q and not any(k in q for k in ["why", "evidence", "reason"]):
            return RetrievedEvidence(
                intent="TX_RISK_SCORE",
                target_entity=txid,
                target_type="TXID",
                evidence_items=[
                    f"TXID: {txid}",
                    f"Risk Score: {int(tx['risk_score'])}/100",
                    f"Risk Tier: {tx['risk_level']}",
                    f"Correlation Confidence: {tx['correlation_confidence_pct']}"
                ],
                structured_data=tx
            )

        # Intent 2: Anomaly Evidence / Why Critical / Why High Risk
        if any(k in q for k in ["why", "critical", "high risk", "anomaly evidence", "evidence", "reason", "signals", "factor", "factors", "explain", "risk factor"]):
            evidence = base_items.copy()
            c_noise = tx.get('hdbscan_is_noise', tx.get('dbscan_is_noise', False))
            c_id = tx.get('hdbscan_cluster', tx.get('dbscan_cluster', -1))
            evidence.append(f"HDBSCAN Cluster: {'Noise Outlier (-1)' if c_noise else f'Cluster #{c_id}'}")
            evidence.append(f"24h Transaction Velocity: {tx['velocity_24h']:.1f} tx/day")
            evidence.append(f"Inter-transaction Time Gap: {tx['time_gap_min']:.2f} min")
            for r in tx["risk_reasons"]:
                evidence.append(f"Forensic Evidence Signal: {r}")
            return RetrievedEvidence(
                intent="TX_EXPLANATION",
                target_entity=txid,
                target_type="TXID",
                evidence_items=evidence,
                structured_data=tx
            )

        # Intent 3: Connected Wallets
        if "wallet" in q:
            evidence = [
                f"TXID: {txid}",
                f"Source Wallet: {tx['source_wallet']}",
                f"Destination Wallet: {tx['destination_wallet']}",
                f"Wallet Degree Centrality: {tx['wallet_degree']} links in graph",
                f"Graph Community: #{tx['community_id']}"
            ]
            return RetrievedEvidence(
                intent="TX_WALLETS",
                target_entity=txid,
                target_type="TXID",
                evidence_items=evidence,
                structured_data=tx
            )

        # Intent 4: Related Transactions
        if any(k in q for k in ["related transaction", "related tx", "connected transaction"]):
            src_w_info = self.store.get_wallet(tx['source_wallet'])
            dst_w_info = self.store.get_wallet(tx['destination_wallet'])
            all_related = set()
            if src_w_info:
                all_related.update(src_w_info["txids"])
            if dst_w_info:
                all_related.update(dst_w_info["txids"])
            all_related.discard(txid)

            evidence = [
                f"TXID: {txid}",
                f"Related Transactions Found: {len(all_related)} transactions connected via shared source/destination wallets",
                f"Connected TXIDs: {', '.join(sorted(list(all_related))[:6]) if all_related else 'None found directly in local dataset'}",
                f"Shared Source Wallet: {tx['source_wallet']}",
                f"Shared Destination Wallet: {tx['destination_wallet']}"
            ]
            return RetrievedEvidence(
                intent="TX_RELATED",
                target_entity=txid,
                target_type="TXID",
                evidence_items=evidence,
                structured_data=tx
            )

        # Intent 5: Temporal Timeline / Preceding / Following
        if any(k in q for k in ["before", "after", "preceding", "following", "timeline", "happened", "when"]):
            timeline_builder = getattr(self.store, 'timeline_builder', None)
            events = timeline_builder.build_transaction_timeline(txid) if timeline_builder else []
            evidence = base_items[:4]
            evidence.append(f"Recorded Transaction Timestamp: {tx['timestamp']}")
            evidence.append(f"Network Broadcast Timestamp: {tx['network_timestamp']}")
            for ev in events[:6]:
                evidence.append(f"Timeline Event [{ev['timestamp']}]: {ev['badge']} - {ev['entity']} ({ev['description']})")

            return RetrievedEvidence(
                intent="TX_TIMELINE",
                target_entity=txid,
                target_type="TXID",
                evidence_items=evidence,
                structured_data={"tx": tx, "events": events}
            )

        # Intent 6: Investigation Path
        if any(k in q for k in ["path", "trail", "flow", "reconstruction", "hop"]):
            path_recon = getattr(self.store, 'path_reconstructor', None)
            paths = path_recon.reconstruct_path(txid, max_depth=3) if path_recon else []
            evidence = base_items[:4]
            if paths:
                p0 = paths[0]
                evidence.append(f"Primary Investigation Path (#{p0['path_id']}): Priority Score {p0['priority_score']:.1f}, {p0['total_hops']} Hops")
                for s in p0['steps']:
                    evidence.append(f"Step {s['step']} [{s['entity_type']}]: {s['entity_id']} - {s['relationship']}")
            return RetrievedEvidence(
                intent="TX_PATH",
                target_entity=txid,
                target_type="TXID",
                evidence_items=evidence,
                structured_data={"tx": tx, "paths": paths}
            )

        # Intent 7: Recommendations / What to examine next
        if any(k in q for k in ["examine next", "recommend", "next steps", "suggested action", "what to do next", "investigate next"]):
            evidence = base_items.copy()
            evidence.append(f"Recommended Action 1: Trace fund flow from destination wallet {tx['destination_wallet']}.")
            evidence.append(f"Recommended Action 2: Inspect other transactions broadcasting from network vantage IP {tx['src_ip']}.")
            evidence.append(f"Recommended Action 3: Review all transactions clustered inside {tx['case_id']}.")
            return RetrievedEvidence(
                intent="TX_RECOMMENDATION",
                target_entity=txid,
                target_type="TXID",
                evidence_items=evidence,
                structured_data=tx
            )

        # Default Transaction Overview
        return RetrievedEvidence(
            intent="TX_OVERVIEW",
            target_entity=txid,
            target_type="TXID",
            evidence_items=base_items + [f"Forensic Evidence Signals: {'; '.join(tx['risk_reasons'])}"],
            structured_data=tx
        )

    # -------------------------------------------------------------------------
    # CASE EVIDENCE BUILDER
    # -------------------------------------------------------------------------
    def _retrieve_case_evidence(self, q: str, case: Dict[str, Any]) -> RetrievedEvidence:
        c_id = case["case_id"]
        base_items = [
            f"Case ID: {c_id}",
            f"Case Title: {case['title']}",
            f"Severity Tier: {case['severity']} (Max Risk Score: {int(case['max_risk_score'])}/100)",
            f"Transaction Count: {case['transaction_count']} transactions",
            f"Primary Anomaly TXID: {case['primary_txid']}",
            f"Connected Wallets: {len(case['wallets'])} unique addresses",
            f"Network Vantage IPs: {len(case['ips'])} observation points",
            f"Total BTC Transferred: {case['total_btc_volume']:.4f} BTC",
            f"Activity Span: From {case['earliest_timestamp']} to {case['latest_timestamp']}"
        ]

        # Intent 1: Case Grouping Reasons
        if any(k in q for k in ["why", "group", "grouped", "reason", "rationale"]):
            evidence = base_items.copy()
            for r in case["grouping_reasons"]:
                evidence.append(f"Grouping Evidence Rationale: {r}")
            return RetrievedEvidence(
                intent="CASE_GROUPING_RATIONALE",
                target_entity=c_id,
                target_type="CASE",
                evidence_items=evidence,
                structured_data=case
            )

        # Intent 2: Case Transactions
        if any(k in q for k in ["transaction", "txid", "txs", "list"]):
            evidence = base_items.copy()
            evidence.append(f"Member Transactions: {', '.join(case['txids'][:10])}")
            return RetrievedEvidence(
                intent="CASE_TRANSACTIONS",
                target_entity=c_id,
                target_type="CASE",
                evidence_items=evidence,
                structured_data=case
            )

        # Intent 3: Highest Risk in Case
        if any(k in q for k in ["highest risk", "most suspicious", "worst", "maximum risk"]):
            primary_tx = self.store.get_transaction(case['primary_txid'])
            evidence = [
                f"Case ID: {c_id}",
                f"Highest Risk Transaction: {case['primary_txid']}",
                f"Risk Score: {int(case['max_risk_score'])}/100 ({case['severity']} Severity)",
                f"AI Anomaly Score: {primary_tx['anomaly_score']:.3f}" if primary_tx else f"Risk Score: {case['max_risk_score']}",
                f"Key Signals: {'; '.join(primary_tx['risk_reasons'][:3])}" if primary_tx else "Key Signals: Multiple correlated anomaly indicators"
            ]
            return RetrievedEvidence(
                intent="CASE_HIGHEST_RISK",
                target_entity=c_id,
                target_type="CASE",
                evidence_items=evidence,
                structured_data=case
            )

        # Intent 4: Case Timeline
        if any(k in q for k in ["timeline", "events", "sequence", "chronolog"]):
            timeline_builder = getattr(self.store, 'timeline_builder', None)
            events = timeline_builder.build_case_timeline(case['txids']) if timeline_builder else []
            evidence = base_items[:4]
            for ev in events[:8]:
                evidence.append(f"Case Event [{ev['timestamp']}]: {ev['badge']} - {ev['entity']} ({ev['description']})")
            return RetrievedEvidence(
                intent="CASE_TIMELINE",
                target_entity=c_id,
                target_type="CASE",
                evidence_items=evidence,
                structured_data={"case": case, "events": events}
            )

        # Default Case Summary
        evidence = base_items.copy()
        for r in case["grouping_reasons"]:
            evidence.append(f"Grouping Evidence: {r}")
        return RetrievedEvidence(
            intent="CASE_SUMMARY",
            target_entity=c_id,
            target_type="CASE",
            evidence_items=evidence,
            structured_data=case
        )

    # -------------------------------------------------------------------------
    # WALLET & IP EVIDENCE BUILDERS
    # -------------------------------------------------------------------------
    def _retrieve_wallet_evidence(self, q: str, w: Dict[str, Any]) -> RetrievedEvidence:
        evidence = [
            f"Wallet Entity: {w['wallet']}",
            f"Associated Transactions: {len(w['txids'])} transactions ({', '.join(w['txids'][:6])})",
            f"Maximum Risk Observed: {int(w['max_risk'])}/100 (Tiers: {', '.join(w['risk_levels'])})",
            f"Total Transferred Volume: {w['total_btc']:.4f} BTC",
            f"Associated Network Vantage IPs: {', '.join(w['associated_ips'][:5]) if w['associated_ips'] else 'None'}",
            f"Counterparty Wallets Connected: {', '.join(w['counterpart_wallets'][:5]) if w['counterpart_wallets'] else 'None'}"
        ]
        return RetrievedEvidence(
            intent="WALLET_SUMMARY",
            target_entity=w["wallet"],
            target_type="WALLET",
            evidence_items=evidence,
            structured_data=w
        )

    def _retrieve_ip_evidence(self, q: str, ip: Dict[str, Any]) -> RetrievedEvidence:
        evidence = [
            f"Network Vantage Point IP: {ip['ip']}",
            f"GeoIP Location: {ip['country']} (ASN: {ip['asn']})",
            f"Associated Transactions: {len(ip['txids'])} transactions ({', '.join(ip['txids'][:6])})",
            f"Total Network Packets Observed: {ip['total_packets']:,} packets",
            f"Maximum Transaction Risk Score: {int(ip['max_risk'])}/100",
            f"Associated Wallet Entities: {', '.join(ip['associated_wallets'][:6]) if ip['associated_wallets'] else 'None'}",
            "Non-Attribution Note: Network observation associated with transaction broadcast. Does not imply wallet ownership."
        ]
        return RetrievedEvidence(
            intent="IP_SUMMARY",
            target_entity=ip["ip"],
            target_type="IP",
            evidence_items=evidence,
            structured_data=ip
        )

    # -------------------------------------------------------------------------
    # GLOBAL INVESTIGATION SUMMARY
    # -------------------------------------------------------------------------
    def _retrieve_global_evidence(self, q: str) -> RetrievedEvidence:
        s = self.store.global_summary
        top_cases = self.store.get_top_cases(3)
        top_anoms = self.store.get_top_anomalies(3)

        evidence = [
            f"Total Monitored Transactions: {s.get('total_transactions', 0):,}",
            f"AI Detected Anomalies: {s.get('detected_anomalies', 0):,}",
            f"Critical Risk Alerts: {s.get('critical_alerts', 0):,}",
            f"High Risk Alerts: {s.get('high_alerts', 0):,}",
            f"Deduplicated Cases Discovered: {s.get('total_cases', 0):,}",
            f"Mean Network-Blockchain Correlation Confidence: {s.get('mean_correlation_confidence', 85)}%",
            f"Top Priority Case: {top_cases[0]['case_id']} ({top_cases[0]['severity']} - {top_cases[0]['title']})" if top_cases else "Top Case: None",
            f"Top Priority Anomaly TXID: {top_anoms[0]['txid']} (Risk: {int(top_anoms[0]['risk_score'])}/100)" if top_anoms else "Top Anomaly: None"
        ]
        return RetrievedEvidence(
            intent="GLOBAL_SUMMARY",
            target_entity="GLOBAL",
            target_type="GLOBAL",
            evidence_items=evidence,
            structured_data=s
        )
