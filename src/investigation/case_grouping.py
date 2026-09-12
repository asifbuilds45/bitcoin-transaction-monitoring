"""
Alert Deduplication & Case Grouping Module
SIH Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Groups strongly connected anomalous transactions into cohesive, deduplicated investigation cases
using real graph connectivity, shared wallets, temporal-network correlation, and DBSCAN clusters.
Strict O(N) linear-time graph construction for instant sub-second processing.
"""

from typing import List, Dict, Any, Optional, Set
import pandas as pd
import networkx as nx


class AlertCaseGrouper:
    """
    Consolidates correlated anomalous alerts into actionable, deduplicated investigation cases
    with stable session-level case IDs (CASE-001, CASE-002, ...).
    """

    def __init__(self, scored_df: pd.DataFrame, graph_engine: Optional[Any] = None):
        """
        Args:
            scored_df: Dataframe containing scored transactions, anomalies, and risk metrics.
            graph_engine: Optional BitcoinGraphEngine for topological connectivity lookups.
        """
        self.scored_df = scored_df.copy()
        self.graph_engine = graph_engine
        self.cases: List[Dict[str, Any]] = []
        self._build_cases()

    def _build_cases(self) -> None:
        """
        Group anomalous transactions into connected cases using linear-time star/path connections.
        Considers transactions with is_anomaly == True or risk_level in ['HIGH', 'CRITICAL'].
        """
        is_anom_col = self.scored_df.get('is_anomaly', False)
        risk_lvl_col = self.scored_df.get('risk_level_normalized', self.scored_df.get('risk_level', 'LOW')).astype(str).str.upper()

        mask = is_anom_col | risk_lvl_col.isin(['CRITICAL', 'HIGH'])
        anom_df = self.scored_df[mask].copy()

        if anom_df.empty:
            self.cases = []
            return

        alert_graph = nx.Graph()
        tx_rows = {}

        wallet_src_idx: Dict[str, List[str]] = {}
        wallet_dst_idx: Dict[str, List[str]] = {}
        ip_src_idx: Dict[str, List[tuple]] = {}
        cluster_idx: Dict[int, List[tuple]] = {}

        for _, row in anom_df.iterrows():
            txid = str(row['txid'])
            rec = row.to_dict()
            ts = pd.to_datetime(rec.get('timestamp'), errors='coerce')
            rec['parsed_ts'] = ts
            tx_rows[txid] = rec
            alert_graph.add_node(txid)

            src_w = str(rec.get('source_wallet', ''))
            dst_w = str(rec.get('destination_wallet', ''))
            src_ip = str(rec.get('src_ip', ''))
            c_label = rec.get('dbscan_cluster', -1)

            if src_w and src_w != 'Unknown':
                wallet_src_idx.setdefault(src_w, []).append(txid)
            if dst_w and dst_w != 'Unknown':
                wallet_dst_idx.setdefault(dst_w, []).append(txid)
            if src_ip and src_ip != 'Unknown':
                ip_src_idx.setdefault(src_ip, []).append((txid, ts))
            if c_label is not None and c_label >= 0:
                cluster_idx.setdefault(int(c_label), []).append((txid, ts))

        def add_case_edge(u: str, v: str, reason: str):
            if u == v:
                return
            if alert_graph.has_edge(u, v):
                alert_graph[u][v].setdefault('reasons', set()).add(reason)
            else:
                alert_graph.add_edge(u, v, reasons={reason})

        # 1. Group by shared source wallet (Linear star-graph connection O(N))
        for w, tx_group in wallet_src_idx.items():
            if len(tx_group) > 1:
                anchor = tx_group[0]
                for node in tx_group[1:]:
                    add_case_edge(anchor, node, f"Shared source wallet: {w}")

        # 2. Group by shared destination wallet (Linear star-graph connection O(N))
        for w, tx_group in wallet_dst_idx.items():
            if len(tx_group) > 1:
                anchor = tx_group[0]
                for node in tx_group[1:]:
                    add_case_edge(anchor, node, f"Shared destination wallet: {w}")

        # 3. Group by sequential fund-flow chain (destination of A is source of B)
        common_wallets = set(wallet_dst_idx.keys()) & set(wallet_src_idx.keys())
        for w in common_wallets:
            anchor_in = wallet_dst_idx[w][0]
            anchor_out = wallet_src_idx[w][0]
            add_case_edge(anchor_in, anchor_out, f"Sequential fund flow: {w}")

        # 4. Group by shared network IP within 2 hours (Linear temporal path connection O(N))
        for ip, group in ip_src_idx.items():
            if len(group) > 1:
                valid_entries = [g for g in group if pd.notna(g[1])]
                valid_entries.sort(key=lambda x: x[1])
                for i in range(len(valid_entries) - 1):
                    diff = (valid_entries[i + 1][1] - valid_entries[i][1]).total_seconds()
                    if diff <= 7200:  # <= 2 hours
                        add_case_edge(
                            valid_entries[i][0],
                            valid_entries[i + 1][0],
                            f"Shared network vantage point ({ip}) within {int(diff/60)}m window"
                        )

        # 5. Group by dense DBSCAN Behavioural Cluster within 4 hours (Linear path connection O(N))
        for c_id, group in cluster_idx.items():
            if len(group) > 1:
                valid_entries = [g for g in group if pd.notna(g[1])]
                valid_entries.sort(key=lambda x: x[1])
                for i in range(len(valid_entries) - 1):
                    diff = (valid_entries[i + 1][1] - valid_entries[i][1]).total_seconds()
                    if diff <= 14400:  # <= 4 hours
                        add_case_edge(
                            valid_entries[i][0],
                            valid_entries[i + 1][0],
                            f"DBSCAN Behavioural Cluster #{c_id} temporal alignment"
                        )

        # Extract connected components (each component forms a Case)
        components = list(nx.connected_components(alert_graph))

        case_candidates = []
        for comp in components:
            txids_in_case = list(comp)
            case_tx_rows = [tx_rows[t] for t in txids_in_case]

            max_risk = max(float(r.get('risk_score', 0)) for r in case_tx_rows)
            total_btc = sum(float(r.get('total_output_amount_btc', 0)) for r in case_tx_rows)

            risk_levels = [str(r.get('risk_level', 'LOW')).upper() for r in case_tx_rows]
            if 'CRITICAL' in risk_levels:
                case_severity = 'CRITICAL'
            elif 'HIGH' in risk_levels:
                case_severity = 'HIGH'
            elif 'MEDIUM' in risk_levels:
                case_severity = 'MEDIUM'
            else:
                case_severity = 'LOW'

            wallets = set()
            ips = set()
            timestamps = []

            for r in case_tx_rows:
                sw = str(r.get('source_wallet', ''))
                dw = str(r.get('destination_wallet', ''))
                if sw and sw != 'Unknown':
                    wallets.add(sw)
                if dw and dw != 'Unknown':
                    wallets.add(dw)
                src_ip = str(r.get('src_ip', ''))
                if src_ip and src_ip != 'Unknown':
                    ips.add(src_ip)
                ts = r.get('parsed_ts')
                if pd.notna(ts):
                    timestamps.append(ts)

            # Primary transaction: highest risk, then highest amount
            sorted_case_rows = sorted(
                case_tx_rows,
                key=lambda r: (float(r.get('risk_score', 0)), float(r.get('total_output_amount_btc', 0))),
                reverse=True
            )
            primary_txid = str(sorted_case_rows[0]['txid'])

            # Aggregate grouping reasons from the subgraph edges
            case_reasons = set()
            sub_g = alert_graph.subgraph(comp)
            for u, v, data in sub_g.edges(data=True):
                case_reasons.update(data.get('reasons', set()))

            if not case_reasons:
                case_reasons.add(f"Individual high-priority anomaly incident: {primary_txid}")

            # Descriptive title
            if len(txids_in_case) > 1:
                if any("Shared source wallet" in r for r in case_reasons):
                    case_title = "Coordinated Multi-Output Source Wallet Activity"
                elif any("Sequential fund flow" in r for r in case_reasons):
                    case_title = "Multi-Hop Chained Transaction Fund-Flow"
                elif any("Shared network vantage point" in r for r in case_reasons):
                    case_title = "Correlated Network Vantage Point Broadcast Activity"
                elif any("DBSCAN" in r for r in case_reasons):
                    case_title = "Dense Behavioural Anomaly Cluster Activity"
                else:
                    case_title = f"Correlated Alert Group ({len(txids_in_case)} Transactions)"
            else:
                case_title = f"Individual High-Risk Anomaly Incident ({primary_txid})"

            earliest_ts = str(min(timestamps)) if timestamps else "Unknown"
            latest_ts = str(max(timestamps)) if timestamps else "Unknown"

            case_candidates.append({
                "severity": case_severity,
                "max_risk_score": max_risk,
                "transaction_count": len(txids_in_case),
                "title": case_title,
                "primary_txid": primary_txid,
                "txids": txids_in_case,
                "transactions": sorted_case_rows,
                "wallets": sorted(list(wallets)),
                "ips": sorted(list(ips)),
                "total_btc_volume": round(total_btc, 6),
                "earliest_timestamp": earliest_ts,
                "latest_timestamp": latest_ts,
                "grouping_reasons": sorted(list(case_reasons))
            })

        # Deterministic sorting for stable case IDs
        case_candidates.sort(
            key=lambda c: (
                c["max_risk_score"],
                c["transaction_count"],
                c["total_btc_volume"]
            ),
            reverse=True
        )

        # Assign stable session IDs: CASE-001, CASE-002, ...
        self.cases = []
        for idx, c in enumerate(case_candidates, 1):
            c["case_id"] = f"CASE-{idx:03d}"
            self.cases.append(c)

    def get_cases(self, min_transactions: int = 1) -> List[Dict[str, Any]]:
        """Retrieve all cases filtered by minimum transaction count."""
        if min_transactions <= 1:
            return self.cases
        return [c for c in self.cases if c["transaction_count"] >= min_transactions]

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific case by its stable ID (e.g., 'CASE-001')."""
        case_id = str(case_id).strip().upper()
        for c in self.cases:
            if c["case_id"] == case_id:
                return c
        return None

    def get_case_summary_dataframe(self) -> pd.DataFrame:
        """Return a high-level summary DataFrame of all investigation cases."""
        if not self.cases:
            return pd.DataFrame()

        records = []
        for c in self.cases:
            records.append({
                "Case ID": c["case_id"],
                "Severity": c["severity"],
                "Title": c["title"],
                "Max Risk": int(c["max_risk_score"]),
                "Transactions": c["transaction_count"],
                "Wallets": len(c["wallets"]),
                "Network IPs": len(c["ips"]),
                "Total BTC": f"{c['total_btc_volume']:.4f}",
                "Primary TXID": c["primary_txid"],
                "Earliest Activity": c["earliest_timestamp"],
                "Latest Activity": c["latest_timestamp"]
            })
        return pd.DataFrame(records)
