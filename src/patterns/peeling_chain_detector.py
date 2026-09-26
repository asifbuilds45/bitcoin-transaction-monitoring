"""
Peeling-Chain Detector
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Detects peeling-chain patterns in Bitcoin transaction graphs: sequential single-continuation
fund flows where a large "continuing" output of one transaction becomes the input of the next,
with small "peeled" amounts dispersed at each hop.

Result is a PATTERN INDICATOR (not a criminal verdict).
Language: "Peeling-chain pattern detected" / "Investigation evidence" / "Requires further investigation"
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PeelingChainDetector:
    """
    Detects peeling-chain structural patterns using multi-condition graph traversal.

    Algorithm:
        1. Build a wallet→txid index (output wallets → txid, input wallets → txid).
        2. For each transaction, check if it is a valid peeling hop:
           - Exactly 1 "large continuing" output (>= continuation_ratio × total_out)
           - That large output wallet appears as an input wallet of another tx
           - That next tx also satisfies the split pattern
           - Chain repeats for min_chain_length transactions minimum
           - Amount is reduced by no more than max_fee_ratio per hop (fee bounds)
           - Timestamps are ordered (if available)
        3. Assign evidence score per detected chain member (0.0–1.0).
        4. Results are cached and returned as a DataFrame column.

    NOT simply: "largest output = peeling chain"
    Uses MULTIPLE structural conditions to avoid false positives.
    """

    def __init__(
        self,
        min_chain_length: int = 3,
        continuation_ratio: float = 0.40,
        max_fee_ratio: float = 0.15,
        min_outputs: int = 2,
        max_outputs: int = 8,
        require_ordered_timestamps: bool = False,
    ):
        """
        Args:
            min_chain_length: Minimum number of sequential transactions to qualify as a chain.
            continuation_ratio: Minimum fraction of total output that the largest output must hold (0.40 = 40%).
            max_fee_ratio: Maximum per-hop amount reduction allowed (as fraction of input amount; 0.15 = 15%).
            min_outputs: Minimum number of outputs per hop (must have at least one peel).
            max_outputs: Maximum outputs per hop (too many outputs is fan-out/CoinJoin, not peeling).
            require_ordered_timestamps: If True, timestamps must increase monotonically across chain hops.
        """
        self.min_chain_length = max(2, min_chain_length)
        self.continuation_ratio = float(continuation_ratio)
        self.max_fee_ratio = float(max_fee_ratio)
        self.min_outputs = max(2, min_outputs)
        self.max_outputs = max_outputs
        self.require_ordered_timestamps = require_ordered_timestamps

        # Internal indexes built during fit()
        self._output_wallet_to_txids: Dict[str, List[str]] = {}   # output wallet → [txid]
        self._input_wallet_to_txids: Dict[str, List[str]] = {}    # input wallet → [txid]
        self._tx_map: Dict[str, Dict[str, Any]] = {}               # txid → parsed row data

        # Results: txid → chain_info
        self._peeling_chain_members: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build indexes and run peeling-chain detection.

        Args:
            df: Transaction DataFrame with pipe-separated output_addresses/input_addresses fields.

        Returns:
            df enriched with:
                - peeling_chain_detected (bool)
                - peeling_chain_evidence (float 0.0–1.0)
                - peeling_chain_id (str, chain identifier or "")
                - peeling_chain_position (int, hop position within chain, 0 if not in chain)
                - peeling_chain_length (int, total chain length, 0 if not in chain)
                - peeling_chain_reasons (str, human-readable indicator)
        """
        logger.info("PeelingChainDetector: building wallet indexes from %d transactions...", len(df))
        self._build_indexes(df)
        logger.info("PeelingChainDetector: running chain traversal...")
        self._detect_all_chains()
        logger.info(
            "PeelingChainDetector: found %d transactions in peeling chains.",
            len(self._peeling_chain_members)
        )
        return self._annotate_dataframe(df)

    def get_chains_summary(self) -> List[Dict[str, Any]]:
        """Return a list of detected peeling chains with member txids."""
        chains: Dict[str, Dict[str, Any]] = {}
        for txid, info in self._peeling_chain_members.items():
            cid = info["chain_id"]
            if cid not in chains:
                chains[cid] = {
                    "chain_id": cid,
                    "chain_length": info["chain_length"],
                    "members": [],
                    "total_btc_peeled": 0.0,
                }
            chains[cid]["members"].append({
                "txid": txid,
                "position": info["position"],
                "evidence_score": info["evidence_score"],
            })
            chains[cid]["total_btc_peeled"] += info.get("peeled_amount", 0.0)
        return list(chains.values())

    # -------------------------------------------------------------------------
    # INTERNAL METHODS
    # -------------------------------------------------------------------------

    def _parse_pipe_list(self, val: Any) -> List[str]:
        """Parse pipe-separated string to list of non-empty tokens."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return []
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", ""):
            return []
        return [x.strip() for x in s.split("|") if x.strip()]

    def _parse_pipe_floats(self, val: Any) -> List[float]:
        """Parse pipe-separated float string to list of floats."""
        tokens = self._parse_pipe_list(val)
        result = []
        for t in tokens:
            try:
                result.append(float(t))
            except ValueError:
                pass
        return result

    def _build_indexes(self, df: pd.DataFrame) -> None:
        """Build wallet → txid lookup indexes and tx_map."""
        self._output_wallet_to_txids.clear()
        self._input_wallet_to_txids.clear()
        self._tx_map.clear()

        for _, row in df.iterrows():
            txid = str(row.get("txid", "")).strip()
            if not txid:
                continue

            out_wallets = self._parse_pipe_list(
                row.get("output_addresses", row.get("parsed_output_addresses", ""))
            )
            in_wallets = self._parse_pipe_list(
                row.get("input_addresses", row.get("parsed_input_addresses", ""))
            )
            out_amounts = self._parse_pipe_floats(
                row.get("output_amounts", row.get("parsed_output_amounts", ""))
            )
            in_amounts = self._parse_pipe_floats(
                row.get("input_amounts", row.get("parsed_input_amounts", ""))
            )

            # Fallback: use total amounts if parse yields nothing
            total_out = float(row.get("total_output_amount_btc", 0.0))
            if not out_amounts and total_out > 0:
                out_amounts = [total_out]

            num_outputs = int(row.get("num_outputs", len(out_wallets) or 1))
            num_inputs = int(row.get("num_inputs", len(in_wallets) or 1))

            ts_raw = row.get("timestamp")
            try:
                ts = pd.to_datetime(ts_raw)
            except Exception:
                ts = None

            self._tx_map[txid] = {
                "txid": txid,
                "out_wallets": out_wallets,
                "in_wallets": in_wallets,
                "out_amounts": out_amounts,
                "in_amounts": in_amounts,
                "total_out": total_out,
                "num_outputs": num_outputs,
                "num_inputs": num_inputs,
                "timestamp": ts,
                "fee_btc": float(row.get("fee_btc", 0.0)),
            }

            for w in out_wallets:
                self._output_wallet_to_txids.setdefault(w, []).append(txid)
            for w in in_wallets:
                self._input_wallet_to_txids.setdefault(w, []).append(txid)

    def _is_valid_peeling_hop(self, tx: Dict[str, Any]) -> Tuple[bool, Optional[str], float, float]:
        """
        Check if a transaction is a valid peeling hop.

        Returns:
            (is_valid, continuing_wallet, continuing_amount, peeled_amount)
        """
        num_out = tx["num_outputs"]
        out_amounts = tx["out_amounts"]
        out_wallets = tx["out_wallets"]
        total_out = tx["total_out"]

        # Condition 1: Must have between min_outputs and max_outputs outputs
        if num_out < self.min_outputs or num_out > self.max_outputs:
            return False, None, 0.0, 0.0

        # Condition 2: Must have at least 1 input (consolidation check)
        if tx["num_inputs"] < 1:
            return False, None, 0.0, 0.0

        if not out_amounts or total_out <= 0:
            return False, None, 0.0, 0.0

        # Condition 3: Find the largest output (candidate "continuing" output)
        max_out_idx = int(np.argmax(out_amounts))
        max_out_amount = out_amounts[max_out_idx]

        # Condition 4: Largest output must be at least continuation_ratio of total_out
        if max_out_amount < self.continuation_ratio * total_out:
            return False, None, 0.0, 0.0

        # Condition 5: Continuing output wallet must exist and appear in the input index
        if max_out_idx < len(out_wallets):
            continuing_wallet = out_wallets[max_out_idx]
        else:
            # If wallet list is shorter (mismatched), use source_wallet as heuristic
            if out_wallets:
                continuing_wallet = out_wallets[0]
            else:
                return False, None, 0.0, 0.0

        if continuing_wallet not in self._input_wallet_to_txids:
            return False, None, 0.0, 0.0

        peeled_amount = total_out - max_out_amount
        return True, continuing_wallet, max_out_amount, peeled_amount

    def _get_next_hop(self, continuing_wallet: str, current_txid: str) -> Optional[str]:
        """Find the next transaction in the chain from the continuing wallet."""
        candidates = self._input_wallet_to_txids.get(continuing_wallet, [])
        for cand_txid in candidates:
            if cand_txid != current_txid:
                return cand_txid
        return None

    def _check_amount_continuity(
        self,
        prev_continuing: float,
        curr_tx: Dict[str, Any]
    ) -> bool:
        """
        Check that the amount reduction from hop N to hop N+1 is within bounds.
        The next hop's total output should not be more than max_fee_ratio less than
        the previous hop's continuing amount (accounting for miner fees).
        """
        curr_total_in = sum(curr_tx["in_amounts"]) if curr_tx["in_amounts"] else curr_tx["total_out"]
        if prev_continuing <= 0:
            return True
        reduction_ratio = abs(prev_continuing - curr_total_in) / prev_continuing
        return reduction_ratio <= self.max_fee_ratio

    def _check_timestamp_order(
        self,
        chain_txids: List[str]
    ) -> bool:
        """Verify timestamps are non-decreasing across the chain (if require_ordered_timestamps=True)."""
        if not self.require_ordered_timestamps:
            return True
        timestamps = [self._tx_map[t]["timestamp"] for t in chain_txids if self._tx_map[t]["timestamp"] is not None]
        if len(timestamps) < 2:
            return True
        for i in range(len(timestamps) - 1):
            if timestamps[i + 1] < timestamps[i]:
                return False
        return True

    def _traverse_chain_from(self, start_txid: str, visited: Set[str]) -> List[str]:
        """
        Traverse a peeling chain starting from start_txid.
        Returns the list of txids in chain order (including start_txid).
        """
        chain = [start_txid]
        current_txid = start_txid
        prev_continuing_amount = None

        while True:
            tx = self._tx_map.get(current_txid)
            if tx is None:
                break

            is_valid, continuing_wallet, continuing_amount, peeled_amount = self._is_valid_peeling_hop(tx)
            if not is_valid:
                break

            # Check amount continuity from previous hop
            if prev_continuing_amount is not None:
                if not self._check_amount_continuity(prev_continuing_amount, tx):
                    break

            next_txid = self._get_next_hop(continuing_wallet, current_txid)
            if next_txid is None or next_txid in visited:
                break

            # Verify the next tx is also a valid hop
            next_tx = self._tx_map.get(next_txid)
            if next_tx is None:
                break

            chain.append(next_txid)
            visited.add(next_txid)
            prev_continuing_amount = continuing_amount
            current_txid = next_txid

            # Safety: max chain length cap to avoid infinite traversal
            if len(chain) > 100:
                break

        return chain

    def _compute_evidence_score(self, chain_length: int, position: int) -> float:
        """
        Compute evidence score (0.0–1.0) for a member of a peeling chain.
        Score reflects chain confidence: longer chains → higher score.
        Members near the middle of long chains score highest.
        """
        # Base score from chain length (longer = higher confidence)
        length_score = min((chain_length - self.min_chain_length + 1) / 7.0, 0.7) + 0.3
        # Position modifier: early and middle hops score highest
        pos_ratio = position / max(chain_length - 1, 1)
        pos_modifier = 1.0 - 0.2 * abs(pos_ratio - 0.5)
        return float(np.clip(length_score * pos_modifier, 0.0, 1.0))

    def _detect_all_chains(self) -> None:
        """Main detection loop: find all peeling chains in the tx graph."""
        self._peeling_chain_members.clear()
        visited: Set[str] = set()
        chain_counter = 0

        # Sort txids for deterministic output
        all_txids = sorted(self._tx_map.keys())

        for txid in all_txids:
            if txid in visited:
                continue

            tx = self._tx_map[txid]
            is_valid, continuing_wallet, continuing_amount, peeled_amount = self._is_valid_peeling_hop(tx)
            if not is_valid:
                continue

            # Only start a chain from a tx that has no INCOMING peeling link
            # (i.e., it is a chain head, not a mid-chain member)
            is_chain_head = True
            for in_wallet in tx["in_wallets"]:
                if in_wallet in self._output_wallet_to_txids:
                    for prev_txid in self._output_wallet_to_txids[in_wallet]:
                        if prev_txid != txid and prev_txid in self._tx_map:
                            prev_is_valid, prev_cont_wallet, _, _ = self._is_valid_peeling_hop(self._tx_map[prev_txid])
                            if prev_is_valid and prev_cont_wallet == in_wallet:
                                is_chain_head = False
                                break
                if not is_chain_head:
                    break

            if not is_chain_head:
                continue

            visited.add(txid)
            chain = self._traverse_chain_from(txid, visited)

            if len(chain) < self.min_chain_length:
                continue

            # Validate timestamp ordering
            if not self._check_timestamp_order(chain):
                continue

            chain_counter += 1
            chain_id = f"PEEL-{chain_counter:04d}"
            chain_length = len(chain)

            for position, member_txid in enumerate(chain):
                member_tx = self._tx_map.get(member_txid, {})
                _, _, cont_amount, peeled_amt = self._is_valid_peeling_hop(member_tx) if member_tx else (False, None, 0.0, 0.0)
                evidence_score = self._compute_evidence_score(chain_length, position)

                self._peeling_chain_members[member_txid] = {
                    "chain_id": chain_id,
                    "position": position + 1,
                    "chain_length": chain_length,
                    "evidence_score": round(evidence_score, 4),
                    "peeled_amount": round(peeled_amt, 6),
                    "continuing_amount": round(cont_amount, 6),
                }

    def _annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the dataframe with peeling chain detection results."""
        df = df.copy()

        peeling_detected = []
        peeling_evidence = []
        peeling_chain_id = []
        peeling_position = []
        peeling_length = []
        peeling_reasons = []

        for _, row in df.iterrows():
            txid = str(row.get("txid", "")).strip()
            info = self._peeling_chain_members.get(txid)

            if info:
                peeling_detected.append(True)
                peeling_evidence.append(info["evidence_score"])
                peeling_chain_id.append(info["chain_id"])
                peeling_position.append(info["position"])
                peeling_length.append(info["chain_length"])
                peeling_reasons.append(
                    f"Peeling-chain pattern detected (Chain {info['chain_id']}, "
                    f"Hop {info['position']}/{info['chain_length']}). "
                    f"Investigation evidence — Requires further investigation."
                )
            else:
                peeling_detected.append(False)
                peeling_evidence.append(0.0)
                peeling_chain_id.append("")
                peeling_position.append(0)
                peeling_length.append(0)
                peeling_reasons.append("")

        df["peeling_chain_detected"] = peeling_detected
        df["peeling_chain_evidence"] = peeling_evidence
        df["peeling_chain_id"] = peeling_chain_id
        df["peeling_chain_position"] = peeling_position
        df["peeling_chain_length"] = peeling_length
        df["peeling_chain_reasons"] = peeling_reasons

        return df
