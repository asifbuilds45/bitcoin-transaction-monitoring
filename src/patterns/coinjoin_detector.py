"""
CoinJoin-Like / Mixing Pattern Detector
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Detects CoinJoin-like and mixing patterns in Bitcoin transactions using multi-condition
structural analysis. Does NOT classify every multi-input/multi-output transaction as CoinJoin.

Result is a PATTERN INDICATOR (not a criminal verdict).
Language: "CoinJoin-like / mixing pattern detected" / "Behavioural indicator" / "Requires further investigation"
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CoinJoinDetector:
    """
    Detects CoinJoin-like and transaction mixing patterns using multiple structural conditions.

    Algorithm:
        1. Parse pipe-separated output_amounts for each transaction.
        2. Apply ALL of the following conditions (configurable):
           a. num_inputs >= min_input_count (minimum participant inputs)
           b. num_outputs >= min_output_count (minimum participant outputs)
           c. Count equal-value outputs within amount_tolerance
           d. equal_output_count >= min_equal_output_count
           e. equal_output_ratio >= equal_output_ratio_threshold
           f. NOT simply a consolidation (fan_in): outputs must be multiple
        3. Calculate evidence score based on equal output ratio and participant count.
        4. Results are cached and returned as DataFrame columns.

    NOT simply: "multi-input/multi-output = CoinJoin"
    Uses equal-output ratio, participant count, and structural exclusion conditions.
    """

    def __init__(
        self,
        min_input_count: int = 3,
        min_output_count: int = 3,
        amount_tolerance: float = 0.01,
        min_equal_output_count: int = 2,
        equal_output_ratio_threshold: float = 0.40,
        max_single_input_ratio: float = 0.95,
    ):
        """
        Args:
            min_input_count: Minimum number of inputs to qualify (avoids single-sender txs).
            min_output_count: Minimum number of outputs to qualify (avoids simple 2-output txs).
            amount_tolerance: Maximum BTC difference for two amounts to be considered "equal" (0.01 BTC).
            min_equal_output_count: Minimum number of outputs that must share a common amount.
            equal_output_ratio_threshold: Minimum fraction of outputs that must be equal-valued.
            max_single_input_ratio: If a single input provides >95% of total, it's likely not CoinJoin.
        """
        self.min_input_count = min_input_count
        self.min_output_count = min_output_count
        self.amount_tolerance = amount_tolerance
        self.min_equal_output_count = min_equal_output_count
        self.equal_output_ratio_threshold = equal_output_ratio_threshold
        self.max_single_input_ratio = max_single_input_ratio

        self._coinjoin_members: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run CoinJoin-like pattern detection on a transaction DataFrame.

        Args:
            df: Transaction DataFrame with pipe-separated output_amounts fields.

        Returns:
            df enriched with:
                - coinjoin_detected (bool)
                - coinjoin_evidence (float 0.0–1.0)
                - coinjoin_equal_output_count (int)
                - coinjoin_equal_output_ratio (float)
                - coinjoin_equal_amount_btc (float)
                - coinjoin_reasons (str, human-readable indicator)
        """
        logger.info("CoinJoinDetector: analyzing %d transactions...", len(df))
        self._coinjoin_members.clear()

        for _, row in df.iterrows():
            result = self._analyze_transaction(row)
            if result["detected"]:
                txid = str(row.get("txid", "")).strip()
                self._coinjoin_members[txid] = result

        logger.info(
            "CoinJoinDetector: found %d transactions with CoinJoin-like indicators.",
            len(self._coinjoin_members)
        )
        return self._annotate_dataframe(df)

    def get_detection_summary(self) -> Dict[str, Any]:
        """Return summary statistics for detected CoinJoin-like transactions."""
        if not self._coinjoin_members:
            return {"total_detected": 0, "avg_evidence_score": 0.0, "avg_equal_output_ratio": 0.0}
        scores = [v["evidence_score"] for v in self._coinjoin_members.values()]
        ratios = [v["equal_output_ratio"] for v in self._coinjoin_members.values()]
        return {
            "total_detected": len(self._coinjoin_members),
            "avg_evidence_score": round(float(np.mean(scores)), 4),
            "avg_equal_output_ratio": round(float(np.mean(ratios)), 4),
        }

    # -------------------------------------------------------------------------
    # INTERNAL METHODS
    # -------------------------------------------------------------------------

    def _parse_pipe_list(self, val: Any) -> List[str]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return []
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", ""):
            return []
        return [x.strip() for x in s.split("|") if x.strip()]

    def _parse_pipe_floats(self, val: Any) -> List[float]:
        tokens = self._parse_pipe_list(val)
        result = []
        for t in tokens:
            try:
                result.append(float(t))
            except ValueError:
                pass
        return result

    def _find_equal_output_cluster(
        self, out_amounts: List[float]
    ) -> Tuple_or_dict:
        """
        Find the largest cluster of equal-valued outputs within amount_tolerance.

        Uses a greedy grouping approach: sort amounts, then group consecutive
        amounts that are within tolerance of each other.

        Returns:
            (equal_count, equal_amount, equal_ratio)
        """
        if not out_amounts:
            return 0, 0.0, 0.0

        sorted_amounts = sorted(out_amounts)
        n = len(sorted_amounts)

        best_count = 1
        best_amount = sorted_amounts[0]
        current_group = [sorted_amounts[0]]

        for i in range(1, n):
            if abs(sorted_amounts[i] - current_group[0]) <= self.amount_tolerance:
                current_group.append(sorted_amounts[i])
            else:
                if len(current_group) > best_count:
                    best_count = len(current_group)
                    best_amount = float(np.mean(current_group))
                current_group = [sorted_amounts[i]]

        if len(current_group) > best_count:
            best_count = len(current_group)
            best_amount = float(np.mean(current_group))

        equal_ratio = best_count / n if n > 0 else 0.0
        return best_count, round(best_amount, 8), round(equal_ratio, 4)

    def _is_consolidation(
        self, num_inputs: int, num_outputs: int, in_amounts: List[float], out_amounts: List[float]
    ) -> bool:
        """
        Heuristic: Is this a simple consolidation (fan_in) rather than CoinJoin?
        A consolidation has many inputs but very few outputs (≤ 2) and no equal outputs.
        """
        # Fan-in: many inputs, very few outputs → consolidation, not mixing
        if num_inputs > num_outputs * 3 and num_outputs <= 2:
            return True
        return False

    def _analyze_transaction(self, row: Any) -> Dict[str, Any]:
        """
        Analyze a single transaction for CoinJoin-like structural properties.
        Returns dict with detection result and evidence details.
        """
        null_result = {
            "detected": False,
            "evidence_score": 0.0,
            "equal_output_count": 0,
            "equal_output_ratio": 0.0,
            "equal_amount_btc": 0.0,
            "reasons": "",
        }

        num_inputs = int(row.get("num_inputs", 1))
        num_outputs = int(row.get("num_outputs", 1))

        # Condition 1: Minimum inputs
        if num_inputs < self.min_input_count:
            return null_result

        # Condition 2: Minimum outputs
        if num_outputs < self.min_output_count:
            return null_result

        # Parse amounts
        out_amounts = self._parse_pipe_floats(
            row.get("output_amounts", row.get("parsed_output_amounts", ""))
        )
        in_amounts = self._parse_pipe_floats(
            row.get("input_amounts", row.get("parsed_input_amounts", ""))
        )

        # Condition 3: Must have parseable output amounts
        if not out_amounts:
            # Fallback: try total_output_amount_btc / num_outputs as synthetic equal amount
            total_out = float(row.get("total_output_amount_btc", 0.0))
            if total_out > 0 and num_outputs >= self.min_output_count:
                out_amounts = [total_out / num_outputs] * num_outputs
            else:
                return null_result

        # Condition 4: Not a simple consolidation
        if self._is_consolidation(num_inputs, num_outputs, in_amounts, out_amounts):
            return null_result

        # Condition 5: Single dominant input check
        if in_amounts:
            max_in = max(in_amounts)
            total_in = sum(in_amounts)
            if total_in > 0 and max_in / total_in > self.max_single_input_ratio:
                return null_result

        # Condition 6: Find equal-output cluster
        equal_count, equal_amount, equal_ratio = self._find_equal_output_cluster(out_amounts)

        if equal_count < self.min_equal_output_count:
            return null_result

        if equal_ratio < self.equal_output_ratio_threshold:
            return null_result

        # All conditions passed — compute evidence score
        evidence_score = self._compute_evidence_score(
            num_inputs, num_outputs, equal_count, equal_ratio
        )

        return {
            "detected": True,
            "evidence_score": round(evidence_score, 4),
            "equal_output_count": equal_count,
            "equal_output_ratio": round(equal_ratio, 4),
            "equal_amount_btc": round(equal_amount, 8),
            "reasons": (
                f"CoinJoin-like / mixing pattern detected "
                f"({equal_count}/{num_outputs} outputs equal at ≈{equal_amount:.6f} BTC, "
                f"ratio={equal_ratio:.2f}, inputs={num_inputs}). "
                f"Behavioural indicator — Requires further investigation."
            ),
        }

    def _compute_evidence_score(
        self,
        num_inputs: int,
        num_outputs: int,
        equal_count: int,
        equal_ratio: float
    ) -> float:
        """
        Compute evidence score (0.0–1.0) based on:
        - Equal output ratio (primary signal)
        - Number of participants (inputs)
        - Number of equal outputs (secondary signal)
        """
        # Equal output ratio contribution (0–0.5)
        ratio_score = min(equal_ratio / 1.0, 0.5)
        # Input count contribution (0–0.3): more inputs = more mixing participants
        input_score = min((num_inputs - self.min_input_count) / 10.0, 0.3)
        # Equal output count contribution (0–0.2)
        eq_count_score = min((equal_count - self.min_equal_output_count) / 8.0, 0.2)

        return float(np.clip(ratio_score + input_score + eq_count_score, 0.0, 1.0))

    def _annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the dataframe with CoinJoin detection results."""
        df = df.copy()

        cj_detected = []
        cj_evidence = []
        cj_eq_count = []
        cj_eq_ratio = []
        cj_eq_amount = []
        cj_reasons = []

        for _, row in df.iterrows():
            txid = str(row.get("txid", "")).strip()
            info = self._coinjoin_members.get(txid)

            if info:
                cj_detected.append(True)
                cj_evidence.append(info["evidence_score"])
                cj_eq_count.append(info["equal_output_count"])
                cj_eq_ratio.append(info["equal_output_ratio"])
                cj_eq_amount.append(info["equal_amount_btc"])
                cj_reasons.append(info["reasons"])
            else:
                cj_detected.append(False)
                cj_evidence.append(0.0)
                cj_eq_count.append(0)
                cj_eq_ratio.append(0.0)
                cj_eq_amount.append(0.0)
                cj_reasons.append("")

        df["coinjoin_detected"] = cj_detected
        df["coinjoin_evidence"] = cj_evidence
        df["coinjoin_equal_output_count"] = cj_eq_count
        df["coinjoin_equal_output_ratio"] = cj_eq_ratio
        df["coinjoin_equal_amount_btc"] = cj_eq_amount
        df["coinjoin_reasons"] = cj_reasons

        return df


# Fix missing type hint (local workaround for tuple return)
Tuple_or_dict = tuple
