"""
Pattern Evidence Layer
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Common orchestration layer for all behavioural pattern detectors.
Runs peeling-chain detection and CoinJoin detection, caches results,
and provides persistence hooks for PostgreSQL.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.patterns.peeling_chain_detector import PeelingChainDetector
from src.patterns.coinjoin_detector import CoinJoinDetector

logger = logging.getLogger(__name__)

# PostgreSQL table name for pattern results cache
PATTERN_CACHE_TABLE = "pattern_detection_results"


class PatternEvidenceLayer:
    """
    Orchestrates pattern detection, aggregates results, and provides
    persistence and retrieval interfaces for investigation pipeline integration.

    Usage in pipeline (insert BEFORE RiskScoringEngine):
        pattern_layer = PatternEvidenceLayer()
        beh_df = pattern_layer.run(beh_df)
        # beh_df now contains peeling_chain_evidence + coinjoin_evidence columns
    """

    def __init__(
        self,
        peeling_chain_config: Optional[Dict[str, Any]] = None,
        coinjoin_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            peeling_chain_config: Config overrides for PeelingChainDetector.
            coinjoin_config: Config overrides for CoinJoinDetector.
        """
        peel_cfg = peeling_chain_config or {}
        cj_cfg = coinjoin_config or {}

        self.peeling_detector = PeelingChainDetector(**peel_cfg)
        self.coinjoin_detector = CoinJoinDetector(**cj_cfg)

        self._run_complete = False
        self._n_peeling = 0
        self._n_coinjoin = 0

    def run(self, df: pd.DataFrame, db_manager=None) -> pd.DataFrame:
        """
        Run all pattern detectors on the given DataFrame.

        Args:
            df: Transaction DataFrame (must contain pipe-separated address/amount fields).
            db_manager: Optional DatabaseManager for PostgreSQL caching.

        Returns:
            df enriched with all pattern evidence columns.
        """
        logger.info("PatternEvidenceLayer: starting pattern detection on %d transactions...", len(df))

        # Check if cached results exist in DB first
        if db_manager is not None and hasattr(db_manager, 'is_connected') and db_manager.is_connected:
            cached = self._load_from_db(db_manager, df)
            if cached is not None:
                logger.info("PatternEvidenceLayer: loaded pattern results from PostgreSQL cache.")
                self._run_complete = True
                return cached

        # Step 1: Peeling-chain detection
        try:
            df = self.peeling_detector.fit_detect(df)
            self._n_peeling = int(df["peeling_chain_detected"].sum())
            logger.info("PatternEvidenceLayer: %d transactions detected in peeling chains.", self._n_peeling)
        except Exception as e:
            logger.warning("PatternEvidenceLayer: peeling chain detection failed: %s", e)
            df["peeling_chain_detected"] = False
            df["peeling_chain_evidence"] = 0.0
            df["peeling_chain_id"] = ""
            df["peeling_chain_position"] = 0
            df["peeling_chain_length"] = 0
            df["peeling_chain_reasons"] = ""

        # Step 2: CoinJoin-like detection
        try:
            df = self.coinjoin_detector.fit_detect(df)
            self._n_coinjoin = int(df["coinjoin_detected"].sum())
            logger.info("PatternEvidenceLayer: %d transactions detected with CoinJoin-like indicators.", self._n_coinjoin)
        except Exception as e:
            logger.warning("PatternEvidenceLayer: CoinJoin detection failed: %s", e)
            df["coinjoin_detected"] = False
            df["coinjoin_evidence"] = 0.0
            df["coinjoin_equal_output_count"] = 0
            df["coinjoin_equal_output_ratio"] = 0.0
            df["coinjoin_equal_amount_btc"] = 0.0
            df["coinjoin_reasons"] = ""

        # Step 3: Persist to PostgreSQL cache if available
        if db_manager is not None and hasattr(db_manager, 'is_connected') and db_manager.is_connected:
            self._save_to_db(db_manager, df)

        self._run_complete = True
        return df

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of pattern detection results."""
        peel_summary = self.peeling_detector.get_chains_summary() if self._run_complete else []
        cj_summary = self.coinjoin_detector.get_detection_summary() if self._run_complete else {}
        return {
            "peeling_chains_detected": self._n_peeling,
            "coinjoin_detected": self._n_coinjoin,
            "peeling_chain_groups": len(peel_summary),
            "coinjoin_avg_evidence_score": cj_summary.get("avg_evidence_score", 0.0),
            "coinjoin_avg_equal_output_ratio": cj_summary.get("avg_equal_output_ratio", 0.0),
        }

    def get_detected_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Return a list of all transactions with pattern detections,
        suitable for the API /api/patterns/* endpoints.
        """
        patterns = []
        if "peeling_chain_detected" not in df.columns and "coinjoin_detected" not in df.columns:
            return patterns

        for _, row in df.iterrows():
            peel = bool(row.get("peeling_chain_detected", False))
            cj = bool(row.get("coinjoin_detected", False))
            if peel or cj:
                entry = {
                    "txid": str(row.get("txid", "")),
                    "risk_score": float(row.get("risk_score", 0.0)),
                    "risk_level": str(row.get("risk_level", "LOW")).upper(),
                    "peeling_chain_detected": peel,
                    "peeling_chain_evidence": float(row.get("peeling_chain_evidence", 0.0)),
                    "peeling_chain_id": str(row.get("peeling_chain_id", "")),
                    "peeling_chain_position": int(row.get("peeling_chain_position", 0)),
                    "peeling_chain_length": int(row.get("peeling_chain_length", 0)),
                    "peeling_chain_reasons": str(row.get("peeling_chain_reasons", "")),
                    "coinjoin_detected": cj,
                    "coinjoin_evidence": float(row.get("coinjoin_evidence", 0.0)),
                    "coinjoin_equal_output_count": int(row.get("coinjoin_equal_output_count", 0)),
                    "coinjoin_equal_output_ratio": float(row.get("coinjoin_equal_output_ratio", 0.0)),
                    "coinjoin_equal_amount_btc": float(row.get("coinjoin_equal_amount_btc", 0.0)),
                    "coinjoin_reasons": str(row.get("coinjoin_reasons", "")),
                }
                patterns.append(entry)
        return patterns

    # -------------------------------------------------------------------------
    # PERSISTENCE HELPERS
    # -------------------------------------------------------------------------

    def _save_to_db(self, db_manager, df: pd.DataFrame) -> None:
        """Save pattern evidence columns to PostgreSQL for caching."""
        try:
            pattern_cols = ["txid"]
            for col in [
                "peeling_chain_detected", "peeling_chain_evidence", "peeling_chain_id",
                "peeling_chain_position", "peeling_chain_length", "peeling_chain_reasons",
                "coinjoin_detected", "coinjoin_evidence", "coinjoin_equal_output_count",
                "coinjoin_equal_output_ratio", "coinjoin_equal_amount_btc", "coinjoin_reasons",
            ]:
                if col in df.columns:
                    pattern_cols.append(col)

            cache_df = df[pattern_cols].copy()
            db_manager.save_dataframe_table(cache_df, PATTERN_CACHE_TABLE)
            logger.info("PatternEvidenceLayer: saved pattern cache to PostgreSQL table '%s'.", PATTERN_CACHE_TABLE)
        except Exception as e:
            logger.warning("PatternEvidenceLayer: could not save pattern cache to DB: %s", e)

    def _load_from_db(self, db_manager, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Load cached pattern results from PostgreSQL and merge into df.
        Returns merged df if cache exists and has same txid set, else None.
        """
        try:
            if not hasattr(db_manager, 'engine') or db_manager.engine is None:
                return None
            import pandas as pd_inner
            cache_df = pd_inner.read_sql_table(PATTERN_CACHE_TABLE, db_manager.engine)
            if cache_df.empty:
                return None
            # Check txid coverage: at least 90% match
            df_txids = set(df["txid"].astype(str))
            cache_txids = set(cache_df["txid"].astype(str))
            overlap = len(df_txids & cache_txids) / max(len(df_txids), 1)
            if overlap < 0.90:
                logger.info("PatternEvidenceLayer: cache txid overlap %.1f%% < 90%%, re-running detection.", overlap * 100)
                return None
            merged = df.merge(cache_df, on="txid", how="left", suffixes=("", "_cached"))
            # Fill NaN pattern columns with defaults
            for col in [c for c in cache_df.columns if c != "txid"]:
                if col + "_cached" in merged.columns:
                    merged[col] = merged[col + "_cached"].combine_first(merged.get(col, pd_inner.Series()))
                    merged.drop(columns=[col + "_cached"], inplace=True, errors="ignore")
            return merged
        except Exception as e:
            logger.debug("PatternEvidenceLayer: cache load skipped: %s", e)
            return None
