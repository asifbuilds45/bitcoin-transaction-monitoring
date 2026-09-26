"""
Behavioural Pattern Detection Package
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Provides multi-condition structural analysis for:
- Peeling-chain detection (sequential single-output continuation chains)
- CoinJoin-like / mixing pattern detection (equal-output multi-input anonymisation)

All outputs are pattern INDICATORS for investigator review, never criminal verdicts.
"""

from src.patterns.peeling_chain_detector import PeelingChainDetector
from src.patterns.coinjoin_detector import CoinJoinDetector
from src.patterns.pattern_evidence_layer import PatternEvidenceLayer

__all__ = [
    "PeelingChainDetector",
    "CoinJoinDetector",
    "PatternEvidenceLayer",
]
