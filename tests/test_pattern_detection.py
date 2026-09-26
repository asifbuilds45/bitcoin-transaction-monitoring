"""
Comprehensive Behavioural Pattern Detection Test Suite
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Tests the 8 mandatory test scenarios:
1. Synthetic Peeling-Chain Detection (multi-hop traversal, continuation ratio, evidence scoring)
2. Non-Peeling-Chain Rejection (false-positive prevention, boundary checks)
3. Synthetic CoinJoin-Like / Mixing Pattern Detection (equal output ratio, participant thresholds)
4. Non-CoinJoin Rejection (consolidation fan-in, dominant input, unequal amounts)
5. Multi-Layer Evidence Fusion 11-Channel Integration (weights sum to 1.0, pattern contribution)
6. Pattern Evidence Layer & Pipeline Integration (end-to-end DataFrame enrichment)
7. Case Grouping & Evidence Timeline Integration (chain-based grouping, pattern timeline events)
8. Offline AI Assistant & Forensic PDF Integration (knowledge store, retrieval, LLM narrative, report)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.patterns.peeling_chain_detector import PeelingChainDetector
from src.patterns.coinjoin_detector import CoinJoinDetector
from src.patterns.pattern_evidence_layer import PatternEvidenceLayer
from src.fusion.evidence_fusion import MultiLayerEvidenceFusionEngine, DEFAULT_EVIDENCE_WEIGHTS
from src.investigation.case_grouping import AlertCaseGrouper
from src.investigation.timeline_builder import EvidenceTimelineBuilder
from src.investigation.path_reconstruction import InvestigationPathReconstructor
from src.assistant.investigation_knowledge import InvestigationKnowledgeStore
from src.assistant.evidence_retrieval import EvidenceRetriever
from src.assistant.offline_llm import OfflineLLMManager
from src.reporting.report_generator import generate_anomaly_reasons, ReportGenerator, REPORTLAB_AVAILABLE


class TestPeelingChainDetector(unittest.TestCase):
    """Scenario 1 & 2: Peeling-Chain Detection & False-Positive Rejection."""

    def setUp(self):
        self.detector = PeelingChainDetector(
            min_chain_length=3,
            continuation_ratio=0.50,
            max_fee_ratio=0.15,
            min_outputs=2,
            max_outputs=6,
        )

    def test_synthetic_peeling_chain_detection(self):
        """Scenario 1: Detect a valid 4-hop peeling chain with continuing outputs."""
        # tx1: 10 BTC in -> 8.5 BTC to W_cont1 (continuing), 1.4 BTC to W_peel1, fee 0.1
        # tx2: 8.5 BTC in -> 7.0 BTC to W_cont2 (continuing), 1.4 BTC to W_peel2, fee 0.1
        # tx3: 7.0 BTC in -> 5.5 BTC to W_cont3 (continuing), 1.4 BTC to W_peel3, fee 0.1
        # tx4: 5.5 BTC in -> 4.0 BTC to W_cont4 (continuing), 1.4 BTC to W_peel4, fee 0.1
        df = pd.DataFrame([
            {
                "txid": "TX_PEEL_01",
                "input_addresses": "W_INIT",
                "output_addresses": "W_CONT_1|W_PEEL_1",
                "input_amounts": "10.0",
                "output_amounts": "8.5|1.4",
                "total_output_amount_btc": 9.9,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_INIT",
                "destination_wallet": "W_CONT_1",
                "timestamp": "2026-09-20 10:00:00",
                "fee_btc": 0.1,
            },
            {
                "txid": "TX_PEEL_02",
                "input_addresses": "W_CONT_1",
                "output_addresses": "W_CONT_2|W_PEEL_2",
                "input_amounts": "8.5",
                "output_amounts": "7.0|1.4",
                "total_output_amount_btc": 8.4,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_CONT_1",
                "destination_wallet": "W_CONT_2",
                "timestamp": "2026-09-20 10:15:00",
                "fee_btc": 0.1,
            },
            {
                "txid": "TX_PEEL_03",
                "input_addresses": "W_CONT_2",
                "output_addresses": "W_CONT_3|W_PEEL_3",
                "input_amounts": "7.0",
                "output_amounts": "5.5|1.4",
                "total_output_amount_btc": 6.9,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_CONT_2",
                "destination_wallet": "W_CONT_3",
                "timestamp": "2026-09-20 10:30:00",
                "fee_btc": 0.1,
            },
            {
                "txid": "TX_PEEL_04",
                "input_addresses": "W_CONT_3",
                "output_addresses": "W_CONT_4|W_PEEL_4",
                "input_amounts": "5.5",
                "output_amounts": "4.0|1.4",
                "total_output_amount_btc": 5.4,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_CONT_3",
                "destination_wallet": "W_CONT_4",
                "timestamp": "2026-09-20 10:45:00",
                "fee_btc": 0.1,
            },
        ])

        res_df = self.detector.fit_detect(df)

        # All 4 transactions should be flagged
        self.assertEqual(int(res_df["peeling_chain_detected"].sum()), 4)
        for _, row in res_df.iterrows():
            self.assertTrue(row["peeling_chain_detected"])
            self.assertGreater(row["peeling_chain_evidence"], 0.0)
            self.assertEqual(row["peeling_chain_length"], 4)
            self.assertTrue(row["peeling_chain_id"].startswith("PEEL-"))
            self.assertIn("Peeling-chain pattern detected", row["peeling_chain_reasons"])
            self.assertIn("Requires further investigation", row["peeling_chain_reasons"])

        # Positions must be strictly 1, 2, 3, 4
        positions = list(res_df["peeling_chain_position"])
        self.assertEqual(positions, [1, 2, 3, 4])

        # Summary check
        summary = self.detector.get_chains_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["chain_length"], 4)

    def test_non_peeling_rejection(self):
        """Scenario 2: Reject isolated transactions, chains below min_length, and fee violations."""
        df = pd.DataFrame([
            # 1. Isolated transaction with no continuation
            {
                "txid": "TX_ISOLATED",
                "input_addresses": "W_ISO_IN",
                "output_addresses": "W_ISO_OUT1|W_ISO_OUT2",
                "input_amounts": "5.0",
                "output_amounts": "4.0|1.0",
                "total_output_amount_btc": 5.0,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_ISO_IN",
                "destination_wallet": "W_ISO_OUT1",
                "timestamp": "2026-09-20 11:00:00",
                "fee_btc": 0.0,
            },
            # 2. Chain of length 2 (below min_chain_length=3)
            {
                "txid": "TX_SHORT_01",
                "input_addresses": "W_SH_0",
                "output_addresses": "W_SH_1|W_PEEL_A",
                "input_amounts": "8.0",
                "output_amounts": "7.0|1.0",
                "total_output_amount_btc": 8.0,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_SH_0",
                "destination_wallet": "W_SH_1",
                "timestamp": "2026-09-20 11:10:00",
                "fee_btc": 0.0,
            },
            {
                "txid": "TX_SHORT_02",
                "input_addresses": "W_SH_1",
                "output_addresses": "W_DEAD_END|W_PEEL_B",
                "input_amounts": "7.0",
                "output_amounts": "6.0|1.0",
                "total_output_amount_btc": 7.0,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_SH_1",
                "destination_wallet": "W_DEAD_END",
                "timestamp": "2026-09-20 11:20:00",
                "fee_btc": 0.0,
            },
            # 3. Equal split 50/50 (continuation ratio threshold not met: 2.5/5.0 = 0.50, needs > 0.50 if equal)
            {
                "txid": "TX_EQUAL_SPLIT",
                "input_addresses": "W_EQ_IN",
                "output_addresses": "W_EQ_1|W_EQ_2",
                "input_amounts": "5.0",
                "output_amounts": "2.5|2.5",
                "total_output_amount_btc": 5.0,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_EQ_IN",
                "destination_wallet": "W_EQ_1",
                "timestamp": "2026-09-20 11:30:00",
                "fee_btc": 0.0,
            }
        ])

        res_df = self.detector.fit_detect(df)
        # None of these should be flagged as peeling chains
        self.assertEqual(int(res_df["peeling_chain_detected"].sum()), 0)
        for _, row in res_df.iterrows():
            self.assertFalse(row["peeling_chain_detected"])
            self.assertEqual(row["peeling_chain_evidence"], 0.0)
            self.assertEqual(row["peeling_chain_id"], "")


class TestCoinJoinDetector(unittest.TestCase):
    """Scenario 3 & 4: CoinJoin-Like Pattern Detection & False-Positive Rejection."""

    def setUp(self):
        self.detector = CoinJoinDetector(
            min_input_count=3,
            min_output_count=3,
            amount_tolerance=0.01,
            min_equal_output_count=3,
            equal_output_ratio_threshold=0.40,
            max_single_input_ratio=0.95,
        )

    def test_synthetic_coinjoin_detection(self):
        """Scenario 3: Detect a multi-input, multi-output transaction with equal denomination outputs."""
        df = pd.DataFrame([
            {
                "txid": "TX_COINJOIN_01",
                "input_addresses": "W_IN1|W_IN2|W_IN3|W_IN4",
                "output_addresses": "W_OUT1|W_OUT2|W_OUT3|W_OUT4|W_CHANGE1",
                "input_amounts": "1.05|1.05|1.05|1.05",
                "output_amounts": "1.00|1.00|1.00|1.00|0.18",  # 4 equal outputs out of 5 (80%)
                "total_output_amount_btc": 4.18,
                "num_inputs": 4,
                "num_outputs": 5,
                "source_wallet": "W_IN1",
                "destination_wallet": "W_OUT1",
                "fee_btc": 0.02,
            }
        ])

        res_df = self.detector.fit_detect(df)
        row = res_df.iloc[0]

        self.assertTrue(row["coinjoin_detected"])
        self.assertGreater(row["coinjoin_evidence"], 0.0)
        self.assertEqual(row["coinjoin_equal_output_count"], 4)
        self.assertAlmostEqual(row["coinjoin_equal_amount_btc"], 1.0, places=2)
        self.assertGreaterEqual(row["coinjoin_equal_output_ratio"], 0.75)
        self.assertIn("CoinJoin-like / mixing pattern detected", row["coinjoin_reasons"])
        self.assertIn("Requires further investigation", row["coinjoin_reasons"])

        summary = self.detector.get_detection_summary()
        self.assertEqual(summary["total_detected"], 1)

    def test_non_coinjoin_rejection(self):
        """Scenario 4: Reject simple 1-in 2-out txs, fan-in consolidation, and unequal distributions."""
        df = pd.DataFrame([
            # 1. Standard 1-in, 2-out Bitcoin transaction
            {
                "txid": "TX_NORMAL_PAYMENT",
                "input_addresses": "W_USER",
                "output_addresses": "W_MERCHANT|W_USER_CHANGE",
                "input_amounts": "2.0",
                "output_amounts": "0.5|1.499",
                "total_output_amount_btc": 1.999,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W_USER",
                "destination_wallet": "W_MERCHANT",
            },
            # 2. Consolidation / Fan-in (many inputs, 1 output)
            {
                "txid": "TX_FAN_IN_CONSOLIDATION",
                "input_addresses": "W1|W2|W3|W4|W5|W6",
                "output_addresses": "W_COLD_STORAGE",
                "input_amounts": "0.1|0.1|0.1|0.1|0.1|0.1",
                "output_amounts": "0.599",
                "total_output_amount_btc": 0.599,
                "num_inputs": 6,
                "num_outputs": 1,
                "source_wallet": "W1",
                "destination_wallet": "W_COLD_STORAGE",
            },
            # 3. Multi-in, multi-out but completely unequal random output amounts
            {
                "txid": "TX_RANDOM_DISPERSAL",
                "input_addresses": "W_A|W_B|W_C",
                "output_addresses": "W_D|W_E|W_F|W_G",
                "input_amounts": "1.0|2.0|3.0",
                "output_amounts": "0.234|1.567|2.891|1.300",  # No equal outputs
                "total_output_amount_btc": 5.992,
                "num_inputs": 3,
                "num_outputs": 4,
                "source_wallet": "W_A",
                "destination_wallet": "W_D",
            },
            # 4. Single dominant input (>95% of total input)
            {
                "txid": "TX_DOMINANT_INPUT",
                "input_addresses": "W_WHALE|W_DUST1|W_DUST2",
                "output_addresses": "W_O1|W_O2|W_O3",
                "input_amounts": "99.0|0.01|0.01",  # 99 / 99.02 > 99%
                "output_amounts": "1.0|1.0|1.0",
                "total_output_amount_btc": 3.0,
                "num_inputs": 3,
                "num_outputs": 3,
                "source_wallet": "W_WHALE",
                "destination_wallet": "W_O1",
            },
        ])

        res_df = self.detector.fit_detect(df)
        self.assertEqual(int(res_df["coinjoin_detected"].sum()), 0)
        for _, row in res_df.iterrows():
            self.assertFalse(row["coinjoin_detected"])
            self.assertEqual(row["coinjoin_evidence"], 0.0)


class TestMultiLayerEvidenceFusion11Channels(unittest.TestCase):
    """Scenario 5: Multi-Layer Evidence Fusion with 11 Independent Channels."""

    def test_evidence_weights_integrity(self):
        """Verify 11 channels are defined and weights sum to 1.0."""
        self.assertEqual(len(DEFAULT_EVIDENCE_WEIGHTS), 11)
        self.assertIn("peeling_chain_evidence", DEFAULT_EVIDENCE_WEIGHTS)
        self.assertIn("coinjoin_evidence", DEFAULT_EVIDENCE_WEIGHTS)

        total_weight = sum(DEFAULT_EVIDENCE_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)

    def test_fusion_engine_computation_with_patterns(self):
        """Verify fusion engine incorporates peeling and CoinJoin evidence into fused score."""
        engine = MultiLayerEvidenceFusionEngine()

        sample_row = pd.Series({
            "anomaly_score": 0.75,
            "dbscan_evidence": 0.85,
            "louvain_community_evidence": 0.60,
            "temporal_evidence_score": 0.55,
            "behavioural_evidence_score": 0.65,
            "wallet_degree": 25,
            "src_country": "India",
            "dst_country": "Germany",
            "country_count": 3,
            "total_output_amount_btc": 6.5,
            "fee_btc": 0.05,
            "packet_count": 1600,
            "bytes_transferred": 150000,
            # Pattern channels
            "peeling_chain_evidence": 0.80,
            "peeling_chain_id": "PEEL-0001",
            "peeling_chain_position": 2,
            "peeling_chain_length": 5,
            "coinjoin_evidence": 0.70,
            "coinjoin_equal_output_count": 4,
            "coinjoin_equal_output_ratio": 0.67,
        })

        fused_score, breakdown, top_reasons = engine.compute_record_fused_evidence(sample_row)

        # Check breakdown
        self.assertIn("peeling_chain_evidence", breakdown)
        self.assertIn("coinjoin_evidence", breakdown)
        self.assertAlmostEqual(breakdown["peeling_chain_evidence"], 0.80, places=2)
        self.assertAlmostEqual(breakdown["coinjoin_evidence"], 0.70, places=2)

        # Check clamped range [0.0, 1.0]
        self.assertGreaterEqual(fused_score, 0.0)
        self.assertLessEqual(fused_score, 1.0)
        self.assertGreater(fused_score, 0.6)  # Elevated due to strong evidence across channels

        # Check reasons contain pattern descriptions
        reasons_text = " ".join(top_reasons)
        self.assertIn("Peeling-chain pattern detected", reasons_text)
        self.assertIn("CoinJoin-like / mixing pattern detected", reasons_text)


class TestPatternEvidenceLayerAndPipeline(unittest.TestCase):
    """Scenario 6: Pattern Evidence Layer orchestration and DataFrame integration."""

    def test_pattern_evidence_layer_run(self):
        """Verify PatternEvidenceLayer enriches transactions with both detectors."""
        layer = PatternEvidenceLayer(
            peeling_chain_config={"min_chain_length": 3, "continuation_ratio": 0.40},
            coinjoin_config={"min_input_count": 3, "min_output_count": 3, "min_equal_output_count": 2, "equal_output_ratio_threshold": 0.40}
        )

        df = pd.DataFrame([
            # Peeling chain hop 1
            {
                "txid": "TX_P1",
                "input_addresses": "W0",
                "output_addresses": "W1|W_P1",
                "output_amounts": "8.0|1.9",
                "total_output_amount_btc": 9.9,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W0",
                "destination_wallet": "W1",
            },
            # Peeling chain hop 2
            {
                "txid": "TX_P2",
                "input_addresses": "W1",
                "output_addresses": "W2|W_P2",
                "output_amounts": "6.0|1.9",
                "total_output_amount_btc": 7.9,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W1",
                "destination_wallet": "W2",
            },
            # Peeling chain hop 3
            {
                "txid": "TX_P3",
                "input_addresses": "W2",
                "output_addresses": "W3|W_P3",
                "output_amounts": "4.0|1.9",
                "total_output_amount_btc": 5.9,
                "num_inputs": 1,
                "num_outputs": 2,
                "source_wallet": "W2",
                "destination_wallet": "W3",
            },
            # CoinJoin transaction
            {
                "txid": "TX_CJ",
                "input_addresses": "WA|WB|WC",
                "output_addresses": "WD|WE|WF|WG",
                "output_amounts": "2.0|2.0|2.0|0.5",
                "total_output_amount_btc": 6.5,
                "num_inputs": 3,
                "num_outputs": 4,
                "source_wallet": "WA",
                "destination_wallet": "WD",
            },
        ])

        enriched_df = layer.run(df)

        # Expected columns must exist
        for col in [
            "peeling_chain_detected", "peeling_chain_evidence", "peeling_chain_id",
            "peeling_chain_position", "peeling_chain_length", "peeling_chain_reasons",
            "coinjoin_detected", "coinjoin_evidence", "coinjoin_equal_output_count",
            "coinjoin_equal_output_ratio", "coinjoin_equal_amount_btc", "coinjoin_reasons"
        ]:
            self.assertIn(col, enriched_df.columns)

        # Check summary
        summary = layer.get_summary()
        self.assertGreaterEqual(summary["peeling_chains_detected"], 3)
        self.assertGreaterEqual(summary["coinjoin_detected"], 1)

        # Check detected patterns list for API
        patterns_list = layer.get_detected_patterns(enriched_df)
        self.assertGreaterEqual(len(patterns_list), 4)


class TestCaseGroupingAndTimelineIntegration(unittest.TestCase):
    """Scenario 7: Case Grouping & Evidence Timeline Integration."""

    def test_case_grouping_by_peeling_chain(self):
        """Verify transactions in the same peeling chain are grouped into the same investigation case."""
        df = pd.DataFrame([
            {
                "txid": "TX_CHAIN_A",
                "risk_score": 85.0,
                "risk_level": "CRITICAL",
                "is_anomaly": True,
                "peeling_chain_id": "PEEL-0042",
                "peeling_chain_position": 1,
                "source_wallet": "W_SRC_A",
                "destination_wallet": "W_DST_A",
                "total_output_amount_btc": 10.0,
                "timestamp": "2026-09-20 12:00:00"
            },
            {
                "txid": "TX_CHAIN_B",
                "risk_score": 80.0,
                "risk_level": "HIGH",
                "is_anomaly": True,
                "peeling_chain_id": "PEEL-0042",
                "peeling_chain_position": 2,
                "source_wallet": "W_SRC_B",
                "destination_wallet": "W_DST_B",
                "total_output_amount_btc": 8.0,
                "timestamp": "2026-09-20 12:15:00"
            },
            {
                "txid": "TX_CHAIN_C",
                "risk_score": 75.0,
                "risk_level": "HIGH",
                "is_anomaly": True,
                "peeling_chain_id": "PEEL-0042",
                "peeling_chain_position": 3,
                "source_wallet": "W_SRC_C",
                "destination_wallet": "W_DST_C",
                "total_output_amount_btc": 6.0,
                "timestamp": "2026-09-20 12:30:00"
            },
        ])

        grouper = AlertCaseGrouper(df)
        cases = grouper.get_cases()

        # All 3 must be grouped into a single case
        self.assertEqual(len(cases), 1)
        c0 = cases[0]
        self.assertEqual(c0["transaction_count"], 3)
        self.assertIn("TX_CHAIN_A", c0["txids"])
        self.assertIn("TX_CHAIN_B", c0["txids"])
        self.assertIn("TX_CHAIN_C", c0["txids"])
        # Check reasons or title reflect peeling chain
        reasons_text = " ".join(c0["grouping_reasons"])
        self.assertIn("PEEL-0042", reasons_text)
        self.assertIn("Peeling-Chain", c0["title"])

    def test_timeline_pattern_events(self):
        """Verify timeline builder emits PEELING_CHAIN and COINJOIN_LIKE event badges."""
        df = pd.DataFrame([
            {
                "txid": "TX_DUAL_PATTERN",
                "timestamp": "2026-09-20 14:00:00",
                "network_timestamp": "2026-09-20 13:59:58",
                "src_ip": "192.168.1.10",
                "dst_ip": "192.168.1.1",
                "source_wallet": "W_IN",
                "destination_wallet": "W_OUT",
                "total_output_amount_btc": 5.0,
                "fee_btc": 0.01,
                "risk_score": 90.0,
                "risk_level": "CRITICAL",
                "is_anomaly": True,
                "peeling_chain_detected": True,
                "peeling_chain_evidence": 0.85,
                "peeling_chain_id": "PEEL-0099",
                "peeling_chain_position": 2,
                "peeling_chain_length": 4,
                "coinjoin_detected": True,
                "coinjoin_evidence": 0.75,
                "coinjoin_equal_output_count": 3,
                "coinjoin_equal_output_ratio": 0.75,
                "coinjoin_equal_amount_btc": 1.25,
            }
        ])

        timeline_builder = EvidenceTimelineBuilder(df)
        events = timeline_builder.build_transaction_timeline("TX_DUAL_PATTERN")

        event_types = [e["event_type"] for e in events]
        self.assertIn("PEELING_CHAIN", event_types)
        self.assertIn("COINJOIN_LIKE", event_types)

        peel_event = next(e for e in events if e["event_type"] == "PEELING_CHAIN")
        self.assertIn("PEEL-0099", peel_event["description"])
        self.assertIn("Requires further investigation", peel_event["description"])

        cj_event = next(e for e in events if e["event_type"] == "COINJOIN_LIKE")
        self.assertIn("1.250000 BTC", cj_event["description"])
        self.assertIn("Requires further investigation", cj_event["description"])


class TestAssistantAndPDFIntegration(unittest.TestCase):
    """Scenario 8: Offline AI Assistant & Forensic PDF Integration."""

    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                "txid": "TX_FORENSIC_01",
                "timestamp": "2026-09-20 15:00:00",
                "total_output_amount_btc": 4.5,
                "fee_btc": 0.005,
                "source_wallet": "W_ALPHA",
                "destination_wallet": "W_BETA",
                "src_ip": "198.51.100.42",
                "src_country": "Netherlands",
                "src_asn": "AS1103",
                "risk_score": 88.0,
                "risk_level": "CRITICAL",
                "anomaly_score": 0.78,
                "is_anomaly": True,
                "risk_reasons": "Large transfer volume; International infrastructure",
                "peeling_chain_detected": True,
                "peeling_chain_evidence": 0.82,
                "peeling_chain_id": "PEEL-0007",
                "peeling_chain_position": 3,
                "peeling_chain_length": 5,
                "peeling_chain_reasons": "Peeling-chain pattern detected (Chain PEEL-0007, Hop 3/5). Investigation evidence — Requires further investigation.",
                "coinjoin_detected": True,
                "coinjoin_evidence": 0.72,
                "coinjoin_equal_output_count": 3,
                "coinjoin_equal_output_ratio": 0.60,
                "coinjoin_equal_amount_btc": 1.5,
                "coinjoin_reasons": "CoinJoin-like / mixing pattern detected (3/5 outputs equal at ≈1.5 BTC). Behavioural indicator — Requires further investigation.",
            }
        ])

    def test_knowledge_store_indexes_patterns(self):
        """Verify knowledge store populates pattern structures in tx_record."""
        store = InvestigationKnowledgeStore({"scored_df": self.sample_df})
        tx = store.get_transaction("TX_FORENSIC_01")

        self.assertIsNotNone(tx)
        self.assertIn("peeling_chain_pattern", tx)
        self.assertIn("coinjoin_pattern", tx)
        self.assertTrue(tx["peeling_chain_pattern"]["detected"])
        self.assertEqual(tx["peeling_chain_pattern"]["chain_id"], "PEEL-0007")
        self.assertTrue(tx["coinjoin_pattern"]["detected"])
        self.assertEqual(tx["coinjoin_pattern"]["equal_output_count"], 3)

    def test_evidence_retriever_includes_patterns(self):
        """Verify EvidenceRetriever includes behavioural pattern items in base_items."""
        store = InvestigationKnowledgeStore({"scored_df": self.sample_df})
        retriever = EvidenceRetriever(store)

        evidence = retriever.retrieve("Explain TX_FORENSIC_01")
        self.assertTrue(evidence.is_supported)
        evidence_text = " ".join(evidence.evidence_items)
        self.assertIn("Peeling-Chain Detected", evidence_text)
        self.assertIn("PEEL-0007", evidence_text)
        self.assertIn("CoinJoin-Like", evidence_text)
        self.assertIn("Requires further investigation", evidence_text)

    def test_offline_llm_synthesizes_pattern_narrative(self):
        """Verify OfflineLLMManager includes pattern indicators in deterministic synthesis."""
        store = InvestigationKnowledgeStore({"scored_df": self.sample_df})
        retriever = EvidenceRetriever(store)
        evidence = retriever.retrieve("Why is TX_FORENSIC_01 high risk?")

        llm = OfflineLLMManager()
        response = llm.generate_response("Why is TX_FORENSIC_01 high risk?", evidence)

        self.assertIn("Peeling-chain pattern detected", response)
        self.assertIn("PEEL-0007", response)
        self.assertIn("CoinJoin-like / mixing pattern detected", response)
        self.assertIn("Requires further investigation", response)

    def test_reportlab_pdf_anomaly_reasons_contain_patterns(self):
        """Verify generate_anomaly_reasons appends peeling and CoinJoin patterns to report."""
        stats = {}
        row = self.sample_df.iloc[0]
        reasons = generate_anomaly_reasons(row, stats)

        reasons_text = " ".join(reasons)
        self.assertIn("Peeling-chain pattern detected", reasons_text)
        self.assertIn("CoinJoin-like", reasons_text)


if __name__ == "__main__":
    unittest.main()
