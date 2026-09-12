"""
Comprehensive Pipeline Integration & Unit Test Suite
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Supports both standard library `unittest` and `pytest`:
    python3 -m unittest tests/test_pipeline.py
    python3 tests/test_pipeline.py
"""

import os
import sys
import io
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_dataset, preprocess_data
from src.ingestion.network_ingestion import parse_and_validate_network_data, parse_and_validate_network_csv
from src.ingestion.blockchain_ingestion import parse_and_validate_blockchain_data, parse_and_validate_blockchain_csv
from src.geoip_enrichment import GeoIPEnricher, enrich_transactions_with_geoip, get_geoip_status
from src.correlation import EntityCorrelator, DualLayerCorrelator, calculate_record_correlation_confidence
from src.temporal.temporal_analysis import analyze_temporal_patterns
from src.behavioural.behavioural_fingerprinting import extract_behavioural_evidence_scores
from src.graph_analysis import BitcoinGraphEngine
from src.clustering.dbscan_clustering import BitcoinDBSCANClustering
from src.feature_engineering import extract_features
from src.anomaly_detection import BitcoinAnomalyDetector
from src.fusion.evidence_fusion import MultiLayerEvidenceFusionEngine
from src.explainability.shap_explainer import BitcoinSHAPExplainer
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.evaluation import evaluate_prototype_predictions
from src.reporting import ReportGenerator
from src.investigation import InvestigationPathReconstructor, EvidenceTimelineBuilder, AlertCaseGrouper



def get_test_dataframe() -> pd.DataFrame:
    path = os.path.join("data", "sample_200.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return load_dataset()


class TestGeoIPEnrichment(unittest.TestCase):
    """Part A & K: Offline GeoIP2 validation."""

    def setUp(self):
        self.enricher = GeoIPEnricher()

    def tearDown(self):
        self.enricher.close()

    def test_public_ip_lookup(self):
        res = self.enricher.lookup_ip("8.8.8.8")
        self.assertIn(res["lookup_status"], ["geoip_resolved", "synthetic_fallback"])
        if res["lookup_status"] == "geoip_resolved":
            self.assertEqual(res["country"], "United States")
            self.assertEqual(res["country_code"], "US")
            self.assertIn("AS15169", res["asn"])

    def test_private_ip_lookup(self):
        """Private/synthetic IPs must strictly return Unknown."""
        for priv_ip in ["10.0.0.1", "192.168.1.100", "172.16.5.4", "127.0.0.1"]:
            res = self.enricher.lookup_ip(priv_ip)
            self.assertEqual(res["country"], "Unknown")
            self.assertEqual(res["country_code"], "Unknown")
            self.assertEqual(res["asn"], "Unknown")
            self.assertEqual(res["lookup_status"], "private_ip")

    def test_invalid_ip_lookup(self):
        """Invalid IP addresses must return Unknown without crashing."""
        res = self.enricher.lookup_ip("not-an-ip-address")
        self.assertEqual(res["country"], "Unknown")
        self.assertEqual(res["asn"], "Unknown")
        self.assertEqual(res["lookup_status"], "invalid_ip")

    def test_missing_database_fallback(self):
        """Missing DB path must gracefully fallback without exception."""
        dummy_enricher = GeoIPEnricher(
            country_db_path="non_existent_country.mmdb",
            asn_db_path="non_existent_asn.mmdb"
        )
        status = dummy_enricher.get_status()
        self.assertFalse(status["country_db_loaded"])
        self.assertFalse(status["asn_db_loaded"])
        self.assertEqual(status["mode"], "Fallback")

        res = dummy_enricher.lookup_ip("8.8.8.8", fallback_country="FallbackCountry", fallback_asn="FallbackASN")
        self.assertEqual(res["country"], "FallbackCountry")
        dummy_enricher.close()

    def test_enrich_dataframe_offline(self):
        df = pd.DataFrame([{
            "src_ip": "8.8.8.8",
            "dst_ip": "10.0.0.2",
            "src_country": "US",
            "dst_country": "UNKNOWN",
            "src_asn": "AS15169",
            "dst_asn": "UNKNOWN"
        }])
        enriched = self.enricher.enrich_dataframe(df)
        self.assertIn("geoip_src_country", enriched.columns)
        self.assertIn("geoip_src_asn", enriched.columns)
        self.assertIn("geoip_lookup_status", enriched.columns)


class TestMultiFormatIngestion(unittest.TestCase):
    """Part B & C: Multi-format (CSV, JSON, XML) two-dataset ingestion."""

    def test_csv_ingestion(self):
        raw = get_test_dataframe()
        net_df, net_warn = parse_and_validate_network_data(raw)
        self.assertFalse(net_df.empty)
        self.assertIn("src_ip_valid", net_df.columns)

        chain_df, chain_warn = parse_and_validate_blockchain_data(raw)
        self.assertFalse(chain_df.empty)
        self.assertIn("parsed_input_addresses", chain_df.columns)

    def test_json_ingestion(self):
        sample = [{
            "txid": "test_tx_001",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 8333,
            "dst_port": 8333,
            "timestamp": "2026-09-08 12:00:00"
        }]
        json_buf = io.StringIO(pd.DataFrame(sample).to_json(orient="records"))
        json_buf.name = "network_data.json"
        net_df, warn = parse_and_validate_network_data(json_buf)
        self.assertEqual(len(net_df), 1)
        self.assertEqual(net_df["txid"].iloc[0], "test_tx_001")

    def test_xml_ingestion(self):
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <transactions>
            <transaction>
                <txid>xml_tx_001</txid>
                <source_wallet>1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa</source_wallet>
                <destination_wallet>3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy</destination_wallet>
                <total_output_amount_btc>1.5</total_output_amount_btc>
            </transaction>
        </transactions>
        """
        xml_buf = io.BytesIO(xml_content.encode('utf-8'))
        xml_buf.name = "blockchain.xml"
        chain_df, warn = parse_and_validate_blockchain_data(xml_buf)
        self.assertEqual(len(chain_df), 1)
        self.assertEqual(chain_df["txid"].iloc[0], "xml_tx_001")


class TestCorrelationAndConfidence(unittest.TestCase):
    """Part D & I: Correlation & normalized Confidence Score."""

    def test_dual_layer_correlation(self):
        raw = get_test_dataframe()
        net_df, _ = parse_and_validate_network_data(raw)
        chain_df, _ = parse_and_validate_blockchain_data(raw)

        correlator = DualLayerCorrelator(net_df, chain_df)
        corr_df = correlator.get_correlated_dataframe()
        self.assertIn("correlation_confidence", corr_df.columns)
        self.assertTrue((corr_df["correlation_confidence"] >= 0.0).all())
        self.assertTrue((corr_df["correlation_confidence"] <= 1.0).all())

    def test_confidence_score_calculation(self):
        conf = calculate_record_correlation_confidence(
            net_ts="2026-09-08 12:00:00",
            block_ts="2026-09-08 12:00:10",
            has_exact_txid=True,
            src_port=8333,
            protocol="TCP"
        )
        self.assertGreaterEqual(conf, 0.70)
        self.assertLessEqual(conf, 1.0)

    def test_non_attribution_wording(self):
        """Ensure IP-wallet association avoids claiming ownership."""
        raw = get_test_dataframe()
        net_df, _ = parse_and_validate_network_data(raw)
        chain_df, _ = parse_and_validate_blockchain_data(raw)
        correlator = DualLayerCorrelator(net_df, chain_df)
        sample_ip = net_df["src_ip"].iloc[0]
        ip_info = correlator.get_ip_correlations(sample_ip)
        self.assertNotIn("owner", ip_info["correlation_descriptor"].lower())
        self.assertNotIn("belongs to", ip_info["correlation_descriptor"].lower())


class TestMachineLearning(unittest.TestCase):
    """Part E: Isolation Forest & DBSCAN without ground truth leakage."""

    def test_isolation_forest_and_dbscan(self):
        raw = get_test_dataframe()
        clean_df = preprocess_data(raw)
        feat_df = extract_features(clean_df)

        self.assertNotIn("ground_truth", feat_df.columns)
        self.assertNotIn("scenario", feat_df.columns)

        detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
        is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)
        self.assertEqual(len(is_anomaly), len(feat_df))
        self.assertTrue((norm_scores >= 0.0).all() and (norm_scores <= 1.0).all())

        dbscan = BitcoinDBSCANClustering(eps=0.5, min_samples=3)
        labels, noise_mask, dbscan_ev = dbscan.fit_predict(feat_df)
        self.assertEqual(len(labels), len(feat_df))
        self.assertEqual(len(noise_mask), len(feat_df))
        self.assertTrue((dbscan_ev >= 0.0).all() and (dbscan_ev <= 1.0).all())


class TestRiskAndRankedAlerts(unittest.TestCase):
    """Part J: Risk scoring, Confidence score, and exact tier filtering."""

    def test_ranked_alerts_and_filtering(self):
        raw = get_test_dataframe()
        clean_df = preprocess_data(raw)
        feat_df = extract_features(clean_df)

        detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
        is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)

        risk_engine = RiskScoringEngine()
        scored_df = risk_engine.compute_risk_scores(clean_df, norm_scores, raw_scores, is_anomaly)

        # Generate ranked alerts
        alerts_all = generate_ranked_alerts(scored_df, "ALL")
        self.assertIn("confidence_score", alerts_all.columns)
        self.assertIn("correlation_confidence_pct", alerts_all.columns)
        self.assertIn("risk_score", alerts_all.columns)
        self.assertIn("anomaly_score", alerts_all.columns)

        # Test exact filtering
        for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            filtered = generate_ranked_alerts(scored_df, tier)
            if not filtered.empty:
                self.assertTrue((filtered["risk_level"] == tier).all())


class TestReportGeneration(unittest.TestCase):
    """Part K: ReportLab PDF report generation with all anomalies."""

    def test_pdf_report_batch_generation(self):
        raw = get_test_dataframe()
        clean_df = preprocess_data(raw)
        feat_df = extract_features(clean_df)

        detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
        is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)

        risk_engine = RiskScoringEngine()
        scored_df = risk_engine.compute_risk_scores(clean_df, norm_scores, raw_scores, is_anomaly)
        stats = ReportGenerator.compute_dataset_statistics(scored_df)

        # Generate batch report for top 3 anomalies
        pdf_bytes = ReportGenerator.generate_batch_report_pdf(scored_df, stats, max_pages=3)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)


class TestInvestigatorCentricFeatures(unittest.TestCase):
    """Validation of the 3 new investigator-centric capabilities."""

    @classmethod
    def setUpClass(cls):
        raw = get_test_dataframe()
        cls.clean_df = preprocess_data(raw)
        feat_df = extract_features(cls.clean_df)

        detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
        is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)

        dbscan = BitcoinDBSCANClustering(eps=0.5, min_samples=5)
        cluster_labels, noise_mask, dbscan_evidence = dbscan.fit_predict(feat_df)
        cls.clean_df['dbscan_cluster'] = cluster_labels
        cls.clean_df['dbscan_is_noise'] = noise_mask

        temp_df = analyze_temporal_patterns(cls.clean_df)
        beh_df = extract_behavioural_evidence_scores(temp_df)
        cls.graph_engine = BitcoinGraphEngine(beh_df)

        risk_engine = RiskScoringEngine()
        cls.scored_df = risk_engine.compute_risk_scores(
            beh_df, norm_scores, raw_scores, is_anomaly,
            dbscan_evidence=dbscan_evidence
        )

    def test_investigation_path_reconstruction(self):
        """Verify bounded path reconstruction and non-attribution semantics."""
        reconstructor = InvestigationPathReconstructor(self.scored_df, self.graph_engine)
        sample_txid = str(self.scored_df['txid'].iloc[0])

        paths = reconstructor.reconstruct_path(sample_txid, max_depth=3)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

        for p in paths:
            self.assertIn("path_id", p)
            self.assertIn("priority_score", p)
            self.assertIn("steps", p)
            self.assertLessEqual(p["total_hops"], 4)

            steps = p["steps"]
            self.assertEqual(steps[0]["entity_type"], "IP")
            self.assertEqual(steps[1]["entity_type"], "TXID")
            self.assertEqual(steps[1]["entity_id"], sample_txid)

            # Strict non-attribution assertion: IP step describes network observation, not wallet ownership
            self.assertIn("network observation", steps[0]["relationship"].lower())
            self.assertNotIn("belongs to wallet owner", steps[0]["relationship"].lower())

    def test_evidence_timeline_chronological_sequence(self):
        """Verify chronological sequence and event metadata in evidence timeline."""
        builder = EvidenceTimelineBuilder(self.scored_df)
        sample_txid = str(self.scored_df['txid'].iloc[0])

        events = builder.build_transaction_timeline(sample_txid)
        self.assertIsInstance(events, list)
        self.assertGreater(len(events), 0)

        event_types = [ev["event_type"] for ev in events]
        self.assertIn("TRANSACTION_OBSERVED", event_types)

        # Assert monotonic chronological ordering where timestamps exist
        valid_ts = [ev["raw_timestamp"] for ev in events if pd.notna(ev.get("raw_timestamp"))]
        for i in range(len(valid_ts) - 1):
            self.assertLessEqual(valid_ts[i], valid_ts[i + 1])

        # Test case timeline across multiple transactions
        multi_txs = self.scored_df['txid'].head(3).tolist()
        case_events = builder.build_case_timeline(multi_txs)
        self.assertIsInstance(case_events, list)
        self.assertGreaterEqual(len(case_events), len(events))

    def test_alert_case_grouping_and_deduplication(self):
        """Verify deterministic case grouping, stable IDs, and explainable rationale."""
        case_grouper = AlertCaseGrouper(self.scored_df, self.graph_engine)
        cases = case_grouper.get_cases()

        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)

        # Verify stable case IDs (CASE-001, CASE-002, ...)
        for idx, c in enumerate(cases, 1):
            expected_id = f"CASE-{idx:03d}"
            self.assertEqual(c["case_id"], expected_id)
            self.assertIn(c["severity"], ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
            self.assertGreaterEqual(c["transaction_count"], 1)
            self.assertGreater(len(c["wallets"]), 0)
            self.assertGreater(len(c["grouping_reasons"]), 0)

        # Test lookup by ID
        c1 = case_grouper.get_case_by_id("CASE-001")
        self.assertIsNotNone(c1)
        self.assertEqual(c1["case_id"], "CASE-001")

        # Test summary dataframe
        summary_df = case_grouper.get_case_summary_dataframe()
        self.assertFalse(summary_df.empty)
        self.assertIn("Case ID", summary_df.columns)
        self.assertIn("Severity", summary_df.columns)
        self.assertIn("Transactions", summary_df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)

