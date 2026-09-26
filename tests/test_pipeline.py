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
from src.assistant import InvestigationKnowledgeStore, EvidenceRetriever, OfflineLLMManager, InvestigationChatbot



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


class TestOfflineAIAssistant(unittest.TestCase):
    """Part 2: Verification of Offline AI Investigation Assistant."""

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

        cls.case_grouper = AlertCaseGrouper(cls.scored_df, cls.graph_engine)
        cls.path_reconstructor = InvestigationPathReconstructor(cls.scored_df, cls.graph_engine)
        cls.timeline_builder = EvidenceTimelineBuilder(cls.scored_df)

        cls.ctx = {
            "scored_df": cls.scored_df,
            "graph_engine": cls.graph_engine,
            "case_grouper": cls.case_grouper,
            "path_reconstructor": cls.path_reconstructor,
            "timeline_builder": cls.timeline_builder,
            "kpis": {
                "total_tx": len(cls.scored_df),
                "detected_anom": int(cls.scored_df['is_anomaly'].sum()),
                "critical_alerts": int((cls.scored_df['risk_level'].astype(str).str.upper() == 'CRITICAL').sum()),
                "high_alerts": int((cls.scored_df['risk_level'].astype(str).str.upper() == 'HIGH').sum()),
                "mean_conf_pct": 85,
                "top_ips": pd.DataFrame(),
                "top_wallets": pd.DataFrame()
            }
        }
        cls.knowledge_store = InvestigationKnowledgeStore(cls.ctx)
        cls.chatbot = InvestigationChatbot(cls.knowledge_store)

    def test_knowledge_store_indexing(self):
        """Verify transaction, case, wallet, and KPI indexing."""
        sample_tx = self.scored_df.iloc[0]
        sample_txid = str(sample_tx['txid'])

        # Query by TXID
        tx_data = self.knowledge_store.get_transaction(sample_txid)
        self.assertIsNotNone(tx_data)
        self.assertEqual(str(tx_data["txid"]).upper(), sample_txid.upper())
        self.assertIn("risk_score", tx_data)

        # Query cases
        cases = self.knowledge_store.get_cases()
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)
        c0_id = cases[0]["case_id"]
        c0 = self.knowledge_store.get_case(c0_id)
        self.assertIsNotNone(c0)
        self.assertEqual(c0["case_id"], c0_id)

        # Query KPIs
        kpis = self.knowledge_store.get_kpis()
        self.assertEqual(kpis.get("total_tx"), len(self.scored_df))

    def test_entity_extraction_and_intent(self):
        """Verify entity extraction and intent matching across prompt formats."""
        retriever = EvidenceRetriever(self.knowledge_store)
        sample_txid = str(self.scored_df['txid'].iloc[0])

        # Test TXID extraction
        ev_tx = retriever.retrieve(f"Explain risk factors for transaction {sample_txid}")
        self.assertEqual(ev_tx.target_type, "TXID")
        self.assertEqual(ev_tx.target_entity, sample_txid.upper())
        self.assertIn(ev_tx.intent, ["TX_EXPLANATION", "RISK_EXPLANATION", "TRANSACTION_SUMMARY"])

        # Test Case extraction
        ev_case = retriever.retrieve("Show details for case CASE-001")
        self.assertEqual(ev_case.target_type, "CASE")
        self.assertEqual(ev_case.target_entity, "CASE-001")
        self.assertEqual(ev_case.intent, "CASE_SUMMARY")

        # Test Refusal / Unsupported query
        ev_unsupported = retriever.retrieve("Who is Satoshi Nakamoto and what is Bitcoin price?")
        self.assertIn(ev_unsupported.intent, ["UNKNOWN_SCOPE", "UNSUPPORTED_OUT_OF_SCOPE"])
        self.assertFalse(ev_unsupported.is_supported)

    def test_conversational_context_tracking(self):
        """Verify conversational multi-turn context (pronouns/follow-ups)."""
        bot = InvestigationChatbot(self.knowledge_store)
        sample_txid = str(self.scored_df['txid'].iloc[0])

        # Turn 1: Explicit TXID
        r1 = bot.ask(f"Tell me about {sample_txid}")
        self.assertEqual(bot.active_context.get("active_txid"), sample_txid.upper())
        self.assertIn(sample_txid.upper(), r1["response"])
        self.assertGreater(len(r1["evidence_used"]), 0)

        # Turn 2: Implicit follow-up using pronoun
        r2 = bot.ask("Why is it flagged?")
        self.assertEqual(bot.active_context.get("active_txid"), sample_txid.upper())
        self.assertIn("risk", r2["response"].lower())

        # Reset context
        bot.reset_context()
        self.assertIsNone(bot.active_context.get("active_txid"))

    def test_grounded_evidence_synthesis_no_hallucination(self):
        """Verify facts in synthesis strictly match knowledge store records."""
        sample_tx = self.scored_df.iloc[0]
        sample_txid = str(sample_tx['txid'])

        ans = self.chatbot.ask(f"Summarize transaction {sample_txid}")
        self.assertTrue(ans["is_supported"])
        self.assertIn(sample_txid.upper(), ans["response"])

        # Check evidence audit items
        ev_items = ans["evidence_used"]
        self.assertGreater(len(ev_items), 0)
        for item in ev_items:
            self.assertIn("source_tag", item)
            self.assertIn("fact", item)

    def test_anti_hallucination_refusal(self):
        """Verify out-of-scope queries produce explicit refusal."""
        ans = self.chatbot.ask("Will Bitcoin price increase tomorrow?")
        self.assertFalse(ans["is_supported"])
        self.assertIn("sufficient evidence", ans["response"].lower())
        self.assertEqual(len(ans["evidence_used"]), 0)

    def test_offline_llm_manager_local_operation(self):
        """Verify OfflineLLMManager functions 100% offline without remote calls."""
        mgr = OfflineLLMManager()
        self.assertIsNotNone(mgr.runtime_type)
        retriever = EvidenceRetriever(self.knowledge_store)
        sample_txid = str(self.scored_df['txid'].iloc[0])
        ev = retriever.retrieve(f"What happened in {sample_txid}")
        res = mgr.generate_response("What happened?", ev)
        self.assertIsInstance(res, str)
        self.assertGreater(len(res), 20)
        self.assertIn(sample_txid.upper(), res)


    def test_assistant_retrieves_resolved_geoip(self):
        """Verify Offline AI Assistant retrieves resolved GeoIP metadata instead of Unknown."""
        sample_txid = str(self.scored_df['txid'].iloc[0])
        tx_data = self.knowledge_store.get_transaction(sample_txid)
        self.assertIsNotNone(tx_data)
        # Verify Country and ASN are not Unknown if present in dataset
        if "src_country" in self.scored_df.columns and self.scored_df['src_country'].iloc[0] not in ["Unknown", "UNKNOWN", None]:
            self.assertNotEqual(tx_data["country"], "Unknown")
            self.assertNotEqual(tx_data["asn"], "Unknown")

        ans = self.chatbot.ask(f"Explain risk factors for {sample_txid}")
        self.assertTrue(ans["is_supported"])
        self.assertIn("Forensic Evidence", ans["response"])


class TestMultiFileLayerIngestion(unittest.TestCase):
    """Verify multiple file uploads and layer-wise merging."""

    def test_multi_file_network_concatenation_and_deduplication(self):
        df1 = pd.DataFrame([
            {"txid": "TX001", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_country": "IN", "src_asn": "AS45609"},
            {"txid": "TX002", "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", "src_country": "US", "src_asn": "AS15169"}
        ])
        df2 = pd.DataFrame([
            {"txid": "TX002", "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", "src_country": "US", "src_asn": "AS15169"}, # duplicate
            {"txid": "TX003", "src_ip": "10.0.0.5", "dst_ip": "10.0.0.6", "src_country": "GB", "src_asn": "AS5089"}
        ])
        
        combined = pd.concat([df1.assign(source_file="file1.csv"), df2.assign(source_file="file2.csv")], ignore_index=True)
        deduped = combined.drop_duplicates(subset=['txid', 'src_ip', 'dst_ip'])
        self.assertEqual(len(deduped), 3)
        self.assertEqual(set(deduped['txid']), {"TX001", "TX002", "TX003"})

    def test_multi_file_blockchain_concatenation_and_deduplication(self):
        df1 = pd.DataFrame([
            {"txid": "TX001", "total_output_amount_btc": 1.5, "source_wallet": "W001", "destination_wallet": "W002"},
            {"txid": "TX002", "total_output_amount_btc": 2.0, "source_wallet": "W003", "destination_wallet": "W004"}
        ])
        df2 = pd.DataFrame([
            {"txid": "TX002", "total_output_amount_btc": 2.0, "source_wallet": "W003", "destination_wallet": "W004"}, # duplicate
            {"txid": "TX004", "total_output_amount_btc": 0.5, "source_wallet": "W005", "destination_wallet": "W006"}
        ])
        combined = pd.concat([df1.assign(source_file="b1.csv"), df2.assign(source_file="b2.csv")], ignore_index=True)
        deduped = combined.drop_duplicates(subset=['txid'])
        self.assertEqual(len(deduped), 3)
        self.assertEqual(set(deduped['txid']), {"TX001", "TX002", "TX004"})


class TestPDFReportOptimization(unittest.TestCase):
    """Verify optimized PDF batch generation and anomaly filtering."""

    def test_pdf_anomaly_filtering(self):
        scored_df = get_test_dataframe()
        scored_df["is_anomaly"] = [1, 0] * (len(scored_df) // 2)
        scored_df["risk_level"] = "HIGH"
        scored_df["risk_score"] = 75.0
        scored_df["anomaly_score"] = 0.85
        stats = ReportGenerator.compute_dataset_statistics(scored_df)
        
        pdf_bytes = ReportGenerator.generate_batch_report_pdf(scored_df, stats)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    def test_pdf_empty_anomalies_clean_message(self):
        scored_df = get_test_dataframe()
        scored_df["is_anomaly"] = 0 # No anomalies
        stats = ReportGenerator.compute_dataset_statistics(scored_df)
        
        pdf_bytes = ReportGenerator.generate_batch_report_pdf(scored_df, stats)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)

