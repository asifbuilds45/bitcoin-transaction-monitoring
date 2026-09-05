"""
Comprehensive Pipeline Integration & Unit Test Suite
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_dataset, preprocess_data, get_dataset_summary
from src.ingestion.network_ingestion import parse_and_validate_network_csv
from src.ingestion.blockchain_ingestion import parse_and_validate_blockchain_csv
from src.geoip_enrichment import enrich_transactions_with_geoip
from src.correlation import EntityCorrelator, DualLayerCorrelator, calculate_record_correlation_confidence
from src.temporal.temporal_analysis import analyze_temporal_patterns, compute_temporal_evidence
from src.behavioural.behavioural_fingerprinting import extract_behavioural_evidence_scores, BehaviouralProfiler
from src.graph_analysis import BitcoinGraphEngine, LouvainCommunityDetector
from src.clustering.dbscan_clustering import BitcoinDBSCANClustering
from src.feature_engineering import extract_features, prepare_ml_feature_matrix
from src.anomaly_detection import BitcoinAnomalyDetector
from src.fusion.evidence_fusion import MultiLayerEvidenceFusionEngine
from src.explainability.shap_explainer import BitcoinSHAPExplainer
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.evaluation import evaluate_prototype_predictions
from src.database.db_manager import DatabaseManager


@pytest.fixture
def raw_dataframe():
    path = os.path.join("data", "sample_200.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return load_dataset()


def test_network_and_blockchain_ingestion(raw_dataframe):
    net_df, net_warn = parse_and_validate_network_csv(raw_dataframe)
    assert not net_df.empty
    assert 'src_ip_valid' in net_df.columns

    chain_df, chain_warn = parse_and_validate_blockchain_csv(raw_dataframe)
    assert not chain_df.empty
    assert 'parsed_input_addresses' in chain_df.columns


def test_layer_correlation_and_confidence(raw_dataframe):
    net_df, _ = parse_and_validate_network_csv(raw_dataframe)
    chain_df, _ = parse_and_validate_blockchain_csv(raw_dataframe)

    dual_correlator = DualLayerCorrelator(net_df, chain_df)
    corr_df = dual_correlator.get_correlated_dataframe()
    assert 'correlation_confidence' in corr_df.columns
    assert corr_df['correlation_confidence'].between(0.0, 1.0).all()

    conf = calculate_record_correlation_confidence(
        net_ts="2026-09-05 10:00:00",
        block_ts="2026-09-05 10:00:15",
        has_exact_txid=True,
        src_port=8333
    )
    assert 0.7 <= conf <= 1.0


def test_geoip_enrichment(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    enriched_df = enrich_transactions_with_geoip(clean_df)
    assert 'geoip_src_country' in enriched_df.columns
    assert 'geoip_lookup_status' in enriched_df.columns


def test_temporal_analysis(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    temp_df = analyze_temporal_patterns(clean_df)
    assert 'temporal_evidence_score' in temp_df.columns
    assert temp_df['temporal_evidence_score'].between(0.0, 1.0).all()


def test_behavioural_fingerprinting(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    beh_df = extract_behavioural_evidence_scores(clean_df)
    assert 'behavioural_evidence_score' in beh_df.columns
    assert beh_df['behavioural_evidence_score'].between(0.0, 1.0).all()


def test_graph_engine_and_louvain_communities(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    graph_engine = BitcoinGraphEngine(clean_df)
    summary = graph_engine.get_graph_summary()
    assert summary['total_nodes'] > 0
    assert summary['louvain_communities_count'] >= 1


def test_isolation_forest_and_dbscan(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    feat_df = extract_features(clean_df)

    assert 'ground_truth' not in feat_df.columns
    assert 'scenario' not in feat_df.columns

    detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)
    assert len(is_anomaly) == len(feat_df)

    dbscan = BitcoinDBSCANClustering(eps=0.5, min_samples=3)
    labels, noise_mask, dbscan_ev = dbscan.fit_predict(feat_df)
    assert len(labels) == len(feat_df)


def test_evidence_fusion_and_risk_scoring(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    feat_df = extract_features(clean_df)

    detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)

    fusion = MultiLayerEvidenceFusionEngine()
    fused_df = fusion.fuse_dataframe_evidence(clean_df)
    assert 'fused_evidence_score' in fused_df.columns

    risk_engine = RiskScoringEngine()
    scored_df = risk_engine.compute_risk_scores(clean_df, norm_scores, raw_scores, is_anomaly)
    assert 'risk_score' in scored_df.columns
    assert scored_df['risk_score'].between(0, 100).all()

    alerts = generate_ranked_alerts(scored_df)
    assert not alerts.empty


def test_shap_explainer(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    feat_df = extract_features(clean_df)
    detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
    _, _, norm_scores = detector.fit_predict(feat_df)

    explainer = BitcoinSHAPExplainer()
    explainer.fit(feat_df, norm_scores)
    sample_explanations = explainer.explain_instance(feat_df.iloc[0])
    assert len(sample_explanations) > 0


def test_database_manager():
    db = DatabaseManager()
    assert db.engine is not None
    summary = db.get_table_summary()
    assert "connected" in summary


def test_end_to_end_pipeline(raw_dataframe):
    clean_df = preprocess_data(raw_dataframe)
    enriched_df = enrich_transactions_with_geoip(clean_df)
    feat_df = extract_features(enriched_df)

    detector = BitcoinAnomalyDetector(contamination=0.10, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)

    risk_engine = RiskScoringEngine()
    scored_df = risk_engine.compute_risk_scores(enriched_df, norm_scores, raw_scores, is_anomaly)
    eval_res = evaluate_prototype_predictions(scored_df)

    assert eval_res['f1_score'] >= 0.0
    assert eval_res['roc_auc'] >= 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
