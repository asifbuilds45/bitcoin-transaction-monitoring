"""
Pipeline Integration Test
NTRO Problem Statement 26146
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from src.preprocessing import load_dataset, preprocess_data, get_dataset_summary
from src.geoip_enrichment import enrich_transactions_with_geoip
from src.correlation import EntityCorrelator
from src.graph_analysis import BitcoinGraphEngine
from src.feature_engineering import extract_features, prepare_ml_feature_matrix
from src.anomaly_detection import BitcoinAnomalyDetector
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.evaluation import evaluate_prototype_predictions


def run_pipeline_test():
    print("==================================================")
    print("STARTING BITCOIN MONITORING PIPELINE TEST")
    print("==================================================")

    # 1. Ingestion & Preprocessing
    print("\n[Step 1] Ingesting and Preprocessing Dataset...")
    raw_df = load_dataset()
    print(f"Loaded raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")
    
    clean_df = preprocess_data(raw_df)
    print(f"Preprocessed dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns.")
    summary = get_dataset_summary(clean_df)
    print(f"Summary: {summary}")

    # 2. GeoIP Enrichment
    print("\n[Step 2] Offline GeoIP & ASN Enrichment...")
    enriched_df = enrich_transactions_with_geoip(clean_df)
    print(f"Enriched columns added. Status breakdown:\n{enriched_df['geoip_lookup_status'].value_counts()}")

    # 3. Entity Correlation
    print("\n[Step 3] Initializing Entity Correlator...")
    correlator = EntityCorrelator(enriched_df)
    sample_ip = enriched_df['src_ip'].iloc[0]
    sample_wallet = enriched_df['source_wallet'].iloc[0]
    ip_info = correlator.get_ip_correlations(sample_ip)
    wallet_info = correlator.get_wallet_correlations(sample_wallet)
    print(f"Sample IP ({sample_ip}) correlation: {ip_info['transaction_count']} txs, {ip_info['unique_wallets_count']} wallets.")
    print(f"Sample Wallet ({sample_wallet}) correlation: {wallet_info['total_transactions']} txs, {wallet_info['unique_ip_count']} IPs.")

    # 4. Graph Engine
    print("\n[Step 4] Building NetworkX Graph...")
    graph_engine = BitcoinGraphEngine(enriched_df)
    g_summary = graph_engine.get_graph_summary()
    print(f"Graph Topology: {g_summary}")
    high_deg = graph_engine.get_high_degree_entities(top_n=3)
    print(f"Top High Degree IPs: {high_deg['top_ips']}")
    print(f"Top High Degree Wallets: {high_deg['top_wallets']}")

    # 5. Feature Engineering (Strictly Unsupervised)
    print("\n[Step 5] Feature Engineering...")
    features_df = extract_features(enriched_df)
    print(f"Engineered feature matrix: {features_df.shape[0]} rows, {features_df.shape[1]} features.")
    print(f"Features: {list(features_df.columns)}")
    assert 'ground_truth' not in features_df.columns, "Data Leakage Error: ground_truth found in ML features!"
    assert 'scenario' not in features_df.columns, "Data Leakage Error: scenario found in ML features!"

    # 6. AI Anomaly Detection (Isolation Forest)
    print("\n[Step 6] Running Isolation Forest...")
    detector = BitcoinAnomalyDetector(contamination=0.10, n_estimators=150, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(features_df)
    print(f"Predictions completed. Detected anomalies: {int(is_anomaly.sum())} / {len(is_anomaly)}")

    # 7. Risk Scoring & Explainability
    print("\n[Step 7] Calculating 0-100 Risk Scores and Generating Explanations...")
    risk_engine = RiskScoringEngine(ml_weight=0.45, heuristic_weight=0.55)
    scored_df = risk_engine.compute_risk_scores(enriched_df, norm_scores, raw_scores, is_anomaly)
    print("Risk Level Breakdown:\n", scored_df['risk_level'].value_counts())
    
    alerts_df = generate_ranked_alerts(scored_df, min_risk="MEDIUM")
    print(f"Ranked Alerts (Medium+): {len(alerts_df)} records.")
    print("Top Alert Reason:\n", alerts_df['risk_reasons'].iloc[0])

    # 8. Prototype Evaluation (Post-Prediction vs Ground Truth)
    print("\n[Step 8] Evaluating Prototype Performance against Ground Truth...")
    eval_results = evaluate_prototype_predictions(scored_df)
    print(f"Precision: {eval_results['precision']:.4f}")
    print(f"Recall:    {eval_results['recall']:.4f}")
    print(f"F1-Score:  {eval_results['f1_score']:.4f}")
    print(f"ROC-AUC:   {eval_results['roc_auc']:.4f}")
    print(f"Confusion Matrix: TP={eval_results['confusion_matrix']['true_positives']}, FP={eval_results['confusion_matrix']['false_positives']}, TN={eval_results['confusion_matrix']['true_negatives']}, FN={eval_results['confusion_matrix']['false_negatives']}")
    print("\nScenario Breakdown:")
    print(eval_results['scenario_breakdown'])

    print("\n==================================================")
    print("PIPELINE TEST PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_pipeline_test()
