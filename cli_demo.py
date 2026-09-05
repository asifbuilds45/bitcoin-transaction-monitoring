"""
Terminal CLI Investigation & Demonstration Tool
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Designed specifically for command-line terminal demonstration.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import load_dataset, preprocess_data, get_dataset_summary
from src.geoip_enrichment import enrich_transactions_with_geoip
from src.correlation import EntityCorrelator
from src.graph_analysis import BitcoinGraphEngine
from src.feature_engineering import extract_features
from src.anomaly_detection import BitcoinAnomalyDetector
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.evaluation import evaluate_prototype_predictions

# ANSI Terminal Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"{CYAN}{BOLD}")
    print("=" * 70)
    print("  [+] BITCOIN SENTINEL -- CYBER INVESTIGATION TERMINAL")
    print("      NTRO Problem Statement ID: 26146")
    print("      AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic")
    print("=" * 70)
    print(f"{RESET}")


def run_cli_pipeline():
    print(f"{YELLOW}[*] Initializing offline dataset and processing pipeline...{RESET}")
    raw_df = load_dataset()
    clean_df = preprocess_data(raw_df)
    enriched_df = enrich_transactions_with_geoip(clean_df)
    feat_df = extract_features(enriched_df)
    
    detector = BitcoinAnomalyDetector(contamination=0.10, n_estimators=150, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)
    
    risk_engine = RiskScoringEngine(ml_weight=0.45, heuristic_weight=0.55)
    scored_df = risk_engine.compute_risk_scores(enriched_df, norm_scores, raw_scores, is_anomaly)
    eval_results = evaluate_prototype_predictions(scored_df)
    correlator = EntityCorrelator(scored_df)
    graph_engine = BitcoinGraphEngine(scored_df)
    
    print(f"{GREEN}[✓] Pipeline loaded: 10,000 records processed successfully in offline mode.{RESET}\n")
    return scored_df, eval_results, correlator, graph_engine


def menu_overview(scored_df, eval_results):
    total = len(scored_df)
    anomalies = int(scored_df['is_anomaly'].sum())
    normal = total - anomalies
    critical = int((scored_df['risk_level'] == 'CRITICAL').sum())
    high = int((scored_df['risk_level'] == 'HIGH').sum())

    print(f"\n{BOLD}{CYAN}--- [1] EXECUTIVE SYSTEM OVERVIEW & KPIS ---{RESET}")
    print(f"Total Transactions Monitored: {BOLD}{total:,}{RESET}")
    print(f"Normal Baseline Traffic:      {GREEN}{normal:,} ({normal/total*100:.1f}%){RESET}")
    print(f"AI Detected Anomalies:        {RED}{anomalies:,} ({anomalies/total*100:.1f}%){RESET}")
    print(f"Critical Severity Alerts:     {RED}{BOLD}{critical:,}{RESET}")
    print(f"High Severity Alerts:         {YELLOW}{high:,}{RESET}")
    print("-" * 50)
    print(f"{BOLD}Risk Tier Breakdown:{RESET}")
    for tier, count in scored_df['risk_level'].value_counts().items():
        color = RED if tier == "CRITICAL" else YELLOW if tier == "HIGH" else GREEN
        print(f"  * {tier:<10} : {color}{count:,} records{RESET}")


def menu_ranked_alerts(scored_df):
    print(f"\n{BOLD}{CYAN}--- [2] TOP PRIORITIZED INVESTIGATION ALERTS ---{RESET}")
    alerts = generate_ranked_alerts(scored_df, min_risk="HIGH").head(5)
    
    for _, row in alerts.iterrows():
        print(f"\n{RED}{BOLD}[ALERT #{row['Rank']}] TXID: {row['txid']} | Risk: {row['risk_score']}/100 ({row['risk_level']}) | Anomaly Score: {row['anomaly_score']:.3f}{RESET}")
        print(f"  * Source IP:      {row['src_ip']} ({row['src_country']}) -> Dest IP: {row['dst_ip']} ({row['dst_country']})")
        print(f"  * Source Wallet:  {row['source_wallet']} -> Dest Wallet: {row['destination_wallet']}")
        print(f"  * Volume:         {row['total_output_amount_btc']:.4f} BTC")
        print(f"  * {BOLD}Forensic Evidence:{RESET}")
        for reason in str(row['risk_reasons']).split(';'):
            if reason.strip():
                print(f"    - {YELLOW}{reason.strip()}{RESET}")


def menu_inspect_tx(scored_df, correlator):
    print(f"\n{BOLD}{CYAN}--- [3] FORENSIC TRANSACTION INSPECTOR ---{RESET}")
    txid = input(f"{BOLD}Enter TXID to inspect (e.g., TX10000137 or press Enter for default): {RESET}").strip()
    if not txid:
        txid = scored_df[scored_df['risk_level'] == 'CRITICAL']['txid'].iloc[0]

    matches = scored_df[scored_df['txid'].str.upper() == txid.upper()]
    if matches.empty:
        print(f"{RED}Transaction '{txid}' not found.{RESET}")
        return

    row = matches.iloc[0]
    print(f"\n{BOLD}Forensic Dossier for: {CYAN}{row['txid']}{RESET}")
    print(f"* Timestamp:          {row['timestamp']}")
    print(f"* Risk Score:         {RED if row['risk_score'] >= 75 else YELLOW}{row['risk_score']}/100 ({row['risk_level']}){RESET}")
    print(f"* AI Anomaly Score:   {row['anomaly_score']:.4f}")
    print(f"* Network Telemetry:  {row['src_ip']} ({row['src_country']}) -> {row['dst_ip']} ({row['dst_country']})")
    print(f"* GeoIP Status:       {GREEN}{row['geoip_lookup_status']}{RESET}")
    print(f"* Packets / Bytes:    {row['packet_count']:,} packets / {row['bytes_transferred']:,} bytes")
    print(f"* Blockchain Ledger:  {row['source_wallet']} -> {row['destination_wallet']} ({row['total_output_amount_btc']:.4f} BTC, fee: {row['fee_btc']:.6f} BTC)")
    print(f"* Behavioral Metrics: Velocity={row['transaction_frequency_24h']} tx/24h, TimeGap={row['avg_time_gap_min']:.1f}m, WalletDegree={row['wallet_degree']}, AssociatedIPs={row['unique_ip_count']}")
    print(f"* {BOLD}Automated Evidence Reasons:{RESET}")
    for r in str(row['risk_reasons']).split(';'):
        if r.strip():
            print(f"  {YELLOW}> {r.strip()}{RESET}")


def menu_graph_stats(graph_engine):
    print(f"\n{BOLD}{CYAN}--- [4] NETWORK TOPOLOGY & GRAPH CENTRALITY ---{RESET}")
    summary = graph_engine.get_graph_summary()
    print(f"* Total Graph Nodes:        {summary['total_nodes']:,} (IPs: {summary['ip_nodes']}, TXIDs: {summary['tx_nodes']}, Wallets: {summary['wallet_nodes']})")
    print(f"* Total Directed Edges:     {summary['total_edges']:,}")
    print(f"* Connected Components:     {summary['connected_components_count']}")
    print(f"* Largest Cluster Size:     {summary['largest_component_size']:,} entities")
    
    high_deg = graph_engine.get_high_degree_entities(top_n=5)
    print(f"\n{BOLD}Top Hub IPs (High Degree Broadcast Vantage Points):{RESET}")
    for item in high_deg['top_ips']:
        print(f"  * IP: {item['entity']:<15} | Connections: {GREEN}{item['degree']}{RESET}")

    print(f"\n{BOLD}Top Hub Wallets (High Degree Financial Aggregation Nodes):{RESET}")
    for item in high_deg['top_wallets']:
        print(f"  * Wallet: {item['entity']:<10} | Connections: {MAGENTA}{item['degree']}{RESET}")


def menu_evaluation(eval_results):
    print(f"\n{BOLD}{CYAN}--- [5] PROTOTYPE EVALUATION & BENCHMARKS ---{RESET}")
    print(f"{YELLOW}Note: ground_truth is strictly used for post-prediction evaluation.{RESET}\n")
    print(f"* Precision:             {GREEN}{eval_results['precision']*100:.2f}%{RESET}")
    print(f"* Recall:                {GREEN}{eval_results['recall']*100:.2f}%{RESET}")
    print(f"* F1-Score:              {GREEN}{eval_results['f1_score']*100:.2f}%{RESET}")
    print(f"* ROC-AUC Score:         {GREEN}{eval_results['roc_auc']:.4f}{RESET}")
    
    cm = eval_results['confusion_matrix']
    print(f"\n{BOLD}Confusion Matrix:{RESET}")
    print(f"  True Positives (TP):   {GREEN}{cm['true_positives']:,}{RESET}  |  False Positives (FP):  {RED}{cm['false_positives']:,}{RESET}")
    print(f"  False Negatives (FN):  {RED}{cm['false_negatives']:,}{RESET}  |  True Negatives (TN):   {GREEN}{cm['true_negatives']:,}{RESET}")
    
    print(f"\n{BOLD}Scenario Detection Rates:{RESET}")
    scenario_df = eval_results['scenario_breakdown']
    for _, row in scenario_df.iterrows():
        name = row['Scenario']
        rate = row['Detection Rate (%)']
        if name != 'normal':
            print(f"  * {name:<20}: {GREEN if rate >= 90 else YELLOW}{rate:>6.2f}%{RESET} (Avg Risk: {row['Avg Risk Score']:.1f})")


def main():
    print_banner()
    scored_df, eval_results, correlator, graph_engine = run_cli_pipeline()

    # Direct non-interactive execution test
    menu_overview(scored_df, eval_results)
    menu_ranked_alerts(scored_df)
    menu_evaluation(eval_results)


if __name__ == "__main__":
    main()
