import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from app import run_investigation_pipeline

net_df = pd.read_csv('demo_network_10000.csv')
chain_df = pd.read_csv('demo_blockchain_10000.csv')
ctx = run_investigation_pipeline(net_df, chain_df)
df = ctx['scored_df']

print("=== TOP CRITICAL TXIDS ===")
crit = df[df['risk_level_normalized'] == 'CRITICAL'].sort_values('risk_score', ascending=False)
for idx, r in crit.head(5).iterrows():
    print(f"TXID: {r['txid']} | Risk: {r['risk_score']:.1f} | Scenario: {r.get('scenario')} | IP: {r['src_ip']} | Wallet: {r['source_wallet']}")

print("\n=== TOP SUSPICIOUS IPS ===")
top_ips = df[df['risk_level_normalized'].isin(['HIGH', 'CRITICAL'])].groupby('src_ip').size().sort_values(ascending=False).head(5)
print(top_ips)

print("\n=== TOP SUSPICIOUS WALLETS ===")
top_wallets = df.groupby('source_wallet')['risk_score'].max().sort_values(ascending=False).head(5)
print(top_wallets)

print("\n=== PEELING CHAIN SAMPLE ===")
peel = df[df.get('peeling_chain_detected', False) == True]
if not peel.empty:
    for idx, r in peel.head(3).iterrows():
        print(f"TXID: {r['txid']} | Chain: {r.get('peeling_chain_id')} | Hop: {r.get('peeling_hop_index')} | Risk: {r['risk_score']:.1f}")
else:
    print("None flagged directly as peeling chain")

print("\n=== COINJOIN SAMPLE ===")
cj = df[df.get('coinjoin_detected', False) == True]
if not cj.empty:
    for idx, r in cj.head(3).iterrows():
        print(f"TXID: {r['txid']} | Wallet: {r['source_wallet']} | Inputs: {r['num_inputs']} | Outputs: {r['num_outputs']} | Risk: {r['risk_score']:.1f}")

print("\n=== FORENSIC CASES ===")
cases = ctx['case_grouper'].get_cases(min_transactions=1)
for c in cases[:3]:
    print(f"Case ID: {c['case_id']} | Severity: {c['severity']} | Title: {c['title']} | TXs: {c['transaction_count']} | Primary TXID: {c['primary_txid']}")
