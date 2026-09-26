"""
Generate Distinct 10,000-Row Demonstration Datasets for Bitcoin Monitoring
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Generates high-fidelity, schema-compliant, correlated Network and Blockchain datasets
with distinct timestamps (March 2026), distinct TXIDs (TX20000001 - TX20010000),
distinct IPs (198.51.100.x, 203.0.113.x, etc.), distinct Wallets (W008000 - W009999),
and clear anomalous behavioral patterns (CoinJoin, Peeling Chains, Fan-out, Rapid-fire).
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_datasets(n_rows: int = 10000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating {n_rows} correlated Network & Blockchain records...")

    # Time parameters: March 2026
    start_time = datetime(2026, 3, 1, 0, 0, 0)
    timestamps = []
    current_time = start_time
    for _ in range(n_rows):
        delta = timedelta(seconds=random.randint(15, 300))
        current_time += delta
        timestamps.append(current_time)

    # Entity pools
    txids = [f"TX200{i+1:05d}" for i in range(n_rows)]

    # Wallets
    wallet_pool = [f"W008{i:03d}" for i in range(500)] + [f"W009{i:03d}" for i in range(500)]
    hub_wallets = wallet_pool[:20]  # High degree hubs

    # IPs (distinct public ranges: 198.51.100.0/24, 203.0.113.0/24, 192.0.2.0/24)
    ip_pool = [f"198.51.100.{i}" for i in range(1, 150)] + \
              [f"203.0.113.{i}" for i in range(1, 100)] + \
              [f"192.0.2.{i}" for i in range(1, 80)]
    
    hub_ips = ["198.51.100.44", "203.0.113.99", "198.51.100.88", "192.0.2.15"]

    countries = ["US", "DE", "SG", "JP", "CH", "IN", "GB", "NL", "CA", "FR"]
    country_weights = [0.25, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04]

    asns = ["AS15169", "AS16509", "AS13335", "AS24940", "AS8075", "AS14061", "AS9009"]
    script_types = ["P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR"]

    # Generate records
    net_records = []
    chain_records = []

    # Scenario distributions: ~85% normal, ~15% anomalous
    anom_scenarios = ["coinjoin_like", "peeling_chain", "fan_out", "rapid_fire", "ip_wallet_reuse", "high_fee_anomaly"]

    # Pre-build a peeling chain sequence for 200 hops
    peel_wallet_src = "W008042"
    peel_current_balance = 50.0

    for i in range(n_rows):
        txid = txids[i]
        ts = timestamps[i]
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        # Network timestamp with slight realistic variance (-15s to +2s)
        net_ts = ts - timedelta(seconds=random.uniform(0.5, 12.0))
        net_ts_str = net_ts.strftime("%Y-%m-%d %H:%M:%S")

        # Decide scenario
        if 500 <= i < 700:
            scenario = "peeling_chain"
            is_anom = True
        elif i % 25 == 0:
            scenario = "coinjoin_like"
            is_anom = True
        elif i % 18 == 0:
            scenario = "fan_out"
            is_anom = True
        elif i % 15 == 0:
            scenario = "ip_wallet_reuse"
            is_anom = True
        elif i % 22 == 0:
            scenario = "rapid_fire"
            is_anom = True
        elif i % 30 == 0:
            scenario = "high_fee_anomaly"
            is_anom = True
        else:
            scenario = "normal"
            is_anom = False

        ground_truth = "synthetic_anomalous" if is_anom else "synthetic_normal"

        # Construct layer attributes based on scenario
        if scenario == "peeling_chain":
            src_ip = "198.51.100.44"
            src_wallet = peel_wallet_src
            dst_wallet = f"W008{1000 + (i - 500):04d}"
            change_wallet = f"W009{1000 + (i - 500):04d}"
            peel_spent = 0.20
            fee_btc = 0.0005
            continue_amount = round(peel_current_balance - peel_spent - fee_btc, 6)
            
            num_inputs = 1
            num_outputs = 2
            input_addrs = [src_wallet]
            output_addrs = [dst_wallet, change_wallet]
            input_amounts = [round(peel_current_balance, 6)]
            output_amounts = [peel_spent, continue_amount]
            tot_in = round(peel_current_balance, 6)
            tot_out = round(continue_amount + peel_spent, 6)
            duration = 1.2
            packets = 45
            bytes_tx = 18400
            src_port = 8333
            dst_port = 8333
            peel_wallet_src = change_wallet  # Chain forward
            peel_current_balance = continue_amount

        elif scenario == "coinjoin_like":
            src_ip = random.choice(hub_ips)
            num_inputs = random.choice([3, 4, 5])
            num_outputs = num_inputs
            src_wallet = random.choice(hub_wallets)
            dst_wallet = random.choice(wallet_pool)
            input_addrs = random.sample(wallet_pool, num_inputs)
            output_addrs = random.sample(wallet_pool, num_outputs)
            eq_amount = round(random.choice([0.1, 0.25, 0.5, 1.0]), 6)
            input_amounts = [round(eq_amount + 0.002, 6) for _ in range(num_inputs)]
            output_amounts = [eq_amount for _ in range(num_outputs)]
            fee_btc = round(0.002 * num_inputs, 6)
            tot_in = sum(input_amounts)
            tot_out = sum(output_amounts)
            duration = random.uniform(80.0, 350.0)
            packets = random.randint(300, 1200)
            bytes_tx = packets * random.randint(120, 250)
            src_port = random.choice([8333, 18333])
            dst_port = 8333

        elif scenario == "fan_out":
            src_ip = "203.0.113.99"
            src_wallet = "W008001"
            num_inputs = random.randint(1, 2)
            num_outputs = random.randint(8, 16)
            input_addrs = [src_wallet] if num_inputs == 1 else [src_wallet, random.choice(wallet_pool)]
            output_addrs = random.sample(wallet_pool, num_outputs)
            dst_wallet = output_addrs[0]
            tot_out = round(random.uniform(10.0, 45.0), 6)
            sub_amt = round(tot_out / num_outputs, 6)
            input_amounts = [tot_out + 0.005] if num_inputs == 1 else [tot_out * 0.6, tot_out * 0.4 + 0.005]
            output_amounts = [sub_amt for _ in range(num_outputs)]
            fee_btc = 0.005
            tot_in = sum(input_amounts)
            duration = random.uniform(5.0, 45.0)
            packets = random.randint(500, 2500)
            bytes_tx = packets * 320
            src_port = 8333
            dst_port = 8333

        elif scenario == "high_fee_anomaly":
            src_ip = random.choice(ip_pool)
            src_wallet = random.choice(wallet_pool)
            dst_wallet = random.choice(wallet_pool)
            num_inputs = 1
            num_outputs = 1
            input_addrs = [src_wallet]
            output_addrs = [dst_wallet]
            tot_out = round(random.uniform(0.5, 3.0), 6)
            fee_btc = round(random.uniform(0.15, 0.85), 6)  # Exorbitant fee!
            input_amounts = [round(tot_out + fee_btc, 6)]
            output_amounts = [tot_out]
            tot_in = input_amounts[0]
            duration = random.uniform(0.8, 12.0)
            packets = random.randint(20, 80)
            bytes_tx = packets * 180
            src_port = random.choice([8333, 18444])
            dst_port = 8333

        elif scenario == "ip_wallet_reuse":
            src_ip = "198.51.100.88"
            src_wallet = "W008005"
            dst_wallet = random.choice(wallet_pool)
            num_inputs = random.randint(2, 4)
            num_outputs = random.randint(2, 4)
            input_addrs = [src_wallet] + random.sample(wallet_pool, num_inputs - 1)
            output_addrs = random.sample(wallet_pool, num_outputs)
            tot_out = round(random.uniform(1.0, 8.0), 6)
            sub_amt = round(tot_out / num_outputs, 6)
            input_amounts = [round((tot_out + 0.001) / num_inputs, 6) for _ in range(num_inputs)]
            output_amounts = [sub_amt for _ in range(num_outputs)]
            fee_btc = 0.001
            tot_in = sum(input_amounts)
            duration = random.uniform(2.0, 25.0)
            packets = random.randint(150, 600)
            bytes_tx = packets * 240
            src_port = 8333
            dst_port = 8333

        else:  # Normal transaction
            src_ip = random.choice(ip_pool)
            src_wallet = random.choice(wallet_pool)
            dst_wallet = random.choice(wallet_pool)
            num_inputs = random.choice([1, 2, 3])
            num_outputs = random.choice([1, 2])
            input_addrs = [src_wallet] if num_inputs == 1 else [src_wallet] + random.sample(wallet_pool, num_inputs - 1)
            output_addrs = [dst_wallet] if num_outputs == 1 else [dst_wallet, random.choice(wallet_pool)]
            tot_out = round(random.uniform(0.01, 2.5), 6)
            fee_btc = round(random.uniform(0.0001, 0.0015), 6)
            tot_in = round(tot_out + fee_btc, 6)
            input_amounts = [tot_in] if num_inputs == 1 else [round(tot_in / num_inputs, 6) for _ in range(num_inputs)]
            output_amounts = [tot_out] if num_outputs == 1 else [round(tot_out * 0.7, 6), round(tot_out * 0.3, 6)]
            duration = round(random.uniform(0.5, 120.0), 3)
            packets = random.randint(10, 350)
            bytes_tx = packets * random.randint(150, 450)
            src_port = random.choice([8333, 18333, 18444])
            dst_port = 8333

        src_country = np.random.choice(countries, p=country_weights)
        dst_country = np.random.choice(countries, p=country_weights)
        src_asn = random.choice(asns)
        dst_asn = random.choice(asns)
        dst_ip = f"10.{random.randint(10, 20)}.{random.randint(10, 30)}.{random.randint(1, 10)}"

        # Network row
        net_records.append({
            "txid": txid,
            "timestamp": ts_str,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": "TCP",
            "network_timestamp": net_ts_str,
            "connection_duration_sec": duration,
            "packet_count": packets,
            "bytes_transferred": bytes_tx,
            "src_country": src_country,
            "dst_country": dst_country,
            "src_asn": src_asn,
            "dst_asn": dst_asn
        })

        # Blockchain row
        chain_records.append({
            "txid": txid,
            "timestamp": ts_str,
            "block_height": 860000 + (i // 50),
            "time_step": (i // 100) + 1,
            "input_addresses": "|".join(input_addrs),
            "output_addresses": "|".join(output_addrs),
            "input_amounts": "|".join(f"{a:.6f}" for a in input_amounts),
            "output_amounts": "|".join(f"{a:.6f}" for a in output_amounts),
            "fee_btc": fee_btc,
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "total_input_amount_btc": round(tot_in, 6),
            "total_output_amount_btc": round(tot_out, 6),
            "script_type": random.choice(script_types),
            "source_wallet": src_wallet,
            "destination_wallet": dst_wallet,
            "transaction_frequency_24h": random.randint(2, 25) if not is_anom else random.randint(40, 180),
            "avg_time_gap_min": round(random.uniform(5.0, 180.0), 3) if not is_anom else round(random.uniform(0.5, 10.0), 3),
            "wallet_degree": random.randint(2, 15) if not is_anom else random.randint(30, 120),
            "unique_ip_count": random.randint(1, 4) if not is_anom else random.randint(8, 35),
            "country_count": random.randint(1, 3) if not is_anom else random.randint(4, 10),
            "asn_count": random.randint(1, 2) if not is_anom else random.randint(3, 8),
            "ground_truth": ground_truth,
            "scenario": scenario
        })

    net_df = pd.DataFrame(net_records)
    chain_df = pd.DataFrame(chain_records)

    # Save to primary demo files
    net_df.to_csv("demo_network_10000.csv", index=False)
    chain_df.to_csv("demo_blockchain_10000.csv", index=False)
    print("Saved -> demo_network_10000.csv and demo_blockchain_10000.csv")

    # Also save as explicit batch 2 files for multi-dataset ingestion demonstration
    net_df.to_csv("demo_batch2_network_10000.csv", index=False)
    chain_df.to_csv("demo_batch2_blockchain_10000.csv", index=False)
    print("Saved -> demo_batch2_network_10000.csv and demo_batch2_blockchain_10000.csv")

    anom_count = (chain_df['ground_truth'] == 'synthetic_anomalous').sum()
    print(f"Generated {len(net_df)} records with {anom_count} anomalies ({anom_count/len(net_df)*100:.1f}%)")

if __name__ == "__main__":
    generate_datasets(10000, seed=42)
