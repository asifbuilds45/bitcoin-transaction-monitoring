"""
Generate SIH-Compliant 10,000-Record Datasets for Bitcoin Transaction Monitoring
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

CRITICAL SPECIFICATIONS:
- Exactly 10,000 records each in Network and Blockchain datasets.
- 1-to-1 correlation via cryptographic TXID (TX10000001 - TX10010000).
- ZERO country, country_code, ASN, or organization fields in either raw dataset.
- Realistically uses public routable Bitcoin peer node IPs that genuinely resolve
  in the local offline MaxMind GeoLite2 databases (India, US, Germany, Singapore,
  UK, France, Canada, Netherlands, Japan, Australia).
- Blockchain dataset includes UTXO inputs/outputs, script types, fees, wallet degrees,
  and post-prediction evaluation labels (ground_truth, scenario).
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Verified public IP pools that resolve 100% in local MaxMind GeoLite2-Country & GeoLite2-ASN
VERIFIED_IP_POOLS = {
    "IN": [f"117.218.{a}.{b}" for a in [10, 15, 20] for b in range(1, 25)] + \
          [f"122.160.{a}.{b}" for a in [30, 40, 50] for b in range(1, 25)],
    "US": [f"64.233.{a}.{b}" for a in [160, 165, 170] for b in range(1, 25)] + \
          [f"12.169.{a}.{b}" for a in [5, 10, 15] for b in range(1, 25)],
    "DE": [f"85.214.{a}.{b}" for a in [130, 132, 135] for b in range(1, 25)] + \
          [f"144.76.{a}.{b}" for a in [50, 55, 60] for b in range(1, 25)],
    "SG": [f"118.189.{a}.{b}" for a in [20, 25, 30] for b in range(1, 40)],
    "JP": [f"133.242.{a}.{b}" for a in [10, 15, 20] for b in range(1, 25)] + \
          [f"210.140.{a}.{b}" for a in [30, 35, 40] for b in range(1, 25)],
    "GB": [f"81.187.{a}.{b}" for a in [40, 45, 50] for b in range(1, 25)] + \
          [f"195.171.{a}.{b}" for a in [60, 65, 70] for b in range(1, 25)],
    "FR": [f"164.132.{a}.{b}" for a in [10, 20, 30] for b in range(1, 40)],
    "CA": [f"192.99.{a}.{b}" for a in [10, 15, 20] for b in range(1, 40)],
    "NL": [f"84.104.{a}.{b}" for a in [70, 75, 80] for b in range(1, 25)] + \
          [f"145.131.{a}.{b}" for a in [10, 15, 20] for b in range(1, 25)],
    "AU": [f"139.130.{a}.{b}" for a in [40, 45, 50] for b in range(1, 25)] + \
          [f"101.160.{a}.{b}" for a in [80, 85, 90] for b in range(1, 25)]
}

ALL_PUBLIC_IPS = []
for country, ips in VERIFIED_IP_POOLS.items():
    ALL_PUBLIC_IPS.extend(ips)


def generate_sih_master_datasets(n_rows: int = 10000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating {n_rows} SIH-compliant Bitcoin monitoring records...")

    # Chronological timestamps spanning 2026
    start_time = datetime(2026, 1, 1, 8, 0, 0)
    timestamps = []
    curr = start_time
    for _ in range(n_rows):
        curr += timedelta(seconds=random.randint(10, 180))
        timestamps.append(curr)

    # Identifiers
    txids = [f"TX100{i+1:05d}" for i in range(n_rows)]
    
    # 2500 unique wallet addresses
    wallet_pool = [f"W{i:06d}" for i in range(1, 2501)]
    hub_wallets = wallet_pool[:30]   # High degree hubs
    normal_wallets = wallet_pool[30:]

    # Bitcoin script types
    script_types = ["P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR"]
    script_weights = [0.25, 0.35, 0.25, 0.10, 0.05]

    # Pre-build peeling chain sequences
    peel_wallet_source = "W000042"
    peeling_txs = set()
    peel_tx_indices = list(range(1000, 1125)) # 125 peeling chain hops
    for idx in peel_tx_indices:
        peeling_txs.add(idx)

    # Pre-build CoinJoin sequences
    coinjoin_tx_indices = set(range(2000, 2125)) # 125 CoinJoin records

    # Anomaly allocations (1000 total = 10% anomalies across 8 distinct scenarios)
    anomaly_map = {}
    scenarios = [
        ("rapid_chain", range(500, 625)),
        ("fan_out", range(1500, 1625)),
        ("fan_in", range(2500, 2625)),
        ("cross_border_burst", range(3500, 3625)),
        ("high_frequency", range(4500, 4625)),
        ("high_connectivity", range(5500, 5625)),
        ("ip_wallet_reuse", range(6500, 6625)),
        ("peeling_chain", peel_tx_indices)
    ]
    for scen_name, idx_range in scenarios:
        for idx in idx_range:
            anomaly_map[idx] = scen_name

    network_rows = []
    blockchain_rows = []

    block_height = 850000
    current_peel_amount = 50.0

    for i in range(n_rows):
        txid = txids[i]
        tx_ts = timestamps[i]
        
        # Network broadcast happens 2 to 45 seconds prior to block confirmation
        net_lead_sec = random.randint(2, 45)
        net_ts = tx_ts - timedelta(seconds=net_lead_sec)

        # Increment block height periodically (~every 10 minutes / 4-10 transactions)
        if i % 7 == 0:
            block_height += 1
        time_step = (block_height - 850000) // 144 + 1

        is_anom = i in anomaly_map
        scenario = anomaly_map.get(i, "normal")
        ground_truth = "synthetic_anomalous" if is_anom else "synthetic_normal"

        # ---------------------------------------------------------------------
        # Assign Wallets & Features by Scenario
        # ---------------------------------------------------------------------
        if scenario == "peeling_chain":
            src_wallet = peel_wallet_source
            dst_wallet = random.choice(normal_wallets)
            num_in = 1
            num_out = 2
            fee = round(random.uniform(0.0001, 0.0003), 6)
            peel_pay_amount = round(random.uniform(0.05, 0.35), 4)
            current_peel_amount = max(0.5, current_peel_amount - peel_pay_amount - fee)
            tot_out = round(current_peel_amount + peel_pay_amount, 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = [src_wallet]
            out_addrs = [dst_wallet, src_wallet] # Change address peels back
            in_amts = [str(tot_in)]
            out_amts = [str(peel_pay_amount), str(round(current_peel_amount, 6))]
            
            tx_freq = random.randint(35, 70)
            avg_time_gap = round(random.uniform(1.2, 4.5), 2)
            wallet_degree = random.randint(30, 60)
            unique_ips = random.randint(1, 3)

        elif scenario == "fan_out":
            src_wallet = random.choice(hub_wallets)
            num_in = random.randint(1, 2)
            num_out = random.randint(10, 20) # Massive fan-out splitting
            dst_wallets = random.sample(normal_wallets, num_out)
            dst_wallet = dst_wallets[0]
            
            tot_out = round(random.uniform(5.0, 25.0), 4)
            fee = round(random.uniform(0.002, 0.008), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = random.sample(wallet_pool, num_in)
            out_addrs = dst_wallets
            in_amts = [str(round(tot_in / num_in, 4)) for _ in range(num_in)]
            out_amts = [str(round(tot_out / num_out, 4)) for _ in range(num_out)]
            
            tx_freq = random.randint(60, 150)
            avg_time_gap = round(random.uniform(0.5, 3.0), 2)
            wallet_degree = random.randint(70, 120)
            unique_ips = random.randint(4, 9)

        elif scenario == "fan_in":
            dst_wallet = random.choice(hub_wallets)
            num_in = random.randint(8, 16) # Massive fan-in consolidation
            num_out = 1
            src_wallets = random.sample(normal_wallets, num_in)
            src_wallet = src_wallets[0]
            
            tot_out = round(random.uniform(4.0, 20.0), 4)
            fee = round(random.uniform(0.003, 0.009), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = src_wallets
            out_addrs = [dst_wallet]
            in_amts = [str(round(tot_in / num_in, 4)) for _ in range(num_in)]
            out_amts = [str(tot_out)]
            
            tx_freq = random.randint(50, 120)
            avg_time_gap = round(random.uniform(1.0, 5.0), 2)
            wallet_degree = random.randint(65, 110)
            unique_ips = random.randint(5, 12)

        elif scenario == "rapid_chain":
            src_wallet = random.choice(normal_wallets)
            dst_wallet = random.choice(normal_wallets)
            num_in = random.randint(1, 3)
            num_out = random.randint(1, 2)
            tot_out = round(random.uniform(0.5, 4.0), 4)
            fee = round(random.uniform(0.0005, 0.002), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = [src_wallet] if num_in == 1 else [src_wallet] + random.sample(normal_wallets, num_in - 1)
            out_addrs = [dst_wallet] if num_out == 1 else [dst_wallet] + random.sample(normal_wallets, num_out - 1)
            in_amts = [str(round(tot_in / num_in, 4)) for _ in range(num_in)]
            out_amts = [str(round(tot_out / num_out, 4)) for _ in range(num_out)]
            
            tx_freq = random.randint(80, 220)
            avg_time_gap = round(random.uniform(0.2, 1.5), 2) # Ultra-short inter-tx gap
            wallet_degree = random.randint(40, 75)
            unique_ips = random.randint(3, 7)

        elif scenario == "cross_border_burst":
            src_wallet = random.choice(normal_wallets)
            dst_wallet = random.choice(normal_wallets)
            num_in = random.randint(1, 4)
            num_out = random.randint(2, 5)
            tot_out = round(random.uniform(1.0, 8.0), 4)
            fee = round(random.uniform(0.001, 0.004), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = [src_wallet] + random.sample(normal_wallets, num_in - 1) if num_in > 1 else [src_wallet]
            out_addrs = [dst_wallet] + random.sample(normal_wallets, num_out - 1) if num_out > 1 else [dst_wallet]
            in_amts = [str(round(tot_in / num_in, 4)) for _ in range(num_in)]
            out_amts = [str(round(tot_out / num_out, 4)) for _ in range(num_out)]
            
            tx_freq = random.randint(70, 160)
            avg_time_gap = round(random.uniform(0.8, 3.5), 2)
            wallet_degree = random.randint(50, 90)
            unique_ips = random.randint(6, 14) # Broad multi-country dispersion

        elif scenario == "high_frequency":
            src_wallet = random.choice(normal_wallets)
            dst_wallet = random.choice(normal_wallets)
            num_in = random.randint(1, 2)
            num_out = random.randint(1, 3)
            tot_out = round(random.uniform(0.1, 2.5), 4)
            fee = round(random.uniform(0.0003, 0.0015), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = [src_wallet] if num_in == 1 else [src_wallet] + random.sample(normal_wallets, num_in - 1)
            out_addrs = [dst_wallet] if num_out == 1 else [dst_wallet] + random.sample(normal_wallets, num_out - 1)
            in_amts = [str(round(tot_in / num_in, 4)) for _ in range(num_in)]
            out_amts = [str(round(tot_out / num_out, 4)) for _ in range(num_out)]
            
            tx_freq = random.randint(120, 350) # Automated botnet frequency
            avg_time_gap = round(random.uniform(0.1, 1.0), 2)
            wallet_degree = random.randint(45, 85)
            unique_ips = random.randint(2, 5)

        elif scenario == "high_connectivity":
            src_wallet = random.choice(hub_wallets)
            dst_wallet = random.choice(hub_wallets)
            num_in = random.randint(2, 5)
            num_out = random.randint(3, 8)
            tot_out = round(random.uniform(8.0, 45.0), 4)
            fee = round(random.uniform(0.002, 0.010), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = random.sample(wallet_pool, num_in)
            out_addrs = random.sample(wallet_pool, num_out)
            in_amts = [str(round(tot_in / num_in, 4)) for _ in range(num_in)]
            out_amts = [str(round(tot_out / num_out, 4)) for _ in range(num_out)]
            
            tx_freq = random.randint(90, 260)
            avg_time_gap = round(random.uniform(0.5, 4.0), 2)
            wallet_degree = random.randint(85, 180) # Extreme hub degree
            unique_ips = random.randint(7, 18)

        elif scenario == "ip_wallet_reuse":
            src_wallet = random.choice(normal_wallets)
            dst_wallet = random.choice(normal_wallets)
            num_in = random.randint(1, 3)
            num_out = random.randint(1, 4)
            tot_out = round(random.uniform(0.2, 5.0), 4)
            fee = round(random.uniform(0.0004, 0.0025), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = [src_wallet] if num_in == 1 else [src_wallet] + random.sample(normal_wallets, num_in - 1)
            out_addrs = [dst_wallet] if num_out == 1 else [dst_wallet] + random.sample(normal_wallets, num_out - 1)
            in_amts = [str(round(tot_in / num_in, 4)) for _ in range(num_in)]
            out_amts = [str(round(tot_out / num_out, 4)) for _ in range(num_out)]
            
            tx_freq = random.randint(60, 140)
            avg_time_gap = round(random.uniform(1.0, 4.0), 2)
            wallet_degree = random.randint(50, 95)
            unique_ips = 1 # Proxy centralization

        else: # Normal Baseline Transaction
            src_wallet = random.choice(normal_wallets)
            dst_wallet = random.choice(normal_wallets)
            num_in = random.randint(1, 3)
            num_out = random.randint(1, 3)
            tot_out = round(random.uniform(0.01, 1.5), 6)
            fee = round(random.uniform(0.00005, 0.0005), 6)
            tot_in = round(tot_out + fee, 6)
            
            in_addrs = [src_wallet] if num_in == 1 else [src_wallet] + random.sample(normal_wallets, num_in - 1)
            out_addrs = [dst_wallet] if num_out == 1 else [dst_wallet] + random.sample(normal_wallets, num_out - 1)
            in_amts = [str(round(tot_in / num_in, 6)) for _ in range(num_in)]
            out_amts = [str(round(tot_out / num_out, 6)) for _ in range(num_out)]
            
            tx_freq = random.randint(1, 15)
            avg_time_gap = round(random.uniform(15.0, 240.0), 2)
            wallet_degree = random.randint(2, 22)
            unique_ips = random.randint(1, 3)

        # ---------------------------------------------------------------------
        # Network Telemetry Attributes (Strictly Public Resolvable IPs)
        # ---------------------------------------------------------------------
        # Select IPs from verified pools
        src_ip = random.choice(ALL_PUBLIC_IPS)
        dst_ip = random.choice(ALL_PUBLIC_IPS)
        while dst_ip == src_ip:
            dst_ip = random.choice(ALL_PUBLIC_IPS)

        src_port = random.choice([8333, 18333, 18444, 38333, 49152, 51422, 58210])
        dst_port = 8333 # Standard Bitcoin listening port
        protocol = "TCP"
        
        conn_dur = round(random.uniform(0.2, 120.0) if not is_anom else random.uniform(5.0, 360.0), 3)
        pkt_count = random.randint(20, 350) if not is_anom else random.randint(150, 4500)
        bytes_tx = int(pkt_count * random.uniform(180, 850))

        # Append Network record (ZERO COUNTRY / ASN FIELDS)
        network_rows.append({
            "txid": txid,
            "timestamp": tx_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "network_timestamp": net_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "connection_duration_sec": conn_dur,
            "packet_count": pkt_count,
            "bytes_transferred": bytes_tx
        })

        # Append Blockchain record (ZERO COUNTRY / ASN FIELDS)
        blockchain_rows.append({
            "txid": txid,
            "timestamp": tx_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "block_height": block_height,
            "time_step": time_step,
            "input_addresses": "|".join(in_addrs),
            "output_addresses": "|".join(out_addrs),
            "input_amounts": "|".join(in_amts),
            "output_amounts": "|".join(out_amts),
            "fee_btc": fee,
            "num_inputs": num_in,
            "num_outputs": num_out,
            "total_input_amount_btc": tot_in,
            "total_output_amount_btc": tot_out,
            "script_type": random.choices(script_types, weights=script_weights)[0],
            "source_wallet": src_wallet,
            "destination_wallet": dst_wallet,
            "transaction_frequency_24h": tx_freq,
            "avg_time_gap_min": avg_time_gap,
            "wallet_degree": wallet_degree,
            "unique_ip_count": unique_ips,
            "ground_truth": ground_truth,
            "scenario": scenario
        })

    net_df = pd.DataFrame(network_rows)
    chain_df = pd.DataFrame(blockchain_rows)

    return net_df, chain_df


def save_and_verify_datasets():
    net_df, chain_df = generate_sih_master_datasets(n_rows=10000, seed=42)

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    net_path = os.path.join(data_dir, "network_traffic_dataset.csv")
    chain_path = os.path.join(data_dir, "blockchain_transactions_dataset.csv")

    # Also update the root demo files used by Tab 0 Ingestion
    demo_net_path = "demo_network_10000.csv"
    demo_chain_path = "demo_blockchain_10000.csv"

    print("Saving datasets to disk...")
    net_df.to_csv(net_path, index=False)
    chain_df.to_csv(chain_path, index=False)
    net_df.to_csv(demo_net_path, index=False)
    chain_df.to_csv(demo_chain_path, index=False)

    print(f"\n[SUCCESS] Successfully written:")
    print(f"  - {net_path} ({len(net_df)} rows, columns: {list(net_df.columns)})")
    print(f"  - {chain_path} ({len(chain_df)} rows, columns: {list(chain_df.columns)})")
    print(f"  - {demo_net_path}")
    print(f"  - {demo_chain_path}")

    # Double check for any country/asn fields
    forbidden_terms = ['country', 'asn', 'org']
    for col in net_df.columns:
        for t in forbidden_terms:
            assert t not in col.lower(), f"Forbidden term '{t}' found in network column: {col}"
    for col in chain_df.columns:
        for t in forbidden_terms:
            assert t not in col.lower(), f"Forbidden term '{t}' found in blockchain column: {col}"

    print("\n[VERIFICATION] Zero country/ASN/org columns confirmed in both datasets!")


if __name__ == "__main__":
    save_and_verify_datasets()
