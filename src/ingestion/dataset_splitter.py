"""
Dataset Splitter Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Splits the combined prototype dataset into two independent layer CSV files:
1. network_traffic_dataset.csv (Network layer telemetry & observation attributes)
2. blockchain_transactions_dataset.csv (Blockchain layer UTXO & wallet transactions)
"""

import os
import pandas as pd

NETWORK_COLUMNS = [
    'txid', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
    'protocol', 'network_timestamp', 'connection_duration_sec',
    'packet_count', 'bytes_transferred', 'src_country', 'dst_country',
    'src_asn', 'dst_asn'
]

BLOCKCHAIN_COLUMNS = [
    'txid', 'timestamp', 'block_height', 'time_step',
    'input_addresses', 'output_addresses', 'input_amounts', 'output_amounts',
    'fee_btc', 'num_inputs', 'num_outputs', 'total_input_amount_btc',
    'total_output_amount_btc', 'script_type', 'source_wallet',
    'destination_wallet', 'transaction_frequency_24h', 'avg_time_gap_min',
    'wallet_degree', 'unique_ip_count', 'country_count', 'asn_count',
    'ground_truth', 'scenario'
]


def split_master_dataset(
    master_path: str = os.path.join("data", "bitcoin_monitoring_final_10000.csv"),
    output_dir: str = "data"
) -> tuple[str, str]:
    """
    Split master dataset into network and blockchain CSV files.
    """
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master dataset not found at {master_path}")

    df = pd.read_csv(master_path)
    
    net_cols = [c for c in NETWORK_COLUMNS if c in df.columns]
    chain_cols = [c for c in BLOCKCHAIN_COLUMNS if c in df.columns]

    df_net = df[net_cols].copy()
    df_chain = df[chain_cols].copy()

    net_path = os.path.join(output_dir, "network_traffic_dataset.csv")
    chain_path = os.path.join(output_dir, "blockchain_transactions_dataset.csv")

    df_net.to_csv(net_path, index=False)
    df_chain.to_csv(chain_path, index=False)

    print(f"Generated Network CSV ({len(df_net)} rows) -> {net_path}")
    print(f"Generated Blockchain CSV ({len(df_chain)} rows) -> {chain_path}")

    return net_path, chain_path


if __name__ == "__main__":
    split_master_dataset()
