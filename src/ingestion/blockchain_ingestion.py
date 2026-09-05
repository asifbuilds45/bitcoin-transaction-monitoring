"""
Blockchain Data Ingestion & Preprocessing Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Union

REQUIRED_BLOCKCHAIN_FIELDS = [
    'txid', 'source_wallet', 'destination_wallet'
]


def parse_pipe_list(value: Any) -> List[str]:
    """Parse pipe-separated string into a list of strings."""
    if pd.isna(value) or not isinstance(value, str):
        return []
    return [item.strip() for item in value.split('|') if item.strip()]


def parse_pipe_float_list(value: Any) -> List[float]:
    """Parse pipe-separated string into a list of floats."""
    if pd.isna(value) or not isinstance(value, str):
        return []
    res = []
    for item in value.split('|'):
        try:
            res.append(float(item.strip()))
        except ValueError:
            continue
    return res


def parse_and_validate_blockchain_csv(
    data_source: Union[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Ingest, validate, and normalize Blockchain-Layer CSV data.
    
    Args:
        data_source: Path to CSV file or existing DataFrame.
        
    Returns:
        Tuple of (clean_blockchain_df, list_of_validation_warnings)
    """
    warnings = []

    if isinstance(data_source, str):
        df_raw = pd.read_csv(data_source)
    else:
        df_raw = data_source.copy(deep=True)

    df = df_raw.copy(deep=True)

    # 1. Validate required columns
    missing_req = [col for col in REQUIRED_BLOCKCHAIN_FIELDS if col not in df.columns]
    if missing_req:
        warnings.append({
            "level": "ERROR",
            "type": "MISSING_REQUIRED_COLUMNS",
            "message": f"Blockchain dataset is missing required columns: {missing_req}"
        })
        for col in missing_req:
            df[col] = "UNKNOWN"

    # 2. Datetime parsing
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().any():
            null_cnt = int(df['timestamp'].isna().sum())
            warnings.append({
                "level": "WARNING",
                "type": "MALFORMED_TIMESTAMPS",
                "message": f"Found {null_cnt} malformed block timestamps. Imputed using ffill/bfill."
            })
            df['timestamp'] = df['timestamp'].bfill().ffill()
    else:
        df['timestamp'] = pd.Timestamp.now()

    # 3. TXID & Wallet String Normalization
    str_cols = ['txid', 'source_wallet', 'destination_wallet', 'script_type', 'ground_truth', 'scenario']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()
        else:
            df[col] = "UNKNOWN"

    # 4. Numeric conversions
    numeric_defaults = {
        'block_height': 700000,
        'time_step': 1,
        'fee_btc': 0.0001,
        'num_inputs': 1,
        'num_outputs': 1,
        'total_input_amount_btc': 0.1,
        'total_output_amount_btc': 0.1,
        'transaction_frequency_24h': 1,
        'avg_time_gap_min': 60.0,
        'wallet_degree': 1,
        'unique_ip_count': 1,
        'country_count': 1,
        'asn_count': 1
    }

    for col, default_val in numeric_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
        else:
            df[col] = default_val

    # Int casting
    int_cols = [
        'block_height', 'time_step', 'num_inputs', 'num_outputs',
        'transaction_frequency_24h', 'wallet_degree', 'unique_ip_count',
        'country_count', 'asn_count'
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    # 5. Parse pipe-separated UTXO address and amount lists
    if 'input_addresses' in df.columns:
        df['parsed_input_addresses'] = df['input_addresses'].apply(parse_pipe_list)
    else:
        df['parsed_input_addresses'] = df['source_wallet'].apply(lambda w: [w])

    if 'output_addresses' in df.columns:
        df['parsed_output_addresses'] = df['output_addresses'].apply(parse_pipe_list)
    else:
        df['parsed_output_addresses'] = df['destination_wallet'].apply(lambda w: [w])

    if 'input_amounts' in df.columns:
        df['parsed_input_amounts'] = df['input_amounts'].apply(parse_pipe_float_list)
    else:
        df['parsed_input_amounts'] = df['total_input_amount_btc'].apply(lambda a: [float(a)])

    if 'output_amounts' in df.columns:
        df['parsed_output_amounts'] = df['output_amounts'].apply(parse_pipe_float_list)
    else:
        df['parsed_output_amounts'] = df['total_output_amount_btc'].apply(lambda a: [float(a)])

    return df, warnings
