"""
Data Ingestion and Preprocessing Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import ipaddress
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

REQUIRED_COLUMNS = [
    'txid', 'timestamp', 'block_height', 'time_step', 'src_ip', 'dst_ip',
    'src_port', 'dst_port', 'protocol', 'network_timestamp',
    'connection_duration_sec', 'packet_count', 'bytes_transferred',
    'src_country', 'dst_country', 'src_asn', 'dst_asn',
    'input_addresses', 'output_addresses', 'input_amounts', 'output_amounts',
    'total_input_amount_btc', 'total_output_amount_btc', 'fee_btc',
    'num_inputs', 'num_outputs', 'script_type',
    'source_wallet', 'destination_wallet',
    'transaction_frequency_24h', 'avg_time_gap_min', 'wallet_degree',
    'unique_ip_count', 'country_count', 'asn_count',
    'ground_truth', 'scenario'
]

NUMERIC_COLUMNS = [
    'block_height', 'time_step', 'src_port', 'dst_port',
    'connection_duration_sec', 'packet_count', 'bytes_transferred',
    'total_input_amount_btc', 'total_output_amount_btc', 'fee_btc',
    'num_inputs', 'num_outputs', 'transaction_frequency_24h',
    'avg_time_gap_min', 'wallet_degree', 'unique_ip_count',
    'country_count', 'asn_count'
]


def validate_ip(ip_str: str) -> bool:
    """Validate if a string is a syntactically valid IPv4 or IPv6 address."""
    if not isinstance(ip_str, str) or not ip_str.strip():
        return False
    try:
        ipaddress.ip_address(ip_str.strip())
        return True
    except ValueError:
        return False


def parse_pipe_list(value: Any) -> List[str]:
    """Parse pipe-separated string into a list of strings."""
    if pd.isna(value) or not isinstance(value, str):
        return []
    return [item.strip() for item in value.split('|') if item.strip()]


def parse_pipe_float_list(value: Any) -> List[float]:
    """Parse pipe-separated string into a list of floats."""
    if pd.isna(value) or not isinstance(value, str):
        return []
    result = []
    for item in value.split('|'):
        try:
            result.append(float(item.strip()))
        except ValueError:
            continue
    return result


def load_dataset(file_path: str = os.path.join("data", "bitcoin_monitoring_final_10000.csv")) -> pd.DataFrame:
    """
    Load raw transaction dataset without modifying the underlying source file.
    
    Args:
        file_path: Relative or absolute path to the master CSV dataset.
        
    Returns:
        pd.DataFrame: Loaded raw DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Master dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess and clean the raw Bitcoin traffic and transaction dataset.
    
    Operations:
    1. Deep copy to ensure zero modification of original source.
    2. Column validation against REQUIRED_COLUMNS.
    3. Missing value imputation / cleaning.
    4. Type casting for numerical fields.
    5. Datetime parsing for transaction and network timestamps.
    6. IP validation checks.
    7. List parsing for address and amount fields.
    
    Args:
        df_raw: Input raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned and structured DataFrame ready for analytics.
    """
    df = df_raw.copy(deep=True)
    
    # 1. Column presence validation
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    
    # 2. Datetime conversions
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['network_timestamp'] = pd.to_datetime(df['network_timestamp'], errors='coerce')
    
    # Fill any NaT with fallback timestamps if corrupted
    if df['timestamp'].isna().any():
        df['timestamp'] = df['timestamp'].bfill().ffill()
    if df['network_timestamp'].isna().any():
        df['network_timestamp'] = df['timestamp']
        
    # 3. Numeric type casting and missing value handling
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(median_val) else median_val)
            
    # Integer columns casting
    int_cols = [
        'block_height', 'time_step', 'src_port', 'dst_port',
        'packet_count', 'bytes_transferred', 'num_inputs', 'num_outputs',
        'transaction_frequency_24h', 'wallet_degree', 'unique_ip_count',
        'country_count', 'asn_count'
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)
        
    # 4. Categorical / string cleaning
    str_cols = [
        'txid', 'src_ip', 'dst_ip', 'protocol', 'src_country', 'dst_country',
        'src_asn', 'dst_asn', 'script_type', 'source_wallet', 'destination_wallet',
        'ground_truth', 'scenario'
    ]
    for col in str_cols:
        df[col] = df[col].fillna('UNKNOWN').astype(str).str.strip()
        
    # 5. IP Validation Flags
    df['src_ip_valid'] = df['src_ip'].apply(validate_ip)
    df['dst_ip_valid'] = df['dst_ip'].apply(validate_ip)
    
    # 6. Parse structured list fields
    df['parsed_input_addresses'] = df['input_addresses'].apply(parse_pipe_list)
    df['parsed_output_addresses'] = df['output_addresses'].apply(parse_pipe_list)
    df['parsed_input_amounts'] = df['input_amounts'].apply(parse_pipe_float_list)
    df['parsed_output_amounts'] = df['output_amounts'].apply(parse_pipe_float_list)
    
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate high-level metadata summary of the preprocessed dataset."""
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "unique_txids": df['txid'].nunique(),
        "unique_src_ips": df['src_ip'].nunique(),
        "unique_source_wallets": df['source_wallet'].nunique(),
        "unique_dest_wallets": df['destination_wallet'].nunique(),
        "total_volume_btc": round(float(df['total_output_amount_btc'].sum()), 4),
        "avg_fee_btc": round(float(df['fee_btc'].mean()), 6),
        "date_range": (
            str(df['timestamp'].min()),
            str(df['timestamp'].max())
        )
    }
