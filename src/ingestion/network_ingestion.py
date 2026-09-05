"""
Network Data Ingestion & Preprocessing Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import ipaddress
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Union

REQUIRED_NETWORK_FIELDS = [
    'txid', 'src_ip', 'dst_ip'
]

OPTIONAL_NETWORK_FIELDS = [
    'timestamp', 'network_timestamp', 'src_port', 'dst_port',
    'protocol', 'connection_duration_sec', 'packet_count',
    'bytes_transferred', 'src_country', 'dst_country', 'src_asn', 'dst_asn'
]


def validate_ip_address(ip_str: Any) -> bool:
    """Validate if value is a valid IPv4 or IPv6 address string."""
    if pd.isna(ip_str) or not isinstance(ip_str, str):
        return False
    clean = str(ip_str).strip()
    if not clean:
        return False
    try:
        ipaddress.ip_address(clean)
        return True
    except ValueError:
        return False


def parse_and_validate_network_csv(
    data_source: Union[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Ingest, validate, and normalize Network-Layer CSV data.
    
    Args:
        data_source: Path to CSV file or existing DataFrame.
        
    Returns:
        Tuple of (clean_network_df, list_of_validation_warnings)
    """
    warnings = []
    
    if isinstance(data_source, str):
        df_raw = pd.read_csv(data_source)
    else:
        df_raw = data_source.copy(deep=True)
        
    df = df_raw.copy(deep=True)
    
    # 1. Validate required field presence
    missing_req = [col for col in REQUIRED_NETWORK_FIELDS if col not in df.columns]
    if missing_req:
        warnings.append({
            "level": "ERROR",
            "type": "MISSING_REQUIRED_COLUMNS",
            "message": f"Network dataset is missing required columns: {missing_req}"
        })
        # If critical required columns are missing, add placeholder string columns
        for col in missing_req:
            df[col] = "UNKNOWN"

    # 2. Datetime parsing
    if 'network_timestamp' in df.columns:
        df['network_timestamp'] = pd.to_datetime(df['network_timestamp'], errors='coerce')
    elif 'timestamp' in df.columns:
        df['network_timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['network_timestamp'] = pd.Timestamp.now()
        warnings.append({
            "level": "WARNING",
            "type": "MISSING_TIMESTAMP",
            "message": "Neither 'network_timestamp' nor 'timestamp' column was provided. Using current timestamp."
        })

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = df['network_timestamp']

    # Impute missing timestamps
    if df['network_timestamp'].isna().any():
        null_count = int(df['network_timestamp'].isna().sum())
        warnings.append({
            "level": "WARNING",
            "type": "MALFORMED_TIMESTAMPS",
            "message": f"Found {null_count} malformed network timestamps. Imputed using forward/backward fill."
        })
        df['network_timestamp'] = df['network_timestamp'].bfill().ffill()
        df['timestamp'] = df['timestamp'].fillna(df['network_timestamp'])

    # 3. IP Normalization & Validation
    df['src_ip'] = df['src_ip'].fillna("0.0.0.0").astype(str).str.strip()
    df['dst_ip'] = df['dst_ip'].fillna("0.0.0.0").astype(str).str.strip()

    df['src_ip_valid'] = df['src_ip'].apply(validate_ip_address)
    df['dst_ip_valid'] = df['dst_ip'].apply(validate_ip_address)

    invalid_src_count = int((~df['src_ip_valid']).sum())
    if invalid_src_count > 0:
        warnings.append({
            "level": "WARNING",
            "type": "INVALID_IP_ADDRESSES",
            "message": f"Found {invalid_src_count} invalid or unparseable source IP addresses."
        })

    # 4. Port & Numeric Normalization
    numeric_defaults = {
        'src_port': 8333,
        'dst_port': 8333,
        'connection_duration_sec': 1.0,
        'packet_count': 1,
        'bytes_transferred': 512
    }

    for col, default_val in numeric_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
        else:
            df[col] = default_val

    # 5. TXID Normalization
    df['txid'] = df['txid'].fillna("UNKNOWN_TXID").astype(str).str.strip()

    # 6. String / Categorical Normalization
    string_defaults = {
        'protocol': 'TCP',
        'src_country': 'UNKNOWN',
        'dst_country': 'UNKNOWN',
        'src_asn': 'UNKNOWN',
        'dst_asn': 'UNKNOWN'
    }

    for col, default_val in string_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val).astype(str).str.strip()
        else:
            df[col] = default_val

    return df, warnings
