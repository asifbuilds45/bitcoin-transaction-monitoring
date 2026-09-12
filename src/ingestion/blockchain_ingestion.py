import io
import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Union

REQUIRED_BLOCKCHAIN_FIELDS = [
    'txid', 'source_wallet', 'destination_wallet'
]


def _parse_xml_to_dataframe(source: Any) -> pd.DataFrame:
    """Parse XML string, buffer, or file path into a pandas DataFrame."""
    try:
        return pd.read_xml(source)
    except Exception:
        pass

    if isinstance(source, str) and os.path.exists(source):
        tree = ET.parse(source)
        root = tree.getroot()
    elif isinstance(source, (str, bytes)):
        root = ET.fromstring(source)
    elif hasattr(source, 'read'):
        if hasattr(source, 'seek'):
            source.seek(0)
        content = source.read()
        root = ET.fromstring(content)
    else:
        raise ValueError("Unsupported XML source")

    records = []
    for child in root:
        rec = {}
        for sub in child:
            rec[sub.tag] = sub.text.strip() if sub.text else ""
        if rec:
            records.append(rec)
    if not records:
        rec = {c.tag: (c.text.strip() if c.text else "") for c in root}
        if rec:
            records.append(rec)
    return pd.DataFrame(records)


def _load_data_source(data_source: Any) -> pd.DataFrame:
    """Load raw DataFrame from CSV, JSON, or XML format."""
    if isinstance(data_source, pd.DataFrame):
        return data_source.copy(deep=True)

    if isinstance(data_source, str):
        lower_path = data_source.lower()
        if lower_path.endswith('.json'):
            return pd.read_json(data_source)
        elif lower_path.endswith('.xml'):
            return _parse_xml_to_dataframe(data_source)
        else:
            return pd.read_csv(data_source)

    name = getattr(data_source, 'name', '').lower()
    if hasattr(data_source, 'seek'):
        data_source.seek(0)

    if name.endswith('.json'):
        return pd.read_json(data_source)
    elif name.endswith('.xml'):
        return _parse_xml_to_dataframe(data_source)
    else:
        return pd.read_csv(data_source)


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


def parse_and_validate_blockchain_data(
    data_source: Any
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Ingest, validate, and normalize Blockchain-Layer data in CSV, JSON, or XML formats.
    
    Args:
        data_source: Path to file, UploadedFile, or existing DataFrame.
        
    Returns:
        Tuple of (clean_blockchain_df, list_of_validation_warnings)
    """
    warnings = []

    try:
        df_raw = _load_data_source(data_source)
    except Exception as e:
        warnings.append({
            "level": "ERROR",
            "type": "INGESTION_ERROR",
            "message": f"Failed to parse blockchain data: {e}"
        })
        df_raw = pd.DataFrame(columns=REQUIRED_BLOCKCHAIN_FIELDS)

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


def parse_and_validate_blockchain_csv(
    data_source: Union[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Backward compatible wrapper for parse_and_validate_blockchain_data."""
    return parse_and_validate_blockchain_data(data_source)
