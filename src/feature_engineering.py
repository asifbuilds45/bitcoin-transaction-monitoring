"""
Feature Engineering Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

CRITICAL NOTE:
Never use 'ground_truth' or 'scenario' as input features for machine learning.
All features must represent purely unsupervised behavioral and network metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler

# Base behavioral and network feature column names
CORE_FEATURE_NAMES = [
    'transaction_frequency_24h',
    'avg_time_gap_min',
    'wallet_degree',
    'unique_ip_count',
    'country_count',
    'asn_count',
    'num_inputs',
    'num_outputs',
    'total_output_amount_btc',
    'fee_btc',
    'connection_duration_sec',
    'packet_count',
    'bytes_transferred'
]

# Engineered derived feature names
DERIVED_FEATURE_NAMES = [
    'input_output_ratio',
    'bytes_per_packet',
    'fee_to_amount_ratio',
    'is_cross_border',
    'duration_per_packet',
    'amount_per_output'
]

ALL_ML_FEATURES = CORE_FEATURE_NAMES + DERIVED_FEATURE_NAMES


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and engineer unsupervised behavioral features from the preprocessed DataFrame.
    
    Args:
        df: Preprocessed DataFrame.
        
    Returns:
        pd.DataFrame: Feature matrix DataFrame containing strictly behavioral/network features.
    """
    feat_df = pd.DataFrame(index=df.index)

    # 1. Copy core behavioral and network features
    for col in CORE_FEATURE_NAMES:
        if col in df.columns:
            feat_df[col] = df[col].astype(float)
        else:
            feat_df[col] = 0.0

    # 2. Derive domain-specific interaction ratios
    # Input / Output Address Shape Ratio
    feat_df['input_output_ratio'] = feat_df['num_inputs'] / np.maximum(feat_df['num_outputs'], 1.0)

    # Network Packet Density (bytes / packet)
    feat_df['bytes_per_packet'] = feat_df['bytes_transferred'] / np.maximum(feat_df['packet_count'], 1.0)

    # Miner Fee to Transaction Volume Ratio
    feat_df['fee_to_amount_ratio'] = feat_df['fee_btc'] / np.maximum(feat_df['total_output_amount_btc'], 0.0001)

    # Cross-Border Network Indicator (1 if src_country != dst_country, else 0)
    if 'src_country' in df.columns and 'dst_country' in df.columns:
        feat_df['is_cross_border'] = (df['src_country'] != df['dst_country']).astype(float)
    else:
        feat_df['is_cross_border'] = 0.0

    # Connection Timing per Packet
    feat_df['duration_per_packet'] = feat_df['connection_duration_sec'] / np.maximum(feat_df['packet_count'], 1.0)

    # Amount per output address
    feat_df['amount_per_output'] = feat_df['total_output_amount_btc'] / np.maximum(feat_df['num_outputs'], 1.0)

    # Clean any inf or NaN values created by division
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feat_df


def prepare_ml_feature_matrix(
    df: pd.DataFrame,
    scale: bool = False
) -> Tuple[pd.DataFrame, Optional[RobustScaler]]:
    """
    Extract features and optionally apply robust scaling for distance-sensitive models.
    
    Args:
        df: Input DataFrame.
        scale: Whether to apply RobustScaler.
        
    Returns:
        Tuple of (feature_df, scaler_object)
    """
    features = extract_features(df)
    scaler = None

    if scale:
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(features)
        features = pd.DataFrame(scaled_values, columns=features.columns, index=features.index)

    return features, scaler
