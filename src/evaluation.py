"""
Prototype Evaluation Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

CRITICAL NOTE:
The 'ground_truth' column is used EXCLUSIVELY for post-prediction evaluation
to validate model performance. It was never used during preprocessing,
feature engineering, or Isolation Forest model fitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_prototype_predictions(
    df_scored: pd.DataFrame,
    ground_truth_col: str = "ground_truth",
    prediction_col: str = "is_anomaly",
    score_col: str = "anomaly_score",
    positive_label_str: str = "synthetic_anomalous"
) -> Dict[str, Any]:
    """
    Evaluate unsupervised Isolation Forest predictions against ground truth labels.
    
    Args:
        df_scored: DataFrame containing model predictions and ground_truth.
        ground_truth_col: Name of column containing true labels.
        prediction_col: Name of column containing binary anomaly prediction (1=anomaly, 0=normal).
        score_col: Name of column containing continuous normalized anomaly score.
        positive_label_str: Value in ground_truth indicating true anomaly.
        
    Returns:
        Dict containing comprehensive evaluation metrics and scenario breakdown.
    """
    if ground_truth_col not in df_scored.columns:
        raise ValueError(f"Ground truth column '{ground_truth_col}' not found in DataFrame.")

    # Convert string ground_truth into binary labels (1=Anomaly, 0=Normal)
    y_true = (df_scored[ground_truth_col] == positive_label_str).astype(int).values
    y_pred = df_scored[prediction_col].astype(int).values
    y_scores = df_scored[score_col].astype(float).values

    # Calculate standard classification metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.0

    # Compute Confusion Matrix
    # cm format: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Scenario-wise detection breakdown
    scenario_breakdown = []
    if 'scenario' in df_scored.columns:
        for scenario_name, group in df_scored.groupby('scenario'):
            total_count = len(group)
            detected_count = int(group[prediction_col].sum())
            high_risk_count = int((group['risk_level'].isin(['HIGH', 'CRITICAL'])).sum())
            detection_rate = round((detected_count / total_count) * 100.0, 2) if total_count > 0 else 0.0
            avg_anomaly_score = round(float(group[score_col].mean()), 4)
            avg_risk_score = round(float(group['risk_score'].mean()), 2)

            scenario_breakdown.append({
                "Scenario": scenario_name,
                "Total Records": total_count,
                "Detected Anomalies": detected_count,
                "High/Critical Risk Alerts": high_risk_count,
                "Detection Rate (%)": detection_rate,
                "Avg Anomaly Score": avg_anomaly_score,
                "Avg Risk Score": avg_risk_score
            })

    scenario_df = pd.DataFrame(scenario_breakdown).sort_values(by='Total Records', ascending=False)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "raw_matrix": cm.tolist()
        },
        "scenario_breakdown": scenario_df,
        "total_evaluated": len(df_scored),
        "true_anomalies_count": int(y_true.sum()),
        "predicted_anomalies_count": int(y_pred.sum())
    }
