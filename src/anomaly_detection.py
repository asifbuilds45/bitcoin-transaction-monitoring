"""
AI Anomaly Detection Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

==============================================================================
WHY ISOLATION FOREST IS SUITABLE FOR THIS PROBLEM:
==============================================================================
1. Unsupervised Learning: Real-world Bitcoin network traffic lacks ground-truth
   malicious labels during real-time monitoring. Isolation Forest operates in a
   strictly unsupervised fashion without requiring prior attack signatures.
2. Anomaly Isolation Principle: Anomalous transactions (such as rapid chains,
   fan-out mixing, botnet transaction bursts) have extreme or sparse attribute
   combinations in multi-dimensional feature space. They require significantly
   fewer random partition splits to isolate in tree structures compared to
   dense normal traffic.
3. Multi-Modal Feature Handling: It seamlessly evaluates combinations of
   network telemetry (packet counts, connection duration) and on-chain metrics
   (wallet degree, transaction frequency, UTXO counts).
4. Deterministic & Offline Feasible: Operates efficiently in local/offline
   environments with fast inference times on 10,000+ transaction batches.
5. No Ground Truth Leakage: The model trains exclusively on behavioral and
   network metrics without ever inspecting labels or scenario names.
==============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest


class BitcoinAnomalyDetector:
    """
    Unsupervised Anomaly Detection engine using Isolation Forest.
    """

    def __init__(
        self,
        contamination: float = 0.10,
        n_estimators: int = 150,
        max_samples: str = 'auto',
        random_state: int = 42
    ):
        """
        Initialize the Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset (~10%).
            n_estimators: Number of isolation trees in the ensemble.
            max_samples: Subsample size drawn from X to train each tree.
            random_state: Seed for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X: pd.DataFrame) -> "BitcoinAnomalyDetector":
        """
        Fit the unsupervised Isolation Forest model on behavioral feature matrix.
        
        Args:
            X: Feature matrix DataFrame (strictly excluding ground_truth/scenario).
        """
        self.feature_names = list(X.columns)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict_and_score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate binary predictions, raw scores, and normalized anomaly scores.
        
        Args:
            X: Feature matrix DataFrame.
            
        Returns:
            Tuple of:
            - is_anomaly (1 for anomaly, 0 for normal)
            - raw_scores (Isolation Forest score_samples output, negative values are more anomalous)
            - normalized_scores (0.0 to 1.0, where 1.0 is maximum anomaly confidence)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before predicting.")

        # sklearn IsolationForest: 1 = normal, -1 = anomaly
        raw_preds = self.model.predict(X)
        is_anomaly = np.where(raw_preds == -1, 1, 0)

        # score_samples: opposite of the anomaly score defined in original paper.
        # Lower (more negative) means more anomalous.
        raw_scores = self.model.score_samples(X)

        # Invert and normalize to [0.0, 1.0] where 1.0 = highly anomalous
        # Negate so higher value = higher abnormality
        inverted_scores = -raw_scores
        min_val = inverted_scores.min()
        max_val = inverted_scores.max()
        if max_val > min_val:
            normalized_scores = (inverted_scores - min_val) / (max_val - min_val)
        else:
            normalized_scores = np.zeros_like(inverted_scores)

        return is_anomaly, raw_scores, normalized_scores

    def fit_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience method to fit and score in a single call."""
        self.fit(X)
        return self.predict_and_score(X)
