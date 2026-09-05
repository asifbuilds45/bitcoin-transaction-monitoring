"""
DBSCAN Behavioural Clustering Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

IMPORTANT ARCHITECTURAL RULE:
DBSCAN is a BEHAVIOURAL CLUSTERING ALGORITHM, not a probability generator.
It partitions transactions into dense behavioral clusters and identifies noise (-1).
Noise membership and sparse cluster occupancy are converted into a normalized DBSCAN Evidence Score.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class BitcoinDBSCANClustering:
    """
    Density-based spatial clustering for behavioral group discovery and noise detection.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN clustering engine.
        
        Args:
            eps: Maximum distance between two samples for one to be considered in the neighborhood.
            min_samples: Number of samples in a neighborhood for a point to be considered a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        self.is_fitted = False
        self.cluster_labels_ = np.array([])
        self.noise_mask_ = np.array([])

    def fit_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit DBSCAN on scaled features, predict clusters, and generate normalized DBSCAN Evidence Scores.
        
        Args:
            X: Feature matrix (strictly unsupervised behavioral features).
            
        Returns:
            Tuple of:
            - cluster_labels (-1 for noise, 0..N for cluster IDs)
            - is_noise_flag (1 for noise/outlier, 0 for cluster member)
            - normalized_dbscan_evidence (0.0 to 1.0)
        """
        if X.empty:
            return np.array([]), np.array([]), np.array([])

        X_scaled = self.scaler.fit_transform(X)
        self.cluster_labels_ = self.model.fit_predict(X_scaled)
        self.is_fitted = True

        self.noise_mask_ = np.where(self.cluster_labels_ == -1, 1, 0)
        
        # Calculate cluster sizes to score sparse clusters
        unique_labels, counts = np.unique(self.cluster_labels_, return_counts=True)
        size_dict = dict(zip(unique_labels, counts))
        total_samples = len(X)

        evidence_scores = []
        for label in self.cluster_labels_:
            if label == -1:
                # Unclustered noise point -> high evidence score
                evidence_scores.append(0.85)
            else:
                cluster_size = size_dict.get(label, total_samples)
                size_ratio = cluster_size / total_samples
                # Sparse cluster (< 1% of data) gets moderate evidence score
                if size_ratio < 0.01:
                    evidence_scores.append(0.50)
                elif size_ratio < 0.05:
                    evidence_scores.append(0.25)
                else:
                    evidence_scores.append(0.05)

        normalized_dbscan_evidence = np.array(evidence_scores)
        return self.cluster_labels_, self.noise_mask_, normalized_dbscan_evidence

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary stats of discovered behavioural clusters."""
        if not self.is_fitted:
            return {"status": "Not fitted"}

        unique, counts = np.unique(self.cluster_labels_, return_counts=True)
        noise_cnt = int(counts[unique == -1][0]) if -1 in unique else 0
        valid_clusters = int(len(unique) - (1 if -1 in unique else 0))

        return {
            "total_clusters_found": valid_clusters,
            "noise_points_count": noise_cnt,
            "cluster_size_distribution": dict(zip([int(k) for k in unique], [int(v) for v in counts]))
        }
