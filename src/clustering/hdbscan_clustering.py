"""
HDBSCAN Hierarchical Density-Based Behavioural Clustering Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

IMPORTANT ARCHITECTURAL RULE:
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
is an upgrade over traditional DBSCAN. It automatically discovers clusters of varying
densities without requiring a single global epsilon distance threshold, handles noise robustly,
and generates continuous outlier scores from the cluster hierarchy.

Provides exact signature compatibility with the Bitcoin Monitoring pipeline:
fit_predict(X) -> Tuple[cluster_labels, is_noise_flag, normalized_evidence]
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN, DBSCAN

logger = logging.getLogger(__name__)


class BitcoinHDBSCANClustering:
    """
    Hierarchical Density-Based Spatial Clustering for Bitcoin behavioural group
    discovery, varying-density cluster isolation, and noise/outlier detection.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: Optional[int] = 3, metric: str = "euclidean"):
        """
        Initialize HDBSCAN clustering engine.

        Args:
            min_cluster_size: Minimum number of samples in a group for that group to be considered a cluster.
            min_samples: Number of samples in a neighborhood for a point to be considered a core point.
            metric: Distance metric to use for feature space clustering.
        """
        self.min_cluster_size = max(2, min_cluster_size)
        self.min_samples = max(1, min_samples) if min_samples else 3
        self.metric = metric
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.cluster_labels_ = np.array([])
        self.noise_mask_ = np.array([])
        self.outlier_scores_ = np.array([])
        self.probabilities_ = np.array([])

        try:
            self.model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                n_jobs=-1
            )
            self._using_hdbscan = True
        except Exception as e:
            logger.warning(f"HDBSCAN init fallback to DBSCAN: {e}")
            self.model = DBSCAN(eps=0.5, min_samples=self.min_cluster_size, n_jobs=-1)
            self._using_hdbscan = False

    def fit_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit HDBSCAN on scaled features, predict behavioural clusters, and generate
        normalized HDBSCAN Evidence Scores.

        Args:
            X: Feature matrix (strictly unsupervised behavioral features).

        Returns:
            Tuple of:
            - cluster_labels (-1 for noise, 0..N for cluster IDs)
            - is_noise_flag (1 for noise/outlier, 0 for cluster member)
            - normalized_hdbscan_evidence (0.0 to 1.0)
        """
        if X is None or X.empty:
            return np.array([]), np.array([]), np.array([])

        # Handle datasets smaller than min_cluster_size
        n_samples = len(X)
        if n_samples < self.min_cluster_size:
            labels = np.array([-1] * n_samples)
            noise = np.array([1] * n_samples)
            evidence = np.array([0.5] * n_samples)
            self.cluster_labels_ = labels
            self.noise_mask_ = noise
            self.is_fitted = True
            return labels, noise, evidence

        X_scaled = self.scaler.fit_transform(X)

        try:
            self.cluster_labels_ = self.model.fit_predict(X_scaled)
            self.is_fitted = True
            
            # Extract GLOSH outlier scores if HDBSCAN
            if hasattr(self.model, "outlier_scores_") and self.model.outlier_scores_ is not None:
                self.outlier_scores_ = np.nan_to_num(self.model.outlier_scores_, nan=0.0)
            else:
                self.outlier_scores_ = np.zeros(n_samples)

            if hasattr(self.model, "probabilities_") and self.model.probabilities_ is not None:
                self.probabilities_ = np.nan_to_num(self.model.probabilities_, nan=0.0)
            else:
                self.probabilities_ = np.ones(n_samples)

        except Exception as e:
            logger.warning(f"HDBSCAN fit failed ({e}), falling back to DBSCAN")
            fallback_dbscan = DBSCAN(eps=0.5, min_samples=min(self.min_cluster_size, n_samples), n_jobs=-1)
            self.cluster_labels_ = fallback_dbscan.fit_predict(X_scaled)
            self.outlier_scores_ = np.zeros(n_samples)
            self.is_fitted = True

        self.noise_mask_ = np.where(self.cluster_labels_ == -1, 1, 0)

        # Calculate cluster sizes to score sparse clusters
        unique_labels, counts = np.unique(self.cluster_labels_, return_counts=True)
        size_dict = dict(zip(unique_labels, counts))
        total_samples = len(X)

        evidence_scores = []
        for idx, label in enumerate(self.cluster_labels_):
            if label == -1:
                # Unclustered noise point -> high evidence score
                raw_outlier = float(self.outlier_scores_[idx]) if len(self.outlier_scores_) > idx else 0.85
                evidence_score = max(0.75, min(0.95, 0.75 + (raw_outlier * 0.20)))
                evidence_scores.append(evidence_score)
            else:
                cluster_size = size_dict.get(label, total_samples)
                size_ratio = cluster_size / total_samples
                
                # Combine size rarity with outlier score
                if size_ratio < 0.01:
                    evidence_scores.append(0.50)
                elif size_ratio < 0.05:
                    evidence_scores.append(0.25)
                else:
                    evidence_scores.append(0.05)

        normalized_hdbscan_evidence = np.array(evidence_scores, dtype=float)
        return self.cluster_labels_, self.noise_mask_, normalized_hdbscan_evidence

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary statistics of discovered behavioural clusters."""
        if not self.is_fitted or len(self.cluster_labels_) == 0:
            return {"status": "Not fitted", "total_clusters_found": 0, "noise_points_count": 0}

        unique, counts = np.unique(self.cluster_labels_, return_counts=True)
        noise_cnt = int(counts[unique == -1][0]) if -1 in unique else 0
        valid_clusters = int(len(unique) - (1 if -1 in unique else 0))

        return {
            "algorithm": "HDBSCAN (Hierarchical Density)",
            "total_clusters_found": valid_clusters,
            "noise_points_count": noise_cnt,
            "noise_ratio": round(noise_cnt / max(len(self.cluster_labels_), 1), 4),
            "cluster_size_distribution": dict(zip([int(k) for k in unique], [int(v) for v in counts])),
            "varying_density_capable": True
        }
