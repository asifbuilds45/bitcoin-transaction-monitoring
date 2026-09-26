"""
SHAP Explainability Module (TreeSHAP)
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Uses SHAP (SHapley Additive exPlanations) via TreeSHAP on surrogate decision trees
to provide mathematically rigorous feature contribution scores for unsupervised anomaly detection
and risk scoring decisions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class BitcoinSHAPExplainer:
    """
    Computes SHAP feature contributions for unsupervised anomaly predictions
    using TreeSHAP on a surrogate model.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.surrogate_model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        self.explainer = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.global_importance_: List[Dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, target_anomaly_scores: np.ndarray) -> "BitcoinSHAPExplainer":
        """
        Fit a surrogate decision tree to approximate the Isolation Forest / Fused anomaly scores.

        Args:
            X: Feature matrix DataFrame.
            target_anomaly_scores: Continuous anomaly score targets (0.0 to 1.0).
        """
        if X is None or X.empty or len(target_anomaly_scores) == 0:
            return self

        self.feature_names = list(X.columns)
        self.surrogate_model.fit(X, target_anomaly_scores)

        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(
                    self.surrogate_model,
                    feature_perturbation="tree_path_dependent"
                )
            except Exception as e:
                logger.warning(f"TreeExplainer initialization notice: {e}")
                try:
                    self.explainer = shap.TreeExplainer(self.surrogate_model)
                except Exception:
                    self.explainer = None

        self.is_fitted = True
        self._calculate_global_importance(X)
        return self

    def _calculate_global_importance(self, X: pd.DataFrame) -> None:
        """Calculate global mean absolute SHAP values or surrogate feature importances."""
        try:
            if SHAP_AVAILABLE and self.explainer is not None and not X.empty:
                sample_size = min(len(X), 200)
                sample_X = X.iloc[:sample_size]
                shap_matrix = self.explainer.shap_values(sample_X)
                if isinstance(shap_matrix, list):
                    shap_matrix = shap_matrix[0]
                mean_abs = np.mean(np.abs(shap_matrix), axis=0)
            else:
                mean_abs = self.surrogate_model.feature_importances_

            importance_list = []
            for name, score in zip(self.feature_names, mean_abs):
                importance_list.append({
                    "feature": name,
                    "importance": round(float(score), 4)
                })
            importance_list.sort(key=lambda x: x["importance"], reverse=True)
            self.global_importance_ = importance_list
        except Exception as e:
            logger.debug(f"Global importance calculation: {e}")
            importances = self.surrogate_model.feature_importances_
            self.global_importance_ = [
                {"feature": name, "importance": round(float(score), 4)}
                for name, score in zip(self.feature_names, importances)
            ]
            self.global_importance_.sort(key=lambda x: x["importance"], reverse=True)

    def explain_instance(
        self,
        instance_row: pd.Series,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain a single transaction's anomaly score using TreeSHAP values.

        Returns:
            List of dicts containing feature, value, shap_value, direction, magnitude, and description.
        """
        if not self.is_fitted or not self.feature_names:
            return []

        # Prepare single row DataFrame matching feature names
        x_dict = {col: float(instance_row.get(col, 0.0)) for col in self.feature_names}
        x_df = pd.DataFrame([x_dict], columns=self.feature_names)

        vals = None
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(x_df)
                if isinstance(shap_vals, list):
                    vals = shap_vals[0][0]
                elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 2:
                    vals = shap_vals[0]
                else:
                    vals = shap_vals
            except Exception as e:
                logger.debug(f"Single instance SHAP fallback: {e}")
                vals = None

        if vals is None:
            vals = self._fallback_feature_importance(x_df.iloc[0])

        results = []
        for idx, col_name in enumerate(self.feature_names):
            s_val = float(vals[idx]) if idx < len(vals) else 0.0
            val = float(x_df.iloc[0, idx])

            direction = "increased risk contribution" if s_val > 0 else "decreased risk contribution"
            magnitude = abs(s_val)

            results.append({
                "feature": col_name,
                "feature_value": val,
                "shap_value": round(s_val, 4),
                "direction": direction,
                "magnitude": round(magnitude, 4),
                "explanation": f"{col_name} = {val:.2f} ({direction})"
            })

        results.sort(key=lambda item: item["magnitude"], reverse=True)
        return results[:top_n]

    def get_global_feature_importance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the top globally important features identified by TreeSHAP."""
        return self.global_importance_[:top_n]

    def _fallback_feature_importance(self, row_series: pd.Series) -> np.ndarray:
        """Surrogate feature importance calculation if SHAP TreeExplainer fails."""
        importances = self.surrogate_model.feature_importances_
        vals = row_series.values.astype(float)
        mean_val = np.mean(vals) if len(vals) > 0 else 1.0
        scale = (vals - mean_val) / max(float(np.std(vals)), 1.0)
        return importances * scale
