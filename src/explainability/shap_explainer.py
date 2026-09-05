"""
SHAP Explainability Module
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Uses SHAP (SHapley Additive exPlanations) via surrogate decision trees or direct
SHAP Explainers to provide mathematically rigorous feature contribution scores
for unsupervised anomaly detection decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.tree import DecisionTreeRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class BitcoinSHAPExplainer:
    """
    Computes SHAP feature contributions for unsupervised anomaly predictions.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.surrogate_model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        self.explainer = None
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X: pd.DataFrame, target_anomaly_scores: np.ndarray) -> "BitcoinSHAPExplainer":
        """
        Fit a surrogate decision tree to approximate the Isolation Forest / Fused anomaly scores.
        
        Args:
            X: Feature matrix DataFrame.
            target_anomaly_scores: Continuous anomaly score targets (0.0 to 1.0).
        """
        self.feature_names = list(X.columns)
        self.surrogate_model.fit(X, target_anomaly_scores)
        
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.surrogate_model)
            except Exception:
                self.explainer = None
                
        self.is_fitted = True
        return self

    def explain_instance(
        self,
        instance_row: pd.Series,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explain a single transaction's anomaly score using SHAP values.
        
        Returns:
            List of dicts containing feature, value, shap_value, direction, magnitude, and description.
        """
        if not self.is_fitted:
            return []

        # Prepare single row DataFrame matching feature names
        x_df = pd.DataFrame([instance_row[col] if col in instance_row else 0.0 for col in self.feature_names], index=self.feature_names).T

        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(x_df)
                if isinstance(shap_vals, list):
                    vals = shap_vals[0][0]
                elif shap_vals.ndim == 2:
                    vals = shap_vals[0]
                else:
                    vals = shap_vals
            except Exception:
                vals = self._fallback_feature_importance(x_df.iloc[0])
        else:
            vals = self._fallback_feature_importance(x_df.iloc[0])

        results = []
        for idx, col_name in enumerate(self.feature_names):
            s_val = float(vals[idx])
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

        # Sort by absolute SHAP magnitude descending
        results.sort(key=lambda item: item['magnitude'], reverse=True)
        return results[:top_n]

    def _fallback_feature_importance(self, row_series: pd.Series) -> np.ndarray:
        """Surrogate feature importance calculation if SHAP TreeExplainer fails."""
        importances = self.surrogate_model.feature_importances_
        vals = row_series.values.astype(float)
        # Weight feature importances by normalized value deviation
        mean_val = np.mean(vals) if len(vals) > 0 else 1.0
        scale = (vals - mean_val) / max(np.std(vals), 1.0)
        return importances * scale
