"""
Datashader Large-Scale Telemetry & Graph Renderer
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Uses Datashader to rasterize tens of thousands to millions of network telemetry packets,
transaction amounts, and entity graph topologies into lightweight 2D density heatmaps
and base64-encoded forensic visualization images. Operates 100% offline.
"""

import io
import base64
import logging
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.colors import viridis, inferno
    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False
    viridis = ["#440154", "#31688e", "#35b779", "#fde725"]
    inferno = ["#000004", "#721f81", "#cd4071", "#fd7f6f", "#fcfdbf"]

# High-contrast forensic color palette
FORENSIC_FIRE = ["#0f172a", "#1e293b", "#3b82f6", "#06b6d4", "#f59e0b", "#ef4444"]


class BitcoinDatashaderRenderer:
    """
    Renders large-scale Bitcoin and network telemetry distributions
    into rasterized density visualizations using Datashader.
    """

    def __init__(self, default_width: int = 800, default_height: int = 450):
        self.width = default_width
        self.height = default_height

    def is_available(self) -> bool:
        """Check if Datashader is installed and functional."""
        return DATASHADER_AVAILABLE

    def render_scatter_density(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        cmap_name: str = "inferno"
    ) -> Optional[str]:
        """
        Rasterize a dense scatter distribution (e.g. packet_rate vs amount_btc)
        and return a base64 PNG data URI string.
        """
        if not DATASHADER_AVAILABLE or df is None or df.empty:
            return None

        w = width or self.width
        h = height or self.height

        if x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"Columns {x_col} or {y_col} not in DataFrame.")
            return None

        # Clean numerical values
        plot_df = df[[x_col, y_col]].dropna().copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce").fillna(0.0)
        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce").fillna(0.0)

        if plot_df.empty:
            return None

        try:
            cvs = ds.Canvas(plot_width=w, plot_height=h)
            agg = cvs.points(plot_df, x_col, y_col)

            cmap = inferno if cmap_name == "inferno" else (viridis if cmap_name == "viridis" else FORENSIC_FIRE)
            img = tf.shade(agg, cmap=cmap, how="log")

            buf = io.BytesIO()
            img.to_pil().save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"
        except Exception as e:
            logger.warning(f"Datashader render error: {e}")
            return None

    def render_anomaly_distribution(
        self,
        df: pd.DataFrame,
        width: int = 800,
        height: int = 400
    ) -> Optional[str]:
        """
        Specialized forensic view: Risk Score vs Output BTC or Packet Rate.
        """
        x_col = "total_output_amount_btc" if "total_output_amount_btc" in df.columns else "amount"
        y_col = "risk_score" if "risk_score" in df.columns else "final_risk_score"

        if x_col in df.columns and y_col in df.columns:
            return self.render_scatter_density(df, x_col, y_col, width=width, height=height, cmap_name="inferno")
        return None
