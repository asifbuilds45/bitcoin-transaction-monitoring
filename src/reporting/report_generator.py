# src/reporting/report_generator.py
"""Security Threat Report Generation Module

Generates PDF reports for individual Bitcoin transactions or a batch of
transactions directly from the existing pipeline data.

Uses **ReportLab** for robust, Unicode-safe PDF generation.

* `ReportGenerator` - public class with static helper methods.
* Data-driven "WHY ANOMALOUS" reasons based on feature comparisons to
  dataset statistics (percentiles & IQR).
* Sequential report IDs (BTC-000001, BTC-000002, ...).
* No hard-coded values - every field is sourced from the scored_df row
  or derived from feature engineering output.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import ReportLab
# ---------------------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature names used for data-driven anomaly reasons
# ---------------------------------------------------------------------------
CORE_FEATURE_NAMES = [
    "transaction_frequency_24h",
    "avg_time_gap_min",
    "wallet_degree",
    "unique_ip_count",
    "country_count",
    "asn_count",
    "num_inputs",
    "num_outputs",
    "total_output_amount_btc",
    "fee_btc",
    "connection_duration_sec",
    "packet_count",
    "bytes_transferred",
]

DERIVED_FEATURE_NAMES = [
    "input_output_ratio",
    "bytes_per_packet",
    "fee_to_amount_ratio",
    "is_cross_border",
    "duration_per_packet",
    "amount_per_output",
]

ALL_FEATURE_NAMES = CORE_FEATURE_NAMES + DERIVED_FEATURE_NAMES

# Human-readable descriptions for features
FEATURE_DESCRIPTIONS = {
    "transaction_frequency_24h": "transaction frequency (24h window)",
    "avg_time_gap_min": "average time gap between transactions",
    "wallet_degree": "wallet connectivity degree",
    "unique_ip_count": "unique IP address count",
    "country_count": "country diversity",
    "asn_count": "ASN diversity",
    "num_inputs": "transaction input count",
    "num_outputs": "transaction output count",
    "total_output_amount_btc": "total output amount (BTC)",
    "fee_btc": "transaction fee (BTC)",
    "connection_duration_sec": "network connection duration",
    "packet_count": "network packet count",
    "bytes_transferred": "bytes transferred",
    "input_output_ratio": "input/output ratio",
    "bytes_per_packet": "bytes per packet",
    "fee_to_amount_ratio": "fee-to-amount ratio",
    "is_cross_border": "cross-border indicator",
    "duration_per_packet": "duration per packet",
    "amount_per_output": "amount per output",
}

# Evidence category mapping for features
FEATURE_CATEGORIES = {
    "transaction_frequency_24h": "Transaction Behaviour",
    "avg_time_gap_min": "Temporal Behaviour",
    "wallet_degree": "Wallet Connectivity",
    "unique_ip_count": "Network Behaviour",
    "country_count": "Network Behaviour",
    "asn_count": "Network Behaviour",
    "num_inputs": "Transaction Structure",
    "num_outputs": "Transaction Structure",
    "total_output_amount_btc": "Transaction Behaviour",
    "fee_btc": "Transaction Behaviour",
    "connection_duration_sec": "Network Behaviour",
    "packet_count": "Network Behaviour",
    "bytes_transferred": "Network Behaviour",
    "input_output_ratio": "Transaction Structure",
    "bytes_per_packet": "Network Behaviour",
    "fee_to_amount_ratio": "Transaction Behaviour",
    "is_cross_border": "Network Behaviour",
    "duration_per_packet": "Network Behaviour",
    "amount_per_output": "Transaction Behaviour",
}


# ---------------------------------------------------------------------------
# Helper: compute statistics for all ML features
# ---------------------------------------------------------------------------
def compute_dataset_statistics(scored_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Pre-compute descriptive stats for each feature."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in ALL_FEATURE_NAMES:
        if col not in scored_df.columns:
            continue
        series = pd.to_numeric(scored_df[col], errors="coerce")
        median = float(series.median())
        p5 = float(series.quantile(0.05))
        p25 = float(series.quantile(0.25))
        p75 = float(series.quantile(0.75))
        p95 = float(series.quantile(0.95))
        iqr = p75 - p25
        stats[col] = {
            "median": median,
            "p5": p5,
            "p25": p25,
            "p75": p75,
            "p95": p95,
            "iqr": iqr,
        }
    return stats


def _format_number(val) -> str:
    """Human-friendly formatting for numbers."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return f"{int(val):,}"
    if isinstance(val, (float, np.floating)):
        if abs(val) < 0.001:
            return f"{val:.6f}"
        return f"{val:,.3f}"
    return str(val)


def _safe_str(val) -> str:
    """Convert a value to a display-safe string, replacing NaN/None."""
    if val is None:
        return "Unknown"
    if isinstance(val, float) and np.isnan(val):
        return "Unknown"
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return "Unknown"
    return s


def generate_anomaly_reasons(tx_row: pd.Series, stats: Dict[str, Dict[str, float]]) -> List[str]:
    """Generate data-driven reasons why the transaction is anomalous.

    Categorises reasons by evidence type and picks the strongest signals.
    """
    raw_reasons: List[tuple] = []

    for feature, fstats in stats.items():
        if feature not in tx_row or pd.isna(tx_row[feature]):
            continue
        val = float(tx_row[feature])
        p5 = fstats["p5"]
        p75 = fstats["p75"]
        p95 = fstats["p95"]

        readable = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " "))
        category = FEATURE_CATEGORIES.get(feature, "General")

        if val >= p95:
            reason = f"Extremely elevated {readable} compared with observed transaction patterns."
            raw_reasons.append((category, reason, 3))
        elif val >= p75:
            reason = f"Elevated {readable} relative to normal transaction behaviour."
            raw_reasons.append((category, reason, 2))
        elif val <= p5:
            reason = f"Unusually low {readable} compared with the observed dataset."
            raw_reasons.append((category, reason, 1))

    # Sort by severity (highest first), pick best per category, limit to ~4
    raw_reasons.sort(key=lambda x: x[2], reverse=True)
    seen_categories = set()
    selected: List[str] = []
    for cat, reason, _ in raw_reasons:
        if cat not in seen_categories:
            selected.append(reason)
            seen_categories.add(cat)
        if len(selected) >= 4:
            break

    # If we have fewer than 4, add more from remaining
    if len(selected) < 4:
        for cat, reason, _ in raw_reasons:
            if reason not in selected:
                selected.append(reason)
            if len(selected) >= 4:
                break

    # Include any existing pipeline-generated textual reasons
    for col in ["risk_reasons", "fused_reasons"]:
        if col in tx_row and isinstance(tx_row[col], str):
            extra = [r.strip() for r in tx_row[col].split(";") if r.strip()]
            for e in extra:
                if e not in selected:
                    selected.append(e)

    # Deduplicate
    seen = set()
    uniq = []
    for r in selected:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def _get_remarks(risk_level: str) -> List[str]:
    """Return risk-level-specific remarks."""
    level = risk_level.upper() if isinstance(risk_level, str) else "UNKNOWN"
    if level == "CRITICAL":
        return [
            "Critical-risk activity requiring immediate investigator attention.",
            "Requires further investigation.",
            "Alert supported by network and blockchain evidence.",
        ]
    elif level == "HIGH":
        return [
            "High-risk activity requiring priority investigation.",
            "Requires further investigation.",
            "Alert supported by network and blockchain evidence.",
        ]
    elif level == "MEDIUM":
        return [
            "Medium-risk activity requiring additional review.",
            "Requires further investigation.",
            "Alert supported by network and blockchain evidence.",
        ]
    elif level == "LOW":
        return [
            "Low-risk anomalous activity identified for monitoring and review.",
            "Requires further investigation.",
            "Alert supported by network and blockchain evidence.",
        ]
    else:
        return [
            "Anomalous activity identified for review.",
            "Requires further investigation.",
            "Alert supported by network and blockchain evidence.",
        ]


# ---------------------------------------------------------------------------
# ReportLab PDF Builder helpers
# ---------------------------------------------------------------------------
if REPORTLAB_AVAILABLE:

    CLR_DARK = HexColor("#0f172a")
    CLR_HEADER_BG = HexColor("#1e293b")
    CLR_ACCENT = HexColor("#3b82f6")
    CLR_CRITICAL = HexColor("#dc2626")
    CLR_HIGH = HexColor("#f97316")
    CLR_MEDIUM = HexColor("#eab308")
    CLR_LOW = HexColor("#22c55e")
    CLR_TEXT = HexColor("#1e293b")
    CLR_MUTED = HexColor("#64748b")

    def _risk_color(level: str) -> HexColor:
        m = {"CRITICAL": CLR_CRITICAL, "HIGH": CLR_HIGH, "MEDIUM": CLR_MEDIUM, "LOW": CLR_LOW}
        return m.get(level.upper() if isinstance(level, str) else "", CLR_ACCENT)

    def _safe_add_style(styles, style):
        if style.name in styles.byName:
            styles.byName[style.name] = style
        else:
            styles.add(style)

    def _build_styles():
        """Create custom paragraph styles."""
        styles = getSampleStyleSheet()
        _safe_add_style(styles, ParagraphStyle(
            "ReportTitle", parent=styles["Title"],
            fontSize=16, leading=20, textColor=CLR_DARK,
            spaceAfter=4, alignment=TA_CENTER,
        ))
        _safe_add_style(styles, ParagraphStyle(
            "SectionHead", parent=styles["Heading2"],
            fontSize=11, leading=14, textColor=white,
            backColor=CLR_HEADER_BG, spaceBefore=8, spaceAfter=4,
            leftIndent=6, borderPadding=(3, 3, 3, 3),
        ))
        _safe_add_style(styles, ParagraphStyle(
            "KVKey", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=CLR_MUTED,
            fontName="Helvetica-Bold",
        ))
        _safe_add_style(styles, ParagraphStyle(
            "KVVal", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=CLR_TEXT,
        ))
        _safe_add_style(styles, ParagraphStyle(
            "BtcBullet", parent=styles["Normal"],
            fontSize=9, leading=13, textColor=CLR_TEXT,
            bulletIndent=12, leftIndent=24,
            spaceBefore=1, spaceAfter=1,
        ))
        _safe_add_style(styles, ParagraphStyle(
            "SubtitleStyle", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=CLR_MUTED,
            alignment=TA_CENTER, spaceAfter=6,
        ))
        return styles

    def _add_header_footer(canvas, doc):
        """Draw header and footer on every page."""
        canvas.saveState()
        w, h = A4
        # Header line
        canvas.setStrokeColor(CLR_ACCENT)
        canvas.setLineWidth(2)
        canvas.line(20 * mm, h - 12 * mm, w - 20 * mm, h - 12 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(CLR_MUTED)
        canvas.drawString(20 * mm, h - 11 * mm, "BITCOIN SENTINEL - SECURITY THREAT REPORT")
        canvas.drawRightString(w - 20 * mm, h - 11 * mm, "CONFIDENTIAL")
        # Footer
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 12 * mm, w - 20 * mm, 12 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, 8 * mm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(w - 20 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    def _build_kv_table(pairs: List[tuple], styles) -> Table:
        """Build a two-column key/value table."""
        data = []
        for k, v in pairs:
            data.append([
                Paragraph(f"<b>{k}:</b>", styles["KVKey"]),
                Paragraph(str(v), styles["KVVal"]),
            ])
        t = Table(data, colWidths=[50 * mm, 115 * mm])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING", (0, 0), (0, -1), 6),
        ]))
        return t

    def _build_single_report(idx: int, row: pd.Series, stats: Dict, styles) -> List:
        """Build flowable elements for one transaction report."""
        elements = []

        report_id = f"BTC-{idx:06d}"
        risk_level = _safe_str(row.get("risk_level", "UNKNOWN")).upper()
        risk_score = row.get("risk_score", 0.0)
        anomaly_score = row.get("anomaly_score", 0.0)
        txid = _safe_str(row.get("txid", "Unknown"))

        # Parse timestamp
        ts_raw = row.get("timestamp")
        if isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw
        else:
            try:
                ts = pd.to_datetime(ts_raw)
            except Exception:
                ts = None

        date_str = ts.strftime("%d-%m-%Y") if ts else "Unknown"
        time_str = ts.strftime("%H:%M:%S") if ts else "Unknown"

        # --- Title ---
        rc = _risk_color(risk_level)
        elements.append(Paragraph("SECURITY THREAT REPORT", styles["ReportTitle"]))
        elements.append(Paragraph(
            f'<font color="{rc.hexval()}">[{risk_level}]</font>&nbsp;&nbsp;{report_id}',
            styles["SubtitleStyle"]
        ))
        elements.append(HRFlowable(width="100%", thickness=1, color=CLR_ACCENT, spaceAfter=6))

        # --- PROFILE ---
        elements.append(Paragraph("PROFILE", styles["SectionHead"]))
        elements.append(Spacer(1, 2))
        elements.append(_build_kv_table([
            ("Report ID", report_id),
            ("Transaction ID", txid),
            ("Risk Level", risk_level),
            ("Date", date_str),
            ("Time", time_str),
        ], styles))
        elements.append(Spacer(1, 4))

        # --- ACTORS ---
        elements.append(Paragraph("ACTORS", styles["SectionHead"]))
        elements.append(Spacer(1, 2))
        elements.append(_build_kv_table([
            ("Source IP", _safe_str(row.get("src_ip"))),
            ("Destination IP", _safe_str(row.get("dst_ip"))),
            ("Wallet", _safe_str(row.get("source_wallet", row.get("destination_wallet")))),
            ("Country", _safe_str(row.get("geoip_src_country", row.get("src_country")))),
            ("ASN", _safe_str(row.get("geoip_src_asn", row.get("src_asn")))),
        ], styles))
        elements.append(Spacer(1, 4))

        # --- DIAGNOSTIC ---
        elements.append(Paragraph("DIAGNOSTIC", styles["SectionHead"]))
        elements.append(Spacer(1, 2))
        elements.append(Paragraph("<b>WHY ANOMALOUS</b>", styles["KVKey"]))
        elements.append(Spacer(1, 1))

        reasons = generate_anomaly_reasons(row, stats)
        if not reasons:
            reasons = ["The anomaly detection model classified this transaction as anomalous based on its combined feature profile."]
        for r in reasons:
            safe_r = r.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe_r = safe_r.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
            elements.append(Paragraph(f"\u2022 {safe_r}", styles["BtcBullet"]))

        elements.append(Spacer(1, 3))

        try:
            a_str = f"{float(anomaly_score):.4f}"
        except (ValueError, TypeError):
            a_str = _safe_str(anomaly_score)
        try:
            r_str = f"{float(risk_score):.2f} / 100"
        except (ValueError, TypeError):
            r_str = _safe_str(risk_score)

        elements.append(_build_kv_table([
            ("Anomaly Score", a_str),
            ("Risk Score", r_str),
        ], styles))
        elements.append(Spacer(1, 4))

        # --- Additional Transaction Info ---
        extra_pairs = []
        for col, label in [
            ("num_inputs", "Number of Inputs"),
            ("num_outputs", "Number of Outputs"),
            ("total_output_amount_btc", "Total Output (BTC)"),
            ("fee_btc", "Fee (BTC)"),
            ("packet_count", "Packet Count"),
            ("bytes_transferred", "Bytes Transferred"),
            ("wallet_degree", "Wallet Degree"),
            ("transaction_frequency_24h", "Tx Frequency (24h)"),
        ]:
            if col in row.index and not pd.isna(row.get(col)):
                extra_pairs.append((label, _format_number(row[col])))
        if extra_pairs:
            elements.append(Paragraph("ADDITIONAL TRANSACTION INFORMATION", styles["SectionHead"]))
            elements.append(Spacer(1, 2))
            elements.append(_build_kv_table(extra_pairs, styles))
            elements.append(Spacer(1, 4))

        # --- SOLUTION / RECOMMENDED ACTIONS ---
        elements.append(Paragraph("SOLUTION / RECOMMENDED ACTIONS", styles["SectionHead"]))
        elements.append(Spacer(1, 2))
        actions = [
            "Investigate the associated wallet.",
            "Review related transactions.",
            "Monitor connected network entities.",
            "Trace related transaction activity.",
        ]
        for a in actions:
            elements.append(Paragraph(f"\u2022 {a}", styles["BtcBullet"]))
        elements.append(Spacer(1, 4))

        # --- REMARKS ---
        elements.append(Paragraph("REMARKS", styles["SectionHead"]))
        elements.append(Spacer(1, 2))
        remarks = _get_remarks(risk_level)
        for rm in remarks:
            elements.append(Paragraph(f"\u2022 {rm}", styles["BtcBullet"]))

        return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class ReportGenerator:
    """Public API for generating PDF reports."""

    @staticmethod
    def compute_dataset_statistics(scored_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Delegate to module-level helper."""
        return compute_dataset_statistics(scored_df)

    @staticmethod
    def generate_batch_report_pdf(
        df_scored: pd.DataFrame,
        stats: Dict[str, Dict[str, float]],
        max_pages: Optional[int] = None,
        id_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> bytes:
        """Generate a single PDF containing one forensic report per anomalous transaction.

        Args:
            df_scored: Complete anomaly result DataFrame.
            stats: Feature statistics dictionary.
            max_pages: Optional limit on number of transaction reports (None = all).
            id_map: Unused, kept for backward compatibility.

        Returns:
            Bytes of the generated PDF.
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF report generation. "
                "Install with: pip install reportlab"
            )

        # Handle empty data gracefully
        if df_scored is None or df_scored.empty:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4,
                                    leftMargin=20*mm, rightMargin=20*mm,
                                    topMargin=18*mm, bottomMargin=18*mm)
            styles = _build_styles()
            elems = [
                Spacer(1, 40),
                Paragraph("BITCOIN SENTINEL", styles["ReportTitle"]),
                Spacer(1, 10),
                Paragraph("No anomalous transactions available for report generation.",
                          styles["KVVal"]),
            ]
            doc.build(elems, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
            return buffer.getvalue()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=18*mm, bottomMargin=18*mm,
        )
        styles = _build_styles()
        elements = []

        total = len(df_scored)
        limit = total if max_pages is None else min(max_pages, total)

        for i, (_, row) in enumerate(df_scored.iterrows()):
            if i >= limit:
                break
            report_elems = _build_single_report(i + 1, row, stats, styles)
            elements.extend(report_elems)
            # Page break between reports (not after the last one)
            if i < limit - 1:
                elements.append(PageBreak())

        doc.build(elements, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
        return buffer.getvalue()

    @staticmethod
    def get_stable_report_id(txid: str, id_map: Dict[str, str]) -> str:
        """Get or create a stable report ID for a transaction."""
        if txid in id_map:
            return id_map[txid]
        new_id = f"BTC-{len(id_map) + 1:06d}"
        id_map[txid] = new_id
        return new_id

# End of module
