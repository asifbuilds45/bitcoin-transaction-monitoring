# src/reporting/report_generator.py
"""Security Threat Report Generation Module

Generates multi-page forensic security threat PDF reports for individual Bitcoin
transactions or a batch of anomalous transactions directly from pipeline data.

Uses **ReportLab** for robust, Unicode-safe forensic PDF generation.

* `ReportGenerator` - public class with static helper methods.
* Data-driven "WHY ANOMALOUS" reasons based on feature comparisons to
  dataset statistics (percentiles & IQR).
* Sequential report IDs (BTC-000001, BTC-000002, ...).
* No hard-coded values - every field is sourced from the scored_df row
  or derived from feature engineering output.
* 100% offline GeoIP Country & ASN enrichment using local GeoLite2 databases.
* Strict 1 transaction = 1 page forensic layout.
"""

from __future__ import annotations

import io
import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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
# ISO-3166-1 alpha-2 country code mappings for display
# ---------------------------------------------------------------------------
ISO_COUNTRY_NAMES = {
    "IN": "India", "US": "United States", "GB": "United Kingdom", "CA": "Canada",
    "AU": "Australia", "SG": "Singapore", "DE": "Germany", "FR": "France",
    "JP": "Japan", "CN": "China", "RU": "Russia", "BR": "Brazil",
    "NL": "Netherlands", "CH": "Switzerland", "KR": "South Korea", "SE": "Sweden",
    "NO": "Norway", "FI": "Finland", "ES": "Spain", "IT": "Italy",
    "ZA": "South Africa", "MX": "Mexico", "ID": "Indonesia", "TR": "Turkey",
    "SA": "Saudi Arabia", "AE": "United Arab Emirates", "IL": "Israel",
    "PL": "Poland", "UA": "Ukraine", "RO": "Romania", "NZ": "New Zealand",
    "IE": "Ireland", "AT": "Austria", "BE": "Belgium", "CZ": "Czech Republic",
    "DK": "Denmark", "PT": "Portugal", "GR": "Greece", "HU": "Hungary",
    "TH": "Thailand", "VN": "Vietnam", "MY": "Malaysia", "PH": "Philippines",
    "HK": "Hong Kong", "TW": "Taiwan", "AR": "Argentina", "CL": "Chile",
    "CO": "Colombia", "EG": "Egypt", "NG": "Nigeria", "KE": "Kenya"
}

_GEOIP_ENRICHER_CACHE = None

def _get_geoip_enricher():
    """Lazily load singleton GeoIPEnricher for fallback offline queries."""
    global _GEOIP_ENRICHER_CACHE
    if _GEOIP_ENRICHER_CACHE is None:
        try:
            from src.geoip_enrichment import GeoIPEnricher
            _GEOIP_ENRICHER_CACHE = GeoIPEnricher()
        except Exception as e:
            logger.debug(f"GeoIPEnricher init: {e}")
            _GEOIP_ENRICHER_CACHE = False
    return _GEOIP_ENRICHER_CACHE if _GEOIP_ENRICHER_CACHE else None


def get_geo_country(row: Any) -> str:
    """
    Robust extraction and formatting of Country for a transaction row.
    Checks existing enriched dataframe columns first, then raw dataset columns,
    and falls back to local GeoLite2 Country MMDB for public IPs.
    """
    def _extract_val(keys):
        for k in keys:
            v = None
            if hasattr(row, 'get'):
                v = row.get(k)
            elif hasattr(row, '__getitem__'):
                try:
                    v = row[k]
                except Exception:
                    continue
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                s = str(v).strip()
                if s and s.lower() not in ("unknown", "none", "nan", "null", "n/a", ""):
                    return s
        return None

    c_name = _extract_val(["geoip_src_country", "src_country_name", "country_name", "src_country", "country", "geo_country"])
    c_code = _extract_val(["geoip_src_country_code", "src_country_code", "country_code", "src_country", "geo_country_code"])

    # If c_name is an ISO code, convert to full name
    if c_name and len(c_name) == 2 and c_name.upper() in ISO_COUNTRY_NAMES:
        c_code = c_code or c_name.upper()
        c_name = ISO_COUNTRY_NAMES[c_name.upper()]
    elif c_code and len(c_code) == 2 and not c_name and c_code.upper() in ISO_COUNTRY_NAMES:
        c_name = ISO_COUNTRY_NAMES[c_code.upper()]

    # If still not found, check local GeoLite2 database for public IP
    if not c_name or c_name.lower() in ("unknown", "none", "nan", ""):
        ip = _extract_val(["src_ip", "source_ip", "ip"])
        if ip:
            enricher = _get_geoip_enricher()
            if enricher:
                res = enricher.lookup_ip(ip)
                if res.get("country") and res["country"].lower() not in ("unknown", "none", "nan", ""):
                    c_name = res["country"]
                    c_code = res.get("country_code")

    if not c_name or c_name.lower() in ("unknown", "none", "nan", ""):
        return "Unknown"

    if c_code and c_code != "Unknown" and c_code.upper() != c_name.upper() and len(c_code) <= 3:
        return f"{c_name} ({c_code.upper()})"
    return c_name


def get_geo_asn(row: Any) -> str:
    """
    Robust extraction and formatting of ASN for a transaction row.
    Checks existing enriched dataframe columns first, then raw dataset columns,
    and falls back to local GeoLite2 ASN MMDB for public IPs.
    """
    def _extract_val(keys):
        for k in keys:
            v = None
            if hasattr(row, 'get'):
                v = row.get(k)
            elif hasattr(row, '__getitem__'):
                try:
                    v = row[k]
                except Exception:
                    continue
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                s = str(v).strip()
                if s and s.lower() not in ("unknown", "none", "nan", "null", "n/a", ""):
                    return s
        return None

    asn = _extract_val(["geoip_src_asn", "src_asn", "asn", "geo_asn"])
    org = _extract_val(["geoip_src_org", "src_org", "asn_org", "organization", "geo_org", "isp", "geo_isp"])

    if not asn or asn.lower() in ("unknown", "none", "nan", ""):
        ip = _extract_val(["src_ip", "source_ip", "ip"])
        if ip:
            enricher = _get_geoip_enricher()
            if enricher:
                res = enricher.lookup_ip(ip)
                if res.get("asn") and res["asn"].lower() not in ("unknown", "none", "nan", ""):
                    asn = res["asn"]
                    org = res.get("asn_org")

    if not asn or asn.lower() in ("unknown", "none", "nan", ""):
        return "Unknown"

    # Normalize ASN formatting (e.g. AS45609)
    asn_str = str(asn).strip()
    if asn_str.isdigit():
        asn_str = f"AS{asn_str}"
    elif asn_str.upper().startswith("AS") and asn_str[2:].isdigit():
        asn_str = asn_str.upper()

    if org and org.lower() not in ("unknown", "none", "nan", "unknown org", "n/a", ""):
        return f"{asn_str} ({org})"
    return asn_str


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


def generate_anomaly_reasons(tx_row: Any, stats: Dict[str, Dict[str, float]]) -> List[str]:
    """Generate data-driven reasons why the transaction is anomalous."""
    raw_reasons: List[tuple] = []

    for feature, fstats in stats.items():
        if hasattr(tx_row, 'get'):
            val_raw = tx_row.get(feature)
        elif hasattr(tx_row, '__getitem__'):
            try:
                val_raw = tx_row[feature]
            except Exception:
                continue
        else:
            continue

        if val_raw is None or pd.isna(val_raw):
            continue
        try:
            val = float(val_raw)
        except (ValueError, TypeError):
            continue

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

    if len(selected) < 4:
        for cat, reason, _ in raw_reasons:
            if reason not in selected:
                selected.append(reason)
            if len(selected) >= 4:
                break

    # Include existing pipeline reasons if present
    for col in ["risk_reasons", "fused_reasons"]:
        val_r = tx_row.get(col) if hasattr(tx_row, 'get') else None
        if val_r and isinstance(val_r, str):
            extra = [r.strip() for r in val_r.split(";") if r.strip()]
            for e in extra:
                if e not in selected:
                    selected.append(e)

    # Include pattern detection reasons if detected
    for col in ["peeling_chain_reasons", "coinjoin_reasons"]:
        val_r = tx_row.get(col) if hasattr(tx_row, 'get') else None
        if val_r and isinstance(val_r, str) and val_r.strip():
            if val_r not in selected:
                selected.append(val_r)

    # Deduplicate and limit to 5 reasons to guarantee 1-page forensic format
    seen = set()
    uniq = []
    for r in selected:
        r_clean = re.sub(r'\bH*DBSCAN\b', 'HDBSCAN', str(r), flags=re.IGNORECASE)
        if r_clean not in seen:
            uniq.append(r_clean)
            seen.add(r_clean)
    return uniq[:5]


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
# ReportLab Styling & Flowables
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
        """Create custom paragraph styles for forensic report."""
        styles = getSampleStyleSheet()
        _safe_add_style(styles, ParagraphStyle(
            "ReportTitle", parent=styles["Title"],
            fontSize=16, leading=20, textColor=CLR_DARK,
            spaceAfter=4, alignment=TA_CENTER,
        ))
        _safe_add_style(styles, ParagraphStyle(
            "SectionHead", parent=styles["Heading2"],
            fontSize=11, leading=14, textColor=white,
            backColor=CLR_HEADER_BG, spaceBefore=5, spaceAfter=2,
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
        """Draw professional cyber header and footer on every page."""
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
        """Build two-column key/value table."""
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

    def _build_single_report(idx: int, row: Any, stats: Dict, styles) -> List:
        """Build flowable elements for one transaction forensic report page."""
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
        elements.append(Spacer(1, 1))
        elements.append(_build_kv_table([
            ("Report ID", report_id),
            ("Transaction ID", txid),
            ("Risk Level", risk_level),
            ("Date", date_str),
            ("Time", time_str),
        ], styles))
        elements.append(Spacer(1, 2))

        # --- ACTORS ---
        elements.append(Paragraph("ACTORS", styles["SectionHead"]))
        elements.append(Spacer(1, 1))

        country_display = get_geo_country(row)
        asn_display = get_geo_asn(row)
        wallet_val = _safe_str(row.get("source_wallet", row.get("destination_wallet")))

        elements.append(_build_kv_table([
            ("Source IP", _safe_str(row.get("src_ip"))),
            ("Destination IP", _safe_str(row.get("dst_ip"))),
            ("Wallet", wallet_val),
            ("Country", country_display),
            ("ASN", asn_display),
        ], styles))
        elements.append(Spacer(1, 2))

        # --- DIAGNOSTIC ---
        elements.append(Paragraph("DIAGNOSTIC", styles["SectionHead"]))
        elements.append(Spacer(1, 1))
        elements.append(Paragraph("<b>WHY ANOMALOUS</b>", styles["KVKey"]))
        elements.append(Spacer(1, 1))

        reasons = generate_anomaly_reasons(row, stats)
        if not reasons:
            reasons = ["The anomaly detection model classified this transaction as anomalous based on its combined feature profile."]
        for r in reasons:
            safe_r = r.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe_r = safe_r.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
            elements.append(Paragraph(f"\u2022 {safe_r}", styles["BtcBullet"]))

        elements.append(Spacer(1, 2))

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
        elements.append(Spacer(1, 2))

        # --- ADDITIONAL TRANSACTION INFORMATION ---
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
            val_col = row.get(col) if hasattr(row, 'get') else None
            if val_col is not None and not pd.isna(val_col):
                extra_pairs.append((label, _format_number(val_col)))

        if extra_pairs:
            elements.append(Paragraph("ADDITIONAL TRANSACTION INFORMATION", styles["SectionHead"]))
            elements.append(Spacer(1, 1))
            elements.append(_build_kv_table(extra_pairs, styles))
            elements.append(Spacer(1, 2))

        # --- BEHAVIOURAL PATTERNS ---
        peel_detected = bool(row.get("peeling_chain_detected", False)) if hasattr(row, 'get') else False
        cj_detected = bool(row.get("coinjoin_detected", False)) if hasattr(row, 'get') else False

        if peel_detected or cj_detected:
            elements.append(Paragraph("BEHAVIOURAL PATTERNS", styles["SectionHead"]))
            elements.append(Spacer(1, 1))

            pattern_bullets = []
            if peel_detected:
                peel_ev = float(row.get("peeling_chain_evidence", 0.0))
                peel_id = _safe_str(row.get("peeling_chain_id", ""))
                peel_pos = int(row.get("peeling_chain_position", 0))
                peel_len = int(row.get("peeling_chain_length", 0))
                pattern_bullets.append(
                    f"Peeling-chain pattern detected: Chain {peel_id}, "
                    f"Hop {peel_pos}/{peel_len} (evidence score: {peel_ev:.3f}). "
                    "Investigation evidence — Requires further investigation."
                )
            if cj_detected:
                cj_ev = float(row.get("coinjoin_evidence", 0.0))
                cj_eq = int(row.get("coinjoin_equal_output_count", 0))
                cj_ratio = float(row.get("coinjoin_equal_output_ratio", 0.0))
                cj_amt = float(row.get("coinjoin_equal_amount_btc", 0.0))
                pattern_bullets.append(
                    f"CoinJoin-like / mixing pattern detected: {cj_eq} equal outputs "
                    f"at approx. {cj_amt:.6f} BTC (equal-output ratio: {cj_ratio:.2f}, "
                    f"evidence score: {cj_ev:.3f}). "
                    "Behavioural indicator — Requires further investigation."
                )

            for pb in pattern_bullets:
                safe_pb = pb.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                elements.append(Paragraph(f"\u2022 {safe_pb}", styles["BtcBullet"]))
            elements.append(Spacer(1, 2))

        # --- SOLUTION / RECOMMENDED ACTIONS ---
        elements.append(Paragraph("SOLUTION / RECOMMENDED ACTIONS", styles["SectionHead"]))
        elements.append(Spacer(1, 1))
        actions = [
            "Investigate the associated wallet.",
            "Review related transactions.",
            "Monitor connected network entities.",
            "Trace related transaction activity.",
        ]
        if peel_detected:
            actions.insert(0, "Trace all transactions in the identified peeling chain to map fund flows.")
        if cj_detected:
            actions.insert(0, "Analyse equal-output recipients for potential mixing service off-ramp activity.")
        for a in actions:
            elements.append(Paragraph(f"\u2022 {a}", styles["BtcBullet"]))
        elements.append(Spacer(1, 2))

        # --- REMARKS ---
        elements.append(Paragraph("REMARKS", styles["SectionHead"]))
        elements.append(Spacer(1, 1))
        remarks = _get_remarks(risk_level)
        for rm in remarks:
            elements.append(Paragraph(f"\u2022 {rm}", styles["BtcBullet"]))

        return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class ReportGenerator:
    """Public API for generating forensic PDF reports."""

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
        """
        Generate a single PDF containing one forensic report per anomalous transaction.

        Args:
            df_scored: Complete or filtered result DataFrame.
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

        # 1. Isolate anomalous records (PDF export must include all anomalous records regardless of filter)
        if df_scored is None or df_scored.empty:
            df_anom = pd.DataFrame()
        elif "is_anomaly" in df_scored.columns:
            df_anom = df_scored[df_scored["is_anomaly"] == 1]
            if df_anom.empty and (df_scored["is_anomaly"].dtype == bool or df_scored["is_anomaly"].sum() > 0):
                df_anom = df_scored[df_scored["is_anomaly"].astype(bool)]
        else:
            df_anom = df_scored

        # Handle empty anomalous dataset gracefully
        if df_anom.empty:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer, pagesize=A4,
                leftMargin=20*mm, rightMargin=20*mm,
                topMargin=18*mm, bottomMargin=18*mm
            )
            styles = _build_styles()
            elems = [
                Spacer(1, 40),
                Paragraph("BITCOIN SENTINEL", styles["ReportTitle"]),
                Spacer(1, 10),
                Paragraph("No anomalous transactions available for report generation.", styles["KVVal"]),
            ]
            doc.build(elems, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
            return buffer.getvalue()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=15*mm, bottomMargin=14*mm,
        )
        styles = _build_styles()
        elements = []

        records = df_anom.to_dict(orient="records")
        total = len(records)
        limit = total if max_pages is None else min(max_pages, total)

        for i in range(limit):
            row = records[i]
            report_elems = _build_single_report(i + 1, row, stats, styles)
            elements.extend(report_elems)
            # Exactly one page per anomalous transaction
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
