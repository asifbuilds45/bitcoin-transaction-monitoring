"""
Bitcoin Monitoring & Traffic Analysis Dashboard
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st

def sanitize_hdbscan(text: Any) -> str:
    """Safely collapse any DBSCAN, HHDBSCAN, HHHDBSCAN to single exact HDBSCAN."""
    if not text:
        return ""
    return re.sub(r'\bH*DBSCAN\b', 'HDBSCAN', str(text), flags=re.IGNORECASE)

# Ensure src modules are imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import load_dataset, preprocess_data
from src.ingestion.network_ingestion import parse_and_validate_network_data, parse_and_validate_network_csv
from src.ingestion.blockchain_ingestion import parse_and_validate_blockchain_data, parse_and_validate_blockchain_csv
from src.geoip_enrichment import enrich_transactions_with_geoip, get_geoip_status
from src.correlation import EntityCorrelator, DualLayerCorrelator, calculate_record_correlation_confidence
from src.temporal.temporal_analysis import analyze_temporal_patterns
from src.behavioural.behavioural_fingerprinting import extract_behavioural_evidence_scores
from src.graph_analysis import BitcoinGraphEngine
from src.clustering.hdbscan_clustering import BitcoinHDBSCANClustering
from src.clustering.dbscan_clustering import BitcoinDBSCANClustering
from src.feature_engineering import extract_features
from src.anomaly_detection import BitcoinAnomalyDetector
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.explainability.shap_explainer import BitcoinSHAPExplainer
from src.evaluation import evaluate_prototype_predictions
from src.database.db_manager import DatabaseManager
from src.reporting import ReportGenerator
from src.investigation import InvestigationPathReconstructor, EvidenceTimelineBuilder, AlertCaseGrouper
from src.assistant import InvestigationKnowledgeStore, InvestigationChatbot
from src.patterns.pattern_evidence_layer import PatternEvidenceLayer

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Sentinel — NTRO PS 26146",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Dark Theme CSS
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 2px;
    }
    /* Tab Styling for Instant Switching */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 6px 14px;
        transition: all 0.15s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PIPELINE EXECUTION ENGINE
# -----------------------------------------------------------------------------
def run_investigation_pipeline(net_source, chain_source):
    """
    Parse, validate, correlate, and analyze Network and Blockchain datasets independently.
    Returns complete context dictionary.
    """
    # 1. Independent Ingestion & Validation
    net_df, net_warnings = parse_and_validate_network_data(net_source)
    chain_df, chain_warnings = parse_and_validate_blockchain_data(chain_source)

    # 2. Independent Correlation Layer (No blind row concatenation or index matching)
    dual_correlator = DualLayerCorrelator(net_df, chain_df)
    clean_df = dual_correlator.get_correlated_dataframe()

    # 3. Offline GeoIP Enrichment
    enriched_df = enrich_transactions_with_geoip(clean_df)

    # 4. Feature Engineering
    feat_df = extract_features(enriched_df)

    # 5. Isolation Forest Anomaly Detection
    detector = BitcoinAnomalyDetector(contamination=0.10, n_estimators=150, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)

    # 6. HDBSCAN Behavioural Clustering (with DBSCAN Fallback)
    hdbscan = BitcoinHDBSCANClustering(min_cluster_size=5, min_samples=3)
    cluster_labels, noise_mask, hdbscan_evidence = hdbscan.fit_predict(feat_df)
    enriched_df['hdbscan_cluster'] = cluster_labels
    enriched_df['dbscan_cluster'] = cluster_labels
    enriched_df['hdbscan_is_noise'] = noise_mask
    enriched_df['dbscan_is_noise'] = noise_mask
    enriched_df['hdbscan_evidence'] = hdbscan_evidence
    enriched_df['dbscan_evidence'] = hdbscan_evidence

    # 7. Temporal & Behavioural Evidence
    temp_df = analyze_temporal_patterns(enriched_df)
    beh_df = extract_behavioural_evidence_scores(temp_df)

    # 8. NetworkX Graph Analysis & Leiden Community Detection (Louvain Fallback)
    graph_engine = BitcoinGraphEngine(beh_df, use_leiden=True)

    leiden_ev = []
    community_ids = []
    for txid in beh_df['txid']:
        score, _ = graph_engine.community_detector.compute_community_evidence_score(str(txid))
        leiden_ev.append(score)
        comm_info = graph_engine.get_node_community_info(str(txid))
        community_ids.append(comm_info.get("community_id", -1))

    beh_df['leiden_community_evidence'] = leiden_ev
    beh_df['louvain_community_evidence'] = leiden_ev
    beh_df['community_id'] = community_ids

    # 8b. Behavioural Pattern Detection (Peeling-Chain & CoinJoin-Like)
    pattern_layer = PatternEvidenceLayer()
    beh_df = pattern_layer.run(beh_df, db_manager=DatabaseManager())

    # 9. Multi-Layer Evidence Fusion & Risk Scoring
    risk_engine = RiskScoringEngine(ml_weight=0.45, heuristic_weight=0.55)
    scored_df = risk_engine.compute_risk_scores(
        beh_df, norm_scores, raw_scores, is_anomaly,
        dbscan_evidence=hdbscan_evidence, louvain_evidence=np.array(leiden_ev)
    )

    # Normalize canonical risk_level
    scored_df['risk_level_normalized'] = scored_df['risk_level'].astype(str).str.strip().str.upper()

    # Sanitize any legacy DBSCAN strings in all text columns
    for col in ['risk_reasons', 'fused_reasons']:
            scored_df[col] = scored_df[col].astype(str).str.replace(r'\bH*DBSCAN\b', 'HDBSCAN', regex=True, flags=re.IGNORECASE)

    # Fit SHAP Explainer
    shap_explainer = BitcoinSHAPExplainer()
    shap_explainer.fit(feat_df, scored_df['anomaly_score'].values)

    # Evaluation against Ground Truth
    eval_results = evaluate_prototype_predictions(scored_df)

    # Entity Correlator & DB Manager
    correlator = EntityCorrelator(scored_df)
    db_manager = DatabaseManager()

    # Precompute Graph Summary & High Degree Nodes
    graph_summary = graph_engine.get_graph_summary()
    high_degree_entities = graph_engine.get_high_degree_entities(top_n=10)

    # Precompute Complete Ranked Alerts (Unfiltered Base Feed)
    ranked_alerts_full = generate_ranked_alerts(scored_df, "ALL")

    # Summary KPIs
    total_tx = len(scored_df)
    detected_anom = int(scored_df['is_anomaly'].sum())
    normal_tx = total_tx - detected_anom
    critical_alerts = int((scored_df['risk_level_normalized'] == 'CRITICAL').sum())
    high_alerts = int((scored_df['risk_level_normalized'] == 'HIGH').sum())
    mean_conf_pct = int(round(scored_df['correlation_confidence'].mean() * 100)) if 'correlation_confidence' in scored_df.columns else 85

    anom_counts = pd.DataFrame({
        "Classification": ["Normal Traffic", "Detected Anomalies"],
        "Count": [normal_tx, detected_anom]
    })

    risk_counts = scored_df['risk_level_normalized'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).fillna(0).reset_index()
    risk_counts.columns = ['Risk Tier', 'Count']

    country_agg = scored_df.groupby('src_country').agg(
        total_tx=('txid', 'count'),
        high_risk=('risk_level_normalized', lambda x: (x.isin(['HIGH', 'CRITICAL'])).sum())
    ).reset_index().sort_values(by='total_tx', ascending=False).head(10)

    time_df = scored_df.set_index('timestamp').resample('7D').agg(
        total=('txid', 'count'),
        anomalies=('is_anomaly', 'sum')
    ).reset_index()

    top_ips = scored_df[scored_df['risk_level_normalized'].isin(['HIGH', 'CRITICAL'])].groupby('src_ip').agg(
        critical_alerts=('risk_level_normalized', 'count'),
        total_btc=('total_output_amount_btc', 'sum'),
        wallets_used=('source_wallet', 'nunique'),
        country=('src_country', 'first'),
        asn=('src_asn', 'first')
    ).reset_index().sort_values(by='critical_alerts', ascending=False).head(5)

    top_wallets = scored_df.groupby('source_wallet').agg(
        max_risk=('risk_score', 'max'),
        avg_risk=('risk_score', 'mean'),
        total_tx=('txid', 'count'),
        unique_ips=('src_ip', 'nunique'),
        total_btc=('total_output_amount_btc', 'sum')
    ).reset_index().sort_values(by=['max_risk', 'avg_risk'], ascending=False).head(5)

    # 10. Investigation Cases, Path Reconstruction, & Evidence Timeline Engines
    case_grouper = AlertCaseGrouper(scored_df, graph_engine)
    path_reconstructor = InvestigationPathReconstructor(scored_df, graph_engine)
    timeline_builder = EvidenceTimelineBuilder(scored_df)

    kpis_dict = {
        "total_tx": total_tx,
        "normal_tx": normal_tx,
        "detected_anom": detected_anom,
        "critical_alerts": critical_alerts,
        "high_alerts": high_alerts,
        "mean_conf_pct": mean_conf_pct,
        "anom_counts": anom_counts,
        "risk_counts": risk_counts,
        "country_agg": country_agg,
        "time_df": time_df,
        "top_ips": top_ips,
        "top_wallets": top_wallets
    }

    # 11. Offline AI Investigation Assistant Knowledge Store & Engine
    knowledge_store = InvestigationKnowledgeStore({
        "scored_df": scored_df,
        "case_grouper": case_grouper,
        "path_reconstructor": path_reconstructor,
        "timeline_builder": timeline_builder,
        "graph_engine": graph_engine,
        "correlator": correlator,
        "kpis": kpis_dict
    })
    chatbot = InvestigationChatbot(knowledge_store)

    return {
        "scored_df": scored_df,
        "feat_df": feat_df,
        "net_warnings": net_warnings,
        "chain_warnings": chain_warnings,
        "eval_results": eval_results,
        "correlator": correlator,
        "graph_engine": graph_engine,
        "shap_explainer": shap_explainer,
        "db_manager": db_manager,
        "graph_summary": graph_summary,
        "high_degree_entities": high_degree_entities,
        "ranked_alerts_full": ranked_alerts_full,
        "case_grouper": case_grouper,
        "path_reconstructor": path_reconstructor,
        "timeline_builder": timeline_builder,
        "knowledge_store": knowledge_store,
        "chatbot": chatbot,
        "pattern_layer": pattern_layer,
        "kpis": kpis_dict
    }



# Initialize Session State Context and File Trackers if not present
net_default = os.path.join("data", "network_traffic_dataset.csv")
chain_default = os.path.join("data", "blockchain_transactions_dataset.csv")

if "network_files" not in st.session_state:
    if os.path.exists(net_default):
        st.session_state["network_files"] = [{
            "id": "net_default",
            "name": os.path.basename(net_default),
            "source": net_default,
            "layer": "Network",
            "status": "VALID",
            "rows": 10000,
            "cols": 15,
            "is_default": True
        }]
    else:
        st.session_state["network_files"] = []

if "blockchain_files" not in st.session_state:
    if os.path.exists(chain_default):
        st.session_state["blockchain_files"] = [{
            "id": "chain_default",
            "name": os.path.basename(chain_default),
            "source": chain_default,
            "layer": "Blockchain",
            "status": "VALID",
            "rows": 10000,
            "cols": 17,
            "is_default": True
        }]
    else:
        st.session_state["blockchain_files"] = []

if "net_uploader_ver" not in st.session_state:
    st.session_state["net_uploader_ver"] = 0

if "chain_uploader_ver" not in st.session_state:
    st.session_state["chain_uploader_ver"] = 0

if "context" not in st.session_state:
    if os.path.exists(net_default) and os.path.exists(chain_default):
        st.session_state["context"] = run_investigation_pipeline(net_default, chain_default)
        st.session_state["datasets_validated"] = True
        st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(st.session_state["context"]["scored_df"])
        st.session_state["active_dataset_info"] = {
            "net_files": [os.path.basename(net_default)],
            "chain_files": [os.path.basename(chain_default)],
            "net_rows": 10000,
            "chain_rows": 10000,
            "correlated_tx": len(st.session_state["context"]["scored_df"]),
            "anomalies": int(st.session_state["context"]["kpis"]["detected_anom"]),
            "critical": int(st.session_state["context"]["kpis"]["critical_alerts"]),
            "timestamp": "Preloaded Benchmark"
        }
    else:
        raw_df = load_dataset()
        st.session_state["context"] = run_investigation_pipeline(raw_df, raw_df)
        st.session_state["datasets_validated"] = True
        st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(st.session_state["context"]["scored_df"])
        st.session_state["active_dataset_info"] = {
            "net_files": ["Raw Dataset"],
            "chain_files": ["Raw Dataset"],
            "net_rows": len(raw_df),
            "chain_rows": len(raw_df),
            "correlated_tx": len(st.session_state["context"]["scored_df"]),
            "anomalies": int(st.session_state["context"]["kpis"]["detected_anom"]),
            "critical": int(st.session_state["context"]["kpis"]["critical_alerts"]),
            "timestamp": "Preloaded Raw"
        }

ctx = st.session_state["context"]
scored_df = ctx["scored_df"]
feat_df = ctx["feat_df"]
eval_results = ctx["eval_results"]
correlator = ctx["correlator"]
graph_engine = ctx["graph_engine"]
shap_explainer = ctx["shap_explainer"]
db_manager = ctx["db_manager"]
kpis = ctx["kpis"]
case_grouper = ctx.get("case_grouper", AlertCaseGrouper(scored_df, graph_engine))
path_reconstructor = ctx.get("path_reconstructor", InvestigationPathReconstructor(scored_df, graph_engine))
timeline_builder = ctx.get("timeline_builder", EvidenceTimelineBuilder(scored_df))

knowledge_store = ctx.get("knowledge_store")
if knowledge_store is None:
    knowledge_store = InvestigationKnowledgeStore(ctx)
    ctx["knowledge_store"] = knowledge_store

chatbot = ctx.get("chatbot")
if chatbot is None:
    chatbot = InvestigationChatbot(knowledge_store)
    ctx["chatbot"] = chatbot

if "active_dataset_info" not in st.session_state:
    st.session_state["active_dataset_info"] = {
        "net_files": [f["name"] for f in st.session_state.get("network_files", [])] or [os.path.basename(net_default)],
        "chain_files": [f["name"] for f in st.session_state.get("blockchain_files", [])] or [os.path.basename(chain_default)],
        "net_rows": len(scored_df),
        "chain_rows": len(scored_df),
        "correlated_tx": len(scored_df),
        "anomalies": int(kpis.get("detected_anom", 0)),
        "critical": int(kpis.get("critical_alerts", 0)),
        "timestamp": "Active Session"
    }

# Initialize report ID map and pre-compute feature statistics for report generation
if "report_id_map" not in st.session_state:
    st.session_state["report_id_map"] = {}
if "feature_stats" not in st.session_state:
    st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(st.session_state["context"]["scored_df"])


def render_active_dataset_banner():
    """Render a clean, high-visibility telemetry status banner reflecting currently active correlated datasets."""
    info = st.session_state.get("active_dataset_info", {})
    if info:
        net_str = ", ".join(info.get("net_files", ["Benchmark"]))
        chain_str = ", ".join(info.get("chain_files", ["Benchmark"]))
        corr_count = info.get("correlated_tx", len(scored_df))
        anom_count = info.get("anomalies", int(kpis.get('detected_anom', 0)))
        crit_count = info.get("critical", int(kpis.get('critical_alerts', 0)))
        ts = info.get("timestamp", "Startup")
        st.markdown(f"""
        <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid #0284c7; border-left: 5px solid #38bdf8; border-radius: 8px; padding: 10px 16px; margin: 6px 0 14px 0; display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem;">
            <div>
                <span style="color: #38bdf8; font-weight: 700;">🟢 ACTIVE TELEMETRY SOURCE:</span>
                <span style="color: #f8fafc; margin-left: 8px;">Network: <b>{net_str}</b> &nbsp;|&nbsp; Blockchain: <b>{chain_str}</b></span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span style="background: #0369a1; color: #ffffff; padding: 3px 9px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">{corr_count:,} Correlated TX</span>
                <span style="background: #b45309; color: #ffffff; padding: 3px 9px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">{anom_count:,} Anomalies</span>
                <span style="background: #b91c1c; color: #ffffff; padding: 3px 9px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">{crit_count:,} Critical</span>
                <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 4px;">⏱️ {ts}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
    <span style="font-size: 2.2rem;">🛡️</span>
    <div>
        <h2 style="margin: 0; font-size: 1.3rem; color: #f8fafc;">Bitcoin Sentinel</h2>
        <span style="font-size: 0.75rem; color: #94a3b8;">NTRO PS 26146 Prototype v2.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")
sb_info = st.session_state.get("active_dataset_info", {})
if sb_info:
    sb_net = ", ".join(sb_info.get("net_files", ["Benchmark"]))
    st.sidebar.caption(f"📁 **Telemetry:** `{sb_net}`")
st.sidebar.metric("Monitored Transactions", f"{kpis['total_tx']:,}")
st.sidebar.metric("AI Anomalies Detected", f"{kpis['detected_anom']:,}")
st.sidebar.metric("Critical Alerts", f"{kpis['critical_alerts']:,}")
st.sidebar.metric("Mean Correlation Conf.", f"{kpis['mean_conf_pct']}%")

st.sidebar.markdown("---")
st.sidebar.subheader("System Mode")
st.sidebar.success("🟢 100% Offline Active")
if db_manager.is_connected:
    st.sidebar.info(f"Database: {db_manager.db_type}")
else:
    st.sidebar.warning("Database: Disconnected")

geoip_info = get_geoip_status()
st.sidebar.markdown(f"""
**GeoIP Status:**
- Country DB: `{geoip_info['country_status']}`
- ASN DB: `{geoip_info['asn_status']}`
- Mode: `{geoip_info['mode']}`
""")

st.sidebar.markdown("---")
st.sidebar.subheader("AI Assistant")
st.sidebar.info(f"Engine: {chatbot.llm_manager.runtime_type}")
act_focus = chatbot.active_context.get("active_txid") or chatbot.active_context.get("active_case_id") or chatbot.active_context.get("active_wallet")
if act_focus:
    st.sidebar.caption(f"Active Focus: `{act_focus}`")
else:
    st.sidebar.caption("Active Focus: Global Feed")

st.sidebar.caption("Defensive Cybersecurity Prototype")


# -----------------------------------------------------------------------------
# INSTANT CLIENT-SIDE TAB NAVIGATION (0 MS DELAY)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📥 0. Dataset Ingestion",
    "📊 1. Overview & KPIs",
    "🔍 2. Transaction Investigation",
    "🕸️ 3. IP-TXID-Wallet Correlation",
    "📈 4. Graph Analytics",
    "🤖 5. AI Anomaly Analysis",
    "🚨 6. Ranked Alerts",
    "🎯 7. Prototype Evaluation",
    "💬 8. AI Investigation Assistant",
    "🔗 9. Behavioural Patterns"
])


# =============================================================================
# TAB 0: DATASET INGESTION & MULTI-DATASET DUAL-LAYER CORRELATION
# =============================================================================
with tabs[0]:
    st.title("📥 Two-Layer Dataset Ingestion & Multi-Dataset Correlation")
    st.caption("Upload multiple Network-Layer and Blockchain-Layer datasets independently (CSV, JSON, XML). All uploaded files accumulate without overwriting previous datasets, merge cleanly within their layers, and correlate across layers for deep forensic investigation.")

    # Top Alert & Correlation Status Banner
    if "correlation_status_msg" in st.session_state:
        c_msg = st.session_state["correlation_status_msg"]
        st.success(f"{c_msg['text']} *(Applied at {c_msg['time']})*")

    # Top Control Bar: Reset & Load Defaults
    top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
    with top_col1:
        st.info("ℹ️ **Multi-Dataset Ingestion:** Select or drag multiple files into either upload zone. Newly uploaded files accumulate cleanly. Use sample files `demo_network_10000.csv` and `demo_blockchain_10000.csv` located in the project root for testing.")
    with top_col2:
        if st.button("📂 Load Default 10K Benchmark", width='stretch', help="Load and run preloaded 10,000-record synthetic benchmark datasets"):
            net_default = os.path.join("data", "network_traffic_dataset.csv")
            chain_default = os.path.join("data", "blockchain_transactions_dataset.csv")
            if os.path.exists(net_default) and os.path.exists(chain_default):
                st.session_state["network_files"] = [{
                    "id": "net_default_1",
                    "name": os.path.basename(net_default),
                    "source": net_default,
                    "layer": "Network",
                    "status": "VALID",
                    "rows": 10000,
                    "cols": 15,
                    "is_default": True
                }]
                st.session_state["blockchain_files"] = [{
                    "id": "chain_default_1",
                    "name": os.path.basename(chain_default),
                    "source": chain_default,
                    "layer": "Blockchain",
                    "status": "VALID",
                    "rows": 10000,
                    "cols": 17,
                    "is_default": True
                }]
                st.session_state["datasets_validated"] = True
                st.session_state["combined_net_df"] = None
                st.session_state["combined_chain_df"] = None
                st.session_state["validation_report"] = {
                    "net_rows": 10000,
                    "chain_rows": 10000,
                    "net_warn": [],
                    "chain_warn": []
                }
                with st.spinner("Reloading default benchmark dataset and running pipeline..."):
                    new_ctx = run_investigation_pipeline(net_default, chain_default)
                    st.session_state["context"] = new_ctx
                    st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(new_ctx["scored_df"])
                    st.session_state["active_dataset_info"] = {
                        "net_files": [os.path.basename(net_default)],
                        "chain_files": [os.path.basename(chain_default)],
                        "net_rows": 10000,
                        "chain_rows": 10000,
                        "correlated_tx": len(new_ctx["scored_df"]),
                        "anomalies": int(new_ctx["kpis"]["detected_anom"]),
                        "critical": int(new_ctx["kpis"]["critical_alerts"]),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state["correlation_status_msg"] = {
                        "type": "success",
                        "text": "✅ Default 10K benchmark dataset active! Correlated 10,000 transactions across all tabs.",
                        "time": pd.Timestamp.now().strftime("%H:%M:%S")
                    }
                    for k in ["tab3_tx", "tab3_w", "tab3_ip", "tab2_search", "tab2_risk", "tab6_risk", "tab2_selected_txid"]:
                        st.session_state.pop(k, None)
                    st.rerun()
    with top_col3:
        if st.button("🗑️ Clear All Datasets", width='stretch', help="Remove all loaded datasets to upload fresh files"):
            st.session_state["network_files"] = []
            st.session_state["blockchain_files"] = []
            st.session_state["net_uploader_ver"] = st.session_state.get("net_uploader_ver", 0) + 1
            st.session_state["chain_uploader_ver"] = st.session_state.get("chain_uploader_ver", 0) + 1
            st.session_state["datasets_validated"] = False
            st.session_state["combined_net_df"] = None
            st.session_state["combined_chain_df"] = None
            st.session_state.pop("validation_report", None)
            st.session_state.pop("correlation_status_msg", None)
            st.rerun()

    st.markdown("---")

    col_net_ui, col_chain_ui = st.columns(2)

    # -------------------------------------------------------------------------
    # 1. NETWORK-LAYER DATA CONTROL
    # -------------------------------------------------------------------------
    with col_net_ui:
        st.subheader("1. NETWORK-LAYER DATASETS")
        st.caption("Bitcoin P2P network telemetry, IP broadcast addresses, ports, and traffic patterns.")

        uploaded_net_files = st.file_uploader(
            "Upload Network Files (CSV, JSON, XML) — Multiple Supported",
            type=["csv", "json", "xml"],
            accept_multiple_files=True,
            key=f"uploader_net_multi_{st.session_state.get('net_uploader_ver', 0)}",
            help="Select one or more Network telemetry files. Newly uploaded files accumulate and will not replace previous ones."
        )

        if uploaded_net_files:
            # If user uploaded custom files, replace default benchmark placeholder
            net_files_current = st.session_state.get("network_files", [])
            if len(net_files_current) == 1 and net_files_current[0].get("is_default"):
                st.session_state["network_files"] = []

            for uf in uploaded_net_files:
                existing_names = [f["name"] for f in st.session_state.get("network_files", [])]
                if uf.name not in existing_names:
                    df, warns = parse_and_validate_network_data(uf)
                    if hasattr(uf, 'seek'):
                        uf.seek(0)
                    status = "VALID" if (not df.empty and not any(w.get("level") == "ERROR" for w in warns)) else "INVALID"
                    st.session_state["network_files"].append({
                        "id": f"net_{uf.name}_{len(st.session_state['network_files'])}",
                        "name": uf.name,
                        "source": uf,
                        "df": df,
                        "layer": "Network",
                        "status": status,
                        "rows": len(df),
                        "cols": len(df.columns),
                        "warnings": warns,
                        "is_default": False
                    })
                    st.session_state["datasets_validated"] = False

        # Active Network Files List
        st.markdown("##### 📁 Active Network Files")
        net_files = st.session_state.get("network_files", [])
        if not net_files:
            st.caption("No network files currently loaded. Upload custom files above or click 'Load Default 10K Benchmark'.")
        else:
            net_to_remove = None
            for idx, n_file in enumerate(net_files):
                c_info, c_del = st.columns([5, 1])
                with c_info:
                    status_icon = "🟢" if n_file.get("status") == "VALID" else "🔴"
                    row_txt = f"— **{n_file['rows']:,}** rows, {n_file['cols']} cols" if n_file.get("rows", 0) > 0 else "(pending validation)"
                    st.markdown(f"{status_icon} `{n_file['name']}` {row_txt}")
                    if n_file.get("warnings"):
                        err_warns = [w for w in n_file["warnings"] if w.get("level") == "ERROR"]
                        if err_warns:
                            st.caption(f"⚠️ {err_warns[0]['message']}")
                with c_del:
                    if st.button("🗑️", key=f"del_net_{n_file['id']}_{idx}", help=f"Remove {n_file['name']}"):
                        net_to_remove = idx

            if net_to_remove is not None:
                st.session_state["network_files"].pop(net_to_remove)
                st.session_state["datasets_validated"] = False
                st.rerun()

            valid_net_count = sum(1 for f in net_files if f.get("status") == "VALID")
            total_net_rows = sum(f.get("rows", 0) for f in net_files if f.get("status") == "VALID")
            st.caption(f"Total: **{valid_net_count} valid file(s)** | **{total_net_rows:,} total network rows**")

    # -------------------------------------------------------------------------
    # 2. BLOCKCHAIN-LAYER DATA CONTROL
    # -------------------------------------------------------------------------
    with col_chain_ui:
        st.subheader("2. BLOCKCHAIN-LAYER DATASETS")
        st.caption("Bitcoin transaction ledgers, UTXOs, miner fees, and wallet addresses.")

        uploaded_chain_files = st.file_uploader(
            "Upload Blockchain Files (CSV, JSON, XML) — Multiple Supported",
            type=["csv", "json", "xml"],
            accept_multiple_files=True,
            key=f"uploader_chain_multi_{st.session_state.get('chain_uploader_ver', 0)}",
            help="Select one or more Blockchain transaction files. Newly uploaded files accumulate and will not replace previous ones."
        )

        if uploaded_chain_files:
            # If user uploaded custom files, replace default benchmark placeholder
            chain_files_current = st.session_state.get("blockchain_files", [])
            if len(chain_files_current) == 1 and chain_files_current[0].get("is_default"):
                st.session_state["blockchain_files"] = []

            for uf in uploaded_chain_files:
                existing_names = [f["name"] for f in st.session_state.get("blockchain_files", [])]
                if uf.name not in existing_names:
                    df, warns = parse_and_validate_blockchain_data(uf)
                    if hasattr(uf, 'seek'):
                        uf.seek(0)
                    status = "VALID" if (not df.empty and not any(w.get("level") == "ERROR" for w in warns)) else "INVALID"
                    st.session_state["blockchain_files"].append({
                        "id": f"chain_{uf.name}_{len(st.session_state['blockchain_files'])}",
                        "name": uf.name,
                        "source": uf,
                        "df": df,
                        "layer": "Blockchain",
                        "status": status,
                        "rows": len(df),
                        "cols": len(df.columns),
                        "warnings": warns,
                        "is_default": False
                    })
                    st.session_state["datasets_validated"] = False

        # Active Blockchain Files List
        st.markdown("##### 📁 Active Blockchain Files")
        chain_files = st.session_state.get("blockchain_files", [])
        if not chain_files:
            st.caption("No blockchain files currently loaded. Upload custom files above or click 'Load Default 10K Benchmark'.")
        else:
            chain_to_remove = None
            for idx, c_file in enumerate(chain_files):
                c_info, c_del = st.columns([5, 1])
                with c_info:
                    status_icon = "🟢" if c_file.get("status") == "VALID" else "🔴"
                    row_txt = f"— **{c_file['rows']:,}** rows, {c_file['cols']} cols" if c_file.get("rows", 0) > 0 else "(pending validation)"
                    st.markdown(f"{status_icon} `{c_file['name']}` {row_txt}")
                    if c_file.get("warnings"):
                        err_warns = [w for w in c_file["warnings"] if w.get("level") == "ERROR"]
                        if err_warns:
                            st.caption(f"⚠️ {err_warns[0]['message']}")
                with c_del:
                    if st.button("🗑️", key=f"del_chain_{c_file['id']}_{idx}", help=f"Remove {c_file['name']}"):
                        chain_to_remove = idx

            if chain_to_remove is not None:
                st.session_state["blockchain_files"].pop(chain_to_remove)
                st.session_state["datasets_validated"] = False
                st.rerun()

            valid_chain_count = sum(1 for f in chain_files if f.get("status") == "VALID")
            total_chain_rows = sum(f.get("rows", 0) for f in chain_files if f.get("status") == "VALID")
            st.caption(f"Total: **{valid_chain_count} valid file(s)** | **{total_chain_rows:,} total blockchain rows**")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # MULTI-DATASET CORRELATION PREVIEW & MATCH METRICS
    # -------------------------------------------------------------------------
    net_files = st.session_state.get("network_files", [])
    chain_files = st.session_state.get("blockchain_files", [])
    valid_net = [f for f in net_files if f.get("status") == "VALID"]
    valid_chain = [f for f in chain_files if f.get("status") == "VALID"]

    # Calculate TXID overlap across layers
    net_txids = set()
    for f in valid_net:
        if "df" in f and isinstance(f["df"], pd.DataFrame) and "txid" in f["df"].columns:
            net_txids.update(f["df"]["txid"].dropna().astype(str).unique())
        elif isinstance(f.get("source"), str) and os.path.exists(f["source"]):
            try:
                t_df = pd.read_csv(f["source"], usecols=["txid"])
                net_txids.update(t_df["txid"].dropna().astype(str).unique())
            except Exception:
                pass

    chain_txids = set()
    for f in valid_chain:
        if "df" in f and isinstance(f["df"], pd.DataFrame) and "txid" in f["df"].columns:
            chain_txids.update(f["df"]["txid"].dropna().astype(str).unique())
        elif isinstance(f.get("source"), str) and os.path.exists(f["source"]):
            try:
                t_df = pd.read_csv(f["source"], usecols=["txid"])
                chain_txids.update(t_df["txid"].dropna().astype(str).unique())
            except Exception:
                pass

    common_txids = len(net_txids & chain_txids)
    match_pct = (common_txids / max(len(net_txids), 1)) * 100 if net_txids else 0.0

    # Check if currently staged files differ from actively correlated files
    active_info = st.session_state.get("active_dataset_info", {})
    staged_net_names = sorted([f["name"] for f in valid_net])
    staged_chain_names = sorted([f["name"] for f in valid_chain])
    act_net_names = sorted(active_info.get("net_files", []))
    act_chain_names = sorted(active_info.get("chain_files", []))
    is_synchronized = (staged_net_names == act_net_names and staged_chain_names == act_chain_names and bool(active_info))

    if not valid_net or not valid_chain or common_txids == 0:
        readiness_color = "#ef4444"
        readiness_text = "NEEDS FILES"
        sub_text = "Upload Network & Chain"
    elif not is_synchronized:
        readiness_color = "#f59e0b"
        readiness_text = "PENDING RUN"
        sub_text = "Click Correlate Below"
    else:
        readiness_color = "#22c55e"
        readiness_text = "LIVE / SYNCED"
        sub_text = f"{common_txids:,} Correlated TX"

    st.markdown("### 🔗 Dual-Layer Correlation Preview Across Uploaded Datasets")
    prev_col1, prev_col2, prev_col3, prev_col4 = st.columns(4)
    with prev_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Network Datasets</div>
            <div class="metric-value" style="color: #38bdf8;">{len(valid_net)} files</div>
            <span style="font-size: 0.8rem; color: #94a3b8;">{len(net_txids):,} unique TXIDs</span>
        </div>
        """, unsafe_allow_html=True)
    with prev_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Blockchain Datasets</div>
            <div class="metric-value" style="color: #38bdf8;">{len(valid_chain)} files</div>
            <span style="font-size: 0.8rem; color: #94a3b8;">{len(chain_txids):,} unique TXIDs</span>
        </div>
        """, unsafe_allow_html=True)
    with prev_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Correlatable Transactions</div>
            <div class="metric-value" style="color: #22c55e;">{common_txids:,}</div>
            <span style="font-size: 0.8rem; color: #94a3b8;">{match_pct:.1f}% cross-layer match</span>
        </div>
        """, unsafe_allow_html=True)
    with prev_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Correlation Status</div>
            <div class="metric-value" style="color: {readiness_color};">{readiness_text}</div>
            <span style="font-size: 0.8rem; color: #94a3b8;">{sub_text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if not is_synchronized and valid_net and valid_chain and common_txids > 0:
        st.warning("⚡ **Pending Correlation:** You have staged new dataset files that are not yet active in the dashboard tabs. Click **'🚀 CORRELATE & RUN INVESTIGATION'** below to execute cross-layer correlation and refresh all 10 tabs with your new data!")

    # -------------------------------------------------------------------------
    # ACTION BUTTONS: VALIDATE DATASETS & CORRELATE AND RUN INVESTIGATION
    # -------------------------------------------------------------------------
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])

    with btn_col1:
        if st.button("🔍 VALIDATE ALL DATASETS", width='stretch', type="secondary"):
            if not valid_net or not valid_chain:
                st.error("Please ensure at least one valid Network dataset and one valid Blockchain dataset are loaded.")
            else:
                with st.spinner("Validating Network and Blockchain schemas independently across all uploaded datasets..."):
                    all_net_dfs = []
                    all_chain_dfs = []
                    all_net_warns = []
                    all_chain_warns = []

                    for nf in valid_net:
                        if "df" in nf and isinstance(nf["df"], pd.DataFrame):
                            df = nf["df"].copy()
                            warns = nf.get("warnings", [])
                        else:
                            df, warns = parse_and_validate_network_data(nf["source"])
                        all_net_dfs.append(df.assign(source_file=nf["name"]))
                        all_net_warns.extend(warns)

                    for cf in valid_chain:
                        if "df" in cf and isinstance(cf["df"], pd.DataFrame):
                            df = cf["df"].copy()
                            warns = cf.get("warnings", [])
                        else:
                            df, warns = parse_and_validate_blockchain_data(cf["source"])
                        all_chain_dfs.append(df.assign(source_file=cf["name"]))
                        all_chain_warns.extend(warns)

                    combined_net = pd.concat(all_net_dfs, ignore_index=True)
                    dedup_net_cols = [c for c in ['txid', 'src_ip', 'dst_ip'] if c in combined_net.columns]
                    combined_net = combined_net.drop_duplicates(subset=dedup_net_cols) if dedup_net_cols else combined_net.drop_duplicates()

                    combined_chain = pd.concat(all_chain_dfs, ignore_index=True)
                    dedup_chain_cols = [c for c in ['txid'] if c in combined_chain.columns]
                    combined_chain = combined_chain.drop_duplicates(subset=dedup_chain_cols) if dedup_chain_cols else combined_chain.drop_duplicates()

                    st.session_state["combined_net_df"] = combined_net
                    st.session_state["combined_chain_df"] = combined_chain
                    st.session_state["datasets_validated"] = True
                    st.session_state["validation_report"] = {
                        "net_files_count": len(all_net_dfs),
                        "chain_files_count": len(all_chain_dfs),
                        "net_rows": len(combined_net),
                        "chain_rows": len(combined_chain),
                        "net_warn": all_net_warns,
                        "chain_warn": all_chain_warns
                    }
                    st.success(f"✓ Validated {len(all_net_dfs)} network file(s) ({len(combined_net):,} rows) and {len(all_chain_dfs)} blockchain file(s) ({len(combined_chain):,} rows) successfully!")
                    st.rerun()

    with btn_col2:
        can_run = bool(valid_net and valid_chain and common_txids > 0)
        btn_label = "🚀 CORRELATE & RUN INVESTIGATION"
        if st.button(btn_label, width='stretch', type="primary", disabled=not can_run):
            with st.spinner("Merging multiple datasets, correlating network telemetry with blockchain ledger, and running AI models..."):
                all_net_dfs = []
                for nf in valid_net:
                    if "df" in nf and isinstance(nf["df"], pd.DataFrame):
                        df = nf["df"].copy()
                    else:
                        df, _ = parse_and_validate_network_data(nf["source"])
                    if hasattr(nf.get("source"), 'seek'):
                        nf["source"].seek(0)
                    all_net_dfs.append(df.assign(source_file=nf["name"]))

                all_chain_dfs = []
                for cf in valid_chain:
                    if "df" in cf and isinstance(cf["df"], pd.DataFrame):
                        df = cf["df"].copy()
                    else:
                        df, _ = parse_and_validate_blockchain_data(cf["source"])
                    if hasattr(cf.get("source"), 'seek'):
                        cf["source"].seek(0)
                    all_chain_dfs.append(df.assign(source_file=cf["name"]))

                combined_net = pd.concat(all_net_dfs, ignore_index=True)
                dedup_net_cols = [c for c in ['txid', 'src_ip', 'dst_ip'] if c in combined_net.columns]
                combined_net = combined_net.drop_duplicates(subset=dedup_net_cols) if dedup_net_cols else combined_net.drop_duplicates()

                combined_chain = pd.concat(all_chain_dfs, ignore_index=True)
                dedup_chain_cols = [c for c in ['txid'] if c in combined_chain.columns]
                combined_chain = combined_chain.drop_duplicates(subset=dedup_chain_cols) if dedup_chain_cols else combined_chain.drop_duplicates()

                # Execute investigation pipeline across all correlated datasets
                new_ctx = run_investigation_pipeline(combined_net, combined_chain)
                st.session_state["context"] = new_ctx
                st.session_state["combined_net_df"] = combined_net
                st.session_state["combined_chain_df"] = combined_chain
                st.session_state["datasets_validated"] = True
                st.session_state["cached_pdf_bytes"] = None
                st.session_state["assistant_chat_history"] = []
                st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(new_ctx["scored_df"])

                # Store active dataset tracking information
                active_net_names = [f["name"] for f in valid_net]
                active_chain_names = [f["name"] for f in valid_chain]
                st.session_state["active_dataset_info"] = {
                    "net_files": active_net_names,
                    "chain_files": active_chain_names,
                    "net_rows": len(combined_net),
                    "chain_rows": len(combined_chain),
                    "correlated_tx": len(new_ctx["scored_df"]),
                    "anomalies": int(new_ctx["kpis"]["detected_anom"]),
                    "critical": int(new_ctx["kpis"]["critical_alerts"]),
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                st.session_state["correlation_status_msg"] = {
                    "type": "success",
                    "text": f"✅ Multi-dataset correlation complete! Successfully correlated {len(new_ctx['scored_df']):,} records across Network ({', '.join(active_net_names)}) and Blockchain ({', '.join(active_chain_names)}). All 10 dashboard tabs are actively updated!",
                    "time": pd.Timestamp.now().strftime("%H:%M:%S")
                }

                # Clear stale widget states so new dataset entities take effect immediately
                for k in ["tab3_tx", "tab3_w", "tab3_ip", "tab2_search", "tab2_risk", "tab6_risk", "tab2_selected_txid"]:
                    st.session_state.pop(k, None)

                st.rerun()

    if "validation_report" in st.session_state:
        vr = st.session_state["validation_report"]
        st.info(f"**Validation Report:** Network ({vr['net_rows']:,} records across {vr.get('net_files_count', 1)} file(s), {len(vr['net_warn'])} warnings) | Blockchain ({vr['chain_rows']:,} records across {vr.get('chain_files_count', 1)} file(s), {len(vr['chain_warn'])} warnings)")


# =============================================================================
# TAB 1: OVERVIEW & EXECUTIVE KPIS
# =============================================================================
with tabs[1]:
    st.title("🛡️ Cyber Monitoring & Anomaly Overview")
    st.caption("Real-time forensic monitoring correlating Bitcoin P2P network telemetry with on-chain UTXO transfers.")
    render_active_dataset_banner()

    # KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Total Records</div><div class="metric-value">{kpis['total_tx']:,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Normal Traffic</div><div class="metric-value" style="color: #22c55e;">{kpis['normal_tx']:,}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">AI Anomalies</div><div class="metric-value" style="color: #f59e0b;">{kpis['detected_anom']:,}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">High Risk</div><div class="metric-value" style="color: #f97316;">{kpis['high_alerts']:,}</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Critical Risk</div><div class="metric-value" style="color: #ef4444;">{kpis['critical_alerts']:,}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Row 1 Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("AI Anomaly Classification Breakdown")
        fig_pie = px.pie(
            kpis['anom_counts'],
            values="Count",
            names="Classification",
            color="Classification",
            color_discrete_map={"Normal Traffic": "#10b981", "Detected Anomalies": "#f43f5e"},
            hole=0.45
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark", height=320)
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        st.subheader("Investigative Risk Severity Tiers (0–100)")
        fig_bar = px.bar(
            kpis['risk_counts'],
            x='Risk Tier',
            y='Count',
            color='Risk Tier',
            color_discrete_map={'LOW': '#22c55e', 'MEDIUM': '#eab308', 'HIGH': '#f97316', 'CRITICAL': '#ef4444'},
            text_auto=True
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark", height=320)
        st.plotly_chart(fig_bar, width='stretch')

    # Row 2 Charts
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Source Country Activity & Risk Profile")
        fig_country = px.bar(
            kpis['country_agg'],
            x='src_country',
            y=['total_tx', 'high_risk'],
            barmode='group',
            labels={'value': 'Transactions', 'src_country': 'Country Code', 'variable': 'Category'},
            color_discrete_map={'total_tx': '#38bdf8', 'high_risk': '#ef4444'}
        )
        fig_country.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark", height=320)
        st.plotly_chart(fig_country, width='stretch')

    with c4:
        st.subheader("Temporal Transaction Velocity & Anomalies")
        fig_time = px.line(
            kpis['time_df'],
            x='timestamp',
            y=['total', 'anomalies'],
            labels={'value': 'Weekly Transactions', 'timestamp': 'Date', 'variable': 'Metric'},
            color_discrete_map={'total': '#38bdf8', 'anomalies': '#f43f5e'}
        )
        fig_time.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark", height=320)
        st.plotly_chart(fig_time, width='stretch')

    # Top Suspicious Entities Table
    st.subheader("Top Suspicious Entities Requiring Review")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.caption("Top Suspicious Source IPs (by High/Critical Risk Count)")
        st.dataframe(kpis['top_ips'], width='stretch', hide_index=True)

    with col_t2:
        st.caption("Top Suspicious Source Wallets (by Risk Score)")
        st.dataframe(kpis['top_wallets'], width='stretch', hide_index=True)


# =============================================================================
# TAB 2: TRANSACTION INVESTIGATION
# =============================================================================
with tabs[2]:
    st.title("🔍 Forensic Transaction Deep-Dive")
    st.caption("Inspect network packet telemetry, on-chain UTXO attributes, Multi-Layer Fused Evidence, and SHAP model explainability.")
    render_active_dataset_banner()

    # Search & Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        risk_filter = st.selectbox("Filter Risk Tier", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"], key="tab2_risk")
    with f_col2:
        search_txid = st.text_input("Quick TXID Search", placeholder="e.g. TX10000137", key="tab2_search").strip()
    with f_col3:
        filtered_df = scored_df.copy()
        if risk_filter != "ALL":
            filtered_df = filtered_df[filtered_df['risk_level_normalized'] == risk_filter.upper()]
        if search_txid:
            filtered_df = filtered_df[filtered_df['txid'].str.contains(search_txid, case=False, na=False)]
        
        txid_options = filtered_df['txid'].head(50).tolist() if not filtered_df.empty else []
        selected_txid = st.selectbox("Select Transaction to Inspect", txid_options if txid_options else ["None Found"])

    if filtered_df.empty or selected_txid == "None Found":
        st.warning("No transactions match the specified search parameters.")
    else:
        tx_row = scored_df[scored_df['txid'] == selected_txid].iloc[0]

        risk_color_map = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
        r_level = tx_row['risk_level']
        badge_color = risk_color_map.get(r_level.upper(), "#3b82f6")
        conf_val = tx_row.get('correlation_confidence_pct', '85%')
        comm_id = tx_row.get('community_id', -1)

        st.markdown(f"""
        <div style="background: #1e293b; border-left: 6px solid {badge_color}; padding: 14px 18px; border-radius: 8px; margin: 12px 0 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">TXID: {tx_row['txid']}</span>
                    <span style="background-color: {badge_color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; margin-left: 10px; font-size: 0.85rem;">
                        {r_level} RISK
                    </span>
                    <span style="background-color: #3b82f6; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; margin-left: 6px; font-size: 0.85rem;">
                        Corr. Confidence: {conf_val}
                    </span>
                    <span style="background-color: #8b5cf6; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; margin-left: 6px; font-size: 0.85rem;">
                        Community #{comm_id}
                    </span>
                </div>
                <div>
                    <span style="font-size: 1rem; color: #94a3b8; margin-right: 15px;">Unified Risk Score: <strong style="color: {badge_color}; font-size: 1.2rem;">{tx_row['risk_score']}/100</strong></span>
                    <span style="font-size: 1rem; color: #94a3b8;">AI Score: <strong style="color: #38bdf8;">{tx_row['anomaly_score']:.3f}</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("💡 Automated Forensic Explanation & Evidence")
        reasons_list = [sanitize_hdbscan(r.strip()) for r in str(tx_row['risk_reasons']).split(';') if r.strip()]
        for r in reasons_list:
            st.markdown(f"- **{r}**")

        st.markdown("---")

        # SHAP Model Feature Explanations
        st.subheader("🤖 SHAP Model Feature Contribution Analysis")
        feat_sub = feat_df.loc[tx_row.name] if tx_row.name in feat_df.index else feat_df.iloc[0]
        shap_explanations = shap_explainer.explain_instance(feat_sub, top_n=6)
        
        if shap_explanations:
            shap_data = []
            for item in shap_explanations:
                shap_data.append({
                    "Feature Name": item["feature"],
                    "Observed Value": f"{item['feature_value']:.4f}",
                    "SHAP Impact": item["shap_value"],
                    "Direction": item["direction"]
                })
            st.dataframe(pd.DataFrame(shap_data), width='stretch', hide_index=True)

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🌐 Network Telemetry & GeoIP")
            geo_status_badge = "🟢 Resolved (MaxMind MMDB)" if tx_row.get('geoip_lookup_status') == "geoip_resolved" else "🟡 Synthetic Fallback (Private IP)"
            st.info(f"**GeoIP Status:** {geo_status_badge}")
            
            net_data = {
                "Source IP": tx_row['src_ip'],
                "Destination IP": tx_row['dst_ip'],
                "Source / Dest Port": f"{tx_row.get('src_port', 8333)} / {tx_row.get('dst_port', 8333)}",
                "Network Protocol": tx_row.get('protocol', 'TCP'),
                "Network Timestamp": str(tx_row.get('network_timestamp', tx_row['timestamp'])),
                "Connection Duration": f"{tx_row.get('connection_duration_sec', 1)}s",
                "Packets / Bytes": f"{int(tx_row.get('packet_count', 0)):,} pkts ({int(tx_row.get('bytes_transferred', 0)/1024):,} KB)",
                "Source GeoIP Country": tx_row.get('geoip_src_country') if pd.notna(tx_row.get('geoip_src_country')) and str(tx_row.get('geoip_src_country')).strip().lower() not in ('unknown', 'none', 'nan', '') else tx_row.get('src_country', 'Unknown'),
                "Destination GeoIP Country": tx_row.get('geoip_dst_country') if pd.notna(tx_row.get('geoip_dst_country')) and str(tx_row.get('geoip_dst_country')).strip().lower() not in ('unknown', 'none', 'nan', '') else tx_row.get('dst_country', 'Unknown'),
                "Source ASN": tx_row.get('geoip_src_asn') if pd.notna(tx_row.get('geoip_src_asn')) and str(tx_row.get('geoip_src_asn')).strip().lower() not in ('unknown', 'none', 'nan', '') else tx_row.get('src_asn', 'Unknown')
            }
            st.table(pd.DataFrame(list(net_data.items()), columns=["Attribute", "Value"]))

        with col_right:
            st.subheader("⛓️ Blockchain Layer & UTXO")
            chain_data = {
                "Block Height": str(tx_row.get('block_height', 700000)),
                "Block Timestamp": str(tx_row['timestamp']),
                "Source Wallet Entity": tx_row['source_wallet'],
                "Destination Wallet Entity": tx_row['destination_wallet'],
                "Transferred Volume": f"{tx_row['total_output_amount_btc']:.6f} BTC",
                "Miner Fee": f"{tx_row['fee_btc']:.6f} BTC",
                "Inputs / Outputs": f"{tx_row.get('num_inputs', 1)} in / {tx_row.get('num_outputs', 1)} out",
                "Script Type": tx_row.get('script_type', 'P2PKH'),
                "Wallet Graph Degree": str(tx_row.get('wallet_degree', 1)),
                "24h Velocity": f"{tx_row.get('transaction_frequency_24h', 1)} tx/24h",
                "Inter-tx Time Gap": f"{float(tx_row.get('avg_time_gap_min', 60)):.2f} min"
            }
            st.table(pd.DataFrame(list(chain_data.items()), columns=["Attribute", "Value"]))

        st.markdown("---")

        # ---------------------------------------------------------------------
        # FEATURE 1: INVESTIGATION PATH RECONSTRUCTION
        # ---------------------------------------------------------------------
        st.subheader("🧭 Investigation Path Reconstruction")
        st.caption(
            "Multi-hop forensic trace connecting network observation vantage points, on-chain transactions, "
            "and wallet fund flows. Strict non-attribution semantics applied."
        )

        p_col1, p_col2 = st.columns([1, 3])
        with p_col1:
            path_depth = st.slider("Traversal Depth (Hops)", min_value=1, max_value=5, value=3, key="tab2_path_depth")
            max_paths_sel = st.slider("Max Paths to Trace", min_value=1, max_value=5, value=3, key="tab2_max_paths")

        reconstructed_paths = path_reconstructor.reconstruct_path(
            selected_txid,
            max_depth=path_depth,
            max_paths=max_paths_sel
        )

        with p_col2:
            if not reconstructed_paths:
                st.info(f"No connected multi-hop paths found for transaction {selected_txid} at depth {path_depth}.")
            else:
                for path in reconstructed_paths:
                    st.markdown(f"""
                    <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">Path #{path['path_id']} ({path['total_hops']} Sequential Nodes)</span>
                            <span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Priority Score: {path['priority_score']:.1f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    steps = path['steps']
                    cols = st.columns(len(steps))
                    entity_colors = {
                        "IP": "#06b6d4",
                        "TXID": "#f97316",
                        "WALLET": "#10b981",
                        "RELATED_TXID": "#a855f7",
                        "CONNECTED_WALLET": "#ec4899"
                    }

                    for idx, step in enumerate(steps):
                        color = entity_colors.get(step["entity_type"], "#64748b")
                        with cols[idx]:
                            st.markdown(f"""
                            <div style="background: #1e293b; border-top: 4px solid {color}; border-radius: 6px; padding: 8px 10px; height: 100%; min-height: 120px;">
                                <div style="font-size: 0.7rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Step {step['step']} • {step['entity_type']}</div>
                                <div style="font-size: 0.85rem; font-weight: 700; color: #f1f5f9; word-break: break-all; margin: 4px 0;">{step['entity_id']}</div>
                                <div style="font-size: 0.75rem; color: {color}; font-weight: 500;">{step['label']}</div>
                                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 4px; line-height: 1.2;">{step['relationship']}</div>
                            </div>
                            """, unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # FEATURE 2: EVIDENCE TIMELINE / ACTIVITY SEQUENCE
        # ---------------------------------------------------------------------
        st.subheader("⏳ Forensic Evidence Timeline")
        st.caption(
            "Chronological activity sequence reconstructed from network packet captures, on-chain block confirmations, "
            "wallet fund movements, and AI anomaly detection signals using actual dataset timestamps."
        )

        timeline_events = timeline_builder.build_transaction_timeline(selected_txid)

        if not timeline_events:
            st.info(f"No temporal events recorded for transaction {selected_txid}.")
        else:
            for ev in timeline_events:
                sev_color = {
                    "CRITICAL": "#ef4444",
                    "HIGH": "#f97316",
                    "WARNING": "#eab308",
                    "INFO": "#38bdf8"
                }.get(ev.get("severity", "INFO"), "#38bdf8")

                ev_badge = sanitize_hdbscan(ev.get('badge', ''))
                ev_entity = sanitize_hdbscan(ev.get('entity', ''))
                ev_desc = sanitize_hdbscan(ev.get('description', ''))

                st.markdown(f"""
                <div style="display: flex; gap: 14px; margin-bottom: 10px; align-items: flex-start;">
                    <div style="min-width: 155px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px; text-align: center;">
                        <span style="font-family: monospace; font-size: 0.8rem; color: #94a3b8; font-weight: 600;">{ev['timestamp']}</span>
                    </div>
                    <div style="flex-grow: 1; background: #1e293b; border-left: 4px solid {sev_color}; border-radius: 6px; padding: 8px 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 0.9rem;">{ev_badge}</span>
                            <span style="font-family: monospace; font-size: 0.8rem; color: #cbd5e1; background: #0f172a; padding: 2px 6px; border-radius: 4px;">{ev_entity}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.35;">{ev_desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)



# =============================================================================
# TAB 3: IP–TXID–WALLET CORRELATION GRAPH
# =============================================================================
with tabs[3]:
    st.title("🕸️ IP–TXID–Wallet Multi-Modal Correlation")
    st.caption("Interactive topological graph linking network vantage points (IPs) to transactions (TXIDs) and wallets with Correlation Confidence.")
    render_active_dataset_banner()

    g_col1, g_col2 = st.columns([1, 3])

    with g_col1:
        st.subheader("Subgraph Focus")
        focus_entity_type = st.radio("Focus Node Type", ["TXID", "Wallet", "IP"], key="tab3_type")
        
        crit_txs = scored_df[scored_df['risk_level_normalized'] == 'CRITICAL']
        default_txid = str(crit_txs['txid'].iloc[0]) if not crit_txs.empty else str(scored_df['txid'].iloc[0])
        default_wallet = str(crit_txs['source_wallet'].iloc[0]) if not crit_txs.empty else str(scored_df['source_wallet'].iloc[0])
        default_ip = str(crit_txs['src_ip'].iloc[0]) if not crit_txs.empty else str(scored_df['src_ip'].iloc[0])

        if focus_entity_type == "TXID":
            cur_tx = st.session_state.get("tab3_tx", default_txid)
            if cur_tx not in graph_engine.G:
                st.session_state["tab3_tx"] = default_txid
            selected_focus = st.text_input("Enter TXID", value=default_txid, key="tab3_tx").strip()
        elif focus_entity_type == "Wallet":
            cur_w = st.session_state.get("tab3_w", default_wallet)
            if cur_w not in graph_engine.G:
                st.session_state["tab3_w"] = default_wallet
            selected_focus = st.text_input("Enter Wallet ID", value=default_wallet, key="tab3_w").strip()
        else:
            cur_ip = st.session_state.get("tab3_ip", default_ip)
            if cur_ip not in graph_engine.G:
                st.session_state["tab3_ip"] = default_ip
            selected_focus = st.text_input("Enter IP Address", value=default_ip, key="tab3_ip").strip()

        hop_radius = st.slider("Exploration Radius (Hops)", 1, 3, 2, key="tab3_hop")
        max_nodes_display = st.slider("Max Nodes to Render", 10, 80, 60, step=5, key="tab3_nodes")

    with g_col2:
        if selected_focus not in graph_engine.G:
            st.warning(f"Entity '{selected_focus}' was not found in the active graph ({graph_engine.G.number_of_nodes():,} nodes). Defaulting to critical hub entity '{default_wallet}'.")
            selected_focus = default_wallet

        sub_g = graph_engine.extract_subgraph(selected_focus, radius=hop_radius, max_nodes=max_nodes_display)

        if sub_g.number_of_nodes() == 0:
            st.warning(f"Entity '{selected_focus}' was not found in the transaction graph.")
        else:
            st.caption(f"Visualizing ego subgraph around **{selected_focus}** ({sub_g.number_of_nodes()} nodes, {sub_g.number_of_edges()} edges)")
            k_val = 0.85 / (max(sub_g.number_of_nodes(), 1) ** 0.5) if sub_g.number_of_nodes() > 0 else 0.6
            pos = nx.spring_layout(sub_g, seed=42, k=k_val, iterations=40)

            node_x_ip, node_y_ip, text_ip = [], [], []
            node_x_tx, node_y_tx, text_tx = [], [], []
            node_x_w, node_y_w, text_w = [], [], []

            for node, attr in sub_g.nodes(data=True):
                x, y = pos[node]
                n_type = attr.get('node_type', 'UNKNOWN')
                if n_type == 'IP':
                    node_x_ip.append(x)
                    node_y_ip.append(y)
                    text_ip.append(f"<b>IP:</b> {node}<br>Geo: {attr.get('country', '')}")
                elif n_type == 'TXID':
                    node_x_tx.append(x)
                    node_y_tx.append(y)
                    text_tx.append(f"<b>TXID:</b> {node}<br>Amount: {attr.get('amount_btc', 0):.4f} BTC")
                else:
                    node_x_w.append(x)
                    node_y_w.append(y)
                    text_w.append(f"<b>Wallet:</b> {node}")

            edge_x, edge_y = [], []
            for edge in sub_g.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig_graph = go.Figure()
            fig_graph.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.3, color="#94a3b8"), hoverinfo='none', mode='lines', name='Links'))

            if node_x_ip:
                fig_graph.add_trace(go.Scatter(x=node_x_ip, y=node_y_ip, mode='markers+text', textposition="top center", hoverinfo='text', text=[t.split('<br>')[0].replace('<b>IP:</b> ', '') for t in text_ip], hovertext=text_ip, marker=dict(size=18, color='#06b6d4', symbol='square', line=dict(width=2, color='#ffffff')), name='IP Address'))
            if node_x_tx:
                fig_graph.add_trace(go.Scatter(x=node_x_tx, y=node_y_tx, mode='markers', hoverinfo='text', hovertext=text_tx, marker=dict(size=14, color='#f97316', symbol='circle', line=dict(width=2, color='#ffffff')), name='TXID'))
            if node_x_w:
                fig_graph.add_trace(go.Scatter(x=node_x_w, y=node_y_w, mode='markers+text', textposition="top center", hoverinfo='text', text=[t.split('<br>')[0].replace('<b>Wallet:</b> ', '') for t in text_w], hovertext=text_w, marker=dict(size=16, color='#10b981', symbol='hexagon', line=dict(width=2, color='#ffffff')), name='Wallet'))

            fig_graph.update_layout(template="plotly_dark", showlegend=True, hovermode='closest', margin=dict(b=10, l=10, r=10, t=10), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=520)
            st.plotly_chart(fig_graph, width='stretch')


# =============================================================================
# TAB 4: GRAPH ANALYTICS & LEIDEN COMMUNITIES
# =============================================================================
with tabs[4]:
    algo_used = getattr(graph_engine.community_detector, 'algorithm_used', 'Leiden')
    st.title(f"📊 Network Topology & Graph Communities ({algo_used})")
    st.caption(f"Macro graph metrics, high-degree hubs, and {algo_used} community detection structures (with Louvain fallback).")
    render_active_dataset_banner()

    g_summary = ctx["graph_summary"]
    col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
    col_g1.metric("Total Graph Nodes", f"{g_summary['total_nodes']:,}")
    col_g2.metric("Total Directed Edges", f"{g_summary['total_edges']:,}")
    col_g3.metric("Unique IP Nodes", f"{g_summary['ip_nodes']:,}")
    col_g4.metric("Unique Wallet Nodes", f"{g_summary['wallet_nodes']:,}")
    col_g5.metric(f"{algo_used} Communities", f"{g_summary.get('louvain_communities_count', 0):,}")

    st.markdown("---")
    high_deg = ctx["high_degree_entities"]

    hd1, hd2 = st.columns(2)
    with hd1:
        st.subheader("Top High-Degree Broadcast IPs (Hub Vantage Points)")
        ip_deg_df = pd.DataFrame(high_deg['top_ips'])
        fig_ip_deg = px.bar(ip_deg_df, x='degree', y='entity', orientation='h', labels={'degree': 'Connections', 'entity': 'IP Address'}, color='degree', color_continuous_scale='tealgrn')
        fig_ip_deg.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=320)
        st.plotly_chart(fig_ip_deg, width='stretch')

    with hd2:
        st.subheader("Top High-Degree Wallets (Financial Hubs)")
        w_deg_df = pd.DataFrame(high_deg['top_wallets'])
        fig_w_deg = px.bar(w_deg_df, x='degree', y='entity', orientation='h', labels={'degree': 'Connections', 'entity': 'Wallet ID'}, color='degree', color_continuous_scale='purples')
        fig_w_deg.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=320)
        st.plotly_chart(fig_w_deg, width='stretch')

    st.subheader(f"Discovered {algo_used} Graph Communities")
    comm_df = pd.DataFrame(graph_engine.community_stats)
    if not comm_df.empty:
        st.dataframe(comm_df, width='stretch', hide_index=True)


# =============================================================================
# TAB 5: AI ANOMALY ANALYSIS (ISOLATION FOREST + HDBSCAN)
# =============================================================================
with tabs[5]:
    st.title("🤖 AI Anomaly & Behavioural Clustering Analysis")
    st.caption("Dual ML architecture: Isolation Forest (unsupervised tree isolation) + HDBSCAN (hierarchical density-based behavioural clustering).")
    render_active_dataset_banner()

    anom_sample = scored_df.sample(min(2000, len(scored_df)), random_state=42)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Isolation Forest Anomaly Score Distribution")
        fig_score_dist = px.histogram(anom_sample, x='anomaly_score', color='is_anomaly', nbins=40, labels={'anomaly_score': 'Normalized Anomaly Score', 'is_anomaly': 'AI Flagged'}, color_discrete_map={0: '#10b981', 1: '#f43f5e'})
        fig_score_dist.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_score_dist, width='stretch')

    with col_m2:
        st.subheader("HDBSCAN Behavioural Cluster Membership")
        cluster_col = 'hdbscan_cluster' if 'hdbscan_cluster' in anom_sample.columns else 'dbscan_cluster'
        dbscan_counts = anom_sample[cluster_col].value_counts().reset_index()
        dbscan_counts.columns = ['Cluster ID', 'Count']
        fig_dbscan = px.bar(dbscan_counts, x='Cluster ID', y='Count', color='Cluster ID', text_auto=True)
        fig_dbscan.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_dbscan, width='stretch')

    st.markdown("---")
    st.subheader("Feature Discrepancy Analysis (Normal vs Anomaly)")
    feature_to_compare = st.selectbox("Select Feature to Compare", ["transaction_frequency_24h", "avg_time_gap_min", "wallet_degree", "unique_ip_count", "packet_count", "bytes_transferred", "num_outputs"], key="tab5_feat")
    fig_feat_box = px.box(anom_sample, x='is_anomaly', y=feature_to_compare, color='is_anomaly', labels={'is_anomaly': 'Predicted Anomaly (0=Normal, 1=Anomaly)', feature_to_compare: feature_to_compare}, color_discrete_map={0: '#10b981', 1: '#f43f5e'})
    fig_feat_box.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_feat_box, width='stretch')


# =============================================================================
# TAB 6: RANKED ALERTS (FIXED STRICT EXACT RISK FILTERING)
# =============================================================================
with tabs[6]:
    st.title("🚨 Prioritized Alerts & Investigation Cases")
    st.caption("Consolidated triage queue: Investigate deduplicated high-priority cases or inspect individual ranked alert traffic.")
    render_active_dataset_banner()

    case_tab, alert_feed_tab = st.tabs(["🗂️ Prioritized Investigation Cases", "📑 Full Alert Feed"])

    # -------------------------------------------------------------------------
    # SUB-TAB 1: PRIORITIZED INVESTIGATION CASES (DEDUPLICATION & GROUPING)
    # -------------------------------------------------------------------------
    with case_tab:
        st.subheader("Investigation Cases (Alert Deduplication & Correlation)")
        st.caption(
            "Consolidates strongly connected anomalous alerts into deduplicated cases using real relationships: "
            "shared source/destination wallets, sequential fund flows, temporal network correlation (≤2h), and HDBSCAN behavioural clusters."
        )

        all_cases = case_grouper.get_cases(min_transactions=1)
        multi_cases = [c for c in all_cases if c["transaction_count"] > 1]
        crit_cases = [c for c in all_cases if c["severity"] == "CRITICAL"]
        total_case_btc = sum(c["total_btc_volume"] for c in all_cases)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Cases Discovered", f"{len(all_cases)}")
        mc2.metric("Critical Severity Cases", f"{len(crit_cases)}")
        mc3.metric("Multi-Transaction Incidents", f"{len(multi_cases)}")
        mc4.metric("Total Case BTC Volume", f"{total_case_btc:.4f} BTC")

        c_filter_col1, c_filter_col2 = st.columns([1, 2])
        with c_filter_col1:
            case_scope = st.radio(
                "Case Scope Filter",
                ["All Cases (Including Isolated Anomalies)", "Multi-Transaction Cases Only (≥2 TXs)"],
                key="tab6_case_scope"
            )
        min_tx_filter = 2 if "Multi-Transaction" in case_scope else 1
        filtered_cases = case_grouper.get_cases(min_transactions=min_tx_filter)

        if not filtered_cases:
            st.info("No cases found matching the selected scope.")
        else:
            # Summary Table
            case_summary_df = pd.DataFrame([
                {
                    "Case ID": c["case_id"],
                    "Severity": c["severity"],
                    "Title": c["title"],
                    "Max Risk": int(c["max_risk_score"]),
                    "Transactions": c["transaction_count"],
                    "Wallets": len(c["wallets"]),
                    "Network IPs": len(c["ips"]),
                    "Total BTC": f"{c['total_btc_volume']:.4f}",
                    "Primary TXID": c["primary_txid"]
                }
                for c in filtered_cases
            ])

            st.dataframe(
                case_summary_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Max Risk": st.column_config.ProgressColumn("Max Risk", help="Highest Risk Score in Case", format="%d", min_value=0, max_value=100),
                    "Severity": st.column_config.TextColumn("Severity"),
                    "Total BTC": st.column_config.TextColumn("Total BTC Volume")
                }
            )

            st.markdown("---")
            st.subheader("🔍 Case Drill-Down & Forensic Reconstruction")

            case_options = [
                f"{c['case_id']} — [{c['severity']}] {c['title']} ({c['transaction_count']} TXs, {len(c['wallets'])} Wallets)"
                for c in filtered_cases
            ]
            selected_case_idx = st.selectbox("Select Case to Inspect", range(len(case_options)), format_func=lambda i: case_options[i])
            active_case = filtered_cases[selected_case_idx]

            sev_color_case = {
                "CRITICAL": "#ef4444",
                "HIGH": "#f97316",
                "MEDIUM": "#eab308",
                "LOW": "#22c55e"
            }.get(active_case["severity"], "#3b82f6")

            st.markdown(f"""
            <div style="background: #1e293b; border-left: 6px solid {sev_color_case}; padding: 14px 18px; border-radius: 8px; margin: 12px 0 16px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{active_case['case_id']}: {active_case['title']}</span>
                        <span style="background-color: {sev_color_case}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; margin-left: 10px; font-size: 0.85rem;">
                            {active_case['severity']} SEVERITY
                        </span>
                    </div>
                    <div>
                        <span style="font-size: 0.95rem; color: #94a3b8; margin-right: 15px;">Transactions: <strong style="color: #f8fafc;">{active_case['transaction_count']}</strong></span>
                        <span style="font-size: 0.95rem; color: #94a3b8; margin-right: 15px;">Wallets: <strong style="color: #10b981;">{len(active_case['wallets'])}</strong></span>
                        <span style="font-size: 0.95rem; color: #94a3b8;">Max Risk: <strong style="color: {sev_color_case};">{int(active_case['max_risk_score'])}/100</strong></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Grouping Evidence Rationale
            st.markdown("**📌 Grouping Evidence & Correlation Rationale:**")
            for reason in active_case["grouping_reasons"]:
                clean_r = sanitize_hdbscan(reason)
                st.markdown(f"- {clean_r}")

            # Grouped Transactions Table
            st.markdown("**📑 Member Transactions in Case:**")
            case_txs_df = pd.DataFrame(active_case["transactions"])[[
                'txid', 'risk_score', 'risk_level', 'anomaly_score', 'total_output_amount_btc',
                'src_ip', 'source_wallet', 'destination_wallet', 'timestamp'
            ]]
            st.dataframe(
                case_txs_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn("Risk Score", format="%d", min_value=0, max_value=100),
                    "anomaly_score": st.column_config.NumberColumn("AI Score", format="%.3f"),
                    "total_output_amount_btc": st.column_config.NumberColumn("BTC Amount", format="%.4f")
                }
            )

            # Case Path & Case Timeline sub-expanders
            with st.expander("🧭 Case Primary Investigation Path", expanded=True):
                case_paths = path_reconstructor.reconstruct_path(active_case["primary_txid"], max_depth=3, max_paths=2)
                if case_paths:
                    for cp in case_paths:
                        st.caption(f"Path #{cp['path_id']} (Priority Score: {cp['priority_score']:.1f})")
                        c_steps = cp['steps']
                        c_cols = st.columns(len(c_steps))
                        for s_idx, step in enumerate(c_steps):
                            c_color = {
                                "IP": "#06b6d4",
                                "TXID": "#f97316",
                                "WALLET": "#10b981",
                                "RELATED_TXID": "#a855f7",
                                "CONNECTED_WALLET": "#ec4899"
                            }.get(step["entity_type"], "#64748b")
                            with c_cols[s_idx]:
                                st.markdown(f"""
                                <div style="background: #0f172a; border-top: 3px solid {c_color}; border-radius: 4px; padding: 6px 8px; min-height: 100px;">
                                    <div style="font-size: 0.65rem; color: #94a3b8; font-weight: bold;">{step['entity_type']}</div>
                                    <div style="font-size: 0.8rem; font-weight: bold; color: #f1f5f9; word-break: break-all;">{step['entity_id']}</div>
                                    <div style="font-size: 0.7rem; color: {c_color};">{step['label']}</div>
                                </div>
                                """, unsafe_allow_html=True)

            with st.expander("⏳ Case Consolidated Evidence Timeline", expanded=True):
                case_timeline = timeline_builder.build_case_timeline(active_case["txids"])
                for ev in case_timeline[:15]:  # Display up to 15 key events
                    ev_color = {
                        "CRITICAL": "#ef4444",
                        "HIGH": "#f97316",
                        "WARNING": "#eab308",
                        "INFO": "#38bdf8"
                    }.get(ev.get("severity", "INFO"), "#38bdf8")

                    ev_badge = sanitize_hdbscan(ev.get('badge', ''))
                    ev_entity = sanitize_hdbscan(ev.get('entity', ''))
                    ev_desc = sanitize_hdbscan(ev.get('description', ''))

                    st.markdown(f"""
                    <div style="display: flex; gap: 12px; margin-bottom: 8px; align-items: flex-start;">
                        <div style="min-width: 145px; background: #0f172a; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; text-align: center;">
                            <span style="font-family: monospace; font-size: 0.75rem; color: #94a3b8;">{ev['timestamp']}</span>
                        </div>
                        <div style="flex-grow: 1; background: #0f172a; border-left: 3px solid {ev_color}; border-radius: 4px; padding: 6px 12px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 0.85rem;">{ev_badge}</span>
                            <span style="font-family: monospace; font-size: 0.75rem; color: #94a3b8; margin-left: 8px;">{ev_entity}</span>
                            <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 2px;">{ev_desc}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SUB-TAB 2: FULL ALERT FEED (INDIVIDUAL RANKED TRIAGE)
    # -------------------------------------------------------------------------
    with alert_feed_tab:
        r_col1, r_col2 = st.columns([1, 2])
        with r_col1:
            selected_risk_level = st.selectbox(
                "Filter Risk Tier (Exact Match)",
                ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                index=0,
                key="tab6_risk_filter_exact"
            )
        with r_col2:
            search_filter = st.text_input("Filter Alerts (by TXID, IP, or Wallet)", placeholder="Enter keyword...", key="tab6_search").strip()

        # Fast in-memory filtering from precomputed complete base feed
        if selected_risk_level == "ALL":
            ranked_alerts = ctx["ranked_alerts_full"].copy()
        else:
            norm_target = selected_risk_level.upper()
            ranked_alerts = ctx["ranked_alerts_full"][ctx["ranked_alerts_full"]['risk_level'].astype(str).str.upper() == norm_target].copy()

        if 'risk_reasons' in ranked_alerts.columns:
            ranked_alerts['risk_reasons'] = ranked_alerts['risk_reasons'].astype(str).str.replace(r'\bH*DBSCAN\b', 'HDBSCAN', regex=True, flags=re.IGNORECASE)

        if search_filter:
            mask = (
                ranked_alerts['txid'].str.contains(search_filter, case=False, na=False) |
                ranked_alerts['src_ip'].str.contains(search_filter, na=False) |
                ranked_alerts['source_wallet'].str.contains(search_filter, case=False, na=False) |
                ranked_alerts['destination_wallet'].str.contains(search_filter, case=False, na=False)
            )
            ranked_alerts = ranked_alerts[mask]

        st.write(f"Displaying **{len(ranked_alerts):,}** prioritized alerts for risk tier **{selected_risk_level}** (Showing top 100 below).")

        # ---- Export Buttons: CSV + PDF side by side ----
        csv_data = ranked_alerts.to_csv(index=False).encode('utf-8')

        btn_col1, btn_col2, btn_spacer = st.columns([1, 1, 2])
        with btn_col1:
            st.download_button(
                label="📥 Export Alerts to CSV",
                data=csv_data,
                file_name="Bitcoin_Prioritized_Alerts.csv",
                mime="text/csv",
            )
        with btn_col2:
            # PDF uses the COMPLETE anomaly set, independent of dashboard risk filter
            anom_df = ctx["scored_df"]
            if "is_anomaly" in anom_df.columns:
                anom_df = anom_df[anom_df["is_anomaly"] == 1]

            if anom_df.empty:
                st.info("No anomalous transactions available for PDF export.")
            else:
                if "cached_pdf_bytes" not in st.session_state:
                    st.session_state["cached_pdf_bytes"] = None

                if st.session_state["cached_pdf_bytes"] is None:
                    if st.button("📄 Export Security Reports to PDF", width='stretch', key="btn_prep_pdf", help="Compile single consolidated PDF report for all anomalous transactions (100% offline)"):
                        with st.spinner("Generating forensic threat report PDF for all anomalous transactions..."):
                            try:
                                pdf_bytes = ReportGenerator.generate_batch_report_pdf(
                                    anom_df,
                                    st.session_state["feature_stats"],
                                    max_pages=None,
                                )
                                st.session_state["cached_pdf_bytes"] = pdf_bytes
                                st.rerun()
                            except Exception as e:
                                st.error(f"PDF generation error: {e}")
                else:
                    p_c1, p_c2 = st.columns([3, 1])
                    with p_c1:
                        st.download_button(
                            label="⬇️ Download Threat Reports PDF",
                            data=st.session_state["cached_pdf_bytes"],
                            file_name="Bitcoin_Security_Threat_Reports.pdf",
                            mime="application/pdf",
                            key="btn_download_pdf",
                        )
                    with p_c2:
                        if st.button("🔄", help="Re-generate fresh PDF", key="btn_regen_pdf"):
                            st.session_state["cached_pdf_bytes"] = None
                            st.rerun()

        st.dataframe(
            ranked_alerts.head(100),
            width='stretch',
            hide_index=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk Score", help="Composite Risk (0-100)", format="%d", min_value=0, max_value=100),
                "risk_level": st.column_config.TextColumn("Risk Level"),
                "anomaly_score": st.column_config.NumberColumn("AI Score", format="%.3f"),
                "confidence_score": st.column_config.ProgressColumn("Confidence", help="Correlation Confidence (0.0-1.0)", format="%.2f", min_value=0.0, max_value=1.0),
                "correlation_confidence_pct": st.column_config.TextColumn("Confidence %"),
                "total_output_amount_btc": st.column_config.NumberColumn("BTC Amount", format="%.4f")
            }
        )



# =============================================================================
# TAB 7: PROTOTYPE EVALUATION
# =============================================================================
with tabs[7]:
    st.title("🎯 Prototype Evaluation & Ground Truth Benchmarks")
    st.warning("🔒 **Evaluation Policy:** `ground_truth` and `scenario` are strictly used for post-prediction validation. Never used during training.")

    ev1, ev2, ev3, ev4 = st.columns(4)
    ev1.metric("Precision", f"{eval_results['precision'] * 100:.2f}%")
    ev2.metric("Recall", f"{eval_results['recall'] * 100:.2f}%")
    ev3.metric("F1-Score", f"{eval_results['f1_score'] * 100:.2f}%")
    ev4.metric("ROC-AUC", f"{eval_results['roc_auc']:.4f}")

    st.markdown("---")
    cm_data = eval_results['confusion_matrix']
    c_cm, c_sc = st.columns([1, 2])

    with c_cm:
        st.subheader("Confusion Matrix")
        cm_matrix = [[cm_data['true_negatives'], cm_data['false_positives']], [cm_data['false_negatives'], cm_data['true_positives']]]
        fig_cm = px.imshow(cm_matrix, labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Count"), x=["Normal (0)", "Anomaly (1)"], y=["Normal (0)", "Anomaly (1)"], text_auto=True, color_continuous_scale="Blues")
        fig_cm.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_cm, width='stretch')

        st.caption(f"**TP:** {cm_data['true_positives']:,} | **TN:** {cm_data['true_negatives']:,} | **FP:** {cm_data['false_positives']:,} | **FN:** {cm_data['false_negatives']:,}")

    with c_sc:
        st.subheader("Scenario-Wise Detection Rates")
        scenario_df = eval_results['scenario_breakdown']
        fig_scenario = px.bar(scenario_df[scenario_df['Scenario'] != 'normal'], x='Detection Rate (%)', y='Scenario', orientation='h', color='Avg Risk Score', color_continuous_scale='Reds', text='Detection Rate (%)', labels={'Scenario': 'Anomaly Scenario'})
        fig_scenario.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=320)
        st.plotly_chart(fig_scenario, width='stretch')

    st.subheader("Detailed Scenario Benchmark Table")
    st.dataframe(scenario_df, width='stretch', hide_index=True)


# =============================================================================
# TAB 8: OFFLINE AI INVESTIGATION ASSISTANT
# =============================================================================
with tabs[8]:
    st.title("💬 Offline AI Investigation Assistant")
    st.caption("Investigator-in-the-loop forensic assistant grounded 100% in local telemetry and blockchain evidence.")

    # Status Bar
    col_st1, col_st2, col_st3, col_st4 = st.columns([1.5, 1.5, 1.5, 1.2])
    with col_st1:
        st.markdown(f"**Engine:** `{chatbot.llm_manager.runtime_type}`")
    with col_st2:
        st.markdown("**Network Mode:** `100% Offline (Local)`")
    with col_st3:
        act_ctx = chatbot.active_context
        ctx_desc = act_ctx.get("active_txid") or act_ctx.get("active_case_id") or act_ctx.get("active_wallet") or "Global Scope"
        st.markdown(f"**Active Focus:** `{ctx_desc}`")
    with col_st4:
        if st.button("🗑️ Clear Chat & Context", key="btn_clear_chat", width='stretch'):
            st.session_state["assistant_chat_history"] = []
            chatbot.reset_context()
            st.rerun()

    st.markdown("---")

    # Suggested Prompts
    st.caption("💡 **Quick Investigation Queries:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_query = None
    with q_col1:
        if st.button("📌 Highest Risk TX", key="q_high_risk", width='stretch'):
            quick_query = "What is the highest risk transaction?"
    with q_col2:
        if st.button("📁 Top Critical Cases", key="q_cases", width='stretch'):
            quick_query = "Show top critical cases"
    with q_col3:
        if st.button("🌐 Suspicious IPs", key="q_ips", width='stretch'):
            quick_query = "List top suspicious IP addresses"
    with q_col4:
        if st.button("🚨 Anomaly Breakdown", key="q_anoms", width='stretch'):
            quick_query = "What anomaly patterns were detected?"

    # Initialize chat history in session state
    if "assistant_chat_history" not in st.session_state:
        st.session_state["assistant_chat_history"] = [
            {
                "role": "assistant",
                "content": "👋 **Welcome, Investigator.** I am your 100% offline Bitcoin Forensic Assistant. I am grounded strictly in current investigation evidence (transactions, cases, network telemetry, graph paths, and timelines).\n\nAsk me about specific transactions (e.g., `Tell me about TX10000137`), cases (`What is Case CASE-001?`), suspicious IPs, or risk scoring rationale.",
                "evidence": []
            }
        ]

    # Render Chat History
    for msg in st.session_state["assistant_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("evidence"):
                with st.expander(f"📋 Auditable Evidence Used ({len(msg['evidence'])} verified facts)", expanded=False):
                    for ev_item in msg["evidence"]:
                        st.markdown(f"- **`{ev_item.get('source_tag', 'RECORD')}`**: {ev_item.get('fact', '')}")

    # Chat Input
    user_query = st.chat_input("Ask a forensic question (e.g. 'Tell me about TX10000137', 'Why is it critical?', 'What is Case CASE-001?')...")
    if quick_query:
        user_query = quick_query

    if user_query:
        # Append and render user query
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state["assistant_chat_history"].append({"role": "user", "content": user_query})

        # Generate response grounded in knowledge store
        with st.chat_message("assistant"):
            with st.spinner("Analyzing investigation evidence..."):
                ans = chatbot.ask(user_query)
            st.markdown(ans["response"])
            if ans.get("evidence_used"):
                with st.expander(f"📋 Auditable Evidence Used ({len(ans['evidence_used'])} verified facts)", expanded=False):
                    for ev_item in ans["evidence_used"]:
                        st.markdown(f"- **`{ev_item.get('source_tag', 'RECORD')}`**: {ev_item.get('fact', '')}")

        # Store in conversation history and trigger update
        st.session_state["assistant_chat_history"].append({
            "role": "assistant",
            "content": ans["response"],
            "evidence": ans.get("evidence_used", [])
        })
        st.rerun()


# =============================================================================
# TAB 9: BEHAVIOURAL PATTERN DETECTION (PEELING-CHAIN & COINJOIN MIXING)
# =============================================================================
with tabs[9]:
    st.title("🔗 Behavioural Pattern Detection: Peeling-Chains & CoinJoin Mixing")
    st.caption("Multi-condition structural pattern analysis across transaction graph topologies. Identifies sequential single-continuation fund peels and equal-denomination mixing dispersals.")
    render_active_dataset_banner()

    st.info(
        "ℹ️ **Defensive Investigative Notice:** All detected patterns represent behavioural indicators for investigator prioritization, "
        "not criminal verdicts. All findings require human-in-the-loop forensic corroboration."
    )

    # Extract pattern subsets from scored_df
    peel_mask = scored_df.get('peeling_chain_detected', False) == True
    cj_mask = scored_df.get('coinjoin_detected', False) == True

    peel_df = scored_df[peel_mask].copy() if peel_mask.any() else pd.DataFrame()
    cj_df = scored_df[cj_mask].copy() if cj_mask.any() else pd.DataFrame()

    total_peel = len(peel_df)
    peel_groups = peel_df['peeling_chain_id'].nunique() if not peel_df.empty and 'peeling_chain_id' in peel_df.columns else 0
    total_cj = len(cj_df)
    cj_btc = cj_df['total_output_amount_btc'].sum() if not cj_df.empty else 0.0

    # Top KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peeling-Chain Txs</div>
            <div class="metric-value" style="color: #f59e0b;">{total_peel}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peeling-Chain Groups</div>
            <div class="metric-value" style="color: #f59e0b;">{peel_groups}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">CoinJoin-Like Txs</div>
            <div class="metric-value" style="color: #a855f7;">{total_cj}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mixing Volume (BTC)</div>
            <div class="metric-value" style="color: #a855f7;">{cj_btc:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 1: Peeling-Chain Detections
    st.subheader("🔗 1. Peeling-Chain Fund-Flow Detections")
    st.markdown(
        "A **peeling chain** is a sequential transaction structure where a large unspent output repeatedly funds the next hop, "
        "peeling off a smaller amount at each step. Detected via multi-hop traversal with continuation ratio constraints and miner fee bounds."
    )

    if not peel_df.empty:
        peel_cols_to_show = [c for c in [
            'txid', 'peeling_chain_id', 'peeling_chain_position', 'peeling_chain_length',
            'peeling_chain_evidence', 'risk_level_normalized', 'risk_score',
            'total_output_amount_btc', 'source_wallet', 'destination_wallet', 'peeling_chain_reasons'
        ] if c in peel_df.columns]

        peel_display = peel_df[peel_cols_to_show].copy()
        peel_display.rename(columns={
            'peeling_chain_id': 'Chain ID',
            'peeling_chain_position': 'Hop',
            'peeling_chain_length': 'Chain Length',
            'peeling_chain_evidence': 'Evidence Score',
            'risk_level_normalized': 'Risk Tier',
            'risk_score': 'Risk Score',
            'total_output_amount_btc': 'Amount (BTC)',
            'source_wallet': 'Source Wallet',
            'destination_wallet': 'Destination Wallet',
            'peeling_chain_reasons': 'Forensic Reasons'
        }, inplace=True)

        st.dataframe(
            peel_display,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No peeling-chain patterns detected in the current active dataset.")

    st.markdown("---")

    # Section 2: CoinJoin / Mixing Detections
    st.subheader("🔀 2. CoinJoin-Like / Mixing Pattern Detections")
    st.markdown(
        "A **CoinJoin / mixing pattern** is characterized by multiple participant inputs and multiple equal-denomination outputs, "
        "breaking direct graph linkability. Evaluated by equal-output ratios, participant counts, and consolidation exclusions."
    )

    if not cj_df.empty:
        cj_cols_to_show = [c for c in [
            'txid', 'num_inputs', 'num_outputs', 'coinjoin_equal_output_count',
            'coinjoin_equal_amount_btc', 'coinjoin_equal_output_ratio', 'coinjoin_evidence',
            'risk_level_normalized', 'risk_score', 'total_output_amount_btc', 'coinjoin_reasons'
        ] if c in cj_df.columns]

        cj_display = cj_df[cj_cols_to_show].copy()
        cj_display.rename(columns={
            'num_inputs': 'Inputs',
            'num_outputs': 'Outputs',
            'coinjoin_equal_output_count': 'Equal Outputs',
            'coinjoin_equal_amount_btc': 'Equal Amount (BTC)',
            'coinjoin_equal_output_ratio': 'Equal Ratio',
            'coinjoin_evidence': 'Evidence Score',
            'risk_level_normalized': 'Risk Tier',
            'risk_score': 'Risk Score',
            'total_output_amount_btc': 'Total Output (BTC)',
            'coinjoin_reasons': 'Forensic Reasons'
        }, inplace=True)

        st.dataframe(
            cj_display,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No CoinJoin-like mixing patterns detected in the current active dataset.")
