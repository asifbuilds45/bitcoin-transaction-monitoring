"""
Bitcoin Monitoring & Traffic Analysis Dashboard
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st

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
from src.clustering.dbscan_clustering import BitcoinDBSCANClustering
from src.feature_engineering import extract_features
from src.anomaly_detection import BitcoinAnomalyDetector
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.explainability.shap_explainer import BitcoinSHAPExplainer
from src.evaluation import evaluate_prototype_predictions
from src.database.db_manager import DatabaseManager
from src.reporting import ReportGenerator
from src.investigation import InvestigationPathReconstructor, EvidenceTimelineBuilder, AlertCaseGrouper

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

    # 6. DBSCAN Behavioural Clustering
    dbscan = BitcoinDBSCANClustering(eps=0.5, min_samples=5)
    cluster_labels, noise_mask, dbscan_evidence = dbscan.fit_predict(feat_df)
    enriched_df['dbscan_cluster'] = cluster_labels
    enriched_df['dbscan_is_noise'] = noise_mask
    enriched_df['dbscan_evidence'] = dbscan_evidence

    # 7. Temporal & Behavioural Evidence
    temp_df = analyze_temporal_patterns(enriched_df)
    beh_df = extract_behavioural_evidence_scores(temp_df)

    # 8. NetworkX Graph Analysis & Louvain Community Detection
    graph_engine = BitcoinGraphEngine(beh_df)

    louvain_ev = []
    community_ids = []
    for txid in beh_df['txid']:
        score, _ = graph_engine.louvain_detector.compute_community_evidence_score(str(txid))
        louvain_ev.append(score)
        comm_info = graph_engine.get_node_community_info(str(txid))
        community_ids.append(comm_info.get("community_id", -1))

    beh_df['louvain_community_evidence'] = louvain_ev
    beh_df['community_id'] = community_ids

    # 9. Multi-Layer Evidence Fusion & Risk Scoring
    risk_engine = RiskScoringEngine(ml_weight=0.45, heuristic_weight=0.55)
    scored_df = risk_engine.compute_risk_scores(
        beh_df, norm_scores, raw_scores, is_anomaly,
        dbscan_evidence=dbscan_evidence, louvain_evidence=np.array(louvain_ev)
    )

    # Normalize canonical risk_level
    scored_df['risk_level_normalized'] = scored_df['risk_level'].astype(str).str.strip().str.upper()

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
        "kpis": {
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
    }



# Initialize Session State Context if not present
if "context" not in st.session_state:
    net_default = os.path.join("data", "network_traffic_dataset.csv")
    chain_default = os.path.join("data", "blockchain_transactions_dataset.csv")
    if os.path.exists(net_default) and os.path.exists(chain_default):
        st.session_state["context"] = run_investigation_pipeline(net_default, chain_default)
        st.session_state["datasets_validated"] = True
        st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(st.session_state["context"]["scored_df"])
    else:
        raw_df = load_dataset()
        st.session_state["context"] = run_investigation_pipeline(raw_df, raw_df)
        st.session_state["datasets_validated"] = True
        st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(st.session_state["context"]["scored_df"])


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


# Initialize report ID map and pre-compute feature statistics for report generation
if "report_id_map" not in st.session_state:
    st.session_state["report_id_map"] = {}
if "feature_stats" not in st.session_state:
    st.session_state["feature_stats"] = ReportGenerator.compute_dataset_statistics(st.session_state["context"]["scored_df"])



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
    "🎯 7. Prototype Evaluation"
])


# =============================================================================
# TAB 0: DATASET INGESTION & INDEPENDENT VALIDATION
# =============================================================================
with tabs[0]:
    st.title("📥 Two-Layer Dataset Ingestion & Validation")
    st.caption("Upload separate Network-Layer and Blockchain-Layer CSV files, validate schema integrity independently, and trigger the investigation pipeline.")

    # Preset / Default dataset loader control
    col_def1, col_def2 = st.columns([3, 1])
    with col_def1:
        st.info("ℹ️ **Default Option:** You can upload custom CSV files below or click to load the preloaded 10,000-record synthetic datasets.")
    with col_def2:
        if st.button("📂 Load Default Datasets", width='stretch'):
            net_default = os.path.join("data", "network_traffic_dataset.csv")
            chain_default = os.path.join("data", "blockchain_transactions_dataset.csv")
            if os.path.exists(net_default) and os.path.exists(chain_default):
                st.session_state["net_file_data"] = net_default
                st.session_state["chain_file_data"] = chain_default
                st.session_state["datasets_validated"] = True
                st.success("Loaded preloaded default Network and Blockchain datasets.")

    st.markdown("---")

    col_net_ui, col_chain_ui = st.columns(2)

    # 1. NETWORK-LAYER DATA CONTROL
    with col_net_ui:
        st.subheader("1. NETWORK-LAYER DATA")
        st.caption("Bitcoin P2P network telemetry, IP addresses, ports, and packet traffic.")
        
        net_upload = st.file_uploader("Upload Network File (CSV, JSON, XML)", type=["csv", "json", "xml"], key="uploader_network_file")
        
        if net_upload is not None:
            st.session_state["net_file_data"] = net_upload
            st.session_state["datasets_validated"] = False

        net_source = st.session_state.get("net_file_data", os.path.join("data", "network_traffic_dataset.csv"))
        
        try:
            if isinstance(net_source, str):
                net_preview_df, _ = parse_and_validate_network_data(net_source)
                net_filename = os.path.basename(net_source)
            else:
                net_preview_df, _ = parse_and_validate_network_data(net_source)
                net_filename = getattr(net_source, 'name', 'Uploaded_Network_File')
                if hasattr(net_source, 'seek'):
                    net_source.seek(0)
            
            st.success(f"✓ Network dataset loaded\n\n**Filename:** `{net_filename}`  \n**Records:** `{len(net_preview_df):,} rows` × `{len(net_preview_df.columns)} columns`")
            
            with st.expander("🔍 Preview Network Dataset", expanded=False):
                st.dataframe(net_preview_df.head(5), width='stretch')
                st.caption(f"**Detected Columns:** {', '.join(net_preview_df.columns[:10])}...")
        except Exception as e:
            st.error(f"Error loading Network dataset: {e}")
            net_preview_df = None

    # 2. BLOCKCHAIN-LAYER DATA CONTROL
    with col_chain_ui:
        st.subheader("2. BLOCKCHAIN-LAYER DATA")
        st.caption("Bitcoin transaction ledgers, UTXOs, miner fees, and wallet addresses.")
        
        chain_upload = st.file_uploader("Upload Blockchain File (CSV, JSON, XML)", type=["csv", "json", "xml"], key="uploader_blockchain_file")
        
        if chain_upload is not None:
            st.session_state["chain_file_data"] = chain_upload
            st.session_state["datasets_validated"] = False

        chain_source = st.session_state.get("chain_file_data", os.path.join("data", "blockchain_transactions_dataset.csv"))
        
        try:
            if isinstance(chain_source, str):
                chain_preview_df, _ = parse_and_validate_blockchain_data(chain_source)
                chain_filename = os.path.basename(chain_source)
            else:
                chain_preview_df, _ = parse_and_validate_blockchain_data(chain_source)
                chain_filename = getattr(chain_source, 'name', 'Uploaded_Blockchain_File')
                if hasattr(chain_source, 'seek'):
                    chain_source.seek(0)
                
            st.success(f"✓ Blockchain dataset loaded\n\n**Filename:** `{chain_filename}`  \n**Records:** `{len(chain_preview_df):,} rows` × `{len(chain_preview_df.columns)} columns`")
            
            with st.expander("🔍 Preview Blockchain Dataset", expanded=False):
                st.dataframe(chain_preview_df.head(5), width='stretch')
                st.caption(f"**Detected Columns:** {', '.join(chain_preview_df.columns[:10])}...")
        except Exception as e:
            st.error(f"Error loading Blockchain dataset: {e}")
            chain_preview_df = None

    st.markdown("---")

    # ACTION BUTTONS: VALIDATE DATASETS & RUN INVESTIGATION
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])

    with btn_col1:
        if st.button("🔍 VALIDATE DATASETS", width='stretch', type="secondary"):
            if net_source is None or chain_source is None:
                st.error("Both Network and Blockchain datasets must be provided before validating.")
            else:
                with st.spinner("Validating Network and Blockchain schemas independently..."):
                    net_df, net_warn = parse_and_validate_network_data(net_source)
                    chain_df, chain_warn = parse_and_validate_blockchain_data(chain_source)
                    
                    st.session_state["datasets_validated"] = True
                    st.session_state["validation_report"] = {
                        "net_rows": len(net_df),
                        "chain_rows": len(chain_df),
                        "net_warn": net_warn,
                        "chain_warn": chain_warn
                    }
                    st.success("✓ Both Network and Blockchain datasets validated successfully!")

    with btn_col2:
        is_validated = st.session_state.get("datasets_validated", False)
        if st.button("🚀 RUN INVESTIGATION", width='stretch', type="primary", disabled=not is_validated):
            with st.spinner("Running Multi-Layer Correlation, Graph Engine, AI Models & Risk Scoring..."):
                new_ctx = run_investigation_pipeline(net_source, chain_source)
                st.session_state["context"] = new_ctx
                st.success("✓ Investigation completed successfully! Explore the findings in Tabs 1–7 above.")
                st.rerun()

    if "validation_report" in st.session_state:
        vr = st.session_state["validation_report"]
        st.info(f"**Validation Report:** Network ({vr['net_rows']:,} rows, {len(vr['net_warn'])} warnings) | Blockchain ({vr['chain_rows']:,} rows, {len(vr['chain_warn'])} warnings)")


# =============================================================================
# TAB 1: OVERVIEW & EXECUTIVE KPIS
# =============================================================================
with tabs[1]:
    st.title("🛡️ Cyber Monitoring & Anomaly Overview")
    st.caption("Real-time forensic monitoring correlating Bitcoin P2P network telemetry with on-chain UTXO transfers.")

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
        reasons_list = [r.strip() for r in str(tx_row['risk_reasons']).split(';') if r.strip()]
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
                "Source GeoIP Country": tx_row.get('geoip_src_country', tx_row.get('src_country', 'UNKNOWN')),
                "Destination GeoIP Country": tx_row.get('geoip_dst_country', tx_row.get('dst_country', 'UNKNOWN')),
                "Source ASN": tx_row.get('geoip_src_asn', tx_row.get('src_asn', 'UNKNOWN'))
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

                st.markdown(f"""
                <div style="display: flex; gap: 14px; margin-bottom: 10px; align-items: flex-start;">
                    <div style="min-width: 155px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px; text-align: center;">
                        <span style="font-family: monospace; font-size: 0.8rem; color: #94a3b8; font-weight: 600;">{ev['timestamp']}</span>
                    </div>
                    <div style="flex-grow: 1; background: #1e293b; border-left: 4px solid {sev_color}; border-radius: 6px; padding: 8px 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 0.9rem;">{ev['badge']}</span>
                            <span style="font-family: monospace; font-size: 0.8rem; color: #cbd5e1; background: #0f172a; padding: 2px 6px; border-radius: 4px;">{ev['entity']}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.35;">{ev['description']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)



# =============================================================================
# TAB 3: IP–TXID–WALLET CORRELATION GRAPH
# =============================================================================
with tabs[3]:
    st.title("🕸️ IP–TXID–Wallet Multi-Modal Correlation")
    st.caption("Interactive topological graph linking network vantage points (IPs) to transactions (TXIDs) and wallets with Correlation Confidence.")

    g_col1, g_col2 = st.columns([1, 3])

    with g_col1:
        st.subheader("Subgraph Focus")
        focus_entity_type = st.radio("Focus Node Type", ["TXID", "Wallet", "IP"], key="tab3_type")
        
        crit_txs = scored_df[scored_df['risk_level_normalized'] == 'CRITICAL']
        default_txid = crit_txs['txid'].iloc[0] if not crit_txs.empty else scored_df['txid'].iloc[0]
        default_wallet = crit_txs['source_wallet'].iloc[0] if not crit_txs.empty else scored_df['source_wallet'].iloc[0]
        default_ip = crit_txs['src_ip'].iloc[0] if not crit_txs.empty else scored_df['src_ip'].iloc[0]

        if focus_entity_type == "TXID":
            selected_focus = st.text_input("Enter TXID", value=default_txid, key="tab3_tx").strip()
        elif focus_entity_type == "Wallet":
            selected_focus = st.text_input("Enter Wallet ID", value=default_wallet, key="tab3_w").strip()
        else:
            selected_focus = st.text_input("Enter IP Address", value=default_ip, key="tab3_ip").strip()

        hop_radius = st.slider("Exploration Radius (Hops)", 1, 3, 2, key="tab3_hop")
        max_nodes_display = st.slider("Max Nodes to Render", 10, 60, 30, key="tab3_nodes")

    with g_col2:
        sub_g = graph_engine.extract_subgraph(selected_focus, radius=hop_radius, max_nodes=max_nodes_display)

        if sub_g.number_of_nodes() == 0:
            st.warning(f"Entity '{selected_focus}' was not found in the transaction graph.")
        else:
            st.caption(f"Visualizing ego subgraph around **{selected_focus}** ({sub_g.number_of_nodes()} nodes, {sub_g.number_of_edges()} edges)")
            pos = nx.spring_layout(sub_g, seed=42, k=0.6, iterations=25)

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
            fig_graph.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.2, color="#64748b"), hoverinfo='none', mode='lines', name='Links'))

            if node_x_ip:
                fig_graph.add_trace(go.Scatter(x=node_x_ip, y=node_y_ip, mode='markers+text', textposition="top center", hoverinfo='text', text=[t.split('<br>')[0].replace('<b>IP:</b> ', '') for t in text_ip], hovertext=text_ip, marker=dict(size=18, color='#06b6d4', symbol='square', line=dict(width=2, color='#ffffff')), name='IP Address'))
            if node_x_tx:
                fig_graph.add_trace(go.Scatter(x=node_x_tx, y=node_y_tx, mode='markers', hoverinfo='text', hovertext=text_tx, marker=dict(size=14, color='#f97316', symbol='circle', line=dict(width=2, color='#ffffff')), name='TXID'))
            if node_x_w:
                fig_graph.add_trace(go.Scatter(x=node_x_w, y=node_y_w, mode='markers+text', textposition="top center", hoverinfo='text', text=[t.split('<br>')[0].replace('<b>Wallet:</b> ', '') for t in text_w], hovertext=text_w, marker=dict(size=16, color='#10b981', symbol='hexagon', line=dict(width=2, color='#ffffff')), name='Wallet'))

            fig_graph.update_layout(template="plotly_dark", showlegend=True, hovermode='closest', margin=dict(b=10, l=10, r=10, t=10), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=480)
            st.plotly_chart(fig_graph, width='stretch')


# =============================================================================
# TAB 4: GRAPH ANALYTICS & LOUVAIN COMMUNITIES
# =============================================================================
with tabs[4]:
    st.title("📊 Network Topology & Louvain Communities")
    st.caption("Macro graph metrics, high-degree hubs, and Louvain community detection structures.")

    g_summary = ctx["graph_summary"]
    col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
    col_g1.metric("Total Graph Nodes", f"{g_summary['total_nodes']:,}")
    col_g2.metric("Total Directed Edges", f"{g_summary['total_edges']:,}")
    col_g3.metric("Unique IP Nodes", f"{g_summary['ip_nodes']:,}")
    col_g4.metric("Unique Wallet Nodes", f"{g_summary['wallet_nodes']:,}")
    col_g5.metric("Louvain Communities", f"{g_summary.get('louvain_communities_count', 0):,}")

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

    st.subheader("Discovered Louvain Graph Communities")
    comm_df = pd.DataFrame(graph_engine.community_stats)
    if not comm_df.empty:
        st.dataframe(comm_df, width='stretch', hide_index=True)


# =============================================================================
# TAB 5: AI ANOMALY ANALYSIS (ISOLATION FOREST + DBSCAN)
# =============================================================================
with tabs[5]:
    st.title("🤖 AI Anomaly & Behavioural Clustering Analysis")
    st.caption("Dual ML architecture: Isolation Forest (unsupervised tree isolation) + DBSCAN (density-based behavioural clustering).")

    anom_sample = scored_df.sample(min(2000, len(scored_df)), random_state=42)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Isolation Forest Anomaly Score Distribution")
        fig_score_dist = px.histogram(anom_sample, x='anomaly_score', color='is_anomaly', nbins=40, labels={'anomaly_score': 'Normalized Anomaly Score', 'is_anomaly': 'AI Flagged'}, color_discrete_map={0: '#10b981', 1: '#f43f5e'})
        fig_score_dist.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_score_dist, width='stretch')

    with col_m2:
        st.subheader("DBSCAN Behavioural Cluster Membership")
        dbscan_counts = anom_sample['dbscan_cluster'].value_counts().reset_index()
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

    case_tab, alert_feed_tab = st.tabs(["🗂️ Prioritized Investigation Cases", "📑 Full Alert Feed"])

    # -------------------------------------------------------------------------
    # SUB-TAB 1: PRIORITIZED INVESTIGATION CASES (DEDUPLICATION & GROUPING)
    # -------------------------------------------------------------------------
    with case_tab:
        st.subheader("Investigation Cases (Alert Deduplication & Correlation)")
        st.caption(
            "Consolidates strongly connected anomalous alerts into deduplicated cases using real relationships: "
            "shared source/destination wallets, sequential fund flows, temporal network correlation (≤2h), and DBSCAN behavioural clusters."
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
                st.markdown(f"- {reason}")

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

                    st.markdown(f"""
                    <div style="display: flex; gap: 12px; margin-bottom: 8px; align-items: flex-start;">
                        <div style="min-width: 145px; background: #0f172a; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; text-align: center;">
                            <span style="font-family: monospace; font-size: 0.75rem; color: #94a3b8;">{ev['timestamp']}</span>
                        </div>
                        <div style="flex-grow: 1; background: #0f172a; border-left: 3px solid {ev_color}; border-radius: 4px; padding: 6px 12px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 0.85rem;">{ev['badge']}</span>
                            <span style="font-family: monospace; font-size: 0.75rem; color: #94a3b8; margin-left: 8px;">{ev['entity']}</span>
                            <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 2px;">{ev['description']}</div>
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

        # ALWAYS filter from ORIGINAL COMPLETE dataset (ctx['scored_df']) to avoid stale dataframe state
        complete_scored_df = ctx["scored_df"]
        
        # Generate ranked alerts feed filtered strictly on exact risk_level
        ranked_alerts = generate_ranked_alerts(complete_scored_df, selected_risk_level)

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
            # PDF uses the COMPLETE anomaly set, ignoring the current risk filter
            if "report_id_map" not in st.session_state:
                st.session_state["report_id_map"] = {}
            try:
                final_pdf_bytes = ReportGenerator.generate_batch_report_pdf(
                    ctx["scored_df"],
                    st.session_state["feature_stats"],
                    max_pages=None,
                )
                st.download_button(
                    label="📄 Export Security Reports to PDF",
                    data=final_pdf_bytes,
                    file_name="Bitcoin_Security_Threat_Reports.pdf",
                    mime="application/pdf",
                )
            except ImportError as e:
                st.error(f"PDF generation unavailable: {e}")

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
