"""
Bitcoin Monitoring & Traffic Analysis Dashboard
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Streamlit Offline Investigation Prototype
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

from src.preprocessing import load_dataset, preprocess_data, get_dataset_summary
from src.geoip_enrichment import enrich_transactions_with_geoip
from src.correlation import EntityCorrelator
from src.graph_analysis import BitcoinGraphEngine
from src.feature_engineering import extract_features
from src.anomaly_detection import BitcoinAnomalyDetector
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.evaluation import evaluate_prototype_predictions

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
        padding: 16px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .risk-badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-badge-high {
        background-color: #f97316;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-badge-low {
        background-color: #22c55e;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA PIPELINE WITH CACHING (OFFLINE EXECUTION)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running Offline Analysis Pipeline...")
def load_and_process_all_data():
    # 1. Ingestion & Preprocessing
    raw_df = load_dataset()
    clean_df = preprocess_data(raw_df)
    
    # 2. GeoIP Enrichment
    enriched_df = enrich_transactions_with_geoip(clean_df)
    
    # 3. Unsupervised Feature Engineering
    feat_df = extract_features(enriched_df)
    
    # 4. AI Anomaly Detection (Isolation Forest)
    detector = BitcoinAnomalyDetector(contamination=0.10, n_estimators=150, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)
    
    # 5. Risk Scoring & Reason Generation
    risk_engine = RiskScoringEngine(ml_weight=0.45, heuristic_weight=0.55)
    scored_df = risk_engine.compute_risk_scores(enriched_df, norm_scores, raw_scores, is_anomaly)
    
    # 6. Evaluation against Ground Truth
    eval_results = evaluate_prototype_predictions(scored_df)
    
    return scored_df, feat_df, eval_results


@st.cache_resource(show_spinner="Constructing NetworkX Entity Graph...")
def get_graph_and_correlator(scored_df_sample):
    correlator = EntityCorrelator(scored_df_sample)
    graph_engine = BitcoinGraphEngine(scored_df_sample)
    return correlator, graph_engine


# Load data
scored_df, feat_df, eval_results = load_and_process_all_data()
correlator, graph_engine = get_graph_and_correlator(scored_df)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bitcoin.png", width=64)
st.sidebar.title("Bitcoin Sentinel")
st.sidebar.caption("NTRO PS 26146 — AI Bitcoin Traffic Monitor")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Investigation Navigation",
    [
        "1. Overview & Executive KPIs",
        "2. Transaction Investigation",
        "3. IP-TXID-Wallet Correlation",
        "4. Graph Analytics & Topology",
        "5. AI Anomaly Analysis",
        "6. Ranked Investigation Alerts",
        "7. Prototype Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("Mode: Fully Offline")
geoip_stat = scored_df['geoip_lookup_status'].iloc[0]
if geoip_stat == "geoip_resolved":
    st.sidebar.info("GeoIP: MaxMind MMDB Active")
else:
    st.sidebar.warning("GeoIP: Synthetic Fallback Active")
st.sidebar.caption("Defensive Cybersecurity Prototype")


# =============================================================================
# PAGE 1: OVERVIEW & EXECUTIVE KPIS
# =============================================================================
if page == "1. Overview & Executive KPIs":
    st.title("🛡️ Cyber Monitoring & Anomaly Overview")
    st.markdown(
        "Real-time forensic monitoring dashboard tracking synthetic Bitcoin peer-to-peer "
        "network broadcasts and on-chain transaction flows."
    )

    # Executive Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_tx = len(scored_df)
    detected_anom = int(scored_df['is_anomaly'].sum())
    normal_tx = total_tx - detected_anom
    critical_alerts = int((scored_df['risk_level'] == 'CRITICAL').sum())
    high_alerts = int((scored_df['risk_level'] == 'HIGH').sum())
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{total_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Baseline Normal</div>
            <div class="metric-value" style="color: #22c55e;">{normal_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AI Anomalies</div>
            <div class="metric-value" style="color: #f59e0b;">{detected_anom:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High Risk Alerts</div>
            <div class="metric-value" style="color: #f97316;">{high_alerts:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Critical Alerts</div>
            <div class="metric-value" style="color: #ef4444;">{critical_alerts:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Row 1 Charts: Anomaly Distribution & Risk Tiers
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("AI Anomaly Classification Breakdown")
        anom_counts = pd.DataFrame({
            "Classification": ["Normal Traffic", "Detected Anomalies"],
            "Count": [normal_tx, detected_anom]
        })
        fig_pie = px.pie(
            anom_counts,
            values="Count",
            names="Classification",
            color="Classification",
            color_discrete_map={"Normal Traffic": "#10b981", "Detected Anomalies": "#f43f5e"},
            hole=0.45
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Investigative Risk Severity Tiers (0–100)")
        risk_counts = scored_df['risk_level'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).fillna(0).reset_index()
        risk_counts.columns = ['Risk Tier', 'Count']
        fig_bar = px.bar(
            risk_counts,
            x='Risk Tier',
            y='Count',
            color='Risk Tier',
            color_discrete_map={'LOW': '#22c55e', 'MEDIUM': '#eab308', 'HIGH': '#f97316', 'CRITICAL': '#ef4444'},
            text_auto=True
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 2 Charts: Geographic Traffic & Temporal Volume
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Source Country Activity & Risk Profile")
        country_agg = scored_df.groupby('src_country').agg(
            total_tx=('txid', 'count'),
            high_risk=('risk_level', lambda x: (x.isin(['HIGH', 'CRITICAL'])).sum())
        ).reset_index().sort_values(by='total_tx', ascending=False).head(10)
        
        fig_country = px.bar(
            country_agg,
            x='src_country',
            y=['total_tx', 'high_risk'],
            barmode='group',
            labels={'value': 'Transactions', 'src_country': 'Country Code', 'variable': 'Category'},
            color_discrete_map={'total_tx': '#38bdf8', 'high_risk': '#ef4444'}
        )
        fig_country.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark")
        st.plotly_chart(fig_country, use_container_width=True)

    with c4:
        st.subheader("Temporal Transaction Velocity & Anomaly Occurrence")
        time_df = scored_df.set_index('timestamp').resample('7D').agg(
            total=('txid', 'count'),
            anomalies=('is_anomaly', 'sum')
        ).reset_index()
        fig_time = px.line(
            time_df,
            x='timestamp',
            y=['total', 'anomalies'],
            labels={'value': 'Weekly Transactions', 'timestamp': 'Date', 'variable': 'Metric'},
            color_discrete_map={'total': '#38bdf8', 'anomalies': '#f43f5e'}
        )
        fig_time.update_layout(margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark")
        st.plotly_chart(fig_time, use_container_width=True)

    # Top Suspicious Entities Table
    st.subheader("Top Suspicious Entities Requiring Prioritized Review")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.caption("Top Suspicious Source IPs (by High/Critical Risk TX Count)")
        top_ips = scored_df[scored_df['risk_level'].isin(['HIGH', 'CRITICAL'])].groupby('src_ip').agg(
            critical_alerts=('risk_level', 'count'),
            total_btc=('total_output_amount_btc', 'sum'),
            wallets_used=('source_wallet', 'nunique'),
            country=('src_country', 'first'),
            asn=('src_asn', 'first')
        ).reset_index().sort_values(by='critical_alerts', ascending=False).head(5)
        st.dataframe(top_ips, use_container_width=True, hide_index=True)

    with col_t2:
        st.caption("Top Suspicious Source Wallets (by Risk Score)")
        top_wallets = scored_df.groupby('source_wallet').agg(
            max_risk=('risk_score', 'max'),
            avg_risk=('risk_score', 'mean'),
            total_tx=('txid', 'count'),
            unique_ips=('src_ip', 'nunique'),
            total_btc=('total_output_amount_btc', 'sum')
        ).reset_index().sort_values(by=['max_risk', 'avg_risk'], ascending=False).head(5)
        st.dataframe(top_wallets, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 2: TRANSACTION INVESTIGATION
# =============================================================================
elif page == "2. Transaction Investigation":
    st.title("🔍 Forensic Transaction Deep-Dive")
    st.markdown("Inspect granular network packet telemetry, on-chain UTXO attributes, and AI-generated risk rationales.")

    # Search & Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        risk_filter = st.selectbox("Filter Risk Level", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with f_col2:
        search_txid = st.text_input("Search TXID", placeholder="e.g. TX10000137").strip()
    with f_col3:
        search_wallet = st.text_input("Search Wallet", placeholder="e.g. W000050").strip()
    with f_col4:
        search_ip = st.text_input("Search IP Address", placeholder="e.g. 10.11.20.2").strip()

    filtered_df = scored_df
    if risk_filter != "ALL":
        filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
    if search_txid:
        filtered_df = filtered_df[filtered_df['txid'].str.contains(search_txid, case=False, na=False)]
    if search_wallet:
        filtered_df = filtered_df[
            filtered_df['source_wallet'].str.contains(search_wallet, case=False, na=False) |
            filtered_df['destination_wallet'].str.contains(search_wallet, case=False, na=False)
        ]
    if search_ip:
        filtered_df = filtered_df[
            filtered_df['src_ip'].str.contains(search_ip, na=False) |
            filtered_df['dst_ip'].str.contains(search_ip, na=False)
        ]

    st.write(f"Showing **{len(filtered_df):,}** matching transactions.")
    
    if filtered_df.empty:
        st.warning("No transactions match the specified search parameters.")
    else:
        # Selector for transaction
        txid_options = filtered_df['txid'].head(100).tolist()
        selected_txid = st.selectbox("Select Transaction for Full Forensic Breakdown", txid_options)
        
        tx_row = scored_df[scored_df['txid'] == selected_txid].iloc[0]

        # Top Banner with Risk Level and Scores
        risk_color_map = {
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#22c55e"
        }
        r_level = tx_row['risk_level']
        badge_color = risk_color_map.get(r_level, "#3b82f6")

        st.markdown(f"""
        <div style="background: #1e293b; border-left: 6px solid {badge_color}; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Transaction: {tx_row['txid']}</span>
                    <span style="background-color: {badge_color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; margin-left: 12px; font-size: 0.9rem;">
                        {r_level} RISK
                    </span>
                </div>
                <div>
                    <span style="font-size: 1.1rem; color: #94a3b8; margin-right: 15px;">Composite Risk Score: <strong style="color: {badge_color}; font-size: 1.3rem;">{tx_row['risk_score']}/100</strong></span>
                    <span style="font-size: 1.1rem; color: #94a3b8;">AI Anomaly Score: <strong style="color: #38bdf8;">{tx_row['anomaly_score']:.3f}</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Explainable Reasons Box
        st.subheader("💡 Automated Forensic Explanation & Evidence")
        reasons_list = [r.strip() for r in str(tx_row['risk_reasons']).split(';') if r.strip()]
        for r in reasons_list:
            st.markdown(f"- **{r}**")

        st.markdown("---")

        # Two-Column Detailed Breakdown
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🌐 Network Telemetry & Offline GeoIP Enrichment")
            st.caption("Layer 3/4 P2P Broadcast Metadata & Local GeoIP Resolution")
            
            geo_status_badge = (
                "🟢 Resolved from MaxMind MMDB" if tx_row['geoip_lookup_status'] == "geoip_resolved"
                else "🟡 Offline Synthetic Fallback (Private IP)"
            )
            st.info(f"**GeoIP Status:** {geo_status_badge}")
            
            net_data = {
                "Source IP": tx_row['src_ip'],
                "Destination IP": tx_row['dst_ip'],
                "Source Port / Dest Port": f"{tx_row['src_port']} / {tx_row['dst_port']}",
                "Network Protocol": tx_row['protocol'],
                "Network Timestamp": str(tx_row['network_timestamp']),
                "Connection Duration": f"{tx_row['connection_duration_sec']} seconds",
                "Packet Count": f"{tx_row['packet_count']:,} packets",
                "Bytes Transferred": f"{tx_row['bytes_transferred']:,} bytes",
                "Source Country (GeoIP)": tx_row['geoip_src_country'],
                "Destination Country (GeoIP)": tx_row['geoip_dst_country'],
                "Source ASN / Org": tx_row['geoip_src_asn'],
                "Destination ASN / Org": tx_row['geoip_dst_asn']
            }
            st.table(pd.DataFrame(list(net_data.items()), columns=["Network Attribute", "Value"]))

        with col_right:
            st.subheader("⛓️ Blockchain Layer & UTXO Structure")
            st.caption("Layer 7 On-Chain State & Financial Transfer Metrics")
            
            chain_data = {
                "Block Height": str(tx_row['block_height']),
                "Block Timestamp": str(tx_row['timestamp']),
                "Source Wallet Entity": tx_row['source_wallet'],
                "Destination Wallet Entity": tx_row['destination_wallet'],
                "Total Transferred Volume": f"{tx_row['total_output_amount_btc']:.6f} BTC",
                "Miner Transaction Fee": f"{tx_row['fee_btc']:.6f} BTC",
                "Number of Inputs / Outputs": f"{tx_row['num_inputs']} inputs / {tx_row['num_outputs']} outputs",
                "Script Type": tx_row['script_type'],
                "Wallet Graph Degree": str(tx_row['wallet_degree']),
                "24-Hour Velocity": f"{tx_row['transaction_frequency_24h']} tx / 24h",
                "Average Time Gap": f"{tx_row['avg_time_gap_min']:.2f} minutes",
                "Associated Unique IPs": str(tx_row['unique_ip_count'])
            }
            st.table(pd.DataFrame(list(chain_data.items()), columns=["Blockchain Attribute", "Value"]))

        # Expanded UTXO Address & Amount Lists
        with st.expander("View Granular Participating UTXO Input/Output Addresses"):
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                st.write("**Input Addresses & Amounts:**")
                in_addrs = str(tx_row['input_addresses']).split('|')
                in_amts = str(tx_row['input_amounts']).split('|')
                in_df = pd.DataFrame({"Address": in_addrs, "Amount (BTC)": in_amts})
                st.dataframe(in_df, use_container_width=True, hide_index=True)
            with u_col2:
                st.write("**Output Addresses & Amounts:**")
                out_addrs = str(tx_row['output_addresses']).split('|')
                out_amts = str(tx_row['output_amounts']).split('|')
                out_df = pd.DataFrame({"Address": out_addrs, "Amount (BTC)": out_amts})
                st.dataframe(out_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 3: IP–TXID–WALLET CORRELATION GRAPH
# =============================================================================
elif page == "3. IP-TXID-Wallet Correlation":
    st.title("🕸️ IP–TXID–Wallet Multi-Modal Correlation")
    st.markdown(
        "Interactive topological correlation linking network vantage points (IPs) to "
        "cryptographic transactions (TXIDs) and economic entities (Wallets)."
    )

    g_col1, g_col2 = st.columns([1, 3])

    with g_col1:
        st.subheader("Subgraph Focus")
        focus_entity_type = st.radio("Focus Node Type", ["TXID", "Wallet", "IP"])
        
        if focus_entity_type == "TXID":
            default_val = scored_df[scored_df['risk_level'] == 'CRITICAL']['txid'].iloc[0]
            selected_focus = st.text_input("Enter TXID", value=default_val).strip()
        elif focus_entity_type == "Wallet":
            default_val = scored_df[scored_df['risk_level'] == 'CRITICAL']['source_wallet'].iloc[0]
            selected_focus = st.text_input("Enter Wallet ID", value=default_val).strip()
        else:
            default_val = scored_df[scored_df['risk_level'] == 'CRITICAL']['src_ip'].iloc[0]
            selected_focus = st.text_input("Enter IP Address", value=default_val).strip()

        hop_radius = st.slider("Graph Exploration Radius (Hops)", 1, 3, 2)
        max_nodes_display = st.slider("Max Nodes to Render", 10, 80, 40)

    with g_col2:
        # Extract subgraph
        sub_g = graph_engine.extract_subgraph(selected_focus, radius=hop_radius, max_nodes=max_nodes_display)

        if sub_g.number_of_nodes() == 0:
            st.warning(f"Entity '{selected_focus}' was not found in the transaction graph.")
        else:
            st.caption(f"Visualizing ego subgraph around **{selected_focus}** ({sub_g.number_of_nodes()} nodes, {sub_g.number_of_edges()} edges)")
            
            # Spring layout for 2D positioning
            pos = nx.spring_layout(sub_g, seed=42, k=0.5)

            # Node Traces by Type
            node_x_ip, node_y_ip, text_ip = [], [], []
            node_x_tx, node_y_tx, text_tx = [], [], []
            node_x_w, node_y_w, text_w = [], [], []

            for node, attr in sub_g.nodes(data=True):
                x, y = pos[node]
                n_type = attr.get('node_type', 'UNKNOWN')
                if n_type == 'IP':
                    node_x_ip.append(x)
                    node_y_ip.append(y)
                    text_ip.append(f"<b>IP:</b> {node}<br>Geo: {attr.get('country', '')} ({attr.get('asn', '')})")
                elif n_type == 'TXID':
                    node_x_tx.append(x)
                    node_y_tx.append(y)
                    text_tx.append(f"<b>TXID:</b> {node}<br>Amount: {attr.get('amount_btc', 0):.4f} BTC<br>Risk: {attr.get('risk_level', 'UNKNOWN')}")
                else:
                    node_x_w.append(x)
                    node_y_w.append(y)
                    text_w.append(f"<b>Wallet:</b> {node}")

            # Edge Trace
            edge_x = []
            edge_y = []
            for edge in sub_g.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig_graph = go.Figure()

            # Add Edges
            fig_graph.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.2, color="#64748b"),
                hoverinfo='none',
                mode='lines',
                name='Connections'
            ))

            # Add IP Nodes (Cyan)
            if node_x_ip:
                fig_graph.add_trace(go.Scatter(
                    x=node_x_ip, y=node_y_ip,
                    mode='markers+text',
                    textposition="top center",
                    hoverinfo='text',
                    text=[t.split('<br>')[0].replace('<b>IP:</b> ', '') for t in text_ip],
                    hovertext=text_ip,
                    marker=dict(size=20, color='#06b6d4', symbol='square', line=dict(width=2, color='#ffffff')),
                    name='IP Address'
                ))

            # Add TXID Nodes (Orange/Purple)
            if node_x_tx:
                fig_graph.add_trace(go.Scatter(
                    x=node_x_tx, y=node_y_tx,
                    mode='markers',
                    hoverinfo='text',
                    hovertext=text_tx,
                    marker=dict(size=16, color='#f97316', symbol='circle', line=dict(width=2, color='#ffffff')),
                    name='Transaction (TXID)'
                ))

            # Add Wallet Nodes (Emerald)
            if node_x_w:
                fig_graph.add_trace(go.Scatter(
                    x=node_x_w, y=node_y_w,
                    mode='markers+text',
                    textposition="top center",
                    hoverinfo='text',
                    text=[t.split('<br>')[0].replace('<b>Wallet:</b> ', '') for t in text_w],
                    hovertext=text_w,
                    marker=dict(size=18, color='#10b981', symbol='hexagon', line=dict(width=2, color='#ffffff')),
                    name='Wallet Entity'
                ))

            fig_graph.update_layout(
                template="plotly_dark",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=550
            )

            st.plotly_chart(fig_graph, use_container_width=True)

    # Detailed Correlation Metrics for Selected Focus
    st.subheader("Entity Correlation Profile")
    if focus_entity_type == "IP":
        ip_data = correlator.get_ip_correlations(selected_focus)
        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Total Transactions Relayed", ip_data.get('transaction_count', 0))
        cp2.metric("Distinct Wallets Broadcast", ip_data.get('unique_wallets_count', 0))
        cp3.metric("Total BTC Volume Broadcast", f"{ip_data.get('total_btc_volume', 0):.4f} BTC")
    elif focus_entity_type == "Wallet":
        w_data = correlator.get_wallet_correlations(selected_focus)
        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Total Wallet Transactions", w_data.get('total_transactions', 0))
        cp2.metric("Originating IP Vantage Points", w_data.get('unique_ip_count', 0))
        cp3.metric("Counterparty Wallets Reached", w_data.get('counterparty_count', 0))


# =============================================================================
# PAGE 4: GRAPH ANALYTICS & TOPOLOGY
# =============================================================================
elif page == "4. Graph Analytics & Topology":
    st.title("📊 Network Topology & Graph Analytics")
    st.markdown("Macro graph metrics, centrality distributions, and hub entity identification.")

    g_summary = graph_engine.get_graph_summary()
    
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    col_g1.metric("Total Graph Nodes", f"{g_summary['total_nodes']:,}")
    col_g2.metric("Total Directed Edges", f"{g_summary['total_edges']:,}")
    col_g3.metric("Unique IP Nodes", f"{g_summary['ip_nodes']:,}")
    col_g4.metric("Unique Wallet Nodes", f"{g_summary['wallet_nodes']:,}")

    st.markdown("---")

    # High Degree Entities
    high_deg = graph_engine.get_high_degree_entities(top_n=10)

    hd1, hd2 = st.columns(2)
    with hd1:
        st.subheader("Top High-Degree Broadcast IPs (Hub Vantage Points)")
        ip_deg_df = pd.DataFrame(high_deg['top_ips'])
        fig_ip_deg = px.bar(
            ip_deg_df,
            x='degree',
            y='entity',
            orientation='h',
            labels={'degree': 'Node Degree (Connections)', 'entity': 'IP Address'},
            color='degree',
            color_continuous_scale='tealgrn'
        )
        fig_ip_deg.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_ip_deg, use_container_width=True)

    with hd2:
        st.subheader("Top High-Degree Wallets (Financial Aggregation Hubs)")
        w_deg_df = pd.DataFrame(high_deg['top_wallets'])
        fig_w_deg = px.bar(
            w_deg_df,
            x='degree',
            y='entity',
            orientation='h',
            labels={'degree': 'Node Degree (Connections)', 'entity': 'Wallet ID'},
            color='degree',
            color_continuous_scale='purples'
        )
        fig_w_deg.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_w_deg, use_container_width=True)

    # Degree Distribution Histogram
    st.subheader("Wallet Degree & Connectivity Distribution")
    fig_deg_dist = px.histogram(
        scored_df,
        x='wallet_degree',
        color='risk_level',
        nbins=40,
        labels={'wallet_degree': 'Wallet Graph Degree', 'count': 'Transaction Count'},
        color_discrete_map={'LOW': '#22c55e', 'MEDIUM': '#eab308', 'HIGH': '#f97316', 'CRITICAL': '#ef4444'}
    )
    fig_deg_dist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_deg_dist, use_container_width=True)


# =============================================================================
# PAGE 5: AI ANOMALY ANALYSIS
# =============================================================================
elif page == "5. AI Anomaly Analysis":
    st.title("🤖 Unsupervised AI Anomaly Detection (Isolation Forest)")
    st.markdown(
        "In-depth analysis of the unsupervised machine learning engine isolating anomalous "
        "network behavior without relying on prior attack signatures or labels."
    )

    # Educational Rationale Panel
    with st.expander("ℹ️ Why Isolation Forest for Bitcoin Network Anomaly Detection?", expanded=True):
        st.markdown("""
        - **100% Unsupervised:** In real-world monitoring, new attack patterns (peeling chains, mixer bursts) lack pre-existing labels. Isolation Forest requires zero labeled training data.
        - **Isolation Principle:** Anomalous data points have rare attribute combinations and require fewer random recursive splits to isolate in feature space compared to dense normal clusters.
        - **Multi-Modal Fusion:** Jointly inspects network flow rates (packets, duration) and on-chain UTXO shape (degree, frequency, inputs/outputs).
        - **Offline & Performant:** Highly scalable with $O(n \\log n)$ time complexity, enabling local real-time inference on 10,000+ records in milliseconds.
        - **Zero Leakage:** Evaluated strictly after prediction without ever consuming `ground_truth` or `scenario` labels.
        """)

    # Score Distribution Histogram
    st.subheader("Normalized Anomaly Score Distribution (0.0 to 1.0)")
    fig_score_dist = px.histogram(
        scored_df,
        x='anomaly_score',
        color='is_anomaly',
        nbins=50,
        labels={'anomaly_score': 'Normalized Anomaly Score (Higher = More Anomalous)', 'is_anomaly': 'AI Flagged (1=Anomaly)'},
        color_discrete_map={0: '#10b981', 1: '#f43f5e'},
        marginal="box"
    )
    fig_score_dist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_score_dist, use_container_width=True)

    st.markdown("---")

    # Feature Separation Comparison (Normal vs Anomaly)
    st.subheader("Feature Discrepancy Analysis (Normal Baseline vs Anomaly)")
    feature_to_compare = st.selectbox(
        "Select Feature for Distribution Comparison",
        [
            "transaction_frequency_24h",
            "avg_time_gap_min",
            "wallet_degree",
            "unique_ip_count",
            "country_count",
            "packet_count",
            "bytes_transferred",
            "num_outputs"
        ]
    )

    fig_feat_box = px.box(
        scored_df,
        x='is_anomaly',
        y=feature_to_compare,
        color='is_anomaly',
        labels={'is_anomaly': 'Predicted Anomaly (0=Normal, 1=Anomaly)', feature_to_compare: feature_to_compare},
        color_discrete_map={0: '#10b981', 1: '#f43f5e'}
    )
    fig_feat_box.update_layout(template="plotly_dark")
    st.plotly_chart(fig_feat_box, use_container_width=True)


# =============================================================================
# PAGE 6: RANKED INVESTIGATION ALERTS
# =============================================================================
elif page == "6. Ranked Investigation Alerts":
    st.title("🚨 Prioritized Investigation Alert Feed")
    st.markdown("Ranked triage queue ordering suspicious transactions by composite risk score (0–100).")

    # Filter by Risk Severity
    r_col1, r_col2 = st.columns([1, 2])
    with r_col1:
        min_risk_level = st.selectbox("Minimum Alert Severity Level", ["MEDIUM", "HIGH", "CRITICAL", "LOW"], index=0)
    with r_col2:
        search_filter = st.text_input("Filter Alerts (by TXID, IP, or Wallet)", placeholder="Enter keyword...").strip()

    ranked_alerts = generate_ranked_alerts(scored_df, min_risk=min_risk_level)

    if search_filter:
        mask = (
            ranked_alerts['txid'].str.contains(search_filter, case=False, na=False) |
            ranked_alerts['src_ip'].str.contains(search_filter, na=False) |
            ranked_alerts['source_wallet'].str.contains(search_filter, case=False, na=False) |
            ranked_alerts['destination_wallet'].str.contains(search_filter, case=False, na=False)
        )
        ranked_alerts = ranked_alerts[mask]

    st.write(f"Displaying **{len(ranked_alerts):,}** prioritized alerts.")

    # CSV Download Button
    csv_data = ranked_alerts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Alerts to CSV",
        data=csv_data,
        file_name="bitcoin_investigation_alerts.csv",
        mime="text/csv"
    )

    # Styled Alerts Table
    st.dataframe(
        ranked_alerts,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                help="Composite Risk (0-100)",
                format="%d",
                min_value=0,
                max_value=100,
            ),
            "anomaly_score": st.column_config.NumberColumn(
                "AI Score",
                format="%.3f"
            ),
            "total_output_amount_btc": st.column_config.NumberColumn(
                "BTC Amount",
                format="%.4f"
            )
        }
    )


# =============================================================================
# PAGE 7: PROTOTYPE EVALUATION
# =============================================================================
elif page == "7. Prototype Evaluation":
    st.title("🎯 Prototype Evaluation & Ground Truth Benchmarks")
    
    st.warning(
        "🔒 **Evaluation Policy:** `ground_truth` and `scenario` are strictly used for "
        "post-prediction validation and benchmarking. They were NEVER provided to the "
        "Isolation Forest model during training or feature engineering."
    )

    # Top Metric Tiles
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
        
        # Display Confusion Matrix as Interactive Heatmap
        cm_matrix = [
            [cm_data['true_negatives'], cm_data['false_positives']],
            [cm_data['false_negatives'], cm_data['true_positives']]
        ]
        
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Count"),
            x=["Normal (0)", "Anomaly (1)"],
            y=["Normal (0)", "Anomaly (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(f"""
        - **True Positives (TP):** {cm_data['true_positives']:,}
        - **True Negatives (TN):** {cm_data['true_negatives']:,}
        - **False Positives (FP):** {cm_data['false_positives']:,}
        - **False Negatives (FN):** {cm_data['false_negatives']:,}
        """)

    with c_sc:
        st.subheader("Scenario-Wise Detection Rate Breakdown")
        scenario_df = eval_results['scenario_breakdown']
        
        fig_scenario = px.bar(
            scenario_df[scenario_df['Scenario'] != 'normal'],
            x='Detection Rate (%)',
            y='Scenario',
            orientation='h',
            color='Avg Risk Score',
            color_continuous_scale='Reds',
            text='Detection Rate (%)',
            labels={'Scenario': 'Anomaly Scenario'}
        )
        fig_scenario.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_scenario, use_container_width=True)

    st.subheader("Detailed Scenario Benchmark Table")
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)
