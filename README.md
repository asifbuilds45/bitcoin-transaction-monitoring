# AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
### NTRO Problem Statement ID: 26146 | Air-Gapped Offline Cybersecurity Workstation

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python)
![Air-Gapped](https://img.shields.io/badge/Environment-100%25%20Offline%20Air--Gapped-green?logo=shield)
![Machine Learning](https://img.shields.io/badge/ML-Isolation%20Forest%20%2B%20HDBSCAN-orange)
![Graph Intelligence](https://img.shields.io/badge/Graph-Leiden%20Modularity%20%2B%20NetworkX-purple)
![Explainability](https://img.shields.io/badge/XAI-TreeSHAP%20Waterfall-red)
![Desktop App](https://img.shields.io/badge/Deployment-Windows%20Desktop%20%2B%20Linux%20CLI-cyan)
![Reporting](https://img.shields.io/badge/Dossiers-ReportLab%20Vector%20PDF-brightgreen)

---

## 1. Problem Statement & Operational Context

### Problem Title
**AI-Powered Monitoring and Analysis of Bitcoin Transaction Traffic** (NTRO Problem Statement ID: 26146).

### Operational Background & Threat Landscape
The Bitcoin peer-to-peer network operates as a decentralized, pseudo-anonymous financial layer. While cryptographic transaction ledgers are publicly auditable on-chain, transaction broadcasts originate at the network layer (OSI Layer 3/4) via P2P TCP gossip protocols (ports 8333, 18333, 18444). 

Adversaries, ransomware cartels, illicit mixing syndicates, and botnets exploit this dual-layer separation by utilizing:
- **Multi-Hop Proxying & VPN Dispersion:** Broadcasting transactions across disparate autonomous systems to obscure origin.
- **Rapid Peeling Chains:** Splitting small amounts to payees while sequentially hopping the bulk change across temporary wallets.
- **CoinJoin / Denomination Equalization:** Obfuscating inputs/outputs in high-volume mixing pools.
- **Fan-Out / Fan-In Distribution:** Rapid splitting and consolidation to foil classical heuristic tracing.

### Project Objective
A self-contained, **100% offline air-gapped forensic workstation** that ingests bulk network telemetry and blockchain metadata, performs multi-modal correlation ($IP \leftrightarrow TXID \leftrightarrow Wallet$), applies dual-layer unsupervised machine learning (**Isolation Forest + HDBSCAN**), executes **Leiden** community graph partitioning, reconstructs multi-hop fund-flow trails, groups alerts into cohesive investigative cases, and generates court-admissible vector PDF dossiers.

---

## 2. Complete Technology Stack

| Layer | Technology | Version | Purpose & Implementation |
| :--- | :--- | :--- | :--- |
| **Core Runtime** | Python | 3.10 – 3.14 | Strictly offline, zero cloud calls, multi-threaded analytical pipeline. |
| **Anomaly Detection** | Scikit-Learn | $\ge 1.3.0$ | **Isolation Forest** unsupervised tree isolation on 15+ behavioral features. |
| **Behavioural Clustering** | HDBSCAN / Scikit-Learn | $\ge 0.8.33$ | **Hierarchical Density-Based Clustering** with GLOSH outlier scoring. |
| **Graph Intelligence** | NetworkX & python-igraph | $\ge 3.0 / \ge 0.10$ | Directed bipartite/tripartite entity graphs ($IP \leftrightarrow TXID \leftrightarrow Wallet$). |
| **Community Detection** | Leidenalg / igraph | $\ge 0.9.0$ | **Leiden Modularity Partitioning** (guaranteed internal node connectivity). |
| **Graph DB Integration** | Neo4j Bolt Driver | $\ge 5.0$ | Batched Cypher `UNWIND` persistence with in-memory NetworkX failover. |
| **Explainable AI (XAI)** | SHAP (TreeSHAP) | $\ge 0.44.0$ | Exact Shapley feature contribution values & beeswarm/waterfall plots. |
| **Large-Scale Visualization** | Datashader / Colorcet | $\ge 0.16.0$ | Offline server-side 2D rasterization of dense telemetry points without lag. |
| **GeoIP / Network Vantage**| MaxMind GeoLite2 (`.mmdb`) | Local DBs | Offline Country & ASN resolution with strict non-fabrication of private IPs. |
| **Databases & Persistence** | PostgreSQL / SQLAlchemy | 16/18 / $\ge 2.0$ | Enterprise `QueuePool` connection pooling with automatic fallback to **SQLite3**. |
| **Forensic PDF Dossiers** | ReportLab | $\ge 4.0$ | Deterministic, vector-graphic, multi-page forensic threat dossiers. |
| **Primary Forensic UI** | Streamlit | $\ge 1.30.0$ | 10-Tab client-side dark-themed investigative analyst workstation. |
| **REST API Engine** | FastAPI & Uvicorn | $\ge 0.104$ | Asynchronous REST service exposing forensic pipelines for air-gapped SOCs. |
| **Offline Desktop App** | Windows WebView2 / Edge | Native | Standalone frameless application mode (`.exe` / `.bat` / `.vbs` launchers). |
| **Offline AI Assistant** | Custom Knowledge Engine | Native | Rule-based & local fallback LLM knowledge retriever for natural language Q&A. |
| **Test Automation** | Pytest | $\ge 7.4$ | Comprehensive air-gapped readiness, refresh, and pattern detection tests. |

---

## 3. System Architecture & Evidence Flow

```
                      ┌────────────────────────────────────────────────────────┐
                      │    Raw Multi-Format Ingestion (CSV / JSON / XML)       │
                      │  • Network Telemetry: P2P Gossip, TCP 8333, Packets    │
                      │  • Blockchain Ledger: UTXO Inputs/Outputs, Fees, Blocks│
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                      ┌────────────────────────────────────────────────────────┐
                      │     Offline GeoIP & Autonomous System (ASN) Vantage    │
                      │        (MaxMind GeoLite2-Country & GeoLite2-ASN)       │
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                      ┌────────────────────────────────────────────────────────┐
                      │        IP ⟷ TXID ⟷ Wallet Multi-Modal Correlation      │
                      │      (Normalized Mathematical Confidence Score: 0-100%)│
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                      ┌────────────────────────────────────────────────────────┐
                      │         NetworkX / Leiden Graph Intelligence           │
                      │   • Directed Entity Graph Construction                 │
                      │   • Leiden Algorithm: High-Density Modularity Partition│
                      │   • Degree Centrality & High-Connectivity Hub Profiling│
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                      ┌────────────────────────────────────────────────────────┐
                      │    Dual AI/ML Detection & Behavioural Pattern Layer    │
                      │   • Isolation Forest: Tree-Based Anomaly Scoring       │
                      │   • HDBSCAN: Hierarchical Density & GLOSH Noise Points │
                      │   • Peeling-Chain Traversal & CoinJoin Mixer Detectors │
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                      ┌────────────────────────────────────────────────────────┐
                      │          Multi-Layer Evidence Fusion Engine            │
                      │    11 Independent Normalized Evidence Channels (0-100) │
                      │         + TreeSHAP Mathematical Feature Attribution    │
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                      ┌────────────────────────────────────────────────────────┐
                      │         Investigator Case Triage & Action Layer        │
                      │   • Case Grouping & Alert Deduplication (CASE-001...)  │
                      │   • Chronological Forensic Evidence Timelines          │
                      │   • Bounded Multi-Hop Investigation Path Reconstruction│
                      │   • 1-Click Forensic Security Threat PDF Export        │
                      └────────────────────────────────────────────────────────┘
```

### 11-Channel Evidence Fusion Formula
Composite risk scores ($0 - 100$) are calculated via deterministic multi-channel fusion:

$$\text{Composite Risk} = \sum_{i=1}^{11} w_i \times E_i$$

1. **Isolation Forest ML Score ($w_1 = 0.18$):** Unsupervised tree isolation depth.
2. **HDBSCAN Density Clustering ($w_2 = 0.09$):** GLOSH noise outlier evidence ($\text{cluster} = -1$).
3. **Leiden Community Risk ($w_3 = 0.09$):** Membership in elevated-risk modular graph partitions.
4. **Peeling-Chain Evidence ($w_4 = 0.10$):** Multi-hop consecutive volume stripping indicators.
5. **CoinJoin / Mixing Evidence ($w_5 = 0.09$):** Equal-denomination output clustering.
6. **Temporal Velocity Evidence ($w_6 = 0.11$):** Ultra-short inter-hop time gaps ($\le 120\text{s}$) and burst spikes.
7. **Behavioural Multi-IP Dispersion ($w_7 = 0.11$):** Broadcast hopping across multiple proxy/VPN IPs.
8. **Graph Hub Degree Centrality ($w_8 = 0.08$):** Hub connectivity degree in the entity transaction graph.
9. **Geo/ASN Vantage Dispersion ($w_9 = 0.07$):** Rapid broadcasts across international jurisdictions.
10. **UTXO Transfer Volume & Fee Skew ($w_{10} = 0.08$):** Disproportionate transaction values and miner fee ratios.
11. **Network Packet Burst Telemetry ($w_{11} = 0.00$):** Baseline network packet volume alignment.

---

## 4. 10-Tab Forensic Analyst Workbench

The Streamlit workstation (`app.py`) provides an instant-response, 10-tab forensic interface:

```
[📥 0. Ingestion]  [📊 1. Overview]  [🔍 2. Investigation]  [🕸️ 3. Correlation]  [📈 4. Graph Analytics]
[🤖 5. AI Anomaly] [🚨 6. Ranked Alerts] [🎯 7. Evaluation] [💬 8. Assistant]   [🔗 9. Patterns]
```

### Tab 0: Two-Layer Dataset Ingestion & Multi-Format Support
- Ingests **Network-Layer** (P2P gossip telemetry) and **Blockchain-Layer** (UTXO on-chain transactions) independently.
- Native parsing of **CSV**, **JSON**, and **XML** formats with independent schema validation.
- Multi-dataset accumulation mode allows uploading multiple files sequentially without overwriting previous evidence.

### Tab 1: Executive Overview & SOC KPIs
- High-level telemetry metrics: Total Monitored Transactions, Detected Anomalies, Critical Risk Alerts, and Mean Correlation Confidence.
- Global Choropleth Map visualizing broadcast vantage points across countries and Autonomous Systems (ASNs).
- Temporal anomaly burst velocity chart over rolling 7-day intervals.

### Tab 2: Forensic Transaction Deep-Dive & Timeline
- Complete transaction dossier for any selected TXID: UTXO inputs/outputs, fee ratios, and GeoIP vantage details.
- **TreeSHAP Waterfall Plot:** Displays the mathematical positive/negative feature contributions driving the risk score.
- **Chronological Evidence Timeline:** Step-by-step forensic progression from network packet broadcast to blockchain confirmation.
- **1-Click PDF Generation:** Instant download of a court-admissible forensic security threat report.

### Tab 3: IP–TXID–Wallet Multi-Modal Correlation
- Maps the cross-layer linkages connecting network-layer observations to on-chain financial entities.
- Computes the **Normalized Correlation Confidence Score** (0–100%) factoring cryptographic TXID alignment (45%), temporal proximity $\le 60$s (30%), P2P TCP port adherence (15%), and vantage repeatability (10%).
- Enforces strict **Non-Attribution Semantics**: explicitly documents that IP vantage points reflect network broadcast observation, not verified wallet ownership.

### Tab 4: Interactive Graph Analytics & Leiden Communities
- Directed tripartite NetworkX entity graph visualizing financial fund-flows and network broadcasts.
- **Leiden Community Partitioning:** Automatically clusters connected entities into modular communities while guaranteeing internal subgraph connectivity.
- High-degree entity identification (detecting mixing hubs, peel routers, and high-connectivity nodes).

### Tab 5: AI Anomaly Analysis (Dual-ML Architecture)
- **Isolation Forest:** Distribution histograms of continuous anomaly scores and anomaly decision boundaries.
- **HDBSCAN Behavioural Clustering:** Hierarchical cluster membership bar charts, identifying dense behavioral groups and unclustered noise points.

### Tab 6: Prioritized Ranked Alerts & Automated Case Grouping
- **Ranked Investigation Queue:** Live triage feed sorted strictly by composite risk score descending, with exact risk-tier filtering (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
- **Automated Case Grouping (`CASE-001`, `CASE-002`, ...):** Eliminates alert fatigue by deterministically consolidating strongly connected transactions sharing identical wallets, sequential peeling flows, or temporal IP correlations.
- **Case Consolidated Timeline & Path:** Reconstructs the end-to-end multi-hop trail across all transactions in a case.
- **Unified PDF Dossier Export:** Batch-generates forensic PDF reports for all anomalous cases with a single click.

### Tab 7: Benchmark Evaluation & Ground Truth Validation
- Comprehensive evaluation metrics against synthetic ground truth: Precision, Recall, F1-Score, and ROC-AUC.
- Detailed scenario detection matrix across 8 attack patterns (`rapid_chain`, `fan_out`, `fan_in`, `cross_border_burst`, `high_frequency`, `high_connectivity`, `ip_wallet_reuse`, `layered_transfer`).

### Tab 8: Offline AI Investigation Assistant
- 100% air-gapped conversational intelligence powered by local rule-based knowledge retrieval and local LLM fallbacks.
- Allows investigators to query: *"Why is this IP address suspicious?"*, *"Explain the risk breakdown for TXID X"*, or *"Summarize CASE-001"*.

### Tab 9: Behavioural Patterns (Peeling Chains & CoinJoin Mixers)
- Dedicated heuristic telemetry identifying multi-hop fund-peeling chains and equal-denomination CoinJoin mixing pools.
- Visual breakdown of peeling chain hop indices, remaining change amounts, and mixer participant counts.

---

## 5. Desktop Application Packaging (100% Offline)

For deployment on classified, air-gapped workstations (e.g., NTRO / Law Enforcement forensic labs), the workstation runs as a native desktop application with dedicated window controls, without browser tabs or URL bars:

```
┌────────────────────────────────────────────────────────┐
│  🛡️ NTRO - AI Bitcoin Forensic Workstation  [-] [□] [X]│
├────────────────────────────────────────────────────────┤
│  [📥 Ingestion] [📊 Overview] [🔍 Investigation] ...   │
│                                                        │
│  [ Full Screen Native Air-Gapped Forensic Desktop ]    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Dedicated Launchers Provided:
- [**`launch_desktop.bat`**](file:///c:/Users/shrey/OneDrive/Desktop/Bitcoin%20Monitoring/launch_desktop.bat): One-click Windows desktop launcher with automatic port detection and HTTP readiness handshake.
- [**`launch_silent.vbs`**](file:///c:/Users/shrey/OneDrive/Desktop/Bitcoin%20Monitoring/launch_silent.vbs): Launches the desktop application completely silently without leaving a black terminal window open.
- [**`create_desktop_shortcut.bat`**](file:///c:/Users/shrey/OneDrive/Desktop/Bitcoin%20Monitoring/create_desktop_shortcut.bat): Automatically places a branded shortcut titled **`NTRO Bitcoin Forensic Workstation`** on the Windows Desktop.
- [**`launch_linux.sh`**](file:///c:/Users/shrey/OneDrive/Desktop/Bitcoin%20Monitoring/launch_linux.sh): Shell script for Ubuntu / Debian / RHEL workstations with execution permissions.
- [**`desktop_app.py`**](file:///c:/Users/shrey/OneDrive/Desktop/Bitcoin%20Monitoring/desktop_app.py): Core Python wrapper supporting dual-mode: Win32 native `pywebview` or native frameless Chromium/Edge app-mode.

---

## 6. Quickstart & Installation Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14.
- Git.
- Optional: PostgreSQL 16+ and Neo4j (system automatically falls back to SQLite3 and NetworkX if not installed).

### Installation
```bash
# Clone the repository
git clone https://github.com/asifbuilds45/bitcoin-transaction-monitoring.git
cd bitcoin-transaction-monitoring

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Running the System

#### Option 1: Native Desktop Application (Windows)
Double-click [**`launch_desktop.bat`**](file:///c:/Users/shrey/OneDrive/Desktop/Bitcoin%20Monitoring/launch_desktop.bat) or run:
```powershell
python desktop_app.py
```

#### Option 2: Linux / Ubuntu Workstation
```bash
chmod +x launch_linux.sh
./launch_linux.sh
```

#### Option 3: Standard Browser Streamlit Execution
```bash
streamlit run app.py
```

#### Option 4: Full Enterprise Stack (FastAPI + Static Frontend + Streamlit)
```powershell
# Windows PowerShell:
.\start_offline_system.ps1

# Linux Bash:
./start_offline_system.sh
```

---

## 7. Air-Gapped Test Suite & Verification

The codebase includes an automated test suite verifying that all modules function 100% offline without external network dependencies:

```bash
# Run comprehensive air-gapped readiness test suite
pytest tests/test_offline_readiness.py -v
```

### Automated Verification Coverage (8/8 Passed):
- `test_01_database_postgresql_persistence_and_indexes`: PostgreSQL connection pool & SQLite fallback.
- `test_02_hdbscan_clustering`: HDBSCAN fit, predict, and GLOSH outlier scoring.
- `test_03_leiden_graph_community_detection`: Leiden partition quality and guaranteed connectivity.
- `test_04_neo4j_manager_fallback`: Neo4j Bolt driver connectivity & graceful NetworkX fallback.
- `test_05_treeshap_explainability`: TreeSHAP surrogate explainer & waterfall attribution.
- `test_06_datashader_renderer`: Offline 2D density rasterization.
- `test_07_fastapi_endpoints_and_react_static`: FastAPI endpoints & static frontend serving.
- `test_08_offline_ai_assistant`: Knowledge retrieval engine without cloud API connectivity.

---

## 8. Repository Structure

```text
bitcoin-transaction-monitoring/
├── app.py                             # Primary 10-Tab Forensic Streamlit Workstation
├── desktop_app.py                     # Native Desktop Application Runner
├── launch_desktop.bat                 # Windows One-Click Batch Launcher
├── launch_silent.vbs                  # Windows Silent Background Launcher
├── launch_linux.sh                    # Linux / Ubuntu Executable Launcher
├── create_desktop_shortcut.bat        # Automated Desktop Icon Creator
├── requirements.txt                   # Complete Python dependencies
├── README.md                          # Comprehensive Technical Documentation
├── data/
│   ├── network_traffic_dataset.csv    # 10,000-record Network Telemetry Benchmark
│   ├── blockchain_transactions_dataset.csv # 10,000-record Blockchain UTXO Benchmark
│   ├── bitcoin_monitoring.db          # Offline SQLite3 Database
│   └── geoip/
│       ├── GeoLite2-Country.mmdb      # Offline Country Database (MaxMind)
│       └── GeoLite2-ASN.mmdb          # Offline Autonomous System Database (MaxMind)
├── frontend/                          # Offline React / HTML5 Cyber Dashboard
│   ├── index.html                     # Standalone Zero-CDN Air-Gapped UI
│   └── package.json
├── src/
│   ├── api/                           # FastAPI REST Engine (src/api/main.py)
│   ├── assistant/                     # Offline AI Assistant & Knowledge Retriever
│   ├── behavioural/                   # Behavioural Fingerprinting & Heuristics
│   ├── clustering/                    # HDBSCAN & DBSCAN Clustering Engines
│   │   ├── hdbscan_clustering.py      # HDBSCAN + GLOSH Outlier Engine
│   │   └── dbscan_clustering.py       # Fallback Density Clustering
│   ├── correlation/                   # IP-TXID-Wallet Correlation & Confidence Scorer
│   ├── database/                      # SQLAlchemy & Database Manager (PostgreSQL/SQLite)
│   ├── explainability/                # TreeSHAP Surrogate Explainer
│   ├── fusion/                        # 11-Channel Evidence Fusion Engine
│   ├── geoip_enrichment.py            # Local MaxMind MMDB Parser
│   ├── graph/                         # Graph Analytics & Community Detection
│   │   ├── graph_engine.py            # NetworkX Directed Entity Graph
│   │   ├── leiden_community.py        # Leiden Modularity Partitioning Engine
│   │   └── neo4j_manager.py           # Neo4j Bolt Sync & Fallback
│   ├── ingestion/                     # Multi-Format Ingestors (CSV, JSON, XML)
│   ├── investigation/                 # Case Grouping, Paths & Timeline Builders
│   ├── patterns/                      # Peeling Chain & CoinJoin Detectors
│   ├── reporting/                     # ReportLab Forensic PDF Dossier Generator
│   ├── risk_scoring.py                # Holistic 0-100 Risk Engine & Ranked Alert Queue
│   └── visualization/                 # Datashader Offline Rasterization Engine
└── tests/                             # Automated Air-Gapped Test Suite
    ├── test_offline_readiness.py      # Core 8-Point Air-Gapped Verification
    ├── test_dataset_refresh.py        # Multi-Dataset Ingestion Tests
    ├── test_pattern_detection.py      # Peeling Chain & CoinJoin Tests
    └── test_pipeline.py               # End-to-End Pipeline Regression Tests
```

---

## 9. Research & Defensive Compliance Disclaimer

> [!IMPORTANT]
> **Defensive Cybersecurity & Intelligence Scope:**
> Developed strictly for defensive cybersecurity monitoring, forensic anomaly detection, and lawful financial transaction analysis under **NTRO Problem Statement ID: 26146**. All analyses are performed on local synthetic datasets and offline MaxMind databases with zero external network connectivity. In adherence to forensic best practices, all system outputs (IP linkages, behavioural clusters, and pattern detections) are classified strictly as **investigative leads and corroborating evidence**, maintaining strict non-attribution principles for court and regulatory admissibility.

---
**Author / Team:** asifbuilds45  
**Project:** AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic  
**Repository:** [https://github.com/asifbuilds45/bitcoin-transaction-monitoring](https://github.com/asifbuilds45/bitcoin-transaction-monitoring)
