# AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
### NTRO Problem Statement ID: 26146 | Offline Cybersecurity Prototype

---

## 1. Problem Statement & Background

### Context & Challenge
The Bitcoin peer-to-peer network operates as a decentralized, pseudo-anonymous financial layer. While cryptographic transaction ledgers are publicly auditable on-chain, transaction broadcasts originate at the network layer (OSI Layer 3/4) via P2P TCP gossip protocols (ports 8333, 18333, 18444). 

Threat actors, illicit mixing services, ransomware operators, and botnets exploit this dual-layer separation by utilizing multi-hop proxying, VPN dispersion, rapid peeling chains, fan-out/fan-in splitting, and automated high-frequency transfer patterns.

### Objective
This prototype provides an **offline, AI-driven cybersecurity investigation system** that:
1. Ingests bulk Bitcoin network telemetry and on-chain UTXO metadata.
2. Performs offline geographic and ASN enrichment on IP addresses via local MaxMind GeoLite2 databases.
3. Constructs multi-modal entity correlation graphs ($IP \leftrightarrow TXID \leftrightarrow Wallet$).
4. Applies unsupervised Machine Learning (**Isolation Forest**) on behavioral features without relying on labeled attack data.
5. Computes multi-factor explainable risk scores ($0 - 100$) with natural-language investigative evidence.
6. Presents a prioritized triage queue and interactive forensic investigation dashboard via Streamlit and Plotly.

---

## 2. System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │  Raw Synthetic Master Dataset (data/*.csv)   │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Data Ingestion & Schema Preprocessing      │
                       │          (src/preprocessing.py)              │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Offline GeoIP & ASN Vantage Enrichment     │
                       │         (src/geoip_enrichment.py)            │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   IP-TXID-Wallet Multi-Modal Correlation     │
                       │           (src/correlation.py)               │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │       NetworkX Directed Entity Graph         │
                       │          (src/graph_analysis.py)             │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │    Behavioral & Graph Feature Engineering    │
                       │        (src/feature_engineering.py)          │
                       │    [Strictly Unsupervised — No Leakage]      │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │     Isolation Forest Anomaly Detection       │
                       │        (src/anomaly_detection.py)            │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │  Multi-Factor Risk Engine (0-100) & Evidence │
                       │           (src/risk_scoring.py)              │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │  Prioritized Alert Queue & Forensic Triage   │
                       │         Streamlit Dashboard (app.py)         │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │      Post-Prediction Benchmark Evaluation    │
                       │            (src/evaluation.py)               │
                       └──────────────────────────────────────────────┘
```

---

## 3. Dataset Description

The dataset comprises **10,000 synthetic records** reflecting realistic Bitcoin network and ledger interactions:
- **9,000 Normal Baseline Transactions** (90%)
- **1,000 Anomalous Transactions** (10%) spanning 8 distinct attack/adversarial patterns (125 records each):
  1. `rapid_chain`: High-velocity sequential wallet transfers with minimal inter-hop time gaps.
  2. `fan_out`: Aggressive splitting of funds into numerous destination addresses (mixer distribution).
  3. `fan_in`: Sudden consolidation of micro-utxos from multiple source wallets.
  4. `cross_border_burst`: Transaction broadcasts hopping across multiple countries/ASNs in rapid bursts.
  5. `high_frequency`: Botnet/automated transaction spikes within a short 24-hour window.
  6. `high_connectivity`: High-degree hub wallets acting as central aggregation nodes.
  7. `ip_wallet_reuse`: Unrelated wallet clusters originating from identical proxy/relay IP vantage points.
  8. `layered_transfer`: Multi-hop obfuscation chains attempting to obscure origin of funds.

### Key Fields:
- **Network Fields:** `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `network_timestamp`, `connection_duration_sec`, `packet_count`, `bytes_transferred`.
- **Blockchain Fields:** `txid`, `timestamp`, `block_height`, `input_addresses`, `output_addresses`, `input_amounts`, `output_amounts`, `total_input_amount_btc`, `total_output_amount_btc`, `fee_btc`, `num_inputs`, `num_outputs`, `script_type`, `source_wallet`, `destination_wallet`.
- **Behavioral & Graph Fields:** `transaction_frequency_24h`, `avg_time_gap_min`, `wallet_degree`, `unique_ip_count`, `country_count`, `asn_count`.
- **Post-Prediction Evaluation Fields:** `ground_truth`, `scenario`.

> **Strict Machine Learning Separation:** `ground_truth` and `scenario` are strictly excluded from data ingestion, feature vectors, and model fitting. They are used exclusively in the post-prediction evaluation module.

---

### 4. Two-Dataset Ingestion & Schema Validation

The prototype ingests and validates two independent datasets:
1. **Network-Layer Data:** P2P gossip telemetry, source/destination IPs, ports, connection duration, packet counts, bytes.
2. **Blockchain-Layer Data:** On-chain ledger blocks, UTXO inputs/outputs, miner fees, and wallet addresses.

**Multi-Format Support:** Ingestion modules (`src/ingestion/`) support **CSV**, **JSON**, and **XML** formats natively without internet access. Both datasets must be independently schema-validated before investigation execution is enabled.

---

## 5. Offline GeoIP & Network Enrichment

Offline MaxMind GeoIP2/GeoLite2 integration (`src/geoip_enrichment.py`):
- **Local Databases:** `data/geoip/GeoLite2-Country.mmdb` and `data/geoip/GeoLite2-ASN.mmdb`.
- **Network Isolation:** Operates 100% locally with zero internet connections or external API calls.
- **Strict Non-Fabrication:** Private, loopback, link-local, and synthetic IP ranges (e.g. `10.x.x.x`, `192.168.x.x`, `172.16-31.x.x`, `127.0.0.1`) strictly resolve to `Country = Unknown` and `ASN = Unknown`.
- **Graceful Fallback:** If MMDB files are absent, the application falls back safely without runtime errors.

---

## 6. Network–Blockchain Correlation & Confidence Score

The correlation engine (`src/correlation/correlator.py`) establishes relationships:
- $\text{IP} \rightarrow \text{TXID}$
- $\text{TXID} \rightarrow \text{Input/Output Wallets}$
- $\text{Wallet} \rightarrow \text{Wallet (Fund-flows)}$
- $\text{IP} \rightarrow \text{Wallet (Network vantage point association)}$

> **Non-Attribution Principle:** The system strictly treats IP addresses as *network observations associated with transaction/wallet activity*, never claiming that an IP address identifies the wallet owner.

### Normalized Correlation Confidence Score (0.0 to 1.0 / 0–100%)
A mathematical confidence measure distinct from anomaly and risk scores:
1. **Exact TXID Match (45%):** Direct cryptographic transaction identifier alignment between telemetry and block data.
2. **Temporal Proximity (30%):** Proximity between network packet broadcast timestamp and blockchain block confirmation timestamp ($\le 60\text{s} \rightarrow 30\%$, $\le 300\text{s} \rightarrow 25\%$, $\le 1800\text{s} \rightarrow 15\%$).
3. **P2P Protocol Consistency (15%):** Verification of Bitcoin P2P protocol compliance and standard listening ports (8333, 18333, 38333) over TCP.
4. **Observation Repeatability (10%):** Multi-burst verification across repeated network vantage points.

---

## 7. Dual AI/ML & Graph Community Analytics

### 1. Isolation Forest (Unsupervised Anomaly Detection)
Identifies multi-dimensional behavioral and network anomalies without requiring labeled training datasets. Produces a continuous anomaly score $\in [0.0, 1.0]$.

### 2. DBSCAN (Behavioural Clustering & Noise Discovery)
Partitions transaction traffic into dense behavioral clusters using scaled feature space. Identifies unclustered noise points ($\text{cluster} = -1$) to detect isolated, aberrant transaction behaviors.

### 3. Louvain Modularity Community Detection
NetworkX directed entity graph partitions IP, TXID, and Wallet nodes into modular graph communities, calculating community density, entity counts, and internal link density.

### 4. Multi-Layer Evidence Fusion & Risk Scoring (0–100)
Combines 9 independent normalized evidence channels:
- Isolation Forest Anomaly Evidence (20%)
- DBSCAN Clustering & Noise Evidence (10%)
- Louvain Graph Community Modularity (10%)
- Temporal Velocity Evidence (12%)
- Behavioural Dispersion Evidence (13%)
- Graph Degree Centrality (10%)
- Geo/ASN Multi-Vantage Evidence (8%)
- Blockchain UTXO Volume & Fee Evidence (9%)
- Network Telemetry Burst Evidence (8%)

---

## 8. Investigator-Centric Capabilities

### 1. Investigation Path Reconstruction (`src/investigation/path_reconstruction.py`)
Rather than presenting isolated alerts, the system reconstructs the connected multi-hop investigation trail:
$$\text{Source IP} \xrightarrow{\text{network observation}} \text{TXID} \xrightarrow{\text{input / output}} \text{Wallet} \xrightarrow{\text{fund-flow}} \text{Related Wallet} \xrightarrow{\text{connected transaction}} \text{Next TXID}$$
- **Configurable Traversal Depth:** Bounded search depth (1 to 5 hops, default 3) to prevent UI freezing and graph explosion.
- **Priority Ranking:** Ranks paths by composite transaction risk, transfer amounts, and connection relevance.
- **Strict Non-Attribution Semantics:** Treats IP addresses strictly as *network observations associated with transaction broadcast activity*, never asserting wallet ownership.
- **UI Location:** Accessible in Tab 2 (Forensic Transaction Deep-Dive) and within each Case Drill-Down view.

### 2. Forensic Evidence Timeline (`src/investigation/timeline_builder.py`)
Reconstructs a strictly chronological activity sequence answering: *"What happened first? What happened next? When did suspicious behaviour emerge?"*
- **Actual Timestamps Only:** Built strictly from real dataset timestamps (`network_timestamp`, `timestamp`). No fabricated or synthetic system times.
- **Event Types Captured:**
  - `NETWORK_OBSERVATION`: P2P gossip broadcast observed at source IP vantage point.
  - `TRANSACTION_OBSERVED`: On-chain confirmation with block height, transfer volume, and miner fee.
  - `INPUT_FUNDING`: Source wallet providing input UTXOs.
  - `OUTPUT_DISPERSAL`: Transaction dispersing funds to destination wallet.
  - `BEHAVIOURAL_CLUSTERING`: Unsupervised DBSCAN cluster membership or sparse noise detection (-1).
  - `INVESTIGATION_ALERT`: Multi-layer fusion generating High/Critical risk alerts with key forensic signals.
- **Case Timeline Aggregation:** Chronologically merges events across multiple related transactions in an incident.
- **UI Location:** Accessible in Tab 2 (Forensic Transaction Deep-Dive) and within each Case Drill-Down view.

### 3. Alert Deduplication & Case Grouping (`src/investigation/case_grouping.py`)
Solves alert fatigue by consolidating strongly connected anomalous transactions into cohesive investigation cases (`CASE-001`, `CASE-002`, ...):
- **Deterministic Grouping Evidence:**
  1. *Shared Source / Destination Wallets:* Multiple anomalous transactions utilizing identical wallet entities.
  2. *Sequential Fund-Flow Chains:* Direct on-chain linkage where the output of transaction A feeds the input of transaction B.
  3. *Temporal Network Correlation:* Transactions broadcast from the same network IP within a tight temporal window ($\le 2$ hours).
  4. *Dense Behavioural DBSCAN Clusters:* Common membership in dense behavioural clusters within close temporal proximity ($\le 4$ hours).
- **Linear-Time Graph Clustering:** $O(N)$ star and temporal path construction across inverted indices for instant sub-second grouping.
- **Stable Session Case IDs:** Deterministically sorted by maximum risk score, transaction count, and volume.
- **Investigation Cases Dashboard:** Dedicated view in Tab 6 providing case summary KPIs, multi-transaction filtering, grouped transaction tables, case path reconstruction, and case evidence timelines.

---


## 9. Forensic PDF Threat Report Generation

Built with **ReportLab**, the system generates comprehensive forensic security threat reports directly from Tab 6:
- **Single PDF, Complete Anomaly Coverage:** Generates one unified PDF containing individual forensic reports for all anomalous transactions.
- **Filter Independence:** Always includes the complete set of anomalies regardless of the currently active UI risk filter.
- **Structured Forensic Template:** Each transaction includes:
  - **PROFILE:** Report ID, TXID, Risk Level, Date, Time.
  - **ACTORS:** Source IP, Destination IP, Wallet, Country (with ISO code), ASN (with Organization).
  - **DIAGNOSTIC:** Data-driven "Why Anomalous" evidence, Anomaly Score, Risk Score.
  - **RECOMMENDED ACTIONS:** Wallet tracing, counterparty inspection, and entity monitoring.
  - **REMARKS:** Risk-level specific investigative notes.

---

## 10. Ubuntu / Linux Setup & Execution Guide

```bash
# 1. Clone repository and navigate to folder
cd "Bitcoin Monitoring"

# 2. Activate virtual environment
source ./venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run automated unit & integration test suite
python3 -m unittest -v tests/test_pipeline.py

# 5. Launch interactive Streamlit prototype
streamlit run app.py
```

---

## 11. Research & Defensive Disclaimer


> [!NOTE]
> Developed strictly for defensive cybersecurity monitoring and investigation under SIH / NTRO Problem Statement 26146. All analysis operates on local synthetic datasets and local MaxMind databases with zero external connectivity.
