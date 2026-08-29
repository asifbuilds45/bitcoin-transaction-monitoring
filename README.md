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

## 4. Offline GeoIP & Network Enrichment

The system includes a dedicated offline MaxMind GeoIP2/GeoLite2 integration (`src/geoip_enrichment.py`).
- **Target Files:** `data/GeoLite2-Country.mmdb` and `data/GeoLite2-ASN.mmdb`.
- **Network Isolation:** Operates 100% locally. Zero internet connections or API requests are made.
- **Graceful Fallback Handling:**
  - When valid public IPs match local `.mmdb` databases: `geoip_lookup_status = "geoip_resolved"`.
  - For synthetic or private range IP addresses (e.g., `10.x.x.x`) or if `.mmdb` files are absent: retains pre-populated country/ASN dataset values and assigns `geoip_lookup_status = "synthetic_fallback"`.
  - **The application never crashes** when database files are missing.

---

## 5. Machine Learning & Behavioral Analytics

### Why Isolation Forest?
1. **Unsupervised Principle:** Real-world cyber monitoring lacks real-time attack labels. Isolation Forest learns baseline distributions without requiring pre-labeled training datasets.
2. **Path Length Anomaly Isolation:** In high-dimensional feature space, anomalous behaviors (e.g., extreme velocity, unusual UTXO fan-outs) are isolated with significantly shorter tree paths than normal cluster points.
3. **Multi-Modal Feature Synthesis:** Simultaneously processes network telemetry (`packet_count`, `bytes_transferred`, `connection_duration_sec`) and on-chain graph metrics (`wallet_degree`, `unique_ip_count`, `num_inputs`, `num_outputs`).
4. **Computational Efficiency:** Scales linearly $O(n \log n)$, allowing instant processing of 10,000+ transaction batches on standard commodity CPUs.

### Risk Scoring Framework (0–100)
Risk scores are calculated by combining normalized Isolation Forest anomaly scores with multi-factor heuristic triggers:
- **0–24: LOW RISK** (Normal baseline transactions)
- **25–49: MEDIUM RISK** (Minor deviations or elevated frequency)
- **50–74: HIGH RISK** (Multi-indicator anomalies, suspicious hub connectivity)
- **75–100: CRITICAL RISK** (Severe behavioral anomalies, multi-country hopping, mixer fan-out/fan-in)

---

## 6. Streamlit Dashboard Modules

Launch the interactive dashboard with `streamlit run app.py` to access 7 specialized views:
1. **Overview & Executive KPIs:** High-level summary metrics, anomaly/risk distributions, country activity profiles, and top suspicious entities.
2. **Transaction Investigation:** Detailed inspector with search/filtering by TXID, Wallet, or IP, complete UTXO breakdown, GeoIP enrichment panel, and natural-language risk reasons.
3. **IP-TXID-Wallet Correlation:** Interactive 2D Plotly network graph rendering multi-modal entity links (Cyan=IP, Orange=TXID, Green=Wallet) with customizable exploration radius.
4. **Graph Analytics & Topology:** Hub entity identification, degree distributions, and clustering metrics.
5. **AI Anomaly Analysis:** Isolation Forest score distribution curves, feature separation boxplots, and algorithmic rationale.
6. **Ranked Investigation Alerts:** Prioritized triage queue sorted by risk score with CSV export capabilities.
7. **Prototype Evaluation:** Independent benchmark against ground truth displaying Precision, Recall, F1-Score, Confusion Matrix Heatmap, and Scenario-specific detection rates.

---

## 7. Ubuntu / Linux Setup & Execution Guide

### Step 1: Create and Activate Python Virtual Environment
```bash
# Update package list and install Python 3 venv if needed
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# Navigate to project directory
cd bitcoin-monitoring

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Automated Pipeline & Test Suite
```bash
python tests/test_pipeline.py
```

### Step 4: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Benchmark Evaluation Results

Validation results obtained during post-prediction evaluation:

| Metric | Score |
|---|---|
| **Precision** | **98.90%** |
| **Recall** | **98.90%** |
| **F1-Score** | **98.90%** |
| **ROC-AUC** | **1.0000** |
| **True Positives (TP)** | 989 |
| **True Negatives (TN)** | 8,989 |
| **False Positives (FP)** | 11 |
| **False Negatives (FN)** | 11 |

---

## 9. Research & Cybersecurity Disclaimer

> [!NOTE]
> This prototype is developed for defensive cybersecurity research, threat analysis, and academic demonstration under NTRO Problem Statement 26146.
> 
> The underlying dataset is synthetic. This system identifies anomalous behavioral patterns and produces **investigative leads** for human analysts; it does not constitute conclusive attribution or proof of illicit activity.
