# 🔍 ANVESH
### AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

**Smart India Hackathon 2026 | PS ID: 26146**

**Team:** FUTURE FORGE VI  
**Organization:** National Technical Research Organisation (NTRO)  
**Theme:** Blockchain & Cybersecurity

## 🎯 Problem Statement

Bitcoin's pseudonymous nature makes tracing illicit transactions challenging. Investigators struggle to correlate fragmented network and blockchain evidence, detect suspicious transaction patterns, and reconstruct complex fund movements efficiently.

## 💡 Proposed Solution

**ANVESH** is an offline, AI-powered Bitcoin investigation platform that connects **IP addresses, TXIDs, and wallet addresses** using machine learning, graph analytics, and behavioural analysis.

It generates prioritized, explainable investigative leads through a unified dashboard and structured forensic reports.

## 🚀 Key Innovations

- **Cross-Layer Correlation:** Links network and blockchain evidence using confidence-based matching.
- **AI/ML Detection:** Identifies anomalies using Isolation Forest, HDBSCAN, and graph analysis.
- **Investigation Reconstruction:** Traces connected wallets, transaction paths, and evidence timelines.
- **Smart Alert Grouping:** Consolidates related alerts into investigation cases.
- **Explainable Risk Scoring:** Provides risk levels, contributing indicators, and supporting evidence.
- **Offline AI Assistant:** Supports evidence-grounded investigation without cloud dependency.

## ⚙️ System Workflow

**Data Ingestion (CSV/JSON/XML)** → **Validation** → **IP–TXID–Wallet Correlation** → **Graph & ML Analysis** → **Risk Scoring** → **Investigation Dashboard & Reports**

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Core & Data Processing | Python, Pandas, NumPy |
| Database | PostgreSQL |
| Graph Analytics | Neo4j, Leiden, Centrality |
| Machine Learning | Scikit-learn, Isolation Forest, HDBSCAN |
| Explainable AI | TreeSHAP |
| GeoIP Enrichment | MaxMind GeoLite2 |
| Backend & Frontend | FastAPI + React |
| Current Prototype Dashboard | Streamlit |
| Local AI Model | Llama 3.2 (3B) – Optional |
| Target Platform | Offline Ubuntu/Linux |

*FastAPI + React is the submitted PPT architecture; the repository's main runnable dashboard currently uses Streamlit. The assistant also supports local evidence-based fallback responses.*

## 📊 Prototype Results

*Team-reported results on synthetic benchmark data.*

| Metric | Result |
|---|---|
| Records Analyzed | 10,000 |
| Precision / Recall / F1 | 95.1% each |
| End-to-End Processing | 18.8 seconds |
| Alert Consolidation | 1,000 alerts → 28 cases |
| Reduction in Separate Alerts | **97.2%** |
| Case Reconstruction | 0.40 seconds |

*Results are based on synthetic testing; real-world performance may vary.*

## 💻 Run the Prototype

```bash
git clone https://github.com/asifbuilds45/bitcoin-transaction-monitoring.git
cd bitcoin-transaction-monitoring
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Dependencies must be installed or prepared locally before fully offline execution.

## 🎥 Working Prototype

**[▶ Watch ANVESH Prototype Demonstration](https://youtu.be/ovWgGH7Dt_4)**

## 🌍 Impact & Benefits

ANVESH helps cybercrime investigators and forensic analysts:

- Identify suspicious Bitcoin activity.
- Connect fragmented network and blockchain evidence.
- Reconstruct complex transaction paths.
- Reduce repetitive alert investigation.
- Generate explainable, investigation-ready findings.
- Analyze sensitive data locally without cloud dependency.

---

### ANVESH — From Suspicious Transactions to Actionable Intelligence

**Developed by FUTURE FORGE VI | Smart India Hackathon 2026**
