"""
FastAPI Enterprise REST API Service
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Exposes core analytical backend pipeline capabilities over RESTful HTTP API endpoints.
Provides complete API coverage across all 9 investigation modules:
1. Ingestion & Preprocessing
2. Dual-Layer Correlation
3. Graph Analytics & Leiden Community Detection
4. Machine Learning & HDBSCAN Density Clustering
5. TreeSHAP Explainability
6. Ranked Alerts & Triage
7. Forensic Reporting
8. Offline AI Investigation Assistant
9. Datashader Telemetry Visualization & PostgreSQL Persistence
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_dataset, preprocess_data
from src.ingestion.network_ingestion import parse_and_validate_network_csv
from src.ingestion.blockchain_ingestion import parse_and_validate_blockchain_csv
from src.geoip_enrichment import enrich_transactions_with_geoip
from src.correlation import EntityCorrelator, DualLayerCorrelator
from src.temporal.temporal_analysis import analyze_temporal_patterns
from src.behavioural.behavioural_fingerprinting import extract_behavioural_evidence_scores
from src.graph_analysis import BitcoinGraphEngine
from src.clustering.hdbscan_clustering import BitcoinHDBSCANClustering
from src.clustering.dbscan_clustering import BitcoinDBSCANClustering
from src.feature_engineering import extract_features
from src.anomaly_detection import BitcoinAnomalyDetector
from src.explainability.shap_explainer import BitcoinSHAPExplainer
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.evaluation import evaluate_prototype_predictions
from src.database.db_manager import DatabaseManager
from src.visualization.datashader_renderer import BitcoinDatashaderRenderer
from src.assistant.chatbot import InvestigationChatbot
from src.patterns.pattern_evidence_layer import PatternEvidenceLayer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bitcoin Sentinel Enterprise REST API",
    description="NTRO PS 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic",
    version="2.1.0"
)

# Allow local connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
pipeline_state = {
    "raw_df": None,
    "scored_df": None,
    "feat_df": None,
    "graph_engine": None,
    "correlator": None,
    "shap_explainer": None,
    "hdbscan_clusterer": None,
    "datashader_renderer": BitcoinDatashaderRenderer(),
    "db_manager": DatabaseManager(),
    "chatbot": None,
    "pattern_layer": None,
    "is_analyzed": False
}


class ChatRequest(BaseModel):
    query: str
    selected_txid: Optional[str] = None


def run_full_pipeline(df_raw: pd.DataFrame):
    """Run full analysis pipeline with HDBSCAN, Leiden, and TreeSHAP."""
    clean_df = preprocess_data(df_raw)
    enriched_df = enrich_transactions_with_geoip(clean_df)
    feat_df = extract_features(enriched_df)

    # 1. Isolation Forest Anomaly Detection
    detector = BitcoinAnomalyDetector(contamination=0.10, n_estimators=150, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)

    # 2. HDBSCAN Density-Based Behavioural Clustering
    hdbscan = BitcoinHDBSCANClustering(min_cluster_size=5, min_samples=3)
    cluster_labels, noise_mask, hdbscan_evidence = hdbscan.fit_predict(feat_df)

    # 3. Temporal & Behavioural Fingerprinting
    temp_df = analyze_temporal_patterns(enriched_df)
    beh_df = extract_behavioural_evidence_scores(temp_df)

    # 4. Graph Construction & Leiden Community Detection
    graph_engine = BitcoinGraphEngine(beh_df, use_leiden=True)

    # Calculate Leiden community evidence
    leiden_ev = []
    for txid in beh_df['txid']:
        score, _ = graph_engine.community_detector.compute_community_evidence_score(str(txid))
        leiden_ev.append(score)

    beh_df['leiden_community_evidence'] = leiden_ev
    beh_df['louvain_community_evidence'] = leiden_ev  # backward compat
    beh_df['hdbscan_evidence'] = hdbscan_evidence
    beh_df['dbscan_evidence'] = hdbscan_evidence

    # 4b. Behavioural Pattern Detection (Peeling-Chain + CoinJoin-Like)
    # Must run BEFORE RiskScoringEngine so pattern evidence contributes to fused score.
    pattern_layer = PatternEvidenceLayer()
    beh_df = pattern_layer.run(beh_df, db_manager=pipeline_state["db_manager"])

    # 5. Multi-Layer Evidence Fusion & Risk Scoring
    risk_engine = RiskScoringEngine(ml_weight=0.45, heuristic_weight=0.55)
    scored_df = risk_engine.compute_risk_scores(
        beh_df, norm_scores, raw_scores, is_anomaly,
        dbscan_evidence=hdbscan_evidence, louvain_evidence=np.array(leiden_ev)
    )

    # 6. TreeSHAP Explainability
    shap_explainer = BitcoinSHAPExplainer(max_depth=5)
    target_scores = scored_df['final_risk_score'].values if 'final_risk_score' in scored_df.columns else norm_scores
    shap_explainer.fit(feat_df, target_scores)

    # 7. Entity Correlation
    correlator = EntityCorrelator(scored_df)

    # Cache state
    pipeline_state["raw_df"] = df_raw
    pipeline_state["scored_df"] = scored_df
    pipeline_state["feat_df"] = feat_df
    pipeline_state["graph_engine"] = graph_engine
    pipeline_state["correlator"] = correlator
    pipeline_state["shap_explainer"] = shap_explainer
    pipeline_state["hdbscan_clusterer"] = hdbscan
    pipeline_state["pattern_layer"] = pattern_layer
    pipeline_state["chatbot"] = InvestigationChatbot(scored_df)
    pipeline_state["is_analyzed"] = True

    # Persist to database if connected
    if pipeline_state["db_manager"].is_connected:
        pipeline_state["db_manager"].save_dataframe_table(scored_df, "transactions")
        ranked = generate_ranked_alerts(scored_df, min_risk="LOW")
        pipeline_state["db_manager"].save_dataframe_table(ranked, "alerts")


@app.on_event("startup")
def startup_event():
    try:
        raw_df = load_dataset()
        run_full_pipeline(raw_df)
    except Exception as e:
        logger.warning(f"Startup pipeline initialization warning: {e}")


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "ONLINE",
        "version": "2.1.0",
        "pipeline_ready": pipeline_state["is_analyzed"],
        "database": pipeline_state["db_manager"].health_check()
    }


@app.get("/api/metrics")
def get_metrics():
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    df = pipeline_state["scored_df"]
    eval_res = evaluate_prototype_predictions(df)

    return {
        "total_transactions": len(df),
        "anomalies_detected": int(df['is_anomaly'].sum()),
        "critical_alerts": int((df['risk_level'] == 'CRITICAL').sum()),
        "high_alerts": int((df['risk_level'] == 'HIGH').sum()),
        "medium_alerts": int((df['risk_level'] == 'MEDIUM').sum()),
        "low_alerts": int((df['risk_level'] == 'LOW').sum()),
        "evaluation_benchmarks": {
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1_score": eval_res["f1_score"],
            "roc_auc": eval_res["roc_auc"]
        }
    }


@app.get("/api/alerts")
def get_alerts(min_risk: str = Query("LOW", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"), limit: int = 100):
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    df = pipeline_state["scored_df"]
    ranked = generate_ranked_alerts(df, min_risk=min_risk)
    
    # Clean non-primitive types for JSON serialization
    records = []
    for _, row in ranked.head(limit).iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp, pd.Series)):
                rec[k] = str(v)
            elif hasattr(v, "item"):
                rec[k] = v.item()
            elif isinstance(v, (list, dict)):
                rec[k] = str(v)
            else:
                rec[k] = v
        records.append(rec)
    return records


@app.get("/api/alerts/{txid}")
def get_alert_by_id(txid: str):
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    df = pipeline_state["scored_df"]
    match = df[df['txid'] == txid]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {txid} not found.")

    row = match.iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, 'item') else str(v) if isinstance(v, (pd.Timestamp, pd.Series, list, dict)) else v) for k, v in row.items()}


@app.get("/api/transactions/{txid}")
def get_transaction_details(txid: str):
    if not pipeline_state["correlator"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    details = pipeline_state["correlator"].get_tx_details(txid)
    if not details:
        raise HTTPException(status_code=404, detail=f"Transaction {txid} not found.")
    return details


@app.get("/api/graph/summary")
def get_graph_summary():
    if not pipeline_state["graph_engine"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    return pipeline_state["graph_engine"].get_graph_summary()


@app.get("/api/graph/ego/{entity}")
def get_graph_ego_network(entity: str, radius: int = 1, max_nodes: int = 50):
    if not pipeline_state["graph_engine"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    sub_g = pipeline_state["graph_engine"].extract_subgraph(entity, radius=radius, max_nodes=max_nodes)
    nodes = [{"id": n, **attr} for n, attr in sub_g.nodes(data=True)]
    edges = [{"source": u, "target": v, **attr} for u, v, attr in sub_g.edges(data=True)]
    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph/communities")
def get_communities():
    if not pipeline_state["graph_engine"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    return pipeline_state["graph_engine"].community_stats


@app.get("/api/ml/clusters")
def get_clustering_summary():
    if not pipeline_state["hdbscan_clusterer"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    return pipeline_state["hdbscan_clusterer"].get_cluster_summary()


@app.get("/api/explain/global")
def get_global_shap_importance(top_n: int = 10):
    if not pipeline_state["shap_explainer"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    return pipeline_state["shap_explainer"].get_global_feature_importance(top_n=top_n)


@app.get("/api/explain/{txid}")
def get_instance_shap_explanation(txid: str, top_n: int = 5):
    if not pipeline_state["shap_explainer"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    df = pipeline_state["scored_df"]
    match = df[df['txid'] == txid]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {txid} not found.")

    row = match.iloc[0]
    return pipeline_state["shap_explainer"].explain_instance(row, top_n=top_n)


@app.get("/api/visualize/density")
def get_telemetry_density_image():
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    img = pipeline_state["datashader_renderer"].render_anomaly_distribution(pipeline_state["scored_df"])
    return {"image_data": img, "available": pipeline_state["datashader_renderer"].is_available()}


@app.get("/api/database/status")
def get_database_status():
    return pipeline_state["db_manager"].get_table_summary()



@app.post("/api/assistant/chat")
def chat_with_assistant(req: ChatRequest):
    if not pipeline_state["chatbot"]:
        raise HTTPException(status_code=400, detail="Chatbot assistant not initialized yet.")
    response = pipeline_state["chatbot"].ask(req.query, selected_txid=req.selected_txid)
    return response


# ==============================================================================
# Pattern Detection Endpoints
# ==============================================================================

@app.get("/api/patterns/summary")
def get_pattern_summary():
    """Return overall pattern detection summary (peeling chains + CoinJoin stats)."""
    if not pipeline_state["is_analyzed"] or pipeline_state["pattern_layer"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    return pipeline_state["pattern_layer"].get_summary()


@app.get("/api/patterns/detected")
def get_detected_patterns(limit: int = 200):
    """Return all transactions with pattern detections."""
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    pl = pipeline_state["pattern_layer"]
    df = pipeline_state["scored_df"]
    if pl is None:
        raise HTTPException(status_code=400, detail="Pattern layer not initialized.")
    patterns = pl.get_detected_patterns(df)
    return patterns[:limit]


@app.get("/api/patterns/{txid}")
def get_pattern_for_tx(txid: str):
    """Return pattern detection result for a specific transaction."""
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    df = pipeline_state["scored_df"]
    match = df[df["txid"] == txid]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {txid} not found.")
    row = match.iloc[0]
    return {
        "txid": txid,
        "peeling_chain_detected": bool(row.get("peeling_chain_detected", False)),
        "peeling_chain_evidence": float(row.get("peeling_chain_evidence", 0.0)),
        "peeling_chain_id": str(row.get("peeling_chain_id", "")),
        "peeling_chain_position": int(row.get("peeling_chain_position", 0)),
        "peeling_chain_length": int(row.get("peeling_chain_length", 0)),
        "peeling_chain_reasons": str(row.get("peeling_chain_reasons", "")),
        "coinjoin_detected": bool(row.get("coinjoin_detected", False)),
        "coinjoin_evidence": float(row.get("coinjoin_evidence", 0.0)),
        "coinjoin_equal_output_count": int(row.get("coinjoin_equal_output_count", 0)),
        "coinjoin_equal_output_ratio": float(row.get("coinjoin_equal_output_ratio", 0.0)),
        "coinjoin_equal_amount_btc": float(row.get("coinjoin_equal_amount_btc", 0.0)),
        "coinjoin_reasons": str(row.get("coinjoin_reasons", "")),
    }


# ==============================================================================
# Static File Mounting for React Dashboard
# ==============================================================================
REACT_BUILD_DIR = PROJECT_ROOT / "frontend" / "build"
REACT_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

static_dir = REACT_DIST_DIR if REACT_DIST_DIR.exists() else (REACT_BUILD_DIR if REACT_BUILD_DIR.exists() else None)

if static_dir and (static_dir / "index.html").exists():
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def serve_react_root():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/dashboard")
    def serve_react_dashboard():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/{full_path:path}")
    def serve_react_app(full_path: str):
        target = static_dir / full_path
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(static_dir / "index.html"))
else:
    @app.get("/")
    def index_status():
        return {
            "service": "Bitcoin Sentinel Enterprise REST API",
            "status": "ONLINE",
            "version": "2.1.0",
            "docs": "/docs",
            "database": pipeline_state["db_manager"].get_table_summary(),
            "note": "React dashboard can be accessed via frontend static build or port 3000."
        }
