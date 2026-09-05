"""
FastAPI REST API Service
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Exposes core analytical backend pipeline capabilities over RESTful HTTP API endpoints.
Uses identical core analysis modules to ensure zero code duplication.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing import load_dataset, preprocess_data
from src.ingestion.network_ingestion import parse_and_validate_network_csv
from src.ingestion.blockchain_ingestion import parse_and_validate_blockchain_csv
from src.geoip_enrichment import enrich_transactions_with_geoip
from src.correlation import EntityCorrelator, DualLayerCorrelator
from src.temporal.temporal_analysis import analyze_temporal_patterns
from src.behavioural.behavioural_fingerprinting import extract_behavioural_evidence_scores
from src.graph_analysis import BitcoinGraphEngine
from src.clustering.dbscan_clustering import BitcoinDBSCANClustering
from src.feature_engineering import extract_features
from src.anomaly_detection import BitcoinAnomalyDetector
from src.risk_scoring import RiskScoringEngine, generate_ranked_alerts
from src.evaluation import evaluate_prototype_predictions
from src.database.db_manager import DatabaseManager

app = FastAPI(
    title="Bitcoin Sentinel REST API — NTRO PS 26146",
    description="AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
pipeline_state = {
    "scored_df": None,
    "graph_engine": None,
    "correlator": None,
    "db_manager": DatabaseManager(),
    "is_analyzed": False
}


def run_full_pipeline(df_raw: pd.DataFrame):
    """Run full analysis pipeline and cache state in memory."""
    clean_df = preprocess_data(df_raw)
    enriched_df = enrich_transactions_with_geoip(clean_df)
    feat_df = extract_features(enriched_df)
    
    # Isolation Forest
    detector = BitcoinAnomalyDetector(contamination=0.10, n_estimators=150, random_state=42)
    is_anomaly, raw_scores, norm_scores = detector.fit_predict(feat_df)
    
    # DBSCAN
    dbscan = BitcoinDBSCANClustering(eps=0.5, min_samples=5)
    cluster_labels, noise_mask, dbscan_evidence = dbscan.fit_predict(feat_df)
    
    # Temporal & Behavioural
    temp_df = analyze_temporal_patterns(enriched_df)
    beh_df = extract_behavioural_evidence_scores(temp_df)

    # Graph & Louvain
    graph_engine = BitcoinGraphEngine(beh_df)
    
    # Calculate Louvain community evidence per record
    louvain_ev = []
    for txid in beh_df['txid']:
        score, _ = graph_engine.louvain_detector.compute_community_evidence_score(str(txid))
        louvain_ev.append(score)
        
    beh_df['louvain_community_evidence'] = louvain_ev
    beh_df['dbscan_evidence'] = dbscan_evidence

    # Risk Engine & Fusion
    risk_engine = RiskScoringEngine(ml_weight=0.45, heuristic_weight=0.55)
    scored_df = risk_engine.compute_risk_scores(
        beh_df, norm_scores, raw_scores, is_anomaly,
        dbscan_evidence=dbscan_evidence, louvain_evidence=np.array(louvain_ev)
    )

    correlator = EntityCorrelator(scored_df)

    # Cache state
    pipeline_state["scored_df"] = scored_df
    pipeline_state["graph_engine"] = graph_engine
    pipeline_state["correlator"] = correlator
    pipeline_state["is_analyzed"] = True

    # Persist to database if available
    if pipeline_state["db_manager"].is_connected:
        pipeline_state["db_manager"].save_dataframe_table(scored_df, "investigation_alerts")


# Auto-run baseline analysis on startup
@app.on_event("startup")
def startup_event():
    try:
        raw_df = load_dataset()
        run_full_pipeline(raw_df)
    except Exception as e:
        print(f"Startup pipeline initialization warning: {e}")


@app.get("/")
def read_root():
    return {
        "service": "Bitcoin Sentinel REST API",
        "status": "ONLINE",
        "version": "2.0.0",
        "database": pipeline_state["db_manager"].get_table_summary()
    }


@app.get("/metrics")
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


@app.get("/alerts")
def get_alerts(min_risk: str = Query("MEDIUM", regex="^(LOW|MEDIUM|HIGH|CRITICAL)$"), limit: int = 50):
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    df = pipeline_state["scored_df"]
    ranked = generate_ranked_alerts(df, min_risk=min_risk)
    return ranked.head(limit).to_dict(orient="records")


@app.get("/alerts/{id_val}")
def get_alert_by_id(id_val: str):
    if not pipeline_state["is_analyzed"] or pipeline_state["scored_df"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    df = pipeline_state["scored_df"]
    match = df[df['txid'] == id_val]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Alert with TXID {id_val} not found.")

    row = match.iloc[0].to_dict()
    # Convert numpy types to native Python
    return {k: (v.item() if hasattr(v, 'item') else str(v) if isinstance(v, (pd.Timestamp, pd.Series)) else v) for k, v in row.items()}


@app.get("/transactions/{txid}")
def get_transaction_details(txid: str):
    if not pipeline_state["correlator"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")
    details = pipeline_state["correlator"].get_tx_details(txid)
    if not details:
        raise HTTPException(status_code=404, detail=f"Transaction {txid} not found.")
    return details


@app.get("/graph/{entity}")
def get_graph_ego_network(entity: str, radius: int = 1, max_nodes: int = 40):
    if not pipeline_state["graph_engine"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    sub_g = pipeline_state["graph_engine"].extract_subgraph(entity, radius=radius, max_nodes=max_nodes)
    nodes = []
    for n, attr in sub_g.nodes(data=True):
        nodes.append({"id": n, **attr})

    edges = []
    for u, v, attr in sub_g.edges(data=True):
        edges.append({"source": u, "target": v, **attr})

    return {"nodes": nodes, "edges": edges}


@app.get("/communities")
def get_louvain_communities():
    if not pipeline_state["graph_engine"]:
        raise HTTPException(status_code=400, detail="Pipeline not analyzed yet.")

    stats = pipeline_state["graph_engine"].community_stats
    return stats
