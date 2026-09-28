"""
Integration test for dynamic dataset refresh & multi-dataset correlation
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import run_investigation_pipeline

def test_distinct_demo_dataset_execution():
    """Verify that running the investigation pipeline on demo_network_10000.csv and demo_blockchain_10000.csv produces distinct results."""
    net_path = "demo_network_10000.csv"
    chain_path = "demo_blockchain_10000.csv"

    assert os.path.exists(net_path), "demo_network_10000.csv must exist"
    assert os.path.exists(chain_path), "demo_blockchain_10000.csv must exist"

    net_df = pd.read_csv(net_path)
    chain_df = pd.read_csv(chain_path)

    assert len(net_df) == 10000
    assert len(chain_df) == 10000

    ctx = run_investigation_pipeline(net_df, chain_df)

    # Verify scored dataframe and KPIs
    scored_df = ctx["scored_df"]
    assert len(scored_df) == 10000
    kpis = ctx["kpis"]
    assert kpis["total_tx"] == 10000
    assert kpis["detected_anom"] > 0

    # Verify that diverse IPs and wallets are present in scored_df
    assert len(scored_df['src_ip'].dropna().unique()) > 50
    assert len(scored_df['source_wallet'].dropna().unique()) > 100

    # Verify graph engine builds graph with new entities
    graph_engine = ctx["graph_engine"]
    assert graph_engine.graph.number_of_nodes() > 0
    assert graph_engine.graph.number_of_edges() > 0

    # Extract subgraph around new high-risk entity
    crit_tx = scored_df[scored_df['risk_level_normalized'].isin(['CRITICAL', 'HIGH'])]['txid'].iloc[0]
    sub_g = graph_engine.extract_subgraph(crit_tx, radius=2, max_nodes=50)
    assert sub_g.number_of_nodes() > 0
    assert sub_g.number_of_edges() > 0


def test_multi_dataset_concatenation():
    """Verify that concatenating two distinct 10K datasets produces 20K records."""
    net_path_1 = os.path.join("data", "network_traffic_dataset.csv")
    chain_path_1 = os.path.join("data", "blockchain_transactions_dataset.csv")
    net_path_2 = "demo_batch2_network_10000.csv"
    chain_path_2 = "demo_batch2_blockchain_10000.csv"

    df_net1 = pd.read_csv(net_path_1)
    df_net2 = pd.read_csv(net_path_2)

    combined_net = pd.concat([df_net1, df_net2], ignore_index=True).drop_duplicates(subset=['txid', 'src_ip'])
    assert len(combined_net) == 20000, f"Expected 20,000 combined network rows, got {len(combined_net)}"

    df_chain1 = pd.read_csv(chain_path_1)
    df_chain2 = pd.read_csv(chain_path_2)

    combined_chain = pd.concat([df_chain1, df_chain2], ignore_index=True).drop_duplicates(subset=['txid'])
    assert len(combined_chain) == 20000, f"Expected 20,000 combined blockchain rows, got {len(combined_chain)}"


def test_json_and_xml_demo_ingestion():
    """Verify that demo datasets in JSON and XML formats parse with zero errors across all 10,000 rows."""
    from src.ingestion.network_ingestion import parse_and_validate_network_data
    from src.ingestion.blockchain_ingestion import parse_and_validate_blockchain_data

    # Test JSON
    net_j_df, net_j_warns = parse_and_validate_network_data("demo_network_10000.json")
    chain_j_df, chain_j_warns = parse_and_validate_blockchain_data("demo_blockchain_10000.json")
    assert len(net_j_df) == 10000
    assert len(chain_j_df) == 10000
    assert not any(w.get("level") == "ERROR" for w in net_j_warns)
    assert not any(w.get("level") == "ERROR" for w in chain_j_warns)

    # Test XML
    net_x_df, net_x_warns = parse_and_validate_network_data("demo_network_10000.xml")
    chain_x_df, chain_x_warns = parse_and_validate_blockchain_data("demo_blockchain_10000.xml")
    assert len(net_x_df) == 10000
    assert len(chain_x_df) == 10000
    assert not any(w.get("level") == "ERROR" for w in net_x_warns)
    assert not any(w.get("level") == "ERROR" for w in chain_x_warns)
