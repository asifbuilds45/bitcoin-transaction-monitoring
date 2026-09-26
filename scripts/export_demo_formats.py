"""
Export 10,000-Row Demo Datasets to JSON and XML Formats
NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
"""

import os
import time
import pandas as pd

def export_all():
    print("Loading CSV demo datasets...")
    net_df = pd.read_csv("demo_network_10000.csv")
    chain_df = pd.read_csv("demo_blockchain_10000.csv")

    print(f"Network records: {len(net_df)}, Blockchain records: {len(chain_df)}")

    # 1. JSON Exports
    print("\n[1/4] Exporting demo_network_10000.json...")
    t0 = time.time()
    net_df.to_json("demo_network_10000.json", orient="records", indent=2)
    print(f"[OK] Saved demo_network_10000.json ({os.path.getsize('demo_network_10000.json') / 1024 / 1024:.2f} MB) in {time.time()-t0:.2f}s")

    print("\n[2/4] Exporting demo_blockchain_10000.json...")
    t0 = time.time()
    chain_df.to_json("demo_blockchain_10000.json", orient="records", indent=2)
    print(f"[OK] Saved demo_blockchain_10000.json ({os.path.getsize('demo_blockchain_10000.json') / 1024 / 1024:.2f} MB) in {time.time()-t0:.2f}s")

    # 2. XML Exports
    print("\n[3/4] Exporting demo_network_10000.xml...")
    t0 = time.time()
    net_df.to_xml("demo_network_10000.xml", index=False, root_name="network_records", row_name="record", parser="etree")
    print(f"[OK] Saved demo_network_10000.xml ({os.path.getsize('demo_network_10000.xml') / 1024 / 1024:.2f} MB) in {time.time()-t0:.2f}s")

    print("\n[4/4] Exporting demo_blockchain_10000.xml...")
    t0 = time.time()
    chain_df.to_xml("demo_blockchain_10000.xml", index=False, root_name="blockchain_records", row_name="record", parser="etree")
    print(f"[OK] Saved demo_blockchain_10000.xml ({os.path.getsize('demo_blockchain_10000.xml') / 1024 / 1024:.2f} MB) in {time.time()-t0:.2f}s")

    # Clean up test files if present
    for tf in ['test_net.json', 'test_net.xml', 'test_chain.json', 'test_chain.xml']:
        if os.path.exists(tf):
            os.remove(tf)

    print("\nAll 4 files created successfully in workspace root!")

if __name__ == "__main__":
    export_all()
