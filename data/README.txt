BITCOIN MONITORING — FINAL SYNTHETIC PROTOTYPE DATASET

This package is a synthetic research/prototype dataset aligned to the NTRO problem statement.
It is NOT real Bitcoin network-interception data.

Main file:
bitcoin_monitoring_final_10000.csv
- 10,000 records
- 37 fields
- 9,000 synthetic normal records
- 1,000 synthetic anomalous records

Explicit correlation:
- Repeated synthetic IPs occur across multiple TXIDs.
- Wallets are deliberately reused so IP -> TXID -> Wallet relationships form connected patterns.
- Suspicious hub wallets and IPs create visible graph clusters.
- Additional edge files are supplied for NetworkX/graph analysis.

Anomaly scenarios:
rapid_chain, fan_out, fan_in, cross_border_burst, high_frequency,
high_connectivity, ip_wallet_reuse, layered_transfer.

Files:
1. bitcoin_monitoring_final_10000.csv — master dataset
2. ip_txid_wallet_correlations.csv — explicit IP-TXID-wallet correlation table
3. transaction_wallet_edges.csv — TXID-wallet graph edges
4. wallet_wallet_edges.csv — wallet-wallet graph edges
5. ip_wallet_edges.csv — IP-wallet graph edges
6. sample_200.csv — small inspection sample

Do NOT feed ground_truth or scenario to Isolation Forest.
Use them only for evaluation/validation of the prototype.

Suggested model features:
transaction_frequency_24h, avg_time_gap_min, wallet_degree, unique_ip_count,
country_count, asn_count, num_inputs, num_outputs, total_output_amount_btc,
fee_btc, connection_duration_sec, packet_count, bytes_transferred.

Recommended pipeline:
Data ingestion -> validation/preprocessing -> IP-TXID-wallet correlation ->
NetworkX graph -> feature engineering -> Isolation Forest -> anomaly score ->
risk scoring -> explanation -> Streamlit dashboard.
