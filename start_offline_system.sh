#!/usr/bin/env bash
# ==============================================================================
# Bitcoin Sentinel - Offline System Startup Script (Ubuntu / Linux)
# NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
# ==============================================================================

MODE="${1:-all}"
export OFFLINE_MODE="1"

echo "============================================================"
echo " STARTING BITCOIN SENTINEL - LINUX AIR-GAPPED SUITE"
echo " Mode: $MODE"
echo "============================================================"

if [ "$MODE" = "api" ] || [ "$MODE" = "all" ]; then
    echo "[+] Launching FastAPI REST API & React Dashboard on http://127.0.0.1:8000"
    python3 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
    API_PID=$!
    echo "    FastAPI running (PID: $API_PID)"
fi

if [ "$MODE" = "streamlit" ] || [ "$MODE" = "all" ]; then
    echo "[+] Launching Streamlit Investigator Dashboard on http://127.0.0.1:8501"
    streamlit run app.py --server.port 8501 --server.address 127.0.0.1 &
    STREAMLIT_PID=$!
    echo "    Streamlit running (PID: $STREAMLIT_PID)"
fi

echo "============================================================"
echo " Services running in background. Press Ctrl+C to terminate."
echo "============================================================"

trap "kill $API_PID $STREAMLIT_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
