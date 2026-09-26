#!/usr/bin/env bash
# NTRO Bitcoin Forensic Workstation - Linux (Ubuntu) Launcher
set -e

# Change directory to script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo "  NTRO Bitcoin Forensic Workstation (Linux / Ubuntu)"
echo "============================================================"
echo ""

# Find available python3
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

# Run desktop_app.py or streamlit directly
if [ -f "desktop_app.py" ]; then
    echo "[1/2] Starting offline forensic workstation..."
    $PYTHON_CMD desktop_app.py
else
    echo "[1/2] Starting Streamlit engine..."
    $PYTHON_CMD -m streamlit run app.py --server.port 8501 --server.headless false
fi
