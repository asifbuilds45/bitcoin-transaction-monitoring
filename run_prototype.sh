#!/usr/bin/env bash
# ==============================================================================
# Script: run_prototype.sh
# Purpose: One-command setup & execution for NTRO PS 26146 Bitcoin Monitoring
# ==============================================================================

set -e

echo "=================================================="
echo "🛡️  NTRO PS 26146: Bitcoin Sentinel Offline Monitor"
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3."
    exit 1
fi

# Create virtual environment if it does not exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment (venv)..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing required Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Run automated pipeline tests
echo "🧪 Running pipeline validation test..."
python tests/test_pipeline.py

# Launch Streamlit dashboard
echo "🚀 Launching Streamlit Investigation Dashboard..."
echo "Access the dashboard at http://localhost:8501"
streamlit run app.py
