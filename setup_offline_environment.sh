#!/usr/bin/env bash
# ==============================================================================
# Bitcoin Sentinel - Offline Environment Setup Script (Ubuntu / Linux)
# NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
# ==============================================================================

set -e

echo "============================================================"
echo " BITCOIN SENTINEL - AIR-GAPPED LINUX ENVIRONMENT SETUP"
echo " NTRO Problem Statement 26146"
echo "============================================================"

# 1. Check Python
echo -e "\n[1/5] Checking Python 3..."
python3 --version || { echo "Python 3 is required."; exit 1; }

# 2. Setup .env.local
echo -e "\n[2/5] Configuring environment variables..."
if [ ! -f ".env.local" ]; then
    if [ -f ".env.local.example" ]; then
        cp .env.local.example .env.local
        echo "Created .env.local from template."
    fi
else
    echo ".env.local already exists."
fi

# 3. Check and install Python dependencies
echo -e "\n[3/5] Verifying required analytical packages..."
python3 -c "
import sys
packages = ['pandas', 'numpy', 'sklearn', 'networkx', 'psycopg2', 'neo4j', 'datashader', 'igraph', 'leidenalg', 'shap', 'fastapi', 'uvicorn', 'streamlit', 'reportlab']
missing = [p for p in packages if not __import__('importlib.util').util.find_spec(p)]
if missing:
    print('MISSING: ' + ', '.join(missing))
    sys.exit(1)
print('ALL PACKAGES VERIFIED')
" || {
    echo "Installing missing dependencies from requirements.txt..."
    pip install -r requirements.txt
}

# 4. Check PostgreSQL
echo -e "\n[4/5] Checking PostgreSQL connectivity..."
python3 -c "
from src.database.db_manager import DatabaseManager
db = DatabaseManager()
print(f'Database: {db.db_type} - {db.connection_status_msg}')
"

# 5. Check GeoIP MMDB files
echo -e "\n[5/5] Checking MaxMind GeoIP databases..."
if [ -f "data/geoip/GeoLite2-Country.mmdb" ] && [ -f "data/geoip/GeoLite2-ASN.mmdb" ]; then
    echo "GeoLite2 Country and ASN MMDB files verified."
else
    echo "GeoIP MMDB files not found in data/geoip/. Fallback heuristics active."
fi

echo "============================================================"
echo " SETUP COMPLETED SUCCESSFULLY!"
echo " Run ./start_offline_system.sh to start all services."
echo "============================================================"
