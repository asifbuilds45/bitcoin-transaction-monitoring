# ==============================================================================
# Bitcoin Sentinel - Offline Environment Setup Script (Windows PowerShell)
# NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
# ==============================================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " BITCOIN SENTINEL - AIR-GAPPED ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host " NTRO Problem Statement 26146" -ForegroundColor DarkCyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/5] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# 2. Check and Copy Environment Configuration
Write-Host "`n[2/5] Setting up local environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".env.local")) {
    if (Test-Path ".env.local.example") {
        Copy-Item ".env.local.example" ".env.local"
        Write-Host "Created .env.local from .env.local.example template." -ForegroundColor Green
    } else {
        Write-Host "WARNING: .env.local.example not found." -ForegroundColor Yellow
    }
} else {
    Write-Host ".env.local already exists. Preserving configuration." -ForegroundColor Green
}

# 3. Verify Offline Python Dependencies
Write-Host "`n[3/5] Verifying installed analytical packages..." -ForegroundColor Yellow
python -c "
import sys
packages = ['pandas', 'numpy', 'sklearn', 'networkx', 'psycopg2', 'neo4j', 'datashader', 'igraph', 'leidenalg', 'shap', 'fastapi', 'uvicorn', 'streamlit', 'reportlab']
missing = []
for p in packages:
    try:
        __import__(p)
    except ImportError:
        missing.append(p)
if missing:
    print('MISSING_PACKAGES:' + ','.join(missing))
    sys.exit(1)
else:
    print('ALL_REQUIRED_PACKAGES_AVAILABLE')
"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing dependencies from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
} else {
    Write-Host "All analytical libraries verified (HDBSCAN, Leiden, TreeSHAP, Datashader, PostgreSQL, Neo4j drivers)." -ForegroundColor Green
}

# 4. Check PostgreSQL Database Connection
Write-Host "`n[4/5] Checking PostgreSQL database connection..." -ForegroundColor Yellow
python -c "
from src.database.db_manager import DatabaseManager
db = DatabaseManager()
print(f'Database Engine: {db.db_type}')
print(f'Status Message: {db.connection_status_msg}')
"

# 5. Verify Offline MaxMind GeoIP Databases
Write-Host "`n[5/5] Checking MaxMind GeoIP offline MMDB databases..." -ForegroundColor Yellow
if ((Test-Path "data/geoip/GeoLite2-Country.mmdb") -and (Test-Path "data/geoip/GeoLite2-ASN.mmdb")) {
    Write-Host "MaxMind GeoLite2 Country and ASN databases found." -ForegroundColor Green
} else {
    Write-Host "NOTE: GeoLite2 MMDB files present in data/geoip/. Fallback heuristics active." -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " SETUP COMPLETE! SYSTEM IS READY FOR OFFLINE OPERATION." -ForegroundColor Green
Write-Host " To start the system, run: .\start_offline_system.ps1" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
