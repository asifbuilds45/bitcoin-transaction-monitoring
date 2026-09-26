# ==============================================================================
# Bitcoin Sentinel - Offline System Startup Script (Windows PowerShell)
# NTRO Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
# ==============================================================================

param (
    [string]$Mode = "all"  # Options: all, api, streamlit
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " STARTING BITCOIN SENTINEL - AIR-GAPPED FORENSIC SUITE" -ForegroundColor Cyan
Write-Host " Mode: $Mode" -ForegroundColor DarkCyan
Write-Host "============================================================" -ForegroundColor Cyan

# Set environment to offline mode
$env:OFFLINE_MODE = "1"

if ($Mode -eq "api" -or $Mode -eq "all") {
    Write-Host "`n[+] Starting FastAPI REST Service & React Dashboard..." -ForegroundColor Green
    Write-Host "    URL: http://127.0.0.1:8000" -ForegroundColor Cyan
    Write-Host "    Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
    if ($Mode -eq "all") {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload"
    } else {
        python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
    }
}

if ($Mode -eq "streamlit" -or $Mode -eq "all") {
    Write-Host "`n[+] Starting Streamlit Forensic Investigator Dashboard..." -ForegroundColor Green
    Write-Host "    URL: http://127.0.0.1:8501" -ForegroundColor Cyan
    if ($Mode -eq "all") {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "streamlit run app.py --server.port 8501 --server.address 127.0.0.1"
    } else {
        streamlit run app.py --server.port 8501 --server.address 127.0.0.1
    }
}

if ($Mode -eq "all") {
    Write-Host "`n============================================================" -ForegroundColor Green
    Write-Host " BOTH SERVICES LAUNCHED IN SEPARATE TERMINALS:" -ForegroundColor Green
    Write-Host " 1. React + FastAPI Dashboard : http://127.0.0.1:8000" -ForegroundColor White
    Write-Host " 2. Streamlit Dashboard       : http://127.0.0.1:8501" -ForegroundColor White
    Write-Host "============================================================" -ForegroundColor Green
}
