@echo off
title NTRO Bitcoin Forensic Workstation
cd /d "%~dp0"

echo ============================================================
echo   Launching NTRO Bitcoin Forensic Desktop Workstation...
echo ============================================================
echo.

:: 1. Clear any stuck old processes on port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: 2. Start Streamlit in the background
echo [1/2] Initializing offline analytics engine...
start "" /b python -m streamlit run app.py --server.port 8501 --server.headless true

:: 3. Wait until the local server is actually ready and responding (HTTP 200)
echo [2/2] Waiting for local server to respond...
set /a ATTEMPTS=0

:WAIT_LOOP
set /a ATTEMPTS+=1
powershell -Command "try { $r = [System.Net.WebRequest]::Create('http://127.0.0.1:8501').GetResponse(); if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    goto LAUNCH_WINDOW
)

if %ATTEMPTS% GEQ 30 (
    echo [ERROR] Server took too long to start. Check if streamlit is installed.
    pause
    exit /b 1
)

timeout /t 1 /nobreak >nul
goto WAIT_LOOP

:LAUNCH_WINDOW
echo.
echo Launching Dedicated Desktop Application Window...
start msedge.exe --app="http://127.0.0.1:8501" --window-size=1440,900
exit
