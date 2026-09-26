@echo off
title Creating NTRO Forensic Desktop Shortcut
cd /d "%~dp0"

powershell -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; " ^
  "$desktop = [Environment]::GetFolderPath('Desktop'); " ^
  "$shortcut = $ws.CreateShortcut(\"$desktop\NTRO Bitcoin Forensic Workstation.lnk\"); " ^
  "$shortcut.TargetPath = \"$((Get-Item .).FullName)\launch_desktop.bat\"; " ^
  "$shortcut.WorkingDirectory = \"$((Get-Item .).FullName)\"; " ^
  "$shortcut.IconLocation = 'imageres.dll,102'; " ^
  "$shortcut.Save()"

echo.
echo ============================================================
echo   Success! 'NTRO Bitcoin Forensic Workstation' shortcut
echo   has been placed directly on your Desktop with a shield icon!
echo ============================================================
echo.
pause
