@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "PATH=%~dp0.venv\Scripts;%PATH%"

echo RVCを起動しています...
echo.

"%~dp0.venv\Scripts\python.exe" "%~dp0realtime_gui.py"

echo.
echo RVCが終了しました。
timeout /t 5 /nobreak >nul