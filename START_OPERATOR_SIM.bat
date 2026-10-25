@echo off
echo ========================================================
echo PFDI Project - Live Industrial Operator Console
echo ========================================================
echo.
echo Starting DCS Live Simulation Console...
echo Opening on port 8502 (separate from main dashboard)...
echo URL: http://localhost:8502
echo.
echo Press Ctrl+C in this window to stop the console.
echo.
cd /d "c:\000 VIT\ETHANOL\pfdi_project"
set PYTHONIOENCODING=utf-8
py -3.12 -m streamlit run dashboard/live_operator_sim.py --server.port 8502 --browser.gatherUsageStats false
pause
