@echo off
echo ========================================================
echo PFDI Project - Research Dashboard & Review Presentation
echo ========================================================
echo.
echo Starting Research Dashboard...
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C in this window to stop the dashboard.
echo.
cd /d "c:\000 VIT\ETHANOL\pfdi_project"
set PYTHONIOENCODING=utf-8
py -3.12 -m streamlit run dashboard/app.py --server.port 8501 --browser.gatherUsageStats false
pause
