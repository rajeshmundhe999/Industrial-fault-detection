@echo off
echo ========================================================
echo PFDI Project - Resume Architecture Ablation Experiments
echo ========================================================
echo.
echo Checking environment and resuming where you left off...
echo Completed variants (A and B) will be skipped automatically.
echo.
cd /d "c:\000 VIT\ETHANOL\pfdi_project"
set PYTHONIOENCODING=utf-8
py -3.12 -u "research_audit\ablation\train_ablation.py"
echo.
echo ========================================================
echo Ablation process finished! Results in:
echo research_audit\ablation\results\ablation_results.csv
echo ========================================================
pause
