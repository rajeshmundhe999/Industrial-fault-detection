@echo off
echo ========================================================
echo Installing CUDA-enabled PyTorch for NVIDIA RTX 4080 GPU
echo ========================================================
echo.
echo Detected GPU: NVIDIA GeForce RTX 4080 Laptop GPU (12GB VRAM)
echo Downloading PyTorch with CUDA 12.4 support (~2.5 GB)...
echo Please keep your internet connection active.
echo.
py -3.12 -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124 --user
echo.
echo Testing GPU availability...
py -3.12 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Failed to detect GPU')"
echo.
echo ========================================================
echo Setup complete! Now you can run RESUME_ABLATION.bat
echo ========================================================
pause
