@echo off
echo ======================================================================
echo PCB DEFECT DETECTION - TRAINING
echo ======================================================================
echo.
echo Configuration:
echo - Epochs: 200
echo - Batch size: 16
echo - CUDA: 11.2 + cuDNN 8.1
echo - GPU: RTX 3080 Ti
echo ======================================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated!
) else (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.

REM Check GPU
echo Checking GPU...
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
echo.

REM Check dataset
if not exist "dataset\lulus_qc\" (
    echo ERROR: Folder dataset\lulus_qc not found!
    pause
    exit /b 1
)
if not exist "dataset\cacat_produksi\" (
    echo ERROR: Folder dataset\cacat_produksi not found!
    pause
    exit /b 1
)

echo Starting training...
echo Press Ctrl+C to stop
echo.

REM Run training
set PYTHONIOENCODING=utf-8
set TF_FORCE_GPU_ALLOW_GROWTH=true

python -u train.py 2>&1

echo.
echo ======================================================================
echo TRAINING FINISHED
echo ======================================================================
echo.
pause
