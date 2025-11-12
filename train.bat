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

REM Activate conda environment
call C:\ProgramData\anaconda3\Scripts\activate.bat pcb
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate conda environment 'pcb'
    pause
    exit /b 1
)

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
