@echo off
echo ======================================================================
echo PCB DEFECT DETECTION - SETUP SCRIPT
echo ======================================================================
echo.
echo This script will setup your development environment:
echo 1. Create virtual environment (.venv)
echo 2. Install TensorFlow 2.10.0 (CUDA 11.2 compatible)
echo 3. Install all dependencies
echo 4. Verify GPU availability
echo.
echo ======================================================================
pause
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.9 from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv .venv
    echo Virtual environment created!
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [5/5] Installing dependencies...
echo This may take 5-10 minutes...
echo.
pip install tensorflow==2.10.0
pip install -r requirements.txt
echo.

REM Verify installation
echo ======================================================================
echo SETUP COMPLETE!
echo ======================================================================
echo.
echo Verifying installation...
echo.
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU Available:', len(gpus) > 0); print('GPU List:', gpus)"
echo.
echo ======================================================================
echo.
echo Next steps:
echo 1. Make sure dataset folder exists with structure:
echo    dataset/lulus_qc/ and dataset/cacat_produksi/
echo.
echo 2. To activate virtual environment in future:
echo    .venv\Scripts\activate
echo.
echo 3. To start training:
echo    python train.py
echo    or run: train.bat
echo.
echo 4. To deactivate virtual environment:
echo    deactivate
echo.
echo ======================================================================
pause
