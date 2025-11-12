@echo off
echo ======================================================================
echo CASTING DEFECT DETECTION - REAL-TIME CAMERA VERSION
echo ======================================================================
echo.

REM Activate virtual environment
if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
    echo Virtual environment activated!
) else (
    echo WARNING: Virtual environment not found!
    echo Please run setup.bat first from parent folder.
    pause
    exit /b 1
)

echo.
echo Starting Real-time Camera App...
echo Make sure your camera is connected!
echo.

REM Run app
python app_realtime.py

echo.
pause
