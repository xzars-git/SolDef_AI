@echo off
title Casting Defect Detection - Demo Launcher
color 0A

:MENU
cls
echo ======================================================================
echo.
echo      🔍 CASTING DEFECT DETECTION - DEMO APP LAUNCHER
echo.
echo      Submersible Pump Impeller Inspection System
echo      AI-Powered Quality Control
echo.
echo ======================================================================
echo.
echo  Please select demo version:
echo.
echo  [1] 📁 Upload Image Version
echo      - Upload and analyze impeller images
echo      - Detailed inspection report
echo      - Best for: Batch processing, archive images
echo.
echo  [2] 📹 Real-time Camera Version
echo      - Live detection via webcam
echo      - Real-time overlay and FPS counter
echo      - Capture frames with results
echo      - Best for: Live inspection, QC line
echo.
echo  [3] ℹ️  View Documentation
echo.
echo  [4] ❌ Exit
echo.
echo ======================================================================
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto UPLOAD
if "%choice%"=="2" goto REALTIME
if "%choice%"=="3" goto DOCS
if "%choice%"=="4" goto EXIT

echo Invalid choice! Please enter 1-4.
timeout /t 2 >nul
goto MENU

:UPLOAD
cls
echo ======================================================================
echo Starting Upload Image Version...
echo ======================================================================
echo.
call run_upload.bat
goto MENU

:REALTIME
cls
echo ======================================================================
echo Starting Real-time Camera Version...
echo ======================================================================
echo.
call run_realtime.bat
goto MENU

:DOCS
cls
echo ======================================================================
echo Opening Documentation...
echo ======================================================================
start README.md
timeout /t 2 >nul
goto MENU

:EXIT
cls
echo.
echo Thank you for using Casting Defect Detection!
echo.
timeout /t 2 >nul
exit
