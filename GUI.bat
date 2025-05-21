@echo off
set ENV_NAME=tvips_converter_env

:: Check if virtual environment exists
if not exist "%ENV_NAME%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment "%ENV_NAME%" not found.
    pause
    exit /b
)

:: Activate virtual environment
call %ENV_NAME%\Scripts\activate.bat
:: Run the GUI
python tvipsbloGUI.py

:: Optional: keep window open after execution
pause
