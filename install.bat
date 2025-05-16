@echo off
set ENV_NAME=tvips_converter_env

echo Setting up Python 3.7.3 environment...

:: Step 1: Check Python version
python --version | findstr "3.7.3" >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.7.3 not found. Please install it and add to PATH.
    pause
    exit /b
)

:: Step 2: Create virtual environment
echo Creating virtual environment: %ENV_NAME%
python -m venv %ENV_NAME%

:: Step 3: Activate the virtual environment
call %ENV_NAME%\Scripts\activate.bat

:: Step 4: Upgrade pip
python -m pip install --upgrade pip

:: Step 5: Install required packages
echo Installing dependencies...
pip install numpy h5py matplotlib opencv-python-headless scikit-image scipy pyqt5 libertem

:: Optional: For packaging
pip install pyinstaller

echo.
echo [DONE] Setup complete.
echo To activate the environment later, run:
echo     %ENV_NAME%\Scripts\activate
pause
