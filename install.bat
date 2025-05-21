@echo off
setlocal
set ENV_NAME=tvips_converter_env
set PYTHON_VERSION=3.7.3
set PYTHON_INSTALLER=python-3.7.3-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/3.7.3/%PYTHON_INSTALLER%
set PYTHON_DIR=%ProgramFiles%\Python373

echo Checking for Python %PYTHON_VERSION%...

:: Check if python 3.7.3 is available
for /f "tokens=2 delims=[]" %%i in ('python --version 2^>^&1 ^| findstr "Python"') do (
    set CURRENT_PY_VER=%%i
)

if "%CURRENT_PY_VER%"=="%PYTHON_VERSION%" (
    echo [OK] Python %PYTHON_VERSION% found.
) else (
    echo [INFO] Python %PYTHON_VERSION% not found. Downloading installer...
    curl -o %PYTHON_INSTALLER% %PYTHON_URL%
    
    if not exist %PYTHON_INSTALLER% (
        echo [ERROR] Download failed.
        pause
        exit /b
    )

    echo Installing Python %PYTHON_VERSION%...
    start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 TargetDir="%PYTHON_DIR%"
    
    if exist "%PYTHON_DIR%\python.exe" (
        set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"
    ) else (
        echo [ERROR] Python installation failed.
        pause
        exit /b
    )
)

:: Confirm installation
python --version | findstr "%PYTHON_VERSION%" >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python %PYTHON_VERSION% still not found.
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
endlocal
