@echo off
REM ===================================================
REM AIS Law Enforcement LLM Application Launcher
REM ===================================================

echo ===================================================
echo  AIS Law Enforcement LLM - Starting Application
echo ===================================================

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [INFO] No virtual environment found, using system Python
)

REM Check for .env file
if exist .env (
    echo [OK] Found .env configuration file
) else (
    echo [WARNING] No .env file found in root directory
    
    REM Check for .env in backend directory
    if exist backend\.env (
        echo [OK] Found .env in backend directory
        echo [INFO] Copying .env from backend to root directory
        copy backend\.env .env
    ) else (
        echo [WARNING] No .env file found! Application may not function correctly.
        echo [INFO] Using environment variables and defaults.
    )
)

REM Get mode parameter
set MODE=local
if not "%1"=="" (
    set MODE=%1
)

REM Check if run.py exists
if exist run.py (
    echo [INFO] Starting application in %MODE% mode...
    python run.py --mode %MODE%
) else (
    echo [WARNING] run.py not found, using legacy startup...
    
    REM Use alternative startup based on what's available
    if exist backend\start_local.py (
        echo [INFO] Starting with backend\start_local.py
        cd backend
        python start_local.py
    ) else if exist app.py (
        echo [INFO] Starting with app.py
        uvicorn app:app --reload
    ) else (
        echo [ERROR] No startup script found. Exiting.
        pause
        exit /b 1
    )
)

REM If the application exited with an error, pause to show the message
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Application exited with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo [INFO] Application terminated successfully.
exit /b 0
