@echo off
echo Starting AIS Law Enforcement LLM Application...

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo No virtual environment found, using system Python
)

REM Run the application with local data source by default
python run.py --mode local

REM If there was an error, pause to show the message
if %ERRORLEVEL% neq 0 (
    echo Application failed to start with error code %ERRORLEVEL%
    pause
)
