@echo off
setlocal

echo ===================================================
echo     Epi Predict - Startup Script
echo ===================================================

:: Check if virtual environment exists
if not exist "venv\Scripts\activate" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    echo [2/4] Installing dependencies...
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    echo [1/4] Virtual environment found. Activating...
    call venv\Scripts\activate
)

:: Check if models are trained
if not exist "models\saved_models\metadata.json" (
    echo.
    echo [3/4] Models are not trained yet. Starting initial training...
    echo This will download WHO data and train all AI models. Please wait...
    python scripts\train_models.py --country India --skip-lstm
) else (
    echo [3/4] Models are already trained.
)

echo.
echo [4/4] Starting the Application...

:: Kill existing Python processes on port 8000 and 8501 to prevent conflicts
FOR /F "tokens=5" %%a in ('netstat -aon ^| find "8000" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
FOR /F "tokens=5" %%a in ('netstat -aon ^| find "8501" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: Start FastAPI backend in a new minimized command prompt
start /min "Epi Predict Backend API" cmd /c "call venv\Scripts\activate && uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: Wait a few seconds to let the API start up
timeout /t 3 /nobreak > nul

:: Start Streamlit Dashboard (this will automatically open the browser)
echo Starting Streamlit Dashboard...
streamlit run dashboard\app.py

endlocal
