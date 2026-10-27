@echo off
title SmartChunk — Self-Healing RAG Pipeline
color 0A
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║               SmartChunk RAG Pipeline                    ║
echo  ║        Structure-Aware · Self-Healing · Offline          ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

:: ── Check Python ────────────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% detected

:: ── Check / install packages ────────────────────────────────────────
echo.
echo [*] Checking dependencies...
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] pip install failed.
        pause
        exit /b 1
    )
) else (
    echo [OK] Core packages already installed
)

:: ── Check / download MiniLM model ───────────────────────────────────
echo.
echo [*] Checking local embedding model...
if not exist "outputs\models" mkdir outputs\models

python -c "import os; os.environ['SENTENCE_TRANSFORMERS_HOME']='outputs/models'; os.environ['HF_HOME']='outputs/models'; from sentence_transformers import SentenceTransformer; m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='outputs/models'); print('[OK] Model ready:', m.get_sentence_embedding_dimension(), 'dims')"
if %errorlevel% neq 0 (
    echo [WARN] Model download may have failed - will retry on first request.
)

:: ── Start server ────────────────────────────────────────────────────
echo.
echo ═══════════════════════════════════════════════════════════════
echo   Starting SmartChunk at http://127.0.0.1:8000
echo   Press Ctrl+C to stop
echo ═══════════════════════════════════════════════════════════════
echo.

:: Open browser after 1 second delay
start "" cmd /c "timeout /t 1 /nobreak >nul && start http://127.0.0.1:8000"

:: Launch Uvicorn
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

pause
