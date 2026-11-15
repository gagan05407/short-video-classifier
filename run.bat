@echo off
echo Starting Multimodal Video Moderation System...

start "Moderation Backend" cmd /k "call %~dp0venv\Scripts\activate.bat && cd /d %~dp0backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 /nobreak >nul
start "Moderation Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:3000

