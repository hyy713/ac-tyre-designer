@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
)

if not exist ".venv\Lib\site-packages\numpy" (
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Dependency installation failed. Check your network or PyPI mirror.
    pause
    exit /b 1
  )
)

".venv\Scripts\pythonw.exe" run.py
if errorlevel 1 (
  ".venv\Scripts\python.exe" run.py
  pause
)

