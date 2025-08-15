@echo off
setlocal
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe

REM Prefer your venv if it exists
if exist "%VENV_PY%" (
  "%VENV_PY%" "%SCRIPT_DIR%run_mtf.py"
) else (
  REM fallback to system Python
  py "%SCRIPT_DIR%run_mtf.py"
  if %errorlevel% neq 0 python "%SCRIPT_DIR%run_mtf.py"
)
pause
