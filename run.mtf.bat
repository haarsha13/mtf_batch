@echo off
setlocal
set SCRIPT_DIR=%~dp0
echo Running from: %SCRIPT_DIR%

REM --- 1) Find Python (prefer existing venv, else system python) ---
set "PYEXE="
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" set "PYEXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
if "%PYEXE%"=="" for %%P in (python.exe) do (for /f "delims=" %%I in ('where %%P 2^>NUL') do set "PYEXE=%%I" & goto :gotpy)
:gotpy

if "%PYEXE%"=="" (
  echo.
  echo [ERROR] Python not found.
  echo Install Python 3.11/3.12 from python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)

REM --- 2) Ensure a local venv exists and has deps ---
if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  "%PYEXE%" -m venv "%SCRIPT_DIR%.venv"
  "%SCRIPT_DIR%.venv\Scripts\python.exe" -m pip install --upgrade pip
  if exist "%SCRIPT_DIR%requirements.txt" "%SCRIPT_DIR%.venv\Scripts\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
)

REM --- 3) Run the tool ---
"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%run_mtf.py"
pause
