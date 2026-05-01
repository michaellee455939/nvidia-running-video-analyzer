@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found in PATH.
  echo Install Python 3 from https://www.python.org/downloads/windows/
  echo During installation, enable "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo Setup finished, but ffmpeg was not found in PATH.
  echo Install it with:
  echo   winget install Gyan.FFmpeg
  echo Then reopen this terminal.
) else (
  echo.
  echo Setup finished. ffmpeg is available.
)

echo.
echo To run the keyword extractor GUI:
echo   run_keyword_extractor_windows.bat
pause
