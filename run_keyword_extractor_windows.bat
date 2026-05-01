@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if not defined NVIDIA_API_KEY (
  echo NVIDIA_API_KEY is not set.
  echo Set it in this terminal with:
  echo   set NVIDIA_API_KEY=your_api_key
  echo Or permanently with:
  echo   setx NVIDIA_API_KEY "your_api_key"
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found in PATH.
  echo Install ffmpeg and reopen this terminal, then run this script again.
  echo Recommended: winget install Gyan.FFmpeg
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" running_clip_extractor.py
) else (
  python running_clip_extractor.py
)

pause
