@echo off
setlocal
if not exist "%~dp0outputs\cmu-16_03\index.html" (
  echo Motion preview is missing. Run scripts/render_mocap.py in the motion environment first.
  pause
  exit /b 1
)
start "" "%~dp0outputs\cmu-16_03\index.html"
