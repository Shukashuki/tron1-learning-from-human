@echo off
setlocal
if not exist "%~dp0outputs\mink-cmu-16_03\index.html" (
  echo Mink preview is missing. Generate the IK and render outputs first.
  pause
  exit /b 1
)
start "" "%~dp0outputs\mink-cmu-16_03\index.html"
