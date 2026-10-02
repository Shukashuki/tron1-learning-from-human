@echo off
setlocal
if not exist "%~dp0outputs\sim2sim-cmu-16_03\index.html" (
  echo The sim2sim comparison is missing. Generate the GMR and dynamics runs first.
  pause
  exit /b 1
)
start "" "%~dp0outputs\sim2sim-cmu-16_03\index.html"
