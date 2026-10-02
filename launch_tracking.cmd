@echo off
setlocal
if not exist "%~dp0outputs\tracking-pilot-preview\index.html" (
  if exist "%~dp0results\2026-10-02-tron1-jump\tracking_comparison.mp4" (
    start "" "%~dp0results\2026-10-02-tron1-jump\tracking_comparison.mp4"
    exit /b 0
  )
  echo The learned tracking preview is missing. Copy the saved evaluation preview first.
  pause
  exit /b 1
)
start "" "%~dp0outputs\tracking-pilot-preview\index.html"
