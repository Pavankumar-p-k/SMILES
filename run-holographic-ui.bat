@echo off
setlocal

cd /d "%~dp0holographic-brain-ui"
if errorlevel 1 (
  echo [in_silico_admet] holographic-brain-ui folder not found.
  pause
  exit /b 1
)

call run-ui.bat %*

endlocal
