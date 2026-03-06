@echo off
setlocal

cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=5173"
if not "%~1"=="" set "HOST=%~1"
if not "%~2"=="" set "PORT=%~2"

set "API_HOST=127.0.0.1"
set "API_PORT=5051"
set "SILICO_ROOT=%~dp0.."
for %%I in ("%SILICO_ROOT%") do set "SILICO_ROOT=%%~fI"
set "SILICO_PY=%SILICO_ROOT%\.runtime_venv\Scripts\python.exe"
set "API_SCRIPT=%~dp0tools\admet_api_server.py"

where node >nul 2>&1
if errorlevel 1 (
  echo [holographic-brain-ui] Node.js is not installed or not on PATH.
  pause
  exit /b 1
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo [holographic-brain-ui] npm is not available on PATH.
  pause
  exit /b 1
)

if not exist "%SILICO_ROOT%" (
  echo [holographic-brain-ui] Silico project not found at "%SILICO_ROOT%".
  pause
  exit /b 1
)

if not exist "%SILICO_PY%" (
  echo [holographic-brain-ui] Silico Python runtime not found at "%SILICO_PY%".
  pause
  exit /b 1
)

if not exist "%API_SCRIPT%" (
  echo [holographic-brain-ui] API bridge script not found at "%API_SCRIPT%".
  pause
  exit /b 1
)

if not exist node_modules (
  echo [holographic-brain-ui] Installing dependencies...
  call npm.cmd install
  if errorlevel 1 (
    echo [holographic-brain-ui] Dependency install failed.
    pause
    exit /b 1
  )
)

echo [holographic-brain-ui] Checking ADMET API on http://%API_HOST%:%API_PORT%/health ...
powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://%API_HOST%:%API_PORT%/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
  echo [holographic-brain-ui] Starting ADMET API backend...
  start "ADMET API (holographic-brain-ui)" /min "%SILICO_PY%" "%API_SCRIPT%" --silico-root "%SILICO_ROOT%" --host %API_HOST% --port %API_PORT%
  powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 30;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://%API_HOST%:%API_PORT%/health' -TimeoutSec 2; if($r.StatusCode -eq 200){ $ok=$true; break } } catch {}; Start-Sleep -Milliseconds 500 }; if($ok){ exit 0 } else { exit 1 }"
  if errorlevel 1 (
    echo [holographic-brain-ui] Warning: ADMET API did not become healthy in time.
  ) else (
    echo [holographic-brain-ui] ADMET API is online.
  )
) else (
  echo [holographic-brain-ui] ADMET API already running.
)

echo [holographic-brain-ui] Starting dev server on http://%HOST%:%PORT%/
call npm.cmd run dev -- --host %HOST% --port %PORT%
if errorlevel 1 (
  echo [holographic-brain-ui] Dev server exited with an error.
  pause
  exit /b 1
)

endlocal
