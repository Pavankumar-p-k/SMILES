@echo off
setlocal

cd /d "%~dp0"

set "HOLO_HOST=127.0.0.1"
set "HOLO_PORT=5173"
if not "%~1"=="" set "HOLO_HOST=%~1"
if not "%~2"=="" set "HOLO_PORT=%~2"

set "STREAMLIT_HOST=127.0.0.1"
set "STREAMLIT_PORT=8501"
if not "%~3"=="" set "STREAMLIT_PORT=%~3"

set "SILICO_PY=%~dp0.runtime_venv\Scripts\python.exe"
set "STREAMLIT_APP=%~dp0app.py"
set "HOLO_DIR=%~dp0holographic-brain-ui"

if not exist "%SILICO_PY%" (
  echo [in_silico_admet] Python runtime not found at "%SILICO_PY%".
  pause
  exit /b 1
)

if not exist "%STREAMLIT_APP%" (
  echo [in_silico_admet] Streamlit app not found at "%STREAMLIT_APP%".
  pause
  exit /b 1
)

if not exist "%HOLO_DIR%\run-ui.bat" (
  echo [in_silico_admet] Holographic UI launcher not found at "%HOLO_DIR%\run-ui.bat".
  pause
  exit /b 1
)

echo [in_silico_admet] Checking Streamlit dashboard on http://%STREAMLIT_HOST%:%STREAMLIT_PORT% ...
powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://%STREAMLIT_HOST%:%STREAMLIT_PORT%/_stcore/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
  echo [in_silico_admet] Starting Streamlit dashboard...
  start "Streamlit Dashboard (in_silico_admet)" /min "%SILICO_PY%" -m streamlit run "%STREAMLIT_APP%" --server.address %STREAMLIT_HOST% --server.port %STREAMLIT_PORT% --browser.gatherUsageStats false
  powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 30;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://%STREAMLIT_HOST%:%STREAMLIT_PORT%/_stcore/health' -TimeoutSec 2; if($r.StatusCode -eq 200){ $ok=$true; break } } catch {}; Start-Sleep -Milliseconds 500 }; if($ok){ exit 0 } else { exit 1 }"
  if errorlevel 1 (
    echo [in_silico_admet] Warning: Streamlit did not become healthy in time.
  ) else (
    echo [in_silico_admet] Streamlit is online at http://%STREAMLIT_HOST%:%STREAMLIT_PORT%/
  )
) else (
  echo [in_silico_admet] Streamlit already running.
)

cd /d "%HOLO_DIR%"
call run-ui.bat %HOLO_HOST% %HOLO_PORT%

endlocal
