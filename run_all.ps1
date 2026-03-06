param(
    [switch]$LaunchApp,
    [string]$Data = "data/admet_multitask.csv",
    [switch]$SkipTrain,
    [switch]$SkipTests,
    [switch]$SkipSystemCheck
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
}

$args = @(
    (Join-Path $projectRoot "run_all.py"),
    "--data", $Data
)

if ($LaunchApp) { $args += "--launch-app" }
if ($SkipTrain) { $args += "--skip-train" }
if ($SkipTests) { $args += "--skip-tests" }
if ($SkipSystemCheck) { $args += "--skip-system-check" }

& $python @args
exit $LASTEXITCODE
