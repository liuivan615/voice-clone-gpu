param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimeRoot = Join-Path $projectRoot "runtime\softvc_clean"
$workspace = Join-Path $projectRoot "smoke_train_runtime"

if (-not (Test-Path $venvPython)) {
    throw "Missing .venv. Run tools/setup_env.ps1 first."
}

if (-not (Test-Path $SourceDir)) {
    throw "SourceDir does not exist: $SourceDir"
}

$env:PYTHONPATH = $projectRoot
$env:FAIRSEQ_SKIP_HYDRA_INIT = "1"

& $venvPython (Join-Path $projectRoot "tools\prepare_smoke_training.py") `
    --source-dir $SourceDir `
    --runtime-root $runtimeRoot `
    --workspace $workspace `
    --speaker smoke `
    --limit 8 `
    --f0-predictor dio

if ($LASTEXITCODE -ne 0) {
    throw "prepare_smoke_training.py failed with exit code $LASTEXITCODE"
}

& $venvPython (Join-Path $projectRoot "tools\smoke_train_step.py") `
    --runtime-root $runtimeRoot `
    --workspace $workspace `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "smoke_train_step.py failed with exit code $LASTEXITCODE"
}

Write-Host "Smoke training completed."
