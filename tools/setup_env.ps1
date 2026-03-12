param(
    [string]$PythonCommand = "",
    [switch]$WithPreprocessTools
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"

function Resolve-BasePython {
    param([string]$Requested)

    if ($Requested) {
        if (Get-Command $Requested -ErrorAction SilentlyContinue) {
            return $Requested
        }
        if (Test-Path $Requested) {
            return $Requested
        }
        throw "Requested Python command is unavailable: $Requested"
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -V 1>$null 2>$null
        if ($LASTEXITCODE -eq 0) {
            return "py -3.12"
        }
        return "py -3"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    throw "Python 3 was not found. Install Python 3.10+ first, or pass -PythonCommand."
}

$resolved = Resolve-BasePython -Requested $PythonCommand
if (-not (Test-Path $venvPython)) {
    if ($resolved -eq "py -3.12") {
        & py -3.12 -m venv $venvRoot
    }
    elseif ($resolved -eq "py -3") {
        & py -3 -m venv $venvRoot
    }
    else {
        & $resolved -m venv $venvRoot
    }
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r $requirementsPath

if ($WithPreprocessTools) {
    & $venvPython -m pip install audio-separator
}

Write-Host "Environment ready."
Write-Host ("Python: {0}" -f $venvPython)
Write-Host "Next step:"
Write-Host "  .\\run.bat"
