param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$dotenvPath = Join-Path $projectRoot ".env"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $key, $value = $trimmed.Split("=", 2)
        $key = $key.Trim()
        $value = $value.Trim().Trim("'").Trim('"')
        if ($key -and -not [string]::IsNullOrWhiteSpace($value) -and -not (Test-Path "Env:$key")) {
            Set-Item -Path ("Env:{0}" -f $key) -Value $value
        }
    }
}

function Test-AppImport {
    param([string]$Command, [string[]]$Arguments)

    & $Command @Arguments -c "from app.main import app; print('APP_IMPORT_OK')" 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Resolve-PythonCommand {
    if ($explicitPython) {
        $pythonCandidate = $explicitPython
        if (-not [System.IO.Path]::IsPathRooted($pythonCandidate)) {
            $pythonCandidate = Join-Path $projectRoot $pythonCandidate
        }
        if (-not (Test-Path $pythonCandidate)) {
            throw "WAV_PYTHON points to a missing interpreter: $explicitPython"
        }
        if (Test-AppImport -Command $pythonCandidate -Arguments @()) {
            return @($pythonCandidate)
        }
        throw "WAV_PYTHON exists, but cannot import app.main. Please install dependencies first."
    }

    if (Test-Path $venvPython) {
        if (Test-AppImport -Command $venvPython -Arguments @()) {
            return @($venvPython)
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        if (Test-AppImport -Command "python" -Arguments @()) {
            return @("python")
        }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        if (Test-AppImport -Command "py" -Arguments @("-3.12")) {
            return @("py", "-3.12")
        }
        if (Test-AppImport -Command "py" -Arguments @("-3")) {
            return @("py", "-3")
        }
    }

    throw "No usable Python interpreter could import app.main. Create .venv with tools/setup_env.ps1 or set WAV_PYTHON."
}

Import-DotEnv -Path $dotenvPath

$explicitPython = $env:WAV_PYTHON
$env:PYTHONPATH = $projectRoot
$env:FAIRSEQ_SKIP_HYDRA_INIT = "1"

$pythonCommand = Resolve-PythonCommand
Write-Host ("Selected Python: {0}" -f ($pythonCommand -join " "))

if ($DryRun) {
    return
}

if ($pythonCommand.Length -gt 1) {
    & $pythonCommand[0] @($pythonCommand[1..($pythonCommand.Length - 1)]) -m app.main
}
else {
    & $pythonCommand[0] -m app.main
}
