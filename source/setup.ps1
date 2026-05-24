# Fireball / MOGP environment (Windows): conda creates Python + pip only; pip installs all deps.
# Same strategy as setup.sh + requirements.txt; PyTorch CUDA 12.8 is installed by this script.
# Usage: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Continue"

function Write-Status {
    param(
        [ValidateSet('OK', 'ERROR', 'INFO', 'WARNING')]
        [string]$Level,
        [string]$Message
    )
    Write-Host ('[' + $Level + '] ' + $Message)
}

function Assert-LastExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        Write-Status ERROR "$Step failed (exit code $LASTEXITCODE)"
        exit 1
    }
}

function Invoke-CondaPython {
    param(
        [string]$EnvName,
        [string]$Code
    )

    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & conda run -n $EnvName python -c $Code 2>&1
        return @{
            Output = $output
            ExitCode = $LASTEXITCODE
        }
    } finally {
        $ErrorActionPreference = $prevErrorAction
    }
}

function Test-PyTorchCUDA128 {
    param([string]$EnvName)

    $result = Invoke-CondaPython -EnvName $EnvName -Code @"
import sys, torch
ver = torch.__version__.split('+')[0].split('.')
major, minor = int(ver[0]), int(ver[1])
patch = int(ver[2]) if len(ver) > 2 else 0
cuda = getattr(torch.version, 'cuda', None)
ok = cuda == '12.8' and ((major, minor, patch) >= (2, 5, 1))
sys.exit(0 if ok else 1)
"@
    return ($result.ExitCode -eq 0)
}

function Install-PyTorchCUDA128 {
    param([string]$EnvName)

    Write-Host "Installing PyTorch (Windows / CUDA 12.8)..."
    if (Test-PyTorchCUDA128 -EnvName $EnvName) {
        $info = Invoke-CondaPython -EnvName $EnvName -Code "import torch; print(torch.__version__)"
        Write-Status OK "PyTorch $($info.Output) with CUDA 12.8 already installed, skipping"
        return
    }

    Write-Host "  (This may take a few minutes, downloading ~2-3GB)..."
    conda run -n $EnvName python -m pip install 'torch>=2.5.1,<3' 'torchvision>=0.20.0' --index-url https://download.pytorch.org/whl/cu128
    Assert-LastExitCode "PyTorch CUDA 12.8 install"
    Write-Status OK "PyTorch CUDA 12.8 installed"
    $verify = Invoke-CondaPython -EnvName $EnvName -Code "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', getattr(torch.version, 'cuda', 'N/A'))"
    Write-Host ($verify.Output | Out-String)
}

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ScriptDir

$EnvName = "fireball_calculator"
$PythonVersion = "3.10"
$ReqFile = Join-Path $ScriptDir "requirements.txt"

Write-Host "=========================================="
Write-Host "Fireball Calculator - Environment Setup (Windows / pip)"
Write-Host "Working directory: $ScriptDir"
Write-Host "=========================================="

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Status ERROR "conda not found. Install Miniconda/Anaconda first."
    Write-Host "Reference: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}
Write-Status OK "Conda detected: $(conda --version)"

if (-not (Test-Path $ReqFile)) {
    Write-Status ERROR "Missing file: $ReqFile"
    exit 1
}

Write-Host "Accepting conda Terms of Service for required channels..."
$channels = @(
    "https://repo.anaconda.com/pkgs/main",
    "https://repo.anaconda.com/pkgs/r",
    "https://repo.anaconda.com/pkgs/msys2"
)
foreach ($channel in $channels) {
    $tosOutput = conda tos accept --override-channels --channel $channel 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Accepted ToS for: $channel"
    } else {
        $errorMsg = $tosOutput | Out-String
        if ($errorMsg -notmatch "already accepted") {
            Write-Status INFO "ToS status for ${channel}: $errorMsg"
        }
    }
}

$existingEnvs = conda env list --json | ConvertFrom-Json
$envExists = $false
$envPath = $null

foreach ($env in $existingEnvs.envs) {
    $envNameOnly = Split-Path -Leaf $env
    if ($envNameOnly -eq $EnvName) {
        $envExists = $true
        $envPath = $env
        break
    }
}

if ($envExists) {
    Write-Status OK "Environment exists: $EnvName. Syncing via pip and requirements.txt..."
} else {
    Write-Host "Creating environment: $EnvName (Python $PythonVersion, base packages + pip only)..."
    conda create -n $EnvName -c defaults --override-channels "python=$PythonVersion" pip -y
    Assert-LastExitCode "conda environment create"
    Write-Status OK "Environment created"

    $existingEnvs = conda env list --json | ConvertFrom-Json
    foreach ($env in $existingEnvs.envs) {
        $envNameOnly = Split-Path -Leaf $env
        if ($envNameOnly -eq $EnvName) {
            $envPath = $env
            break
        }
    }
}

conda run -n $EnvName python -m pip install --upgrade pip
Assert-LastExitCode "pip upgrade"

Install-PyTorchCUDA128 -EnvName $EnvName

Write-Host "Running pip install -r requirements.txt (gpytorch and other deps)..."
conda run -n $EnvName python -m pip install -r $ReqFile --upgrade
Assert-LastExitCode "requirements install"
Write-Status OK "pip dependencies installed/updated"

$segmentAnythingPath = Join-Path $ScriptDir "third_party/segment-anything"
if (Test-Path $segmentAnythingPath -PathType Container) {
    Write-Host "Found third-party submodule segment-anything, attempting editable install..."
    conda run -n $EnvName python -m pip install -e $segmentAnythingPath 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Status OK "segment-anything installed"
    } else {
        Write-Status WARNING "segment-anything install failed (ignored)"
    }
}

if (-not $envPath) {
    $existingEnvs = conda env list --json | ConvertFrom-Json
    foreach ($env in $existingEnvs.envs) {
        $envNameOnly = Split-Path -Leaf $env
        if ($envNameOnly -eq $EnvName) {
            $envPath = $env
            break
        }
    }
}

if (-not $envPath) {
    Write-Status WARNING "Could not locate environment path. Skipping env var hook setup."
    exit 0
}

Write-Host ""
Write-Host "Configuring environment variables (OpenMP conflict workaround)..."

$activateDir = Join-Path $envPath "etc\conda\activate.d"
$deactivateDir = Join-Path $envPath "etc\conda\deactivate.d"

New-Item -ItemType Directory -Path $activateDir -Force | Out-Null
New-Item -ItemType Directory -Path $deactivateDir -Force | Out-Null

$activateScript = Join-Path $activateDir "env_vars.ps1"
$activateContent = @'
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
if ($env:FIREBALL_PROJECT_ROOT) {
    $projectSource = Join-Path $env:FIREBALL_PROJECT_ROOT "source"
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$projectSource;$($env:PYTHONPATH)"
    } else {
        $env:PYTHONPATH = $projectSource
    }
}
'@

$deactivateScript = Join-Path $deactivateDir "env_vars.ps1"
$deactivateContent = @'
if (Test-Path Env:KMP_DUPLICATE_LIB_OK) {
    Remove-Item Env:KMP_DUPLICATE_LIB_OK
}
'@

Set-Content -Path $activateScript -Value $activateContent -Encoding UTF8
Set-Content -Path $deactivateScript -Value $deactivateContent -Encoding UTF8

Write-Status OK "Environment variables configured"
Write-Host "  - KMP_DUPLICATE_LIB_OK=TRUE is set on activate and removed on deactivate"

Write-Host ""
Write-Host "=========================================="
Write-Host "Environment ready"
Write-Host "=========================================="
Write-Host ""
Write-Host "Dependencies: $ReqFile (pip) + PyTorch CUDA 12.8 from setup.ps1"
Write-Host "Python $PythonVersion, GPyTorch, Matplotlib, etc."
Write-Host ""
Write-Host "Next steps:"
Write-Host "1) Activate:      conda activate $EnvName"
Write-Host "2) Run app:       cd `"$ProjectRoot`" ; python source/desktop/app.py"
Write-Host "3) Segmentation:  cd `"$ProjectRoot`" ; python source/image_segment/test_complete_propagation.py {json_file}"
Write-Host "4) MOGP CLI:      cd `"$ProjectRoot`" ; `$env:PYTHONPATH='source' ; python -m gp_model.cli train --help"
Write-Host ""
Write-Host "Environment management:"
Write-Host "- Re-sync deps:   powershell -ExecutionPolicy Bypass -File $ScriptDir\setup.ps1"
Write-Host "- Remove env:     conda env remove -n $EnvName -y"
Write-Host ""
Write-Host "Note: environment.yml is for reference only; requirements.txt is the source of truth."
Write-Host "This script does not activate the environment automatically."
Write-Host "If conda activate fails, run: conda init powershell"
