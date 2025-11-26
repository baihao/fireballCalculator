$ErrorActionPreference = "Stop"

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=========================================="
Write-Host "Fireball Calculator - Environment Setup (Windows PowerShell)"
Write-Host "=========================================="

$EnvName = "fireball_calculator"
$PythonVersion = "3.10"

# 1) Check conda
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Conda is not available. Please install Miniconda/Anaconda first."
    Write-Host "Reference: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}
$condaVersion = conda --version
Write-Host "[OK] Conda detected: $condaVersion"

# Accept conda Terms of Service if needed
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
        # ToS might already be accepted, which is fine
        $errorMsg = $tosOutput | Out-String
        if ($errorMsg -notmatch "already accepted") {
            Write-Host "[INFO] ToS status for $channel : $errorMsg"
        }
    }
}

# 2) Check if environment exists
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
    Write-Host "[OK] Environment already exists: $EnvName (will reuse existing environment)"
} else {
    Write-Host "Environment not found, creating: $EnvName"
    if (Test-Path "environment.yml") {
        # Create base environment first with Python only
        Write-Host "Creating base environment with Python only..."
        conda create -n $EnvName python=3.9 -y
        
        # Install conda packages one by one, avoiding packages that trigger Qt installation
        Write-Host "Installing conda packages (avoiding Qt dependencies)..."
        conda install -n $EnvName -y pip numpy>=2.0.0 pandas>=2.3.0 pillow>=11.0.0 scipy>=1.13.0
        
        # Install matplotlib using pip instead of conda to avoid Qt dependencies
        Write-Host "Installing matplotlib via pip to avoid Qt dependencies..."
        conda run -n $EnvName python -m pip install matplotlib>=3.9.0
        
        # Install pip dependencies manually
        Write-Host "Installing pip dependencies..."
        conda run -n $EnvName python -m pip install --upgrade pip
        conda run -n $EnvName python -m pip install --ignore-installed "PySide6==6.10.1" "opencv-python>=4.11.0" "torch>=2.0.0" "torchvision>=0.15.0"
    } else {
        Write-Host "Creating environment with default settings (python=$PythonVersion)..."
        conda create -n $EnvName "python=$PythonVersion" -y
        if (Test-Path "requirements.txt") {
            conda run -n $EnvName python -m pip install --upgrade pip
            conda run -n $EnvName python -m pip install -r requirements.txt
        }
    }
    Write-Host "[OK] Environment created"
    
    # Get environment path after creation
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
    Write-Host "[WARNING] Unable to locate environment path. Skipping env var hook setup."
    exit 0
}

# Install third-party submodule if exists
if (Test-Path "third_party/segment-anything" -PathType Container) {
    Write-Host "Detected third-party submodule segment-anything, attempting editable install..."
    try {
        conda run -n $EnvName python -m pip install -e third_party/segment-anything
    } catch {
        Write-Host "[WARNING] segment-anything install failed (ignored): $($_.Exception.Message)"
    }
}

# 3) Configure environment variables for OpenMP conflicts
Write-Host ""
Write-Host "Configuring environment variables (resolving OpenMP conflicts)..."

$activateDir = Join-Path $envPath "etc\conda\activate.d"
$deactivateDir = Join-Path $envPath "etc\conda\deactivate.d"

New-Item -ItemType Directory -Path $activateDir -Force | Out-Null
New-Item -ItemType Directory -Path $deactivateDir -Force | Out-Null

# Create activation script for PowerShell
$activateScript = Join-Path $activateDir "env_vars.ps1"
$activateContent = @'
# Mitigate OpenMP library conflicts
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Optionally add project root to PYTHONPATH
if ($env:FIREBALL_PROJECT_ROOT) {
    $projectSource = Join-Path $env:FIREBALL_PROJECT_ROOT "source"
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$projectSource;$($env:PYTHONPATH)"
    } else {
        $env:PYTHONPATH = $projectSource
    }
}
'@

# Create deactivation script for PowerShell
$deactivateScript = Join-Path $deactivateDir "env_vars.ps1"
$deactivateContent = @'
# Cleanup environment variables
if (Test-Path Env:KMP_DUPLICATE_LIB_OK) {
    Remove-Item Env:KMP_DUPLICATE_LIB_OK
}
'@

Set-Content -Path $activateScript -Value $activateContent -Encoding UTF8
Set-Content -Path $deactivateScript -Value $deactivateContent -Encoding UTF8

Write-Host "[OK] Environment variables configured"
Write-Host "  - Automatically set KMP_DUPLICATE_LIB_OK=TRUE (resolves OpenMP conflicts)"
Write-Host "  - Applied automatically on activate, cleaned on deactivate"

Write-Host ""
Write-Host "=========================================="
Write-Host "Environment ready"
Write-Host "=========================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "1) Activate the environment:    conda activate $EnvName"
Write-Host "   (Environment variable KMP_DUPLICATE_LIB_OK will be set automatically)"
Write-Host "2) Run the app:                 python source/desktop/app.py"
Write-Host "3) Run segmentation:            python source/image_segment/test_complete_propagation.py <json_file>"
Write-Host ""
Write-Host "Environment management:"
Write-Host "- Remove:    conda env remove -n $EnvName -y"
Write-Host "- List:      conda env list"
Write-Host ""
Write-Host "Note: This script does not activate the environment automatically."
Write-Host "If 'conda activate' doesn't work, initialize conda first: conda init powershell"

