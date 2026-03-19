#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Fireball Calculator - Windows Desktop Build" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Convert-Path (Join-Path $ScriptDir "..\..\..")
Set-Location $ProjectRoot

$EnvName = "fireball_calculator"
$EntryScript = "source\desktop\app.py"
$AppName = "FireballAnalysisApp"

Write-Host "[Info] Project root: $ProjectRoot"
Write-Host "[Info] Conda env: $EnvName"
Write-Host "[Info] Entry script: $EntryScript"

# Conda check
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[Error] Conda is not available. Please install Miniconda/Anaconda." -ForegroundColor Red
    exit 1
}

# Activate conda env (skip if already active)
Write-Host "[Step] Activating conda environment..."
if ($env:CONDA_DEFAULT_ENV -eq $EnvName) {
    Write-Host "[OK] Env already active: $EnvName"
} else {
    $condaBase = (conda info --base 2>$null)
    if (-not $condaBase) {
        Write-Host "[Error] Unable to locate conda base. Ensure conda is initialized." -ForegroundColor Red
        exit 1
    }
    $condaHook = Join-Path $condaBase "shell\condabin\conda-hook.ps1"
    if (Test-Path $condaHook) {
        & $condaHook
        conda activate $EnvName
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[Error] Failed to activate env $EnvName" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[Error] conda-hook.ps1 not found. Ensure conda is initialized." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Env activated: $EnvName"
}

# Install PyInstaller
Write-Host "[Step] Installing PyInstaller..."
python -m pip install --upgrade pip
python -m pip install pyinstaller

# Clean old artifacts
Write-Host "[Step] Cleaning previous build outputs..."
Remove-Item -Path "dist" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$AppName.spec" -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Env vars
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Icon
$IconPath = "source\desktop\icon\fireball_app_icon.ico"
if (-not (Test-Path $IconPath)) {
    $IconPath = "source\desktop\icon\fireball_app_icon.png"
    if (-not (Test-Path $IconPath)) {
        Write-Host "[Warn] Icon not found. Using default icon."
        $IconPath = ""
    }
}

# PyInstaller args
Write-Host "[Step] Building desktop app..."
$PyArgs = @(
    "--onedir",
    "--windowed",
    "--clean",
    "--noconfirm",
    "--additional-hooks-dir", "source\desktop\package_scripts",
    "--paths", "source",
    "--paths", "source\desktop",
    "--hidden-import", "PySide6",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    "--hidden-import", "numpy",
    "--hidden-import", "scipy",
    "--hidden-import", "scipy.optimize",
    "--hidden-import", "matplotlib",
    "--hidden-import", "matplotlib.pyplot",
    "--hidden-import", "matplotlib.backends.backend_qt5agg",
    "--hidden-import", "cv2",
    "--hidden-import", "unittest",
    "--hidden-import", "torch",
    "--hidden-import", "torch.nn",
    "--hidden-import", "torch.nn.functional",
    "--hidden-import", "torch.backends",
    "--hidden-import", "torch.backends.cuda",
    "--hidden-import", "torch.library",
    "--hidden-import", "torch.fx",
    "--hidden-import", "torch.nested",
    "--hidden-import", "torchvision",
    "--hidden-import", "torchvision.transforms",
    "--hidden-import", "torchvision.ops",
    "--hidden-import", "segment_anything",
    "--hidden-import", "segment_anything.sam_model_registry",
    "--hidden-import", "segment_anything.predictor",
    "--collect-all", "torch",
    "--collect-all", "segment_anything",
    "--exclude-module", "tkinter",
    "--exclude-module", "pandas",
    "--exclude-module", "IPython",
    "--exclude-module", "jupyter",
    "--exclude-module", "notebook",
    "--exclude-module", "pytest",
    "--exclude-module", "tensorflow",
    "--name", $AppName
)

if ($IconPath) {
    $PyArgs += "--icon"
    $PyArgs += $IconPath
    Write-Host "[Info] Using icon: $IconPath"
}

if (Test-Path "source\desktop\resources") {
    $PyArgs += "--add-data"
    $PyArgs += "source\desktop\resources;desktop\resources"
    Write-Host "[Info] Adding resources: source\desktop\resources"
}

if (Test-Path "source\desktop\icon") {
    $PyArgs += "--add-data"
    $PyArgs += "source\desktop\icon;desktop\icon"
    Write-Host "[Info] Adding icon folder: source\desktop\icon"
}

# Include SAM third-party minimal set if present
if (Test-Path "source\third_party\segment-anything") {
    $TempSam = Join-Path $ProjectRoot "temp_sam_desktop"
    Remove-Item -Path $TempSam -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path (Join-Path $TempSam "segment-anything\segment_anything") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $TempSam "segment-anything\checkpoints") -Force | Out-Null

    Write-Host "[Step] Preparing minimal SAM package..."
    if (Test-Path "source\third_party\segment-anything\segment_anything") {
        Copy-Item -Path "source\third_party\segment-anything\segment_anything\*" -Destination (Join-Path $TempSam "segment-anything\segment_anything\") -Recurse -Force -ErrorAction SilentlyContinue
    }
    $ckptCount = 0
    $vitB = Get-ChildItem -Path "source\third_party\segment-anything\checkpoints" -Filter "sam_vit_b*.pth" -ErrorAction SilentlyContinue
    if ($vitB) {
        Copy-Item -Path $vitB.FullName -Destination (Join-Path $TempSam "segment-anything\checkpoints\") -Force -ErrorAction SilentlyContinue
        $ckptCount = $vitB.Count
        Write-Host "[Info] Included vit_b checkpoints ($ckptCount)"
    } else {
        Write-Host "[Warn] No SAM checkpoints found. Segmentation may not work."
    }
    $PyArgs += "--add-data"
    $PyArgs += "$TempSam\segment-anything;third_party/segment-anything"
}

# Run PyInstaller
Write-Host "[Step] Running PyInstaller..."
pyinstaller $PyArgs $EntryScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] PyInstaller build failed." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] PyInstaller build finished."

# Create MPS stub to avoid import errors on Windows
$mpsDir = "dist\$AppName\_internal\torch\backends\mps"
$mpsBase = "dist\$AppName\_internal\torch\backends"
if (-not (Test-Path $mpsBase)) { New-Item -ItemType Directory -Path $mpsBase -Force | Out-Null }
if (Test-Path $mpsDir) { Remove-Item -Path $mpsDir -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $mpsDir -Force | Out-Null
$stub = @"
# MPS backend stub for Windows
def is_available():
    return False

def is_built():
    return False
"@
Set-Content -Path (Join-Path $mpsDir "__init__.py") -Value $stub -Encoding UTF8
Write-Host "[OK] MPS stub created."

# Clean temp SAM dir
if ($TempSam -and (Test-Path $TempSam)) {
    Write-Host "[Step] Cleaning temp SAM directory..."
    Remove-Item -Path $TempSam -Recurse -Force -ErrorAction SilentlyContinue
}

# Clean build artifacts in dist
Write-Host "[Step] Cleaning dist artifacts..."
$internalPath = "dist\$AppName\_internal"
if (Test-Path $internalPath) {
    Get-ChildItem -Path $internalPath -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $internalPath -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $internalPath -Recurse -Directory -Filter "tests" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $internalPath -Recurse -Directory -Filter "test" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    # Keep torch.testing (PyTorch may import it); do not remove torch/testing
    # Optional: remove quantization if not used
    Remove-Item -Path "$internalPath\torch\ao" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[Info] Removed pyc, __pycache__, test folders."
}

# Show result
$exePath = "dist\$AppName\$AppName.exe"
if (Test-Path $exePath) {
    $totalSizeMB = [math]::Round((Get-ChildItem -Path "dist\$AppName" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 2)
    Write-Host ""
    Write-Host "Build complete!"
    Write-Host "Output: $ProjectRoot\dist\$AppName\"
    Write-Host "Size: $totalSizeMB MB"
    Write-Host "Executable: $ProjectRoot\$exePath"
    Write-Host ""
    Write-Host "Run:"
    Write-Host "  cd $ProjectRoot"
    Write-Host "  .\\dist\\$AppName\\$AppName.exe"
} else {
    Write-Host "[Error] Executable not found: $exePath" -ForegroundColor Red
    exit 1
}

exit 0

