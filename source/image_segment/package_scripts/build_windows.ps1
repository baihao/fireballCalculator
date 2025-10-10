Param(
  [string]$EnvName = "fireball_calculator",
  [string]$EntryScript = "source/image_segment/test_complete_propagation.py",
  [string]$AppName = "image_segment_propagation"
)

$ErrorActionPreference = "Stop"

# 获取项目根目录（从 source/image_segment/package_scripts 向上三级）
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path (Join-Path (Join-Path $ScriptDir "..") "..") "..")
Set-Location $ProjectRoot

Write-Host "[Windows] 项目根目录: $ProjectRoot"
Write-Host "[Windows] Conda 环境: $EnvName"
Write-Host "[Windows] 打包入口: $EntryScript"

# 若存在项目内激活脚本，先尝试执行
$ActivateScript = Join-Path $ProjectRoot "source/activate_env.sh"
if (Test-Path $ActivateScript) {
  Write-Host "[Windows] 检测到 source/activate_env.sh（仅 *nix），Windows 将忽略。"
}

# 激活 conda 环境
if (Get-Command conda -ErrorAction SilentlyContinue) {
  (& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
  conda activate $EnvName
} else {
  Write-Error "未检测到 conda，请在 Windows 打包机上安装 Anaconda/Miniconda 并创建环境: $EnvName"
}

python -m pip install --upgrade pip
python -m pip install pyinstaller

# 清理旧产物
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Remove-Item -Force "$AppName.spec" -ErrorAction SilentlyContinue

# 打包为目录模式（而非单文件），PyTorch 在目录模式下更容易打包成功
# 添加 source 目录到 Python 路径，确保能找到 image_segment 模块
# 使用 --hidden-import 显式包含可能被遗漏的模块
Write-Host "[Windows] 开始打包（目录模式，便于处理 PyTorch 等大型依赖）..."
pyinstaller `
  --onedir `
  --clean `
  --noconfirm `
  --paths "source" `
  --add-data "source/third_party;third_party" `
  --hidden-import torch `
  --hidden-import torch.nn `
  --hidden-import torch.nn.functional `
  --hidden-import torchvision `
  --hidden-import cv2 `
  --hidden-import numpy `
  --hidden-import matplotlib `
  --hidden-import matplotlib.pyplot `
  --hidden-import segment_anything `
  --hidden-import segment_anything.modeling `
  --hidden-import segment_anything.predictor `
  --hidden-import segment_anything.utils `
  --collect-all torch `
  --collect-all segment_anything `
  --copy-metadata segment_anything `
  --name "$AppName" `
  "$EntryScript"

# 输出打包结果信息
$ExePath = Join-Path $ProjectRoot "dist\$AppName\$AppName.exe"
if (Test-Path $ExePath) {
  Write-Host "[Windows] 打包完成！"
  Write-Host "[Windows] 输出目录: $ProjectRoot\dist\$AppName\"
  Write-Host "[Windows] 可执行文件: $ExePath"
  Write-Host ""
  Write-Host "使用方法:"
  Write-Host "  cd $ProjectRoot"
  Write-Host "  .\dist\$AppName\$AppName.exe test_data\fireball_sequence.json"
} else {
  Write-Host "[Windows] 打包完成，输出文件: $ProjectRoot\dist\$AppName.exe"
}

exit 0

