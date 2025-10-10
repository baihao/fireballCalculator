Param(
  [string]$EnvName = "fireball_calculator",
  [string]$EntryScript = "source/image_segment/test_complete_propagation.py",
  [string]$AppName = "image_segment_propagation"
)

$ErrorActionPreference = "Stop"

# 切换到项目根目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
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

# 打包为单文件可执行
pyinstaller `
  --onefile `
  --clean `
  --noconfirm `
  --name "$AppName" `
  "$EntryScript"

Write-Host "[Windows] 打包完成，输出文件: $ProjectRoot/dist/$AppName.exe"
exit 0


