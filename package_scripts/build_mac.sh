#!/usr/bin/env bash
set -euo pipefail

# macOS: 使用 Conda 环境 fireball_calculator 与 PyInstaller 将
# source/image_segment/test_complete_propagation.py 打包为单文件可执行程序

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="fireball_calculator"
ENTRY_SCRIPT="source/image_segment/test_complete_propagation.py"
APP_NAME="image_segment_propagation"

echo "[macOS] 项目根目录: $PROJECT_ROOT"
echo "[macOS] Conda 环境: $ENV_NAME"
echo "[macOS] 打包入口: $ENTRY_SCRIPT"

# 若项目提供了环境激活脚本，优先尝试
if [ -f "$PROJECT_ROOT/source/activate_env.sh" ]; then
  echo "[macOS] 发现 source/activate_env.sh，尝试执行"
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/source/activate_env.sh" || true
fi

# 使用 conda 激活指定环境（需要安装 conda）
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "[macOS] 未检测到 conda，请先安装 Anaconda/Miniconda 并创建环境: $ENV_NAME" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install pyinstaller

# 清理旧产物
rm -rf dist build "${APP_NAME}.spec" 2>/dev/null || true

# 打包为单文件可执行
pyinstaller \
  --onefile \
  --clean \
  --noconfirm \
  --name "${APP_NAME}" \
  "$ENTRY_SCRIPT"

echo "[macOS] 打包完成，输出文件: $PROJECT_ROOT/dist/${APP_NAME}"
exit 0


