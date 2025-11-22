#!/usr/bin/env bash
set -euo pipefail

# macOS 桌面应用打包脚本
# 使用 PyInstaller 打包 app.py 为独立的 macOS 应用

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="fireball_calculator"
ENTRY_SCRIPT="source/desktop/app.py"
APP_NAME="FireballAnalysisApp"

echo "[macOS 桌面应用] 项目根目录: $PROJECT_ROOT"
echo "[macOS 桌面应用] Conda 环境: $ENV_NAME"
echo "[macOS 桌面应用] 打包入口: $ENTRY_SCRIPT"

# 激活 conda 环境
if [ -f "$PROJECT_ROOT/source/activate_env.sh" ]; then
  source "$PROJECT_ROOT/source/activate_env.sh" || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "[macOS 桌面应用] 未检测到 conda" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install pyinstaller

# 更彻底地清理旧产物
echo "[macOS 桌面应用] 清理旧的构建产物..."

# 完全删除 dist 目录，避免符号链接冲突
if [ -d "dist" ]; then
  echo "[macOS 桌面应用] 完全清理 dist 目录（包括所有符号链接）..."
  # 先删除所有符号链接
  find "dist" -type l -exec rm -f {} \; 2>/dev/null || true
  # 然后删除整个目录
  rm -rf "dist" 2>/dev/null || true
fi

# 清理所有构建产物
rm -rf build "${APP_NAME}.spec" 2>/dev/null || true

# 清理 Python 缓存
rm -rf "__pycache__" 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

# 准备图标路径
ICON_PATH="source/desktop/icon/fireball_app_icon.icns"
if [ ! -f "$ICON_PATH" ]; then
  # 如果没有 .icns 文件，尝试使用 .png
  ICON_PATH="source/desktop/icon/fireball_app_icon.png"
  if [ ! -f "$ICON_PATH" ]; then
    echo "[macOS 桌面应用] 警告: 未找到应用图标，将使用默认图标"
    ICON_PATH=""
  fi
fi

# 打包参数
echo "[macOS 桌面应用] 开始打包桌面应用..."

PYINSTALLER_ARGS=(
  --onedir
  --windowed
  --clean
  --noconfirm
  --paths "source"
  --paths "source/desktop"
  --hidden-import PySide6
  --hidden-import PySide6.QtCore
  --hidden-import PySide6.QtGui
  --hidden-import PySide6.QtWidgets
  --hidden-import numpy
  --hidden-import scipy
  --hidden-import scipy.optimize
  --hidden-import matplotlib
  --hidden-import matplotlib.pyplot
  --hidden-import matplotlib.backends.backend_qt5agg
  --hidden-import cv2
  --hidden-import unittest
  --exclude-module tkinter
  --exclude-module pandas
  --exclude-module IPython
  --exclude-module jupyter
  --exclude-module notebook
  --exclude-module pytest
  --exclude-module torch
  --exclude-module torchvision
  --exclude-module tensorflow
  --name "${APP_NAME}"
)

# 注意：不使用 --collect-all，避免重复打包导致的符号链接冲突
# PyInstaller 会自动检测并包含必要的依赖

# 添加图标（如果存在）
if [ -n "$ICON_PATH" ]; then
  PYINSTALLER_ARGS+=(--icon "$ICON_PATH")
  echo "[macOS 桌面应用] 使用图标: $ICON_PATH"
fi

# 添加资源文件
if [ -d "source/desktop/resources" ]; then
  PYINSTALLER_ARGS+=(--add-data "source/desktop/resources:desktop/resources")
  echo "[macOS 桌面应用] 包含资源文件: source/desktop/resources"
fi

# 添加图标目录
if [ -d "source/desktop/icon" ]; then
  PYINSTALLER_ARGS+=(--add-data "source/desktop/icon:desktop/icon")
  echo "[macOS 桌面应用] 包含图标目录: source/desktop/icon"
fi

# 执行打包
echo "[macOS 桌面应用] 开始执行 PyInstaller 打包..."
pyinstaller "${PYINSTALLER_ARGS[@]}" "$ENTRY_SCRIPT"

# 清理打包目录中的不必要文件
echo "[macOS 桌面应用] 清理不必要的文件..."
if [ -d "dist/${APP_NAME}/_internal" ]; then
  # 删除 .pyc 文件
  find "dist/${APP_NAME}/_internal" -name "*.pyc" -delete
  # 删除 __pycache__ 目录
  find "dist/${APP_NAME}/_internal" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  # 删除测试文件
  find "dist/${APP_NAME}/_internal" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
  find "dist/${APP_NAME}/_internal" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
  
  # 删除 PyTorch 相关目录（如果被打包进来了）
  echo "[macOS 桌面应用] 清理 PyTorch 相关目录..."
  rm -rf "dist/${APP_NAME}/_internal/torch" 2>/dev/null || true
  rm -rf "dist/${APP_NAME}/_internal/torchvision" 2>/dev/null || true
  rm -rf "dist/${APP_NAME}/_internal/tensorflow" 2>/dev/null || true
fi

# 创建 macOS .app 包（可选，如果需要标准的 .app 格式）
if [ -f "dist/${APP_NAME}/${APP_NAME}" ]; then
  chmod +x "dist/${APP_NAME}/${APP_NAME}"
  echo "[macOS 桌面应用] 已添加可执行权限"
  
  # 显示打包结果
  FINAL_SIZE=$(du -sh "dist/${APP_NAME}" | cut -f1)
  echo ""
  echo "[macOS 桌面应用] 打包完成！"
  echo "[macOS 桌面应用] 输出目录: $PROJECT_ROOT/dist/${APP_NAME}/"
  echo "[macOS 桌面应用] 总大小: $FINAL_SIZE"
  echo "[macOS 桌面应用] 可执行文件: $PROJECT_ROOT/dist/${APP_NAME}/${APP_NAME}"
  echo ""
  echo "使用方法:"
  echo "  cd $PROJECT_ROOT"
  echo "  ./dist/${APP_NAME}/${APP_NAME}"
  echo ""
  echo "或者直接双击运行:"
  echo "  open dist/${APP_NAME}/${APP_NAME}"
fi

exit 0

