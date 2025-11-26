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
  --hidden-import torch
  --hidden-import torch.nn
  --hidden-import torch.nn.functional
  --hidden-import torch.backends
  --hidden-import torch.backends.mps
  --hidden-import segment_anything
  --hidden-import segment_anything.sam_model_registry
  --hidden-import segment_anything.predictor
  --collect-all torch
  --collect-all segment_anything
  --exclude-module tkinter
  --exclude-module pandas
  --exclude-module IPython
  --exclude-module jupyter
  --exclude-module notebook
  --exclude-module pytest
  --exclude-module tensorflow
  --name "${APP_NAME}"
)

# 注意：使用 --collect-all torch 和 --collect-all segment_anything
# 以确保包含所有必要的依赖，因为桌面应用直接调用分割模块

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

# 添加 SAM 第三方库（如果需要）
if [ -d "source/third_party/segment-anything" ]; then
  # 创建临时目录，只复制必要的 SAM 文件
  TEMP_SAM_DIR="$PROJECT_ROOT/temp_sam_desktop"
  rm -rf "$TEMP_SAM_DIR"
  mkdir -p "$TEMP_SAM_DIR/segment-anything/segment_anything"
  mkdir -p "$TEMP_SAM_DIR/segment-anything/checkpoints"
  
  echo "[macOS 桌面应用] 准备最小化的 SAM 文件..."
  
  # 只复制 SAM Python 代码（不包括 demo/notebooks/assets）
  cp -r source/third_party/segment-anything/segment_anything/* "$TEMP_SAM_DIR/segment-anything/segment_anything/" 2>/dev/null || true
  
  # 只复制需要的模型检查点（默认 vit_b，约 375MB）
  # 如果需要其他模型，取消注释相应行
  CHECKPOINT_COUNT=0
  if ls source/third_party/segment-anything/checkpoints/sam_vit_b*.pth 1> /dev/null 2>&1; then
    cp source/third_party/segment-anything/checkpoints/sam_vit_b*.pth "$TEMP_SAM_DIR/segment-anything/checkpoints/" 2>/dev/null || true
    CHECKPOINT_COUNT=$((CHECKPOINT_COUNT + 1))
    echo "[macOS 桌面应用] ✓ 已包含 vit_b 模型检查点"
  fi
  
  # 可选：如果需要 vit_l 模型，取消下面的注释
  # if ls source/third_party/segment-anything/checkpoints/sam_vit_l*.pth 1> /dev/null 2>&1; then
  #   cp source/third_party/segment-anything/checkpoints/sam_vit_l*.pth "$TEMP_SAM_DIR/segment-anything/checkpoints/" 2>/dev/null || true
  #   CHECKPOINT_COUNT=$((CHECKPOINT_COUNT + 1))
  #   echo "[macOS 桌面应用] ✓ 已包含 vit_l 模型检查点"
  # fi
  
  # 可选：如果需要 vit_h 模型，取消下面的注释（注意：文件很大，约 2.4GB）
  # if ls source/third_party/segment-anything/checkpoints/sam_vit_h*.pth 1> /dev/null 2>&1; then
  #   cp source/third_party/segment-anything/checkpoints/sam_vit_h*.pth "$TEMP_SAM_DIR/segment-anything/checkpoints/" 2>/dev/null || true
  #   CHECKPOINT_COUNT=$((CHECKPOINT_COUNT + 1))
  #   echo "[macOS 桌面应用] ✓ 已包含 vit_h 模型检查点"
  # fi
  
  if [ $CHECKPOINT_COUNT -eq 0 ]; then
    echo "[macOS 桌面应用] 警告: 未找到任何 SAM 模型检查点文件"
    echo "[macOS 桌面应用] 分割功能可能无法正常工作"
  else
    # 显示检查点文件大小
    CHECKPOINT_SIZE=$(du -sh "$TEMP_SAM_DIR/segment-anything/checkpoints" 2>/dev/null | cut -f1 || echo "未知")
    echo "[macOS 桌面应用] 检查点文件大小: $CHECKPOINT_SIZE (包含 $CHECKPOINT_COUNT 个模型)"
  fi
  
  # 添加临时 SAM 目录到打包参数
  PYINSTALLER_ARGS+=(--add-data "$TEMP_SAM_DIR/segment-anything:third_party/segment-anything")
  echo "[macOS 桌面应用] 包含 SAM 第三方库: source/third_party/segment-anything"
fi

# 执行打包
echo "[macOS 桌面应用] 开始执行 PyInstaller 打包..."
pyinstaller "${PYINSTALLER_ARGS[@]}" "$ENTRY_SCRIPT"

# 清理临时 SAM 目录
if [ -d "$TEMP_SAM_DIR" ]; then
  echo "[macOS 桌面应用] 清理临时 SAM 目录..."
  rm -rf "$TEMP_SAM_DIR"
fi

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
  
  # 删除 TensorFlow 相关目录（不需要）
  echo "[macOS 桌面应用] 清理 TensorFlow 相关目录..."
  rm -rf "dist/${APP_NAME}/_internal/tensorflow" 2>/dev/null || true
  
  # 注意：不再删除 PyTorch 相关目录，因为桌面应用直接调用分割模块需要 PyTorch
  echo "[macOS 桌面应用] 保留 PyTorch 和 SAM 相关依赖（分割模块需要）"
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

