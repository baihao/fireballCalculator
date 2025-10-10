#!/usr/bin/env bash
set -euo pipefail

# macOS 优化版打包脚本
# 只打包必要的库和模型文件，显著减小体积和提升速度

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="fireball_calculator"
ENTRY_SCRIPT="source/image_segment/test_complete_propagation.py"
APP_NAME="image_segment_propagation"

echo "[macOS 优化版] 项目根目录: $PROJECT_ROOT"
echo "[macOS 优化版] Conda 环境: $ENV_NAME"
echo "[macOS 优化版] 打包入口: $ENTRY_SCRIPT"

# 激活 conda 环境
if [ -f "$PROJECT_ROOT/source/activate_env.sh" ]; then
  source "$PROJECT_ROOT/source/activate_env.sh" || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "[macOS 优化版] 未检测到 conda" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install pyinstaller

# 清理旧产物
rm -rf dist build "${APP_NAME}.spec" 2>/dev/null || true

# 创建临时目录，只复制必要的 SAM 文件
TEMP_SAM_DIR="$PROJECT_ROOT/temp_sam_minimal"
rm -rf "$TEMP_SAM_DIR"
mkdir -p "$TEMP_SAM_DIR/segment-anything/segment_anything"
mkdir -p "$TEMP_SAM_DIR/segment-anything/checkpoints"

echo "[macOS 优化版] 准备最小化的 SAM 文件..."

# 只复制 SAM Python 代码（不包括 demo/notebooks/assets）
cp -r source/third_party/segment-anything/segment_anything/* "$TEMP_SAM_DIR/segment-anything/segment_anything/"

# 只复制需要的模型检查点（默认 vit_b，约 375MB）
# 如果需要其他模型，取消注释相应行
cp source/third_party/segment-anything/checkpoints/sam_vit_b*.pth "$TEMP_SAM_DIR/segment-anything/checkpoints/" 2>/dev/null || echo "警告: 未找到 vit_b 模型"
# cp source/third_party/segment-anything/checkpoints/sam_vit_l*.pth "$TEMP_SAM_DIR/segment-anything/checkpoints/" 2>/dev/null || true
# cp source/third_party/segment-anything/checkpoints/sam_vit_h*.pth "$TEMP_SAM_DIR/segment-anything/checkpoints/" 2>/dev/null || true

echo "[macOS 优化版] SAM 文件准备完成"

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

# 优化版打包参数
echo "[macOS 优化版] 开始打包（已优化：排除不必要的模块）..."
pyinstaller \
  --onedir \
  --clean \
  --noconfirm \
  --paths "source" \
  --add-data "$TEMP_SAM_DIR/segment-anything:third_party/segment-anything" \
  --hidden-import torch \
  --hidden-import torch.nn \
  --hidden-import torch.nn.functional \
  --hidden-import cv2 \
  --hidden-import numpy \
  --hidden-import matplotlib \
  --hidden-import matplotlib.pyplot \
  --hidden-import matplotlib.backends.backend_agg \
  --hidden-import segment_anything \
  --hidden-import segment_anything.modeling \
  --hidden-import segment_anything.predictor \
  --hidden-import segment_anything.utils \
  --collect-all torch \
  --collect-all segment_anything \
  --exclude-module tkinter \
  --exclude-module pandas \
  --exclude-module IPython \
  --exclude-module jupyter \
  --exclude-module notebook \
  --name "${APP_NAME}" \
  "$ENTRY_SCRIPT"

# 清理临时目录
rm -rf "$TEMP_SAM_DIR"

# 清理打包目录中的不必要文件
echo "[macOS 优化版] 清理不必要的文件..."
if [ -d "dist/${APP_NAME}/_internal" ]; then
  # 删除 .pyc 文件
  find "dist/${APP_NAME}/_internal" -name "*.pyc" -delete
  # 删除 __pycache__ 目录
  find "dist/${APP_NAME}/_internal" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  # 删除测试文件
  find "dist/${APP_NAME}/_internal" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
  find "dist/${APP_NAME}/_internal" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
fi

# 添加可执行权限
if [ -f "dist/${APP_NAME}/${APP_NAME}" ]; then
  chmod +x "dist/${APP_NAME}/${APP_NAME}"
  echo "[macOS 优化版] 已添加可执行权限"
  
  # 显示打包结果
  FINAL_SIZE=$(du -sh "dist/${APP_NAME}" | cut -f1)
  echo ""
  echo "[macOS 优化版] 打包完成！"
  echo "[macOS 优化版] 输出目录: $PROJECT_ROOT/dist/${APP_NAME}/"
  echo "[macOS 优化版] 总大小: $FINAL_SIZE"
  echo "[macOS 优化版] 可执行文件: $PROJECT_ROOT/dist/${APP_NAME}/${APP_NAME}"
  echo ""
  echo "使用方法:"
  echo "  cd $PROJECT_ROOT"
  echo "  ./dist/${APP_NAME}/${APP_NAME} test_data/fireball_sequence.json --no-viz"
fi

exit 0

