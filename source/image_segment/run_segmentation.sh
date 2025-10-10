#!/usr/bin/env bash
# 图像序列分割工具运行脚本
# 使用方法: ./run_segmentation.sh <json_path> [--no-viz] [--out=DIR]

set -euo pipefail

# 获取脚本所在目录并切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="fireball_calculator"

# 激活 conda 环境
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "错误: 未检测到 conda，请先安装 Anaconda/Miniconda" >&2
  exit 1
fi

# 运行分割脚本
python source/image_segment/test_complete_propagation.py "$@"

