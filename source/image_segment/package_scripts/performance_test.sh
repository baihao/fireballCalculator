#!/usr/bin/env bash
# 性能对比测试脚本
# 比较 Python 脚本、PyInstaller 和 Nuitka 三个版本的性能

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

TEST_FILE="test_data/fireball_sequence.json"
ENV_NAME="fireball_calculator"

echo "=========================================="
echo "火球图像分割 - 性能对比测试"
echo "=========================================="
echo "测试文件: $TEST_FILE"
echo "测试内容: 完整的图像序列分割（无可视化）"
echo ""

# 激活 conda 环境
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

# 创建临时输出目录
mkdir -p test_output_perf_{python,pyinstaller,nuitka}

echo "=========================================="
echo "测试 1/3: Python 脚本（基准性能）"
echo "=========================================="
if [ -f "source/image_segment/test_complete_propagation.py" ]; then
  echo "开始测试..."
  /usr/bin/time -l python source/image_segment/test_complete_propagation.py \
    "$TEST_FILE" --no-viz --out=test_output_perf_python 2>&1 | \
    grep -E "(real|user|sys|maximum resident|成功分割)" | tail -10
  echo "✓ Python 脚本测试完成"
else
  echo "⚠️  未找到 Python 脚本"
fi

echo ""
echo "=========================================="
echo "测试 2/3: PyInstaller 优化版"
echo "=========================================="
if [ -f "dist/image_segment_propagation/image_segment_propagation" ]; then
  echo "开始测试..."
  /usr/bin/time -l ./dist/image_segment_propagation/image_segment_propagation \
    "$TEST_FILE" --no-viz --out=test_output_perf_pyinstaller 2>&1 | \
    grep -E "(real|user|sys|maximum resident|成功分割)" | tail -10
  echo "✓ PyInstaller 版本测试完成"
else
  echo "⚠️  未找到 PyInstaller 版本"
  echo "   运行: ./source/image_segment/package_scripts/build_mac_optimized.sh"
fi

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "说明:"
echo "  real - 总运行时间（墙上时钟时间）"
echo "  user - CPU 用户态时间"
echo "  sys  - CPU 内核态时间"
echo "  maximum resident set size - 峰值内存占用"
echo ""
echo "清理测试输出:"
echo "  rm -rf test_output_perf_*"

# 清理临时输出（可选）
# rm -rf test_output_perf_*

