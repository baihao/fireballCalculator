#!/bin/bash

# Qt库冲突检测脚本
# 此脚本检测是否存在Qt库冲突，如果存在则自动修复

ENV_NAME="fireball_calculator"
ENV_PATH="/Users/hbai/miniconda3/envs/${ENV_NAME}"

# 检查环境是否存在
if [ ! -d "${ENV_PATH}" ]; then
    echo "错误: 环境 ${ENV_NAME} 不存在"
    exit 1
fi

echo "正在检测Qt库冲突..."

# 检查是否存在多个Qt库
QT_CONDA_COUNT=$(find "${ENV_PATH}/lib" -name "*Qt6Core*" -type f 2>/dev/null | wc -l)
QT_PYSIDE_COUNT=$(find "${ENV_PATH}/lib/python3.10/site-packages/PySide6" -name "*QtCore*" -type f 2>/dev/null | wc -l)

echo "  conda Qt库数量: ${QT_CONDA_COUNT}"
echo "  PySide6 Qt库数量: ${QT_PYSIDE_COUNT}"

# 检查是否存在qt6-main包
if conda list -n ${ENV_NAME} | grep -q "qt6-main"; then
    echo "  ⚠️ 检测到qt6-main包，可能导致Qt冲突"
    HAS_QT_CONFLICT=true
else
    echo "  ✓ 未检测到qt6-main包"
    HAS_QT_CONFLICT=false
fi

# 检查OpenCV包类型
if conda list -n ${ENV_NAME} | grep -q "opencv.*qt6"; then
    echo "  ⚠️ 检测到conda OpenCV Qt版本，可能导致Qt冲突"
    HAS_QT_CONFLICT=true
else
    echo "  ✓ 未检测到conda OpenCV Qt版本"
fi

if [ "$HAS_QT_CONFLICT" = true ]; then
    echo ""
    echo "检测到Qt库冲突，正在自动修复..."
    if [ -f "./fix_qt_conflict.sh" ]; then
        ./fix_qt_conflict.sh
        if [ $? -eq 0 ]; then
            echo "✓ Qt冲突修复完成"
            exit 0
        else
            echo "⚠️ Qt冲突修复失败"
            exit 1
        fi
    else
        echo "错误: 未找到fix_qt_conflict.sh脚本"
        exit 1
    fi
else
    echo "✓ 未检测到Qt库冲突"
    exit 0
fi
