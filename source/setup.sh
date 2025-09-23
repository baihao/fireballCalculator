#!/bin/bash

# 简化版环境设置脚本
# 作用：创建或复用名为 fireball_calculator 的conda环境，并安装依赖

set -e

echo "=========================================="
echo "火球分析系统 - 环境设置 (简化版)"
echo "=========================================="

ENV_NAME="fireball_calculator"
PYTHON_VERSION="3.10"

# 1) 检查 conda
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 conda，请先安装 Miniconda/Anaconda"
    echo "参考: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✓ Conda 已安装: $(conda --version)"

# 2) 是否已存在环境
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✓ 已检测到环境: ${ENV_NAME} (将复用现有环境)"
else
    echo "未检测到环境，准备创建: ${ENV_NAME}"
    if [ -f "environment.yml" ]; then
        echo "使用 environment.yml 创建环境..."
        conda env create -f environment.yml -n ${ENV_NAME}
    else
        echo "使用默认配置创建环境 (python=${PYTHON_VERSION})..."
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
        # 安装 requirements.txt（若存在）
        if [ -f "requirements.txt" ]; then
            conda run -n ${ENV_NAME} python -m pip install --upgrade pip
            conda run -n ${ENV_NAME} python -m pip install -r requirements.txt
        fi
    fi
    echo "✓ 环境创建完成"
fi

# 若存在本地子模块，进行可选安装（不强制）
if [ -d "third_party/segment-anything" ]; then
    echo "检测到第三方子模块 segment-anything，尝试以可编辑模式安装..."
    conda run -n ${ENV_NAME} python -m pip install -e third_party/segment-anything || true
fi

echo ""
echo "=========================================="
echo "环境准备完成"
echo "=========================================="
echo ""
echo "下一步："
echo "1) 激活环境:    conda activate ${ENV_NAME}"
echo "2) 运行应用:    python source/desktop/app.py"
echo ""
echo "环境管理："
echo "- 删除环境:    conda env remove -n ${ENV_NAME} -y"
echo "- 查看环境:    conda env list"
echo ""
echo "提示：本脚本不会自动激活环境，请按上面的命令手动激活。"