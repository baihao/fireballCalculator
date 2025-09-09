#!/bin/bash

# Conda 安装脚本
# 用于安装 Miniconda（如果未安装）

echo "=========================================="
echo "Conda 安装脚本"
echo "=========================================="

# 检查系统类型
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        # Intel Mac
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
else
    echo "不支持的操作系统: $OSTYPE"
    exit 1
fi

echo "检测到系统: $OSTYPE"
echo "下载地址: $CONDA_URL"

# 检查conda是否已安装
if command -v conda &> /dev/null; then
    echo "✓ Conda 已安装: $(conda --version)"
    echo "跳过安装步骤"
    exit 0
fi

# 下载并安装Miniconda
echo ""
echo "正在下载 Miniconda..."
wget $CONDA_URL -O miniconda.sh

if [ $? -eq 0 ]; then
    echo "✓ 下载完成"
else
    echo "❌ 下载失败，请手动下载: $CONDA_URL"
    exit 1
fi

echo ""
echo "正在安装 Miniconda..."
bash miniconda.sh -b -p $HOME/miniconda3

if [ $? -eq 0 ]; then
    echo "✓ 安装完成"
    echo ""
    echo "请运行以下命令初始化conda:"
    echo "  $HOME/miniconda3/bin/conda init"
    echo "  然后重新打开终端或运行: source ~/.bashrc"
    echo ""
    echo "或者直接使用完整路径:"
    echo "  $HOME/miniconda3/bin/conda create -n fireball_calculator python=3.10"
else
    echo "❌ 安装失败"
    exit 1
fi

# 清理下载文件
rm miniconda.sh
