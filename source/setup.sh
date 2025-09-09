#!/bin/bash

# 火球半径计算器环境设置脚本
# 此脚本将使用conda创建Python虚拟环境并安装所需依赖

echo "=========================================="
echo "火球半径计算器环境设置 (Conda版本)"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Miniconda或Anaconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda 已安装: $(conda --version)"

# 检查Python版本要求
PYTHON_VERSION="3.10"
ENV_NAME="fireball_calculator"

echo "✓ 将使用 Python ${PYTHON_VERSION} 创建环境: ${ENV_NAME}"

# 检查conda环境是否已存在
echo ""
echo "正在检查conda环境..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，正在删除..."
    conda env remove -n ${ENV_NAME} -y
fi

# 创建conda环境
echo ""
echo "正在创建conda环境..."

# 检查是否存在environment.yml文件
if [ -f "environment.yml" ]; then
    echo "检测到 environment.yml 文件，使用该文件创建环境..."
    conda env create -f environment.yml
else
    echo "使用默认配置创建环境..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

if [ $? -eq 0 ]; then
    echo "✓ Conda环境创建成功"
else
    echo "错误: Conda环境创建失败"
    exit 1
fi

# 激活conda环境
echo ""
echo "正在激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

if [ $? -eq 0 ]; then
    echo "✓ Conda环境激活成功"
    echo "当前Python路径: $(which python)"
    echo "当前Python版本: $(python --version)"
else
    echo "错误: Conda环境激活失败"
    exit 1
fi

# 升级pip
echo ""
echo "正在升级pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "正在安装依赖包..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ 依赖包安装成功"
else
    echo "错误: 依赖包安装失败"
    exit 1
fi

# 如存在本地子模块 segment-anything，则以可编辑模式安装
if [ -d "third_party/segment-anything" ]; then
    echo ""
    echo "检测到本地子模块: third_party/segment-anything"
    echo "正在以可编辑模式安装 Segment Anything ..."
    pip install -e third_party/segment-anything
    if [ $? -eq 0 ]; then
        echo "✓ Segment Anything 安装成功 (editable)"
    else
        echo "⚠️ Segment Anything 安装失败，请检查网络或子模块状态"
    fi
else
    echo ""
    echo "未检测到 third_party/segment-anything。"
    echo "如需使用 SAM，请先添加子模块："
    echo "  git submodule add https://github.com/facebookresearch/segment-anything.git source/third_party/segment-anything"
    echo "  git submodule update --init --recursive"
    echo "然后重新运行本脚本。"
fi

# 如存在本地子模块 sam2，则以可编辑模式安装
if [ -d "third_party/sam2" ]; then
    echo ""
    echo "检测到本地子模块: third_party/sam2"
    echo "正在以可编辑模式安装 SAM2 ..."
    
    # 检查Python版本是否满足SAM2要求
    PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
    PYTHON_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
    
    if [ $PYTHON_MAJOR -gt 3 ] || ([ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -ge 10 ]); then
        pip install -e third_party/sam2
        if [ $? -eq 0 ]; then
            echo "✓ SAM2 安装成功 (editable)"
        else
            echo "⚠️ SAM2 安装失败，请检查网络或子模块状态"
        fi
    else
        echo "⚠️ SAM2 需要 Python 3.10+，当前版本: ${PYTHON_VER}"
        echo "跳过 SAM2 安装"
    fi
else
    echo ""
    echo "未检测到 third_party/sam2。"
    echo "如需使用 SAM2，请先添加子模块："
    echo "  git submodule add git@github.com:facebookresearch/sam2.git source/third_party/sam2"
    echo "  git submodule update --init --recursive"
    echo "然后重新运行本脚本。"
fi

# 验证安装
echo ""
echo "正在验证安装..."
python -c "import numpy; import matplotlib; print('✓ 所有依赖包导入成功')"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "环境设置完成！"
    echo "=========================================="
    echo ""
    echo "使用方法:"
    echo "1. 激活conda环境: conda activate ${ENV_NAME}"
    echo "2. 运行计算器: python fireball_radius_calculator.py"
    echo "3. 退出conda环境: conda deactivate"
    echo ""
    echo "注意: 每次使用前都需要先激活conda环境"
    echo ""
    echo "环境管理命令:"
    echo "- 查看所有环境: conda env list"
    echo "- 删除环境: conda env remove -n ${ENV_NAME}"
    echo "- 导出环境: conda env export > environment.yml"
    echo ""
else
    echo "错误: 依赖包验证失败"
    exit 1
fi 