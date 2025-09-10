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
ENV_PYTHON_PATH="/Users/hbai/miniconda3/envs/${ENV_NAME}/bin/python"
ENV_PIP_PATH="/Users/hbai/miniconda3/envs/${ENV_NAME}/bin/pip"

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
    echo "环境Python路径: ${ENV_PYTHON_PATH}"
    echo "环境Python版本: $(${ENV_PYTHON_PATH} --version)"
    echo "环境pip路径: ${ENV_PIP_PATH}"
else
    echo "错误: Conda环境激活失败"
    exit 1
fi

# 检查pip模块是否已安装
echo ""
echo "正在检查pip模块..."
if ! ${ENV_PYTHON_PATH} -c "import pip" 2>/dev/null; then
    echo "pip模块未安装，正在安装..."
    ${ENV_PYTHON_PATH} -m ensurepip --upgrade
    if [ $? -eq 0 ]; then
        echo "✓ pip模块安装成功"
    else
        echo "错误: pip模块安装失败"
        exit 1
    fi
else
    echo "✓ pip模块已存在"
fi

# 升级pip
echo ""
echo "正在升级pip..."
${ENV_PIP_PATH} install --upgrade pip

# 安装额外依赖（如果requirements.txt存在且与environment.yml不同）
echo ""
echo "正在检查额外依赖..."
if [ -f "requirements.txt" ]; then
    echo "检测到 requirements.txt 文件"
    echo "注意: 基础依赖已通过 environment.yml 安装"
    echo "正在安装 requirements.txt 中的额外依赖..."
    ${ENV_PIP_PATH} install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ 额外依赖安装成功"
    else
        echo "⚠️ 部分依赖安装失败，但环境仍可使用"
    fi
else
    echo "未检测到 requirements.txt 文件，跳过额外依赖安装"
fi

# 如存在本地子模块 segment-anything，则以可编辑模式安装
if [ -d "third_party/segment-anything" ]; then
    echo ""
    echo "检测到本地子模块: third_party/segment-anything"
    echo "正在以可编辑模式安装 Segment Anything ..."
    ${ENV_PIP_PATH} install -e third_party/segment-anything
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
    PYTHON_VER=$(${ENV_PYTHON_PATH} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(${ENV_PYTHON_PATH} -c "import sys; print(sys.version_info.major)")
    PYTHON_MINOR=$(${ENV_PYTHON_PATH} -c "import sys; print(sys.version_info.minor)")
    
    if [ $PYTHON_MAJOR -gt 3 ] || ([ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -ge 10 ]); then
        ${ENV_PIP_PATH} install -e third_party/sam2
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

# 检查并修复Qt库冲突
echo ""
echo "正在检查Qt库冲突..."
if [ -f "./check_qt_conflict.sh" ]; then
    echo "运行Qt冲突检测脚本..."
    ./check_qt_conflict.sh
    if [ $? -eq 0 ]; then
        echo "✓ Qt冲突检查完成"
    else
        echo "⚠️ Qt冲突检测或修复失败，但环境仍可使用"
    fi
else
    echo "未检测到Qt冲突检测脚本，跳过Qt检查"
fi

# 验证安装
echo ""
echo "正在验证安装..."
${ENV_PYTHON_PATH} -c "import numpy; import matplotlib; print('✓ 所有依赖包导入成功')"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "环境设置完成！"
    echo "=========================================="
    echo ""
    echo "✓ 环境信息:"
    echo "  环境名称: ${ENV_NAME}"
    echo "  Python版本: $(${ENV_PYTHON_PATH} --version)"
    echo "  Python路径: ${ENV_PYTHON_PATH}"
    echo "  pip路径: ${ENV_PIP_PATH}"
    echo ""
    
    # 设置当前会话的别名
    echo "正在设置当前会话的别名..."
    alias python310="${ENV_PYTHON_PATH}"
    alias pip310="${ENV_PIP_PATH}"
    echo "✓ 别名设置成功！"
    echo ""
    
    echo "✓ 快速使用命令:"
    echo "1. 激活conda环境: conda activate ${ENV_NAME}"
    echo "2. 设置别名: source ./setup_aliases.sh"
    echo "3. 使用环境Python: python310 fireball_radius_calculator.py"
    echo "4. 使用环境pip: pip310 install package_name"
    echo "5. 退出conda环境: conda deactivate"
    echo ""
    echo "✓ 环境管理命令:"
    echo "- 查看所有环境: conda env list"
    echo "- 删除环境: conda env remove -n ${ENV_NAME}"
    echo "- 导出环境: conda env export > environment.yml"
    echo "- 重新创建环境: ./setup.sh"
    echo "- 设置别名: source ./setup_aliases.sh"
    echo ""
    echo "✓ 永久别名设置 (添加到 ~/.zshrc 或 ~/.bashrc):"
    echo "  alias python310='${ENV_PYTHON_PATH}'"
    echo "  alias pip310='${ENV_PIP_PATH}'"
    echo ""
    echo "✓ 注意事项:"
    echo "- 每次使用前都需要先激活conda环境"
    echo "- 当前会话已设置 python310 和 pip310 别名"
    echo "- 如果遇到Python版本问题，请使用完整路径: ${ENV_PYTHON_PATH}"
    echo ""
else
    echo "错误: 依赖包验证失败"
    exit 1
fi 