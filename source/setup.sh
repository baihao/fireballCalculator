#!/bin/bash

# 火球半径计算器环境设置脚本
# 此脚本将创建Python虚拟环境并安装所需依赖

echo "=========================================="
echo "火球半径计算器环境设置"
echo "=========================================="

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "错误: 未找到pip3，请先安装pip3"
    exit 1
fi

echo "✓ Python3 已安装: $(python3 --version)"
echo "✓ pip3 已安装: $(pip3 --version)"

# 创建虚拟环境
echo ""
echo "正在创建Python虚拟环境..."
if [ -d "venv" ]; then
    echo "虚拟环境已存在，正在删除..."
    rm -rf venv
fi

python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✓ 虚拟环境创建成功"
else
    echo "错误: 虚拟环境创建失败"
    exit 1
fi

# 激活虚拟环境
echo ""
echo "正在激活虚拟环境..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ 虚拟环境激活成功"
    echo "当前Python路径: $(which python)"
    echo "当前Python版本: $(python --version)"
else
    echo "错误: 虚拟环境激活失败"
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
    echo "1. 激活虚拟环境: source venv/bin/activate"
    echo "2. 运行计算器: python fireball_radius_calculator.py"
    echo "3. 退出虚拟环境: deactivate"
    echo ""
    echo "注意: 每次使用前都需要先激活虚拟环境"
    echo ""
else
    echo "错误: 依赖包验证失败"
    exit 1
fi 