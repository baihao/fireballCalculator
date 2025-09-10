#!/bin/bash

# 火球计算器环境别名设置脚本
# 此脚本为当前终端会话设置 python310 和 pip310 别名

ENV_NAME="fireball_calculator"
ENV_PYTHON_PATH="/Users/hbai/miniconda3/envs/${ENV_NAME}/bin/python"
ENV_PIP_PATH="/Users/hbai/miniconda3/envs/${ENV_NAME}/bin/pip"

# 检查环境是否存在
if [ ! -f "${ENV_PYTHON_PATH}" ]; then
    echo "错误: 环境 ${ENV_NAME} 不存在或 Python 路径不正确"
    echo "请先运行 ./setup.sh 创建环境"
    exit 1
fi

# 设置别名
alias python310="${ENV_PYTHON_PATH}"
alias pip310="${ENV_PIP_PATH}"

echo "✓ 别名设置成功！"
echo "  python310 -> ${ENV_PYTHON_PATH}"
echo "  pip310 -> ${ENV_PIP_PATH}"
echo ""
echo "现在您可以使用:"
echo "  python310 --version"
echo "  pip310 list"
echo "  python310 app.py"
echo ""
echo "注意: 这些别名只在当前终端会话中有效"
echo "要永久设置，请将以下行添加到 ~/.zshrc 或 ~/.bashrc:"
echo "  alias python310='${ENV_PYTHON_PATH}'"
echo "  alias pip310='${ENV_PIP_PATH}'"
