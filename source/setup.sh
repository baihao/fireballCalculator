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

# 3) 解决 OpenMP 库冲突（PyTorch 和其他库可能都包含 OpenMP）
echo ""
echo "配置环境变量（解决 OpenMP 冲突）..."
CONDA_ENV_DIR="${CONDA_PREFIX%/*}/${ENV_NAME}"
ACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

# 创建激活脚本
cat > "$ACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/bash
# 解决 OpenMP 库冲突
export KMP_DUPLICATE_LIB_OK=TRUE

# 添加项目根目录到 PYTHONPATH（可选，便于导入模块）
if [ -n "${FIREBALL_PROJECT_ROOT:-}" ]; then
    export PYTHONPATH="${FIREBALL_PROJECT_ROOT}/source:${PYTHONPATH:-}"
fi
EOF

# 创建停用脚本
cat > "$DEACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/bash
# 清理环境变量
unset KMP_DUPLICATE_LIB_OK
EOF

chmod +x "$ACTIVATE_DIR/env_vars.sh"
chmod +x "$DEACTIVATE_DIR/env_vars.sh"

echo "✓ 环境变量配置完成"
echo "  - 自动设置 KMP_DUPLICATE_LIB_OK=TRUE（解决 OpenMP 冲突）"
echo "  - 激活环境时自动应用，停用时自动清理"

echo ""
echo "=========================================="
echo "环境准备完成"
echo "=========================================="
echo ""
echo "下一步："
echo "1) 激活环境:    conda activate ${ENV_NAME}"
echo "   （环境变量 KMP_DUPLICATE_LIB_OK 将自动设置）"
echo "2) 运行应用:    python source/desktop/app.py"
echo "3) 运行分割:    python source/image_segment/test_complete_propagation.py <json_file>"
echo ""
echo "环境管理："
echo "- 删除环境:    conda env remove -n ${ENV_NAME} -y"
echo "- 查看环境:    conda env list"
echo ""
echo "提示：本脚本不会自动激活环境，请按上面的命令手动激活。"