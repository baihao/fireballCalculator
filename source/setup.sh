#!/bin/bash

# 火球分析 / MOGP 环境：用 conda 仅创建「Python + pip」基环境，依赖全部由 pip 安装/更新
#（避免 conda-forge repodata 网络失败；与 document/fireball_gp_mogp_module_design.md 对齐）
# 用法：bash /path/to/source/setup.sh  或  cd source && ./setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "火球分析系统 - 环境设置（pip 安装依赖）"
echo "工作目录: ${SCRIPT_DIR}"
echo "=========================================="

ENV_NAME="fireball_calculator"
PYTHON_VERSION="3.10"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

# 1) 检查 conda
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 conda，请先安装 Miniconda/Anaconda"
    echo "参考: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✓ Conda 已安装: $(conda --version)"

if [ ! -f "${REQ_FILE}" ]; then
    echo "❌ 未找到 ${REQ_FILE}"
    exit 1
fi

# 2) 创建空环境（仅用 defaults channel，避免 conda-forge 元数据拉取失败）
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "✓ 已存在环境: ${ENV_NAME}，使用 pip 升级/同步 requirements.txt ..."
else
    echo "创建环境: ${ENV_NAME}（Python ${PYTHON_VERSION}，仅 base 包 + pip）..."
    # --override-channels：忽略全局 conda 里配置的 conda-forge，减少无效请求
    conda create -n "${ENV_NAME}" -c defaults --override-channels "python=${PYTHON_VERSION}" pip -y
    echo "✓ 环境创建完成"
fi

echo "正在 pip install -r requirements.txt（含 torch / gpytorch 等）..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}" --upgrade

echo "✓ pip 依赖已安装/更新"

# 若存在本地子模块，进行可选安装（不强制）
if [ -d "${SCRIPT_DIR}/third_party/segment-anything" ]; then
    echo "检测到第三方子模块 segment-anything，尝试以可编辑模式安装..."
    conda run -n "${ENV_NAME}" python -m pip install -e "${SCRIPT_DIR}/third_party/segment-anything" || true
fi

# 3) 解决 OpenMP 库冲突（PyTorch 和其他库可能都包含 OpenMP）
echo ""
echo "配置环境变量（解决 OpenMP 冲突）..."
CONDA_BASE="$(conda info --base)"
CONDA_ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
ACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
if [ -n "${FIREBALL_PROJECT_ROOT:-}" ]; then
    export PYTHONPATH="${FIREBALL_PROJECT_ROOT}/source:${PYTHONPATH:-}"
fi
EOF

cat > "$DEACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/bash
unset KMP_DUPLICATE_LIB_OK
EOF

chmod +x "$ACTIVATE_DIR/env_vars.sh"
chmod +x "$DEACTIVATE_DIR/env_vars.sh"

echo "✓ 环境变量配置完成"
echo "  - 自动设置 KMP_DUPLICATE_LIB_OK=TRUE（解决 OpenMP 冲突）"

echo ""
echo "=========================================="
echo "环境准备完成"
echo "=========================================="
echo ""
echo "依赖来源: ${REQ_FILE}（pip）；Python ${PYTHON_VERSION}、PyTorch、GPyTorch、Matplotlib 等"
echo ""
echo "下一步："
echo "1) 激活环境:    conda activate ${ENV_NAME}"
echo "2) 在项目根目录运行应用:  cd \"${PROJECT_ROOT}\" && python source/desktop/app.py"
echo "3) 运行分割:                 cd \"${PROJECT_ROOT}\" && python source/image_segment/test_complete_propagation.py <json_file>"
echo "4) MOGP CLI:                 cd \"${PROJECT_ROOT}\" && PYTHONPATH=source python -m gp_model.cli train --help"
echo ""
echo "环境管理："
echo "- 再次同步依赖:  bash ${SCRIPT_DIR}/setup.sh"
echo "- 删除环境:      conda env remove -n ${ENV_NAME} -y"
echo ""
echo "提示：environment.yml 仍可作为参考；实际安装以 requirements.txt 为准。"
echo "本脚本不会自动激活环境，请手动: conda activate ${ENV_NAME}"
