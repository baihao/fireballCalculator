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
TORCH_MIN_VERSION="2.5.1"
SEGMENT_PATH="${SCRIPT_DIR}/third_party/segment-anything"

conda_python() {
    conda run -n "${ENV_NAME}" python "$@"
}

env_exists() {
    conda env list | grep -qE "^${ENV_NAME}[[:space:]]"
}

python_version_ok() {
    conda_python -c "
import sys
sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)
" >/dev/null 2>&1
}

pytorch_version_ok() {
    conda_python -c "
import sys, torch
ver = torch.__version__.split('+')[0].split('.')
major, minor = int(ver[0]), int(ver[1])
patch = int(ver[2]) if len(ver) > 2 else 0
min_major, min_minor, min_patch = [int(x) for x in '${TORCH_MIN_VERSION}'.split('.')]
ok = (major, minor, patch) >= (min_major, min_minor, min_patch)
sys.exit(0 if ok else 1)
" >/dev/null 2>&1
}

requirements_satisfied() {
    REQ_FILE="${REQ_FILE}" conda_python -c "
import os, re, subprocess, sys
req = os.environ['REQ_FILE']
proc = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-r', req, '--dry-run'],
    capture_output=True,
    text=True,
)
if proc.returncode != 0:
    sys.exit(1)
text = (proc.stdout or '') + (proc.stderr or '')
sys.exit(1 if re.search(r'Would install|Installing collected packages', text) else 0)
" >/dev/null 2>&1
}

segment_anything_ok() {
    if [ ! -d "${SEGMENT_PATH}" ]; then
        return 0
    fi
    conda_python -c "import segment_anything" >/dev/null 2>&1
}

environment_satisfied() {
    env_exists \
        && python_version_ok \
        && pytorch_version_ok \
        && requirements_satisfied \
        && segment_anything_ok
}

install_pytorch_mac() {
    echo "正在安装 PyTorch（macOS / pip，支持 MPS）..."
    if pytorch_version_ok; then
        echo "✓ PyTorch $(conda_python -c 'import torch; print(torch.__version__)') 已满足要求，跳过安装"
        return 0
    fi
    conda_python -m pip install "torch>=${TORCH_MIN_VERSION},<3" "torchvision>=0.20.0"
    echo "✓ PyTorch 安装完成: $(conda_python -c 'import torch; print(torch.__version__)')"
}

install_requirements() {
    echo "正在 pip install -r requirements.txt（含 gpytorch 等）..."
    conda_python -m pip install -r "${REQ_FILE}" --upgrade
    echo "✓ pip 依赖已安装/更新"
}

install_segment_anything() {
    if [ ! -d "${SEGMENT_PATH}" ]; then
        return 0
    fi
    if segment_anything_ok; then
        echo "✓ segment-anything 已安装，跳过"
        return 0
    fi
    echo "检测到第三方子模块 segment-anything，尝试以可编辑模式安装..."
    conda_python -m pip install -e "${SEGMENT_PATH}" || true
}

configure_env_vars() {
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
}

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

# 2) 创建 conda 环境（若不存在）
if env_exists; then
    echo "✓ 已存在环境: ${ENV_NAME}"
else
    echo "创建环境: ${ENV_NAME}（Python ${PYTHON_VERSION}，仅 base 包 + pip）..."
    conda create -n "${ENV_NAME}" -c defaults --override-channels "python=${PYTHON_VERSION}" pip -y
    echo "✓ 环境创建完成"
fi

# 3) 检测依赖是否已满足
if environment_satisfied; then
    echo "✓ 当前环境 ${ENV_NAME} 已满足 Python ${PYTHON_VERSION}、PyTorch 与 requirements.txt 依赖，跳过 pip 安装"
    configure_env_vars
else
    echo "环境需同步依赖（缺失或版本不满足）..."
    conda_python -m pip install --upgrade pip
    install_pytorch_mac
    if requirements_satisfied; then
        echo "✓ requirements.txt 依赖已满足，跳过"
    else
        install_requirements
    fi
    install_segment_anything
    configure_env_vars
fi

echo ""
echo "=========================================="
echo "环境准备完成"
echo "=========================================="
echo ""
echo "依赖来源: ${REQ_FILE}（pip）+ setup.sh 安装 PyTorch；Python ${PYTHON_VERSION}、GPyTorch、Matplotlib 等"
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
