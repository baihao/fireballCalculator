#!/usr/bin/env bash
# 将 source/ 加入 PYTHONPATH 后调用 MOGP CLI；可在任意目录执行本脚本。
# 用法: bash source/gp_model/run_cli.sh train --data-dir training_data
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$(cd "${HERE}/.." && pwd)"
export PYTHONPATH="${SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
exec python -m gp_model.cli "$@"
