#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量验证：固定含铝量 30%，当量取 1..150，对已训练模型做预测并作图。

建议使用（在 ``source`` 目录下）::

    python kernel_regression/run_test.py --model-dir ./krr_outputs/kernel_regression_<timestamp>

等价于在项目根若 ``cd kernel_regression`` 后::

    python run_test.py --model-dir ../krr_outputs/kernel_regression_<timestamp>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _ensure_python_path() -> Path:
    here = Path(__file__).resolve()
    src = here.parent.parent
    desktop = src / "desktop"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(desktop))
    return src


_ensure_python_path()

from kernel_regression.graph import (  # noqa: E402
    DEFAULT_SWEEP_FNAME,
    _resolve_png_out,
    plot_kbc_vs_equivalent,
)
from kernel_regression.train_kbc_kernel_ridge import (  # noqa: E402
    predict_kernel_regression_kbc,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="K/B/C：当量扫 1..150（含铝 30%）+ 绘图")
    ap.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="kernel_regression_<timestamp> artefact 目录",
    )
    ap.add_argument(
        "--al-percent",
        type=float,
        default=30.0,
        help="固定含铝量（%%），默认 30",
    )
    ap.add_argument(
        "--equiv-min",
        type=int,
        default=1,
        help="起始当量（kg TNT），含端点",
    )
    ap.add_argument(
        "--equiv-max",
        type=int,
        default=150,
        help="结束当量（kg TNT），含端点",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"PNG；默认与模型同目录 {DEFAULT_SWEEP_FNAME}；相对路径相对于 --model-dir",
    )
    args = ap.parse_args()

    root = Path(args.model_dir).expanduser().resolve()
    outp = _resolve_png_out(root, args.out, DEFAULT_SWEEP_FNAME)

    equiv = np.arange(int(args.equiv_min), int(args.equiv_max) + 1, dtype=np.float64)
    al = float(args.al_percent)
    K_arr = np.empty_like(equiv)
    B_arr = np.empty_like(equiv)
    C_arr = np.empty_like(equiv)

    for i, e in enumerate(equiv):
        k, b, c = predict_kernel_regression_kbc(root, float(e), al)
        K_arr[i] = k
        B_arr[i] = b
        C_arr[i] = c

    img = plot_kbc_vs_equivalent(
        equiv, K_arr, B_arr, C_arr, al_percent=al, output_path=outp
    )
    print(str(img.resolve()))


if __name__ == "__main__":
    main()
