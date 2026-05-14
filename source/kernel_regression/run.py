#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核岭回归 CLI：加载 ``--data-dir``、调用 ``train_kernel_regression_kbc`` / ``predict_kernel_regression_kbc``。

请在仓库 ``source`` 目录下执行，例如::
    python kernel_regression/run.py train --data-dir ... --out-dir ... [--graph]
    python kernel_regression/run.py predict --model-dir ... --equiv 10 --al-percent 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_python_path() -> Path:
    here = Path(__file__).resolve()
    src = here.parent.parent
    desktop = src / "desktop"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(desktop))
    return src


_ensure_python_path()


from kernel_regression.graph import plot_loocv_gamma_curves  # noqa: E402
from kernel_regression.train_kbc_kernel_ridge import (  # noqa: E402
    predict_kernel_regression_kbc,
    train_kernel_regression_kbc,
)
from training_tab.training_dataset_model import TrainingDatasetModel  # noqa: E402
from training_tab.utils.dataset_io import import_training_folder  # noqa: E402


def cmd_train(args: argparse.Namespace) -> None:
    res = import_training_folder(Path(args.data_dir), recursive=True, strict_drag_fit_success=False)
    if not res.ok or not res.records:
        sys.stderr.write(f"训练失败: {res.error_message or '无有效记录'}\n")
        sys.exit(1)

    tm = TrainingDatasetModel()
    tm.set_loaded_training_folder(res.folder_resolved, res.records)

    try:
        saved_root, _errors = train_kernel_regression_kbc(
            tm,
            Path(args.out_dir),
            alpha=args.alpha,
        )
    except ValueError as e:
        sys.stderr.write(f"训练失败: {e}\n")
        sys.exit(2)

    print(str(saved_root.resolve()))

    if getattr(args, "graph", False):
        try:
            png = plot_loocv_gamma_curves(saved_root)
        except Exception as e:
            sys.stderr.write(f"绘图失败: {e}\n")
            sys.exit(4)
        sys.stderr.write(f"[graph] {png.resolve()}\n")


def cmd_predict(args: argparse.Namespace) -> None:
    try:
        k_hat, b_hat, c_hat = predict_kernel_regression_kbc(
            Path(args.model_dir),
            equiv_kg_tnt=float(args.equiv),
            al_percent=float(args.al_percent),
        )
    except FileNotFoundError as e:
        sys.stderr.write(f"预测失败: {e}\n")
        sys.exit(3)

    print(
        json.dumps(
            {
                "equiv_kg": args.equiv,
                "al_percent": args.al_percent,
                "K": k_hat,
                "B": b_hat,
                "C": c_hat,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="K/B/C Kernel Ridge（命令行入口）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="从训练 JSON 目录拟合并导出模型")
    pt.add_argument("--data-dir", type=str, required=True, help="与 desktop dataset_io 相同语义的目录")
    pt.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="对应 API 中的 model_path（父目录）；其下新建 kernel_regression_<timestamp>",
    )
    pt.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="KernelRidge 正则 alpha；不设则为 10^-3（与 train_kernel_regression_kbc 默认一致）",
    )
    pt.add_argument(
        "--graph",
        "-graph",
        dest="graph",
        action="store_true",
        help=(
            "训练完成后在 artefact 目录写入 γ～LOOCV 误差图（等同于 graph.py loocv）；"
            "PNG 绝对路径写入 stderr [graph] …"
        ),
    )
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="在 kernel_regression_<timestamp> 目录上做点预测")
    pp.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="含 kbc_krr_K/B/C.joblib 的 artifact 目录",
    )
    pp.add_argument("--equiv", type=float, required=True)
    pp.add_argument("--al-percent", type=float, required=True)
    pp.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
