"""CLI：train / predict / plot（见 fireball_gp_mogp_module_design.md §6）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from gp_model.config import (
    DEFAULT_TRAINED_FILENAME,
    DEFAULT_TRAINING_OUTPUT_DIR,
    default_hyperparams,
    load_hyperparams_json,
    load_trained_artifact,
    merge_hyperparams,
)
from gp_model.curve_plot import (
    load_kbc_json,
    plot_config_from_hyperparams,
    plot_diameter_curve,
    save_figure,
)
from gp_model.data_input import load_training_dir, load_x_star_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _import_train_infer():
    """train / predict 依赖 GPyTorch；延迟导入以便 plot/--help 在无 gpytorch 时仍可用。"""
    try:
        from gp_model.train_infer import predict_mogp, run_train_cli

        return predict_mogp, run_train_cli
    except ModuleNotFoundError as e:
        mod = getattr(e, "name", None) or ""
        if mod in ("gpytorch", "linear_operator") or "gpytorch" in str(e).lower():
            logger.error(
                "缺少 GPyTorch（训练/推理需要）。请安装依赖，例如："
                " pip install -r source/requirements.txt"
                "（或见 source/gp_model/README.md）"
            )
            raise SystemExit(1) from e
        raise


def _cmd_train(args: argparse.Namespace) -> int:
    _, run_train_cli = _import_train_infer()
    data_dir = Path(args.data_dir).resolve()
    hp = default_hyperparams()
    if args.config:
        hp = merge_hyperparams(load_hyperparams_json(args.config))
    strict = bool(hp.get("io", {}).get("strict_drag_fit_success", False))
    ds = load_training_dir(data_dir, strict_drag_fit_success=strict, recursive=False)
    out = args.out_hyperparams or (data_dir / DEFAULT_TRAINED_FILENAME)
    run_train_cli(ds, hp, out_path=str(out), data_dir=str(data_dir))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    predict_mogp, _ = _import_train_infer()
    hp_path = Path(args.hyperparams).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.x_json:
        X_star = load_x_star_path(args.x_json)
    else:
        if args.equivalent is None or args.al_percent is None:
            logger.error("需提供 --equivalent 与 --al-percent，或 --x-json")
            return 2
        import numpy as np

        X_star = np.array(
            [[float(args.equivalent), float(args.al_percent)]], dtype=np.float64
        )

    art = load_trained_artifact(str(hp_path))
    pred = predict_mogp(art, X_star)

    result = {
        "X_star": X_star.tolist(),
        "mean_KBC": pred.mean.tolist(),
        "std_KBC": pred.std.tolist(),
        "variance_KBC": pred.variance.tolist(),
        "std_KBC_latent": pred.std_latent.tolist(),
        "variance_KBC_latent": pred.variance_latent.tolist(),
    }
    pred_json = out_dir / "predict_result.json"
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("已写入 %s", pred_json)

    if args.no_plot:
        return 0

    cfg_plot = plot_config_from_hyperparams(art)
    io_cfg = art.get("io") or {}
    plot_name = str(io_cfg.get("predict_plot_filename", "diameter_vs_time.png"))
    # 多点：每张图一个点
    m = X_star.shape[0]
    for i in range(m):
        mu = pred.mean[i]
        K, B, C = float(mu[0]), float(mu[1]), float(mu[2])
        eq = float(X_star[i, 0])
        al = float(X_star[i, 1])
        fig = plot_diameter_curve(
            K, B, C, cfg_plot, equivalent=eq, al_percent=al
        )
        if m == 1:
            out_png = out_dir / plot_name
        else:
            pat = str(io_cfg.get("predict_point_plot_pattern", "diameter_{index}.png"))
            out_png = out_dir / pat.format(index=i)
        save_figure(fig, out_png)
        logger.info("已写入 %s", out_png)

    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    hp = load_trained_artifact(args.hyperparams)
    cfg = plot_config_from_hyperparams(hp)
    kbc = load_kbc_json(args.kbc_json)
    K, B, C = float(kbc["K"]), float(kbc["B"]), float(kbc["C"])
    eq = kbc.get("equivalent")
    al = kbc.get("al_percent")
    eq_f = float(eq) if eq is not None else None
    al_f = float(al) if al is not None else None
    fig = plot_diameter_curve(
        K, B, C, cfg, equivalent=eq_f, al_percent=al_f
    )
    save_figure(fig, args.out)
    logger.info("已写入 %s", args.out)
    return 0


def _cmd_sweep_k(args: argparse.Namespace) -> int:
    _import_train_infer()
    from gp_model.config import load_trained_artifact
    from gp_model.curve_plot import save_figure
    from gp_model.k_sweep import (
        plot_kbc_sweep,
        save_kbc_sweep_csv,
        save_kbc_sweep_json,
        sweep_kbc_vs_equivalent,
    )

    hp_path = Path(args.hyperparams).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    art = load_trained_artifact(str(hp_path))
    eq, pred = sweep_kbc_vs_equivalent(
        art,
        al_percent=float(args.al_percent),
        equivalent_min=float(args.equivalent_min),
        equivalent_max=float(args.equivalent_max),
        num_points=int(args.num_points),
    )
    csv_path = out_dir / "kbc_vs_equivalent.csv"
    json_path = out_dir / "kbc_vs_equivalent.json"
    save_kbc_sweep_csv(csv_path, eq, pred)
    save_kbc_sweep_json(
        json_path,
        al_percent=float(args.al_percent),
        equivalent_min=float(args.equivalent_min),
        equivalent_max=float(args.equivalent_max),
        num_points=int(args.num_points),
        equivalent=eq,
        pred=pred,
    )
    fig = plot_kbc_sweep(eq, pred, al_percent=float(args.al_percent))
    png_path = out_dir / str(args.out_png)
    save_figure(fig, png_path)
    logger.info("已写入 %s, %s, %s", csv_path, json_path, png_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fireball-gp",
        description="火球 K/C 双单任务 GP：train / predict / plot / sweep-k（gp_model）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="分别训练 K-GP 与 C-GP，写出 trained_hyperparams.json（schema 3）")
    t.add_argument("--data-dir", required=True, help="训练 JSON 所在目录")
    t.add_argument("--config", help="可选超参数 JSON；省略则用内置默认")
    t.add_argument(
        "--out-hyperparams",
        help=f"输出路径；默认 <data-dir>/{DEFAULT_TRAINED_FILENAME}",
    )
    t.set_defaults(func=_cmd_train)

    pr = sub.add_parser("predict", help="推理并默认画图到 out-dir")
    pr.add_argument("--hyperparams", required=True, help="train 产出的 JSON")
    pr.add_argument(
        "--out-dir",
        default=DEFAULT_TRAINING_OUTPUT_DIR,
        help=f"输出目录；默认 {DEFAULT_TRAINING_OUTPUT_DIR}/（predict_result.json 与图）",
    )
    pr.add_argument("--equivalent", type=float, help="单点 X（与 al-percent 同用）")
    pr.add_argument("--al-percent", type=float, dest="al_percent")
    pr.add_argument("--x-json", help="x_star.json（单点或 points）")
    pr.add_argument("--no-plot", action="store_true", help="仅写 predict_result.json")
    pr.set_defaults(func=_cmd_predict)

    pl = sub.add_parser("plot", help="离线根据 kbc.json 出图")
    pl.add_argument("--hyperparams", required=True, help="含 plot 段的 JSON")
    pl.add_argument("--kbc-json", required=True)
    pl.add_argument("--out", required=True, help="输出 PNG 路径")
    pl.set_defaults(func=_cmd_plot)

    sk = sub.add_parser(
        "sweep-k",
        help="固定含铝量沿当量扫掠 K、C（两路独立 GP；B 为训练集 b_mean），输出 CSV/JSON/PNG",
    )
    sk.add_argument("--hyperparams", required=True, help="train 产出的 JSON")
    sk.add_argument(
        "--out-dir",
        default=DEFAULT_TRAINING_OUTPUT_DIR,
        help=f"输出目录；默认 {DEFAULT_TRAINING_OUTPUT_DIR}/",
    )
    sk.add_argument("--al-percent", type=float, default=40.0, help="含铝量 %%（默认 40）")
    sk.add_argument("--equivalent-min", type=float, default=1.0, help="当量下限 kg（默认 1）")
    sk.add_argument("--equivalent-max", type=float, default=160.0, help="当量上限 kg（默认 160）")
    sk.add_argument(
        "--num-points",
        type=int,
        default=160,
        help="沿当量采样点数（默认 160，线性插值）",
    )
    sk.add_argument(
        "--out-png",
        default="kbc_vs_equivalent.png",
        help="输出 PNG 文件名（相对于 out-dir，默认 kbc_vs_equivalent.png，含 K/B/C 三子图）",
    )
    sk.set_defaults(func=_cmd_sweep_k)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
