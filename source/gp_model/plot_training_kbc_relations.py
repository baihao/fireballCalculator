"""从 training_data JSON 绘制当量、含铝量与拟合 K,B,C 的关系图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def _setup_matplotlib_cjk() -> None:
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Heiti SC", "STHeiti", "Noto Sans CJK SC", "SimHei"):
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break


def plot_kbc_vs_inputs(
    equivalent_kg: np.ndarray,
    al_percent: np.ndarray,
    K: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    *,
    title_prefix: str = "",
) -> plt.Figure:
    eq = np.asarray(equivalent_kg, dtype=np.float64)
    al = np.asarray(al_percent, dtype=np.float64)
    norm = Normalize(vmin=float(al.min()), vmax=float(al.max()))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    labels = ("K", "B", "C")
    ys = (K, B, C)

    for col, name, y in zip(range(3), labels, ys):
        sc = axes[0, col].scatter(
            eq,
            y,
            c=al,
            cmap="viridis",
            norm=norm,
            s=130,
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
        )
        axes[0, col].set_xscale("log")
        axes[0, col].set_xlabel("当量 (kg)")
        axes[0, col].set_ylabel(name)
        axes[0, col].grid(True, which="both", alpha=0.3)
        axes[0, col].set_title(f"{name} — 颜色 = 含铝量 (%)")

        sc2 = axes[1, col].scatter(
            al,
            y,
            c=eq,
            cmap="plasma",
            norm=Normalize(vmin=float(eq.min()), vmax=float(eq.max())),
            s=130,
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
        )
        axes[1, col].set_xlabel("含铝量 (%)")
        axes[1, col].set_ylabel(name)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_title(f"{name} — 颜色 = 当量 (kg)")

    cbar_al = fig.colorbar(sc, ax=axes[0, :], shrink=0.82, pad=0.02, aspect=30)
    cbar_al.set_label("含铝量 (%)")
    cbar_eq = fig.colorbar(sc2, ax=axes[1, :], shrink=0.82, pad=0.02, aspect=30)
    cbar_eq.set_label("当量 (kg)")

    for col in range(3):
        for i, (e, a) in enumerate(zip(eq, al)):
            axes[0, col].annotate(
                f"{int(a)}%",
                (e, ys[col][i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.85,
            )
            axes[1, col].annotate(
                f"{e:g}",
                (a, ys[col][i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.85,
            )

    tpre = f"{title_prefix}\n" if title_prefix else ""
    fig.suptitle(
        tpre + "训练数据：当量、含铝量与拖曳拟合参数 K, B, C（上排 x=当量；下排 x=含铝量）",
        fontsize=11,
    )
    return fig


def plot_kbc_3d(
    equivalent_kg: np.ndarray,
    al_percent: np.ndarray,
    K: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> plt.Figure:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 注册 3d

    eq = np.asarray(equivalent_kg, dtype=np.float64)
    al = np.asarray(al_percent, dtype=np.float64)
    fig = plt.figure(figsize=(10, 3.4))
    labels = ("K", "B", "C")
    zs = (K, B, C)
    for i, (name, z) in enumerate(zip(labels, zs)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        sc = ax.scatter(eq, al, z, c=z, cmap="coolwarm", s=80, depthshade=True)
        ax.set_xlabel("当量 (kg)")
        ax.set_ylabel("含铝量 (%)")
        ax.set_zlabel(name)
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, shrink=0.55, aspect=12, pad=0.12)
    fig.suptitle("三维散点：当量 × 含铝量 × 参数", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.12, wspace=0.35)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制训练数据中 K,B,C 与当量、含铝量关系")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("training_data"),
        help="含 fireball_diameter_fit_*.json 的目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training_output/kbc_vs_equivalent_al.png"),
        help="主图输出路径",
    )
    parser.add_argument(
        "--out-3d",
        type=Path,
        default=Path("training_output/kbc_vs_equivalent_al_3d.png"),
        help="三维散点图输出路径",
    )
    args = parser.parse_args()

    _setup_matplotlib_cjk()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gp_model.data_input import load_training_dir

    ds = load_training_dir(args.data_dir, strict_drag_fit_success=False)
    if len(ds.X) == 0:
        raise SystemExit("无有效训练样本")

    eq = ds.X[:, 0]
    al = ds.X[:, 1]
    K, C = ds.Y[:, 0], ds.Y[:, 1]
    B = ds.B_train

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_kbc_vs_inputs(eq, al, K, B, C)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig3 = plot_kbc_3d(eq, al, K, B, C)
    fig3.savefig(args.out_3d, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"已写入 {args.out.resolve()}")
    print(f"已写入 {args.out_3d.resolve()}（n={len(eq)}）")


if __name__ == "__main__":
    main()
