#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球温度计算器 — 时间与温度单位毫秒 / 开尔文

两条剖面可选（``profile``）：

1) ``reference_csv``（默认）：以 ``fireball_temperature_reference_curve.csv``（与源码同目录）为基准，
   使用单调保形分段三次插值（PCHIP）在全时间轴上贴合数据；可选用 ``reference_csv_path`` 替换文件。

2) ``legacy``：原先「手工数字化 6 点 + 左侧三次多项式 + 右侧指数拖曳 + blend/c1」的解析近似。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

TEMP_OFFSET = 273.15


def default_temperature_curve_csv_path() -> Path:
    return Path(__file__).resolve().parent / "fireball_temperature_reference_curve.csv"


def _load_reference_curve_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1))
    t_ms = np.asarray(d[:, 0], dtype=np.float64).ravel()
    t_k = np.asarray(d[:, 1], dtype=np.float64).ravel()
    if t_ms.shape[0] < 5:
        raise ValueError(f"温度参考曲线至少需要 5 个控制点：{path}")
    order = np.argsort(t_ms)
    return t_ms[order], t_k[order]


@dataclass
class UpwardCubic:
    coeffs: np.ndarray  # [p3, p2, p1, p0]，t 单位为 ms，T 为 °C

    def T(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        p = np.poly1d(self.coeffs)
        return p(t_ms)

    def dT(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        return np.polyder(np.poly1d(self.coeffs))(t_ms)


@dataclass
class DragDecay:
    A: float
    k: float
    t0: float
    T0: float

    def T(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms)
        return self.A + (self.T0 - self.A) * np.exp(-self.k * (t - self.t0))

    def dT(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms)
        return -(self.T0 - self.A) * self.k * np.exp(-self.k * (t - self.t0))


class FireballTemperatureCalculator:
    def __init__(
        self,
        blend_width_ms: float = 12.0,
        mode: Literal["blend", "c1"] = "blend",
        profile: Literal["reference_csv", "legacy"] = "reference_csv",
        reference_csv_path: Optional[str | Path] = None,
    ):
        """
        Args:
            blend_width_ms / mode: 仅在 ``profile='legacy'`` 时生效。
            profile: ``reference_csv``（默认贴合 vendored CSV）或 ``legacy``。
            reference_csv_path: 覆盖 CSV 路径；若为 None 则用默认同名文件。
        """
        self.profile: Literal["reference_csv", "legacy"] = profile
        self.mode = mode
        self.blend_w = float(blend_width_ms)

        self.reference_curve_path: Optional[Path] = None
        self._interp_degC: Optional[PchipInterpolator] = None

        self.t_ms_all: Optional[np.ndarray] = None
        self.T_degC_all: Optional[np.ndarray] = None
        self.t0 = 35.0
        self.up_model: Optional[UpwardCubic] = None
        self.decay_model: Optional[DragDecay] = None
        self.p_coeffs: Optional[np.ndarray] = None

        self.t1 = 0.0
        self.t2 = 0.0

        if profile == "legacy":
            self._init_legacy_piecewise(blend_width_ms)
        else:
            self._init_reference_pchip(reference_csv_path)

    def _init_reference_pchip(self, csv_path: Optional[str | Path]) -> None:
        path = Path(csv_path) if csv_path is not None else default_temperature_curve_csv_path()
        if not path.is_file():
            warnings.warn(
                f"未找到温度参考 CSV：{path}，将回退 legacy 分段模型。", UserWarning
            )
            self.profile = "legacy"
            self._init_legacy_piecewise(self.blend_w)
            return
        t_ms, T_K = _load_reference_curve_csv(path)
        T_degC = T_K - TEMP_OFFSET

        duplicates = np.zeros(t_ms.shape[0], dtype=bool)
        duplicates[1:] = t_ms[1:] == t_ms[:-1]
        kept = ~duplicates

        self.reference_curve_path = path
        self.t_ms_reference = t_ms[kept]
        self.T_K_reference = T_K[kept]
        self._interp_degC = PchipInterpolator(
            self.t_ms_reference.astype(np.float64),
            (T_degC[kept]).astype(np.float64),
            extrapolate=True,
        )
        peak_i = int(np.argmax(self.T_K_reference))
        self.t0_reference_peak_ms = float(self.t_ms_reference[peak_i])

    def _init_legacy_piecewise(self, blend_width_ms: float) -> None:
        """原 6 点手工近似 + cubic / drag + blend."""
        self.t_ms_all = np.array([0, 20, 35, 70, 105, 140], dtype=float)
        self.T_degC_all = np.array([1180, 1240, 1220, 1015, 820, 740], dtype=float)
        self.t0 = 35.0
        self.blend_w = float(blend_width_ms)
        self.t1 = max(0.0, self.t0 - self.blend_w / 2.0)
        self.t2 = self.t0 + self.blend_w / 2.0

        mask_up = self.t_ms_all <= self.t0
        self.p_coeffs = np.polyfit(
            self.t_ms_all[mask_up], self.T_degC_all[mask_up], deg=3
        )
        self.up_model = UpwardCubic(coeffs=self.p_coeffs)
        T0 = float(self.up_model.T(self.t0))
        dT0 = float(self.up_model.dT(self.t0))

        mask_decay = self.t_ms_all >= self.t0
        x = self.t_ms_all[mask_decay] - self.t0
        T_decay = self.T_degC_all[mask_decay]

        if self.mode == "c1":
            A_min = 500.0
            A_max = float(min(T_decay)) - 1.0
            A_grid = np.linspace(A_min, A_max, 401)
            best = None
            for A in A_grid:
                B = T0 - A
                if B <= 0:
                    continue
                k_c = -dT0 / B
                if k_c <= 0:
                    continue
                T_pred = A + (T0 - A) * np.exp(-k_c * x)
                sse = float(np.sum((T_decay - T_pred) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, A, k_c)
            if best is None:
                raise RuntimeError("C1 模式下未能找到有效的 A")
            _, A_opt, k_opt = best
        else:
            A_min = 500.0
            A_max = float(min(T_decay)) - 1.0
            k_min, k_max = 1e-3, 0.2
            A_grid = np.linspace(A_min, A_max, 401)
            k_grid = np.linspace(k_min, k_max, 400)
            best = None
            for A in A_grid:
                for k_c in k_grid:
                    T_pred = A + (T0 - A) * np.exp(-k_c * x)
                    sse = float(np.sum((T_decay - T_pred) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, A, k_c)
            assert best is not None
            _, A_opt, k_opt = best

        self.decay_model = DragDecay(A=A_opt, k=k_opt, t0=self.t0, T0=T0)

    def _blend_S_and_Sdot(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip((t - self.t1) / self.blend_w, 0.0, 1.0)
        S = 3 * s**2 - 2 * s**3
        dS_dt = (6 * s - 6 * s**2) / self.blend_w
        return S, dS_dt

    def temperature_modified(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(t_ms)
        t = np.asarray(t_ms, dtype=float)
        if self.profile == "reference_csv" and self._interp_degC is not None:
            T_K = self._interp_degC(t) + TEMP_OFFSET
        else:
            assert self.up_model is not None and self.decay_model is not None
            if self.mode == "c1":
                T_C = np.where(
                    t <= self.t0,
                    self.up_model.T(t),
                    self.decay_model.T(t),
                )
            else:
                T_up = self.up_model.T(t)
                T_drag = self.decay_model.T(t)
                S, _ = self._blend_S_and_Sdot(t)
                T_C = (1 - S) * T_up + S * T_drag
            T_K = T_C + TEMP_OFFSET

        return float(T_K) if scalar else T_K

    def rate_modified(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(t_ms)
        t = np.asarray(t_ms, dtype=float)
        if self.profile == "reference_csv" and self._interp_degC is not None:
            deriv = self._interp_degC.derivative()(t)
            dT = np.asarray(deriv, dtype=float)
        else:
            assert self.up_model is not None and self.decay_model is not None
            if self.mode == "c1":
                dT = np.where(
                    t <= self.t0,
                    self.up_model.dT(t),
                    self.decay_model.dT(t),
                )
            else:
                Tup = self.up_model.T(t)
                Tdr = self.decay_model.T(t)
                dTup = self.up_model.dT(t)
                dTdr = self.decay_model.dT(t)
                S, dSdt = self._blend_S_and_Sdot(t)
                dT = (1 - S) * dTup + S * dTdr + dSdt * (Tdr - Tup)
        return float(dT) if scalar else dT

    def print_parameters(self) -> None:
        if self.profile == "reference_csv" and self._interp_degC is not None:
            print("Temperature profile: reference CSV (PCHIP)")
            path = self.reference_curve_path or default_temperature_curve_csv_path()
            print(f"  source: {path}")
            n = self.t_ms_reference.shape[0]
            print(f"  control points: {n}, t_span = [{self.t_ms_reference[0]:.3g}, {self.t_ms_reference[-1]:.3g}] ms")
            print(f"  peak ≈ {self.t0_reference_peak_ms:.3g} ms, T_peak ≈ {float(np.max(self.T_K_reference)):.3f} K")
            Ts = np.asarray(self.temperature_modified(self.t_ms_reference), dtype=float)
            rmse = float(np.sqrt(np.mean((Ts - self.T_K_reference) ** 2)))
            print(f"  RMSE(sampled control times, K): {rmse:g}")
            return

        assert self.p_coeffs is not None and self.decay_model is not None
        print("Temperature profile: legacy (6-point analytic)")
        p3, p2, p1, p0 = self.p_coeffs
        print("Upward polynomial (cubic) coefficients (t in ms, T in °C internally):")
        print(f"  p3 = {p3:.10e}")
        print(f"  p2 = {p2:.10e}")
        print(f"  p1 = {p1:.10e}")
        print(f"  p0 + offset = {p0 + TEMP_OFFSET:.10e} K")
        dm = self.decay_model
        print(f"  t0 = {self.t0:.1f} ms, T0 = {dm.T0 + TEMP_OFFSET:.6f} K")
        print(f"  mode = {self.mode}, blend width = {self.blend_w:.1f} ms")
        print("\nDrag decay (in K endpoints):")
        print(f"  A  = {dm.A + TEMP_OFFSET:.6f} K")
        print(f"  k  = {dm.k:.6f} (/ms)")
        assert self.T_degC_all is not None and self.t_ms_all is not None
        T_pred_K = self.temperature_modified(self.t_ms_all)
        T_true_K = self.T_degC_all + TEMP_OFFSET
        ss_res = float(np.sum((T_true_K - T_pred_K) ** 2))
        ss_tot = float(np.sum((T_true_K - np.mean(T_true_K)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"\nGlobal R^2 (legacy anchor points): {r2:.6f}")

    def plot(self, t_max_ms: Optional[float] = None, show_points: bool = True) -> None:
        if (
            self.profile == "reference_csv"
            and self._interp_degC is not None
            and t_max_ms is None
        ):
            t_max_ms = float(self.t_ms_reference[-1])
        elif t_max_ms is None:
            t_max_ms = 140.0

        n = max(600, int(t_max_ms) + 1)
        t = np.linspace(0.0, t_max_ms, n)
        T_mod_K = self.temperature_modified(t)
        dT_mod = self.rate_modified(t)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            t,
            T_mod_K,
            label=f"Temperature ({self.profile})",
            color="tab:orange",
            linewidth=2,
        )
        if show_points:
            if self.profile == "reference_csv" and self._interp_degC is not None:
                step = max(1, self.t_ms_reference.shape[0] // 42)
                idx = slice(None, None, step)
                plt.scatter(
                    self.t_ms_reference[idx],
                    self.T_K_reference[idx],
                    s=26,
                    label="CSV samples",
                    color="darkred",
                    zorder=4,
                    alpha=0.75,
                )
            elif self.T_degC_all is not None and self.t_ms_all is not None:
                plt.scatter(
                    self.t_ms_all,
                    self.T_degC_all + TEMP_OFFSET,
                    label="Digitized (legacy)",
                    marker="v",
                    color="tab:orange",
                )
        linestyle = "-" if self.profile == "legacy" else "--"
        xline = (
            float(self.t0)
            if self.profile == "legacy"
            else self.t0_reference_peak_ms
        )
        plt.axvline(xline, color="gray", linestyle=linestyle, linewidth=1, label="t_peak")
        plt.xlabel("t (ms)")
        plt.ylabel("T (K)")
        plt.title("Modified temperature vs time")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, dT_mod, label="dT/dt (K/ms)", color="tab:blue", linewidth=2)
        plt.axhline(0.0, color="gray", linewidth=1)
        plt.axvline(xline, color="gray", linestyle=linestyle, linewidth=1, label="t_peak")
        plt.xlabel("t (ms)")
        plt.ylabel("dT/dt (K/ms)")
        plt.title("Temperature rate vs time")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


def main() -> None:
    calc = FireballTemperatureCalculator()
    calc.print_parameters()
    demo_t = np.linspace(0, 140, 6)
    for tid in demo_t[:-1]:
        T = calc.temperature_modified(float(tid))
        dT = calc.rate_modified(float(tid))
        print(f"t = {tid:>6.2f} ms -> T = {T:8.3f} K, dT/dt = {dT:10.6f} K/ms")

    csv_path = default_temperature_curve_csv_path()
    if csv_path.is_file():
        tt, KK = _load_reference_curve_csv(csv_path)
        chk = tt <= 141
        preds = calc.temperature_modified(tt[chk])
        diff = preds - KK[chk]
        print(f"PCHIP vs CSV (t<=140 ms): max |Δ| = {float(np.max(np.abs(diff))):.4f} K")
        preds_all = calc.temperature_modified(tt)
        print(
            f"PCHIP vs CSV (full curve): RMSE(K)={float(np.sqrt(np.mean((preds_all - KK) ** 2))):.4g}; "
            f"max |Δ|={float(np.max(np.abs(preds_all - KK))):.4g} K（插值结点处≈数值零）"
        )
    print(
        "提示：在交互环境中可调用 calc.plot() 查看图形；"
        "无显示环境时请设置 MPLBACKEND=Agg 并在 plot() 中改用 savefig。"
    )


if __name__ == "__main__":
    main()
