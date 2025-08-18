#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球温度计算器（左侧三次多项式 + 右侧拖曳衰减，平滑连接，温度单位：K）

- 时间单位: 毫秒(ms)
- 温度单位: 开尔文(K)

模型：
1) 0 ≤ t ≤ t0：上升段三次多项式 T_up(t) = p3*t^3 + p2*t^2 + p1*t + p0
2) t ≥ t0：拖曳衰减 T_drag(t) = A + (T0 - A) * exp(-k*(t - t0))
3) 过渡两种方案（通过构造参数选择）：
   - mode='blend'（默认）：在 t0 附近宽度 w 内使用 C1 smoothstep 平滑过渡（大致平滑）
   - mode='c1': 直接 C1 连续拼接（t0 处数值与导数一致）
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal

TEMP_OFFSET = 273.15  # °C -> K 偏移（派生单位尺度一致，导数数值不变）

@dataclass
class UpwardCubic:
    coeffs: np.ndarray  # [p3, p2, p1, p0]（以°C拟合）

    def T(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        p = np.poly1d(self.coeffs)
        return p(t_ms)  # °C

    def dT(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        p = np.poly1d(self.coeffs)
        dp = np.polyder(p)
        return dp(t_ms)  # °C/ms == K/ms 数值一致

@dataclass
class DragDecay:
    A: float  # °C
    k: float
    t0: float
    T0: float  # °C

    def T(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms)
        return self.A + (self.T0 - self.A) * np.exp(-self.k * (t - self.t0))  # °C

    def dT(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms)
        return -(self.T0 - self.A) * self.k * np.exp(-self.k * (t - self.t0))  # °C/ms

class FireballTemperatureCalculator:
    def __init__(self, blend_width_ms: float = 12.0, mode: Literal['blend','c1'] = 'blend'):
        # 手工近似读取的“Modified temperature”数据点（°C）
        self.t_ms_all = np.array([0, 20, 35, 70, 105, 140], dtype=float)
        self.T_degC_all = np.array([1180, 1240, 1220, 1015, 820, 740], dtype=float)

        # 分段点（峰值附近）
        self.t0 = 35.0
        self.mode = mode
        self.blend_w = float(blend_width_ms)
        self.t1 = max(0.0, self.t0 - self.blend_w / 2.0)
        self.t2 = self.t0 + self.blend_w / 2.0

        # 上升段拟合（三次，°C）
        mask_up = self.t_ms_all <= self.t0
        t_up = self.t_ms_all[mask_up]
        T_up = self.T_degC_all[mask_up]
        self.p_coeffs = np.polyfit(t_up, T_up, deg=3)
        self.up_model = UpwardCubic(coeffs=self.p_coeffs)
        self.T0 = float(self.up_model.T(self.t0))  # °C
        self.dT0 = float(self.up_model.dT(self.t0))  # °C/ms

        # 衰减段拟合（拖曳函数，°C）
        mask_decay = self.t_ms_all >= self.t0
        t_decay = self.t_ms_all[mask_decay]
        T_decay = self.T_degC_all[mask_decay]
        x = t_decay - self.t0

        if self.mode == 'c1':
            # C1 连续：k 由导数连续性确定，搜索 A（°C）
            A_min = 500.0
            A_max = min(T_decay) - 1.0
            A_grid = np.linspace(A_min, A_max, 401)
            best = None
            for A in A_grid:
                B = self.T0 - A
                if B <= 0:
                    continue
                k = -self.dT0 / B
                if k <= 0:
                    continue
                T_pred = A + (self.T0 - A) * np.exp(-k * x)
                sse = float(np.sum((T_decay - T_pred) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, A, k)
            if best is None:
                raise RuntimeError('C1 模式下未能找到有效的 A')
            _, A_opt, k_opt = best
        else:
            # blend 模式：C0 连续，(A,k) 联合搜索（°C）
            A_min = 500.0
            A_max = min(T_decay) - 1.0
            k_min = 1e-3
            k_max = 0.2
            A_grid = np.linspace(A_min, A_max, 401)
            k_grid = np.linspace(k_min, k_max, 400)
            best = None
            for A in A_grid:
                for k in k_grid:
                    T_pred = A + (self.T0 - A) * np.exp(-k * x)
                    sse = float(np.sum((T_decay - T_pred) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, A, k)
            _, A_opt, k_opt = best

        self.decay_model = DragDecay(A=A_opt, k=k_opt, t0=self.t0, T0=self.T0)

    # smoothstep S(s) = 3s^2 - 2s^3 and its time derivative via ds/dt = 1/w
    def _blend_S_and_Sdot(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip((t - self.t1) / self.blend_w, 0.0, 1.0)
        S = 3 * s**2 - 2 * s**3
        dS_dt = (6 * s - 6 * s**2) / self.blend_w
        return S, dS_dt

    def temperature_modified(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms, dtype=float)
        if self.mode == 'c1':
            T_C = np.where(t <= self.t0, self.up_model.T(t), self.decay_model.T(t))
        else:
            T_up = self.up_model.T(t)
            T_drag = self.decay_model.T(t)
            S, _ = self._blend_S_and_Sdot(t)
            T_C = (1 - S) * T_up + S * T_drag
        T_K = T_C + TEMP_OFFSET
        if np.isscalar(t_ms):
            return float(T_K)
        return T_K

    def rate_modified(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms, dtype=float)
        if self.mode == 'c1':
            dT = np.where(t <= self.t0, self.up_model.dT(t), self.decay_model.dT(t))
        else:
            Tup = self.up_model.T(t)
            Tdr = self.decay_model.T(t)
            dTup = self.up_model.dT(t)
            dTdr = self.decay_model.dT(t)
            S, dSdt = self._blend_S_and_Sdot(t)
            dT = (1 - S) * dTup + S * dTdr + dSdt * (Tdr - Tup)
        # 数值与°C/ms相同，但单位标注为K/ms
        if np.isscalar(t_ms):
            return float(dT)
        return dT

    def print_parameters(self):
        p3, p2, p1, p0 = self.p_coeffs
        print("Upward polynomial (cubic) coefficients (t in ms, T in K):")
        print(f"  p3 = {p3:.10e}")
        print(f"  p2 = {p2:.10e}")
        print(f"  p1 = {p1:.10e}")
        print(f"  p0 = {p0 + TEMP_OFFSET:.10e}  # 已按K平移")
        print(f"  t0 = {self.t0:.1f} ms, T0 = {self.T0 + TEMP_OFFSET:.6f} K, dT0 = {self.dT0:.6f} K/ms")
        print(f"  mode = {self.mode}, blend width w = {self.blend_w:.1f} ms")

        print("\nDrag decay parameters (in K):")
        print(f"  A  = {self.decay_model.A + TEMP_OFFSET:.6f} K")
        print(f"  k  = {self.decay_model.k:.6f}  (per ms)")

        # 全局拟合优度（以°C拟合，对R²无影响）
        T_pred_K = self.temperature_modified(self.t_ms_all)
        T_true_K = self.T_degC_all + TEMP_OFFSET
        ss_res = float(np.sum((T_true_K - T_pred_K) ** 2))
        ss_tot = float(np.sum((T_true_K - np.mean(T_true_K)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f"\nGlobal R^2 (K): {r2:.6f}")

    def plot(self, t_max_ms: float = 140.0, show_points: bool = True):
        t = np.linspace(0, t_max_ms, 800)
        T_mod_K = self.temperature_modified(t)
        dT_mod = self.rate_modified(t)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(t, T_mod_K, label=f"Modified temperature ({self.mode})", color="tab:orange", linewidth=2)
        if show_points:
            plt.scatter(self.t_ms_all, self.T_degC_all + TEMP_OFFSET, label="Digitized points", marker="v", color="tab:orange")
        plt.axvline(self.t0, color="gray", linestyle="--", linewidth=1, label="t0")
        plt.xlabel("t (ms)")
        plt.ylabel("T (K)")
        plt.title("Modified temperature vs time")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, dT_mod, label="dT/dt (K/ms)", color="tab:blue", linewidth=2)
        plt.axhline(0.0, color="gray", linewidth=1)
        plt.axvline(self.t0, color="gray", linestyle="--", linewidth=1, label="t0")
        plt.xlabel("t (ms)")
        plt.ylabel("dT/dt (K/ms)")
        plt.title("Temperature rate vs time")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


def main():
    # 使用默认 'blend'：大致平滑且尾部拟合更好；如需严格连续导数可设置 mode='c1'
    calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
    calc.print_parameters()

    for t_ms in [0, 20, 35, 70, 105, 140]:
        T = calc.temperature_modified(t_ms)
        dT = calc.rate_modified(t_ms)
        print(f"t = {t_ms:>5.1f} ms -> T = {T:8.2f} K, dT/dt = {dT:9.4f} K/ms")

    calc.plot(t_max_ms=140)

if __name__ == "__main__":
    main() 