#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工程计算 — 火球直径（显式 K/B/C 拖曳式）、默认温度、热通量与累积热辐射。

温度时间序列：无训练温度数据时使用 ``FireballTemperatureCalculator`` 内嵌参考曲线，
形状仅作归一化剖面；时长 ``t_d``、峰值 ``T_max`` 由 ``engineering_tab``（当量、含铝率）确定。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 与 model_tab 包同级在 source/
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from fireball_radius_calculator import FireballCalculator
from fireball_temperature_calculator import (
    FireballTemperatureCalculator,
    REFERENCE_EQUIVALENT_KG,
    equivalent_time_scale,
    fireball_total_duration_ms,
    reference_curve_duration_ms,
)
from transmissivity_calculator import TransmissivityParams
from fireball_heat_radiation_calculator import (
    compute_heat_flux_over_time,
    integrate_heat_radiation,
)

DEFAULT_TEMP_BLEND_WIDTH_MS = 12.0
J_TO_KJ = 1000.0
DEFAULT_HEAT_FLUX_DISTANCES_M = (6.0, 7.0, 8.0, 9.0, 10.0)
DEFAULT_RADIATION_X = (6.0, 10.0, 50)

# 侧栏与参数仿真默认工况当量（与标定基准独立）
DEFAULT_SIMULATION_EQUIVALENT_KG = 2000.0
# 参数仿真（K/B/C 显式）默认仿真时长
DEFAULT_PARAMETER_SIMULATION_DURATION_MS = 2000.0
# 温度预测上限 (K)：超过则截断
MAX_TEMPERATURE_K = 3000.0

_DESKTOP_ROOT = os.path.join(_PKG_ROOT, "desktop")


def reference_temperature_duration_ms() -> float:
    """参考温度曲线时间跨度（ms）；来自内嵌数据（可被外部 CSV 覆盖）。"""
    return float(reference_curve_duration_ms())


REFERENCE_DURATION_MS = reference_temperature_duration_ms()


def _clamp_temperature_k(value: float) -> float:
    return min(float(value), MAX_TEMPERATURE_K)


def _clamp_temperature_series(t_k: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(t_k, dtype=np.float64), MAX_TEMPERATURE_K)


def _ensure_desktop_import_path() -> None:
    if _DESKTOP_ROOT not in sys.path:
        sys.path.insert(0, _DESKTOP_ROOT)


def peak_temperature_k_from_engineering(
    equivalent_kg: float,
    al_fraction: float,
    t_amb_k: float,
) -> float:
    """工程公式 ``T(t;M,x)`` 的峰值 ``T_max``（``t_amb_k`` 保留签名兼容，未参与计算）。"""
    _ensure_desktop_import_path()
    from engineering_tab.utils.image_formula_engineering import (
        EngineeringEstimateInputs,
        compute_engineering_estimate,
    )

    result = compute_engineering_estimate(
        EngineeringEstimateInputs(
            equivalent_kg=float(equivalent_kg),
            al_fraction=float(al_fraction),
            t_amb_k=float(t_amb_k),
        )
    )
    return _clamp_temperature_k(float(result.t_max_k))


def default_simulation_duration_ms(equivalent_kg: float) -> float:
    """推荐仿真时长 = 工程火球持续时间 t_d（ms）。"""
    return fireball_total_duration_ms(float(equivalent_kg))


def expansion_velocity_series(t_ms: np.ndarray, diameter_m: np.ndarray) -> np.ndarray:
    """火球膨胀速度 v(t) = dD/dt，对直径时序数值求导（单位 m/ms）。"""
    t = np.asarray(t_ms, dtype=np.float64)
    d = np.asarray(diameter_m, dtype=np.float64)
    if t.size < 2:
        return np.zeros_like(d)
    return np.gradient(d, t)


def expansion_velocity_drag_analytic(
    t_ms: np.ndarray,
    k_diameter_m: float,
    b: float,
    c_material: float,
) -> np.ndarray:
    """
    拖曳曲线 D(t)=K(1−B·exp(−C_eff·t_s²)) 对时间的解析导数（m/ms）。

    t_s = t_ms/1000，C_eff = C × 10⁶（与 ``diameter_drag_series`` 一致）。
    """
    t_s = np.asarray(t_ms, dtype=np.float64) / 1000.0
    c_eff = float(c_material) * 1e6
    return (
        float(k_diameter_m)
        * float(b)
        * c_eff
        * 2.0
        * t_s
        * np.exp(-c_eff * np.square(t_s))
        / 1000.0
    )


def diameter_drag_series(
    t_ms: np.ndarray,
    k_diameter_m: float,
    b: float,
    c_material: float,
) -> np.ndarray:
    """
    火球直径拖曳模型（与 JSON / 核回归目标一致）：\\( D(t) = K(1 - B e^{-C t^2}) \\)。

    ``c_material`` 与 ``FireballCalculator`` 材料库中 C 同量纲（如 0.05）；内部对时间 t 使用秒，
    与同计算器一致地作 ``* 1e6`` 转为与 \\(t^2\\)（s²）相乘。
    """
    t_s = np.asarray(t_ms, dtype=np.float64) / 1000.0
    c_s = float(c_material) * 1e6
    return np.asarray(k_diameter_m, dtype=np.float64) * (
        1.0 - float(b) * np.exp(-c_s * np.square(t_s))
    )


def diameter_series_calculator_scaled(
    t_ms: np.ndarray,
    calculator: FireballCalculator,
    material_name: str,
    equivalent_ratio: float,
) -> np.ndarray:
    """使用 ``FireballCalculator.calculate_diameter`` 与当量比值 M（非显式 KBC 路径）。"""
    t_s = np.asarray(t_ms, dtype=np.float64) / 1000.0
    m = float(equivalent_ratio)
    out = np.empty(t_s.shape[0], dtype=np.float64)
    for i, t in enumerate(t_s):
        out[i] = calculator.calculate_diameter(float(t), material_name, m)
    return out


def temperature_shape_only_series(
    t_ms: np.ndarray,
    duration_ms: float,
    *,
    peak_temperature_k: Optional[float] = None,
    ambient_k: float = 297.15,
) -> np.ndarray:
    """
    参数仿真温度序列：基准曲线按 ``duration_ms`` 拉伸时间轴。

    - 未给 ``peak_temperature_k``：幅值随内嵌参考曲线。
    - 给定峰值：幅值缩放到该峰值 (K)，环境温度为 ``ambient_k``。
    """
    t = np.asarray(t_ms, dtype=np.float64)
    calc = FireballTemperatureCalculator()
    if peak_temperature_k is None or float(peak_temperature_k) <= 0:
        return _clamp_temperature_series(
            calc.temperature_from_reference_shape(t, float(duration_ms))
        )
    return _clamp_temperature_series(
        calc.temperature_baseline_scaled(
            t,
            equivalent_kg=1.0,
            duration_ms=float(duration_ms),
            peak_temperature_k=_clamp_temperature_k(peak_temperature_k),
            ambient_k=float(ambient_k),
        )
    )


def default_temperature_series(
    t_ms: np.ndarray,
    equivalent_kg: float,
    al_fraction: float = 0.30,
    t_amb_k: float = 297.15,
) -> np.ndarray:
    """
    无实验温度序列：CSV 形状 + 工程 t_d 时长 + T_max 峰值。
    """
    t = np.asarray(t_ms, dtype=np.float64)
    duration_ms = fireball_total_duration_ms(float(equivalent_kg))
    peak_k = peak_temperature_k_from_engineering(
        float(equivalent_kg), float(al_fraction), float(t_amb_k)
    )
    calc = FireballTemperatureCalculator()
    return _clamp_temperature_series(
        calc.temperature_baseline_scaled(
            t,
            equivalent_kg=float(equivalent_kg),
            duration_ms=duration_ms,
            peak_temperature_k=peak_k,
            ambient_k=float(t_amb_k),
        )
    )


def temperature_series_from_training(
    t_ms: np.ndarray,
    training_temperature_data: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    train_time_ms, train_temp_K = training_temperature_data
    return _clamp_temperature_series(
        np.interp(
            np.asarray(t_ms, dtype=np.float64),
            np.asarray(train_time_ms, dtype=np.float64),
            np.asarray(train_temp_K, dtype=np.float64),
        )
    )


def heat_flux_bundle(
    t_ms: np.ndarray,
    t_K: np.ndarray,
    diameter_m: np.ndarray,
    env_temp_C: float,
    env_humidity: float,
    env_pressure_Pa: float,
    distances_m: Sequence[float] = DEFAULT_HEAT_FLUX_DISTANCES_M,
) -> Tuple[List[List[Any]], Dict[str, np.ndarray]]:
    """多距离热通量时间序列；返回 chart 用 ``heat_flux_series`` 与 ``prediction_data`` 内 dict。"""
    transmissivity_params = TransmissivityParams(
        Ta_K=env_temp_C + 273.15,
        RH_percent=env_humidity,
        PwSat_Pa=env_pressure_Pa,
    )
    heat_flux_series: List[List[Any]] = []
    store: Dict[str, np.ndarray] = {}
    for dist in distances_m:
        q_t = compute_heat_flux_over_time(
            float(dist), t_ms, t_K, diameter_m, transmissivity_params
        )
        heat_flux_series.append([dist, q_t])
        store[f"{dist:.1f}"] = q_t
    return heat_flux_series, store


def summarize_max_temperature_heat_flux(
    t_ms: np.ndarray,
    t_k: np.ndarray,
    heat_flux_store: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    与 ``heat_flux_bundle`` / ``compute_heat_flux_over_time`` 一致：
    取参与热通量积分的 T(t) 的全局最大值；并给出最近距离处 q(t) 峰值及对应时刻的温度。
    """
    t = np.asarray(t_ms, dtype=np.float64)
    tk = np.asarray(t_k, dtype=np.float64)
    if tk.size == 0:
        return {}
    i_tmax = int(np.argmax(tk))
    t_max_k = float(tk[i_tmax])
    summary: Dict[str, float] = {
        "T_K": t_max_k,
        "T_C": t_max_k - 273.15,
        "at_ms": float(t[i_tmax]),
    }
    if not isinstance(heat_flux_store, dict) or len(heat_flux_store) == 0:
        return summary
    ref_key = min(heat_flux_store.keys(), key=lambda s: float(s))
    ref_x = float(ref_key)
    q = np.asarray(heat_flux_store[ref_key], dtype=np.float64)
    i_q = int(np.argmax(q))
    summary["reference_distance_m"] = ref_x
    summary["peak_heat_flux_W_m2"] = float(q[i_q])
    summary["peak_heat_flux_at_ms"] = float(t[i_q])
    summary["T_at_peak_heat_flux_K"] = float(tk[i_q])
    summary["T_at_peak_heat_flux_C"] = float(tk[i_q]) - 273.15
    return summary


def preview_temperature_series_for_heat_flux(
    duration_ms: float,
    equivalent_kg: Optional[float] = None,
    training_temperature_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    al_fraction: float = 0.30,
    t_amb_k: float = 297.15,
    *,
    parameter_simulation: bool = False,
    peak_temperature_k: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """未跑完整 bundle 时，用与热通量相同的 T(t) 来源做预览。"""
    if duration_ms <= 0:
        return None
    if not parameter_simulation and (equivalent_kg is None or equivalent_kg <= 0):
        return None
    n = max(2, int(duration_ms) + 1)
    t_ms = np.linspace(0.0, float(duration_ms), n)
    if training_temperature_data is not None:
        t_k = temperature_series_from_training(t_ms, training_temperature_data)
    elif parameter_simulation:
        t_k = temperature_shape_only_series(
            t_ms,
            float(duration_ms),
            peak_temperature_k=peak_temperature_k,
            ambient_k=float(t_amb_k),
        )
    else:
        t_k = default_temperature_series(
            t_ms, float(equivalent_kg), al_fraction=al_fraction, t_amb_k=t_amb_k
        )
    return t_ms, t_k


def cumulative_radiation_kjm2(
    t_ms: np.ndarray,
    t_K: np.ndarray,
    diameter_m: np.ndarray,
    env_temp_C: float,
    env_humidity: float,
    env_pressure_Pa: float,
    x_min: float = DEFAULT_RADIATION_X[0],
    x_max: float = DEFAULT_RADIATION_X[1],
    n_points: int = int(DEFAULT_RADIATION_X[2]),
) -> Tuple[np.ndarray, np.ndarray]:
    transmissivity_params = TransmissivityParams(
        Ta_K=env_temp_C + 273.15,
        RH_percent=env_humidity,
        PwSat_Pa=env_pressure_Pa,
    )
    x_values = np.linspace(x_min, x_max, n_points)
    H_kjm2: List[float] = []
    for x in x_values:
        q_t = compute_heat_flux_over_time(x, t_ms, t_K, diameter_m, transmissivity_params)
        h_j = integrate_heat_radiation(q_t, t_ms)
        H_kjm2.append(h_j / J_TO_KJ)
    return x_values, np.asarray(H_kjm2, dtype=np.float64)


def build_prediction_bundle(
    *,
    t_ms: np.ndarray,
    duration_ms: float,
    equivalent: float,
    material_name: str,
    env_temp: float,
    env_humidity: float,
    env_pressure: float,
    calculator: FireballCalculator,
    use_explicit_kbc: bool,
    kbc: Optional[Tuple[float, float, float]],
    training_equivalent: Optional[float],
    training_temperature_data: Optional[Tuple[np.ndarray, np.ndarray]],
    al_fraction: float = 0.30,
    parameter_simulation: bool = False,
    peak_temperature_k: Optional[float] = None,
) -> Dict[str, Any]:
    """
    组装一次仿真所需的 ``prediction_data`` 及中间数组。

    - ``use_explicit_kbc=True`` 且 ``kbc`` 非空：直径由 ``diameter_drag_series``（核岭回归预测的 K,B,C）。
    - 否则：直径由 ``FireballCalculator`` + 当量比值 M（相对 ``training_equivalent`` 或计算器标准当量）。
    - ``parameter_simulation=True`` 时可用 ``peak_temperature_k`` 指定火球最大温度 (K)。
    """
    t_ms = np.asarray(t_ms, dtype=np.float64)
    t_s = t_ms / 1000.0

    if training_equivalent is not None:
        standard_equivalent = float(training_equivalent)
    else:
        standard_equivalent = float(calculator.get_standard_equivalent(material_name))
    m = equivalent / standard_equivalent if standard_equivalent > 0 else 1.0

    if use_explicit_kbc and kbc is not None:
        Kd, b, c = kbc
        d_m = diameter_drag_series(t_ms, Kd, b, c)
        kbc_source = "explicit_kbc"
    else:
        d_m = diameter_series_calculator_scaled(t_ms, calculator, material_name, m)
        kbc_source = "calculator_scaled"
        p = calculator.get_standard_parameters(material_name)
        Kd = 2.0 * float(np.sqrt(m) * p["K"])  # 展示用等效直径系数（与半径换算一致）
        b = p["B"]
        c = p["C"] / m if m > 0 else p["C"]

    t_amb_k = float(env_temp) + 273.15
    peak_k_used: Optional[float] = None
    if training_temperature_data is not None:
        t_k = temperature_series_from_training(t_ms, training_temperature_data)
    elif parameter_simulation:
        peak_k_used = (
            _clamp_temperature_k(peak_temperature_k)
            if peak_temperature_k is not None and float(peak_temperature_k) > 0
            else None
        )
        t_k = temperature_shape_only_series(
            t_ms,
            float(duration_ms),
            peak_temperature_k=peak_k_used,
            ambient_k=t_amb_k,
        )
    else:
        peak_k_used = peak_temperature_k_from_engineering(
            float(equivalent), float(al_fraction), t_amb_k
        )
        t_k = default_temperature_series(
            t_ms,
            float(equivalent),
            al_fraction=float(al_fraction),
            t_amb_k=t_amb_k,
        )

    heat_series, heat_store = heat_flux_bundle(
        t_ms, t_k, d_m, env_temp, env_humidity, env_pressure
    )
    x_rad, h_rad = cumulative_radiation_kjm2(
        t_ms, t_k, d_m, env_temp, env_humidity, env_pressure
    )
    v_ms = expansion_velocity_series(t_ms, d_m)

    return {
        "time_ms": t_ms,
        "time_s": t_s,
        "material_name": material_name,
        "duration": duration_ms,
        "equivalent": None if parameter_simulation else equivalent,
        "equivalent_ratio": None if parameter_simulation else m,
        "simulation_mode": "parameter" if parameter_simulation else "equivalent",
        "env_temp": env_temp,
        "env_humidity": env_humidity,
        "env_pressure": env_pressure,
        "diameter_data": d_m,
        "expansion_velocity_data": v_ms,
        "temperature_data": t_k,
        "heat_flux_data": heat_store,
        "heat_radiation_data": {"distances": x_rad, "heat_radiation": h_rad},
        "kbc_source": kbc_source,
        "kbc_display": (float(Kd), float(b), float(c)),
        "temperature_time_scale": None,
        "temperature_duration_ms": (
            float(duration_ms)
            if parameter_simulation or training_temperature_data is None
            else fireball_total_duration_ms(float(equivalent))
        ),
        "temperature_peak_k": peak_k_used,
        "max_temperature_heat_flux": summarize_max_temperature_heat_flux(
            t_ms, t_k, heat_store
        ),
        "_heat_flux_series_chart": heat_series,
    }
