#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工程计算 — 火球直径（显式 K/B/C 拖曳式）、默认温度、热通量与累积热辐射。

温度时间序列：无训练温度数据时使用 ``FireballTemperatureCalculator`` 参考 CSV（标定当量
100 kg TNT）；其它当量在时间上按 ``(m/100)^(2/3)`` 缩放后再取样。
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
    default_temperature_curve_csv_path,
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

# 默认温度参考曲线标定：100 kg TNT 下的爆炸仿真时间窗（与 reference CSV 末时刻一致）
REFERENCE_EQUIVALENT_KG = 100.0


def reference_temperature_duration_ms() -> float:
    """100 kg 标定工况下参考温度 CSV 的时间跨度（ms）。"""
    path = default_temperature_curve_csv_path()
    if path.is_file():
        t_ms = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0)
        return float(np.max(t_ms))
    return 2000.0


REFERENCE_DURATION_MS = reference_temperature_duration_ms()


def equivalent_time_scale(equivalent_kg: float) -> float:
    """
    相对 100 kg 标定工况的时间缩放因子：``(m / 100)^(2/3)``。

    仿真时刻 t 对应参考曲线时刻 ``t_ref = t / scale``；特征爆炸时长随当量同比例放大。
    """
    m = float(equivalent_kg)
    if m <= 0:
        raise ValueError("当量必须大于 0")
    return float((m / REFERENCE_EQUIVALENT_KG) ** (2.0 / 3.0))


def default_simulation_duration_ms(equivalent_kg: float) -> float:
    """与默认温度缩放一致的推荐仿真时长（ms）。"""
    return REFERENCE_DURATION_MS * equivalent_time_scale(equivalent_kg)


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


def default_temperature_series(t_ms: np.ndarray, equivalent_kg: float) -> np.ndarray:
    """
    无实验温度序列时的默认温度曲线。

    参考剖面为 **100 kg TNT** 下 ``FireballTemperatureCalculator`` 的 CSV/PCHIP 曲线；
    对当量 ``m`` kg，在仿真时刻 ``t`` 上取参考时刻 ``t_ref = t / (m/100)^(2/3)`` 的温度。
    """
    t = np.asarray(t_ms, dtype=np.float64)
    scale = equivalent_time_scale(float(equivalent_kg))
    t_ref = t / scale
    calc = FireballTemperatureCalculator()
    return np.asarray(calc.temperature_modified(t_ref), dtype=np.float64)


def temperature_series_from_training(
    t_ms: np.ndarray,
    training_temperature_data: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    train_time_ms, train_temp_K = training_temperature_data
    return np.interp(
        np.asarray(t_ms, dtype=np.float64),
        np.asarray(train_time_ms, dtype=np.float64),
        np.asarray(train_temp_K, dtype=np.float64),
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
) -> Dict[str, Any]:
    """
    组装一次仿真所需的 ``prediction_data`` 及中间数组。

    - ``use_explicit_kbc=True`` 且 ``kbc`` 非空：直径由 ``diameter_drag_series``（核岭回归预测的 K,B,C）。
    - 否则：直径由 ``FireballCalculator`` + 当量比值 M（相对 ``training_equivalent`` 或计算器标准当量）。
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

    if training_temperature_data is not None:
        t_k = temperature_series_from_training(t_ms, training_temperature_data)
    else:
        t_k = default_temperature_series(t_ms, float(equivalent))

    heat_series, heat_store = heat_flux_bundle(
        t_ms, t_k, d_m, env_temp, env_humidity, env_pressure
    )
    x_rad, h_rad = cumulative_radiation_kjm2(
        t_ms, t_k, d_m, env_temp, env_humidity, env_pressure
    )

    return {
        "time_ms": t_ms,
        "time_s": t_s,
        "material_name": material_name,
        "duration": duration_ms,
        "equivalent": equivalent,
        "equivalent_ratio": m,
        "env_temp": env_temp,
        "env_humidity": env_humidity,
        "env_pressure": env_pressure,
        "diameter_data": d_m,
        "temperature_data": t_k,
        "heat_flux_data": heat_store,
        "heat_radiation_data": {"distances": x_rad, "heat_radiation": h_rad},
        "kbc_source": kbc_source,
        "kbc_display": (float(Kd), float(b), float(c)),
        "temperature_time_scale": (
            None
            if training_temperature_data is not None
            else equivalent_time_scale(float(equivalent))
        ),
        "_heat_flux_series_chart": heat_series,
    }
