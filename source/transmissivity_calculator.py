#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atmospheric transmissivity calculator

Implements the empirical model (see document/transmissivity.md):
    τ = 1.006 − 0.1171 log10(X_H2O) − 0.02368 [log10(X_H2O)]^2
        − 0.03188 log10(X_CO2) + 0.001164 [log10(X_CO2)]^2
with
    X_CO2 = (273 / Ta) * S
    X_H2O = (288.651 / Ta) * (760 / 101325) * [ (RH/100) * PwSat(Ta) ] * S

Units
- Ta: Kelvin (K)
- S: meters (m)
- RH: %
- PwSat: Pa
- τ: dimensionless

Defaults
- Ta = 24 °C → Ta_K = 297.15 K
- RH = 48 %
- PwSat = 2987.87 Pa
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TransmissivityParams:
    Ta_K: float = 24.0 + 273.15  # 24°C in Kelvin
    RH_percent: float = 48.0     # %
    PwSat_Pa: float = 2987.87    # Pa


def compute_X_co2(S_m: np.ndarray | float, Ta_K: float) -> np.ndarray | float:
    return (273.0 / Ta_K) * np.asarray(S_m)


def compute_X_h2o(S_m: np.ndarray | float, Ta_K: float, RH_percent: float, PwSat_Pa: float) -> np.ndarray | float:
    S = np.asarray(S_m)
    return (288.651 / Ta_K) * (760.0 / 101325.0) * ((RH_percent / 100.0) * PwSat_Pa) * S


def transmissivity(S_m: np.ndarray | float, params: TransmissivityParams = TransmissivityParams()) -> np.ndarray | float:
    """Compute τ(S) with safety for non-positive arguments of log10.
    Returns NaN where X terms are non-positive.
    """
    S = np.asarray(S_m, dtype=float)
    Xco2 = compute_X_co2(S, params.Ta_K)
    Xh2o = compute_X_h2o(S, params.Ta_K, params.RH_percent, params.PwSat_Pa)

    # Avoid invalid logs: mask <=0
    mask = (Xco2 > 0) & (Xh2o > 0)
    tau = np.full_like(S, np.nan, dtype=float)
    if np.any(mask):
        lH2O = np.log10(Xh2o[mask])
        lCO2 = np.log10(Xco2[mask])
        tau_mask = (
            1.006
            - 0.1171 * lH2O
            - 0.02368 * (lH2O ** 2)
            - 0.03188 * lCO2
            + 0.001164 * (lCO2 ** 2)
        )
        tau[mask] = tau_mask
    if np.isscalar(S_m):
        return float(tau)
    return tau


def plot_transmissivity(params: TransmissivityParams = TransmissivityParams(), S_min: float = 0.5, S_max: float = 150.0, n_points: int = 500) -> None:
    S = np.linspace(S_min, S_max, n_points)
    tau = transmissivity(S, params)

    plt.figure(figsize=(7, 5))
    plt.plot(S, tau, color='tab:purple', linewidth=2)
    plt.xlabel('Distance S (m)')
    plt.ylabel('Transmissivity τ (dimensionless)')
    plt.title('Atmospheric transmissivity vs distance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    params = TransmissivityParams()
    print('Transmissivity model (defaults):')
    print(f'  Ta = {params.Ta_K:.2f} K  (from 24 °C)')
    print(f'  RH = {params.RH_percent:.1f} %')
    print(f'  PwSat(Ta) = {params.PwSat_Pa:.2f} Pa')

    # Example values
    for S in [1, 5, 10, 20, 50, 100, 150]:
        tau = transmissivity(S, params)
        print(f'S = {S:>5.1f} m -> τ = {tau:.5f}')

    print('\nPlotting τ(S)...')
    plot_transmissivity(params)


if __name__ == '__main__':
    main() 