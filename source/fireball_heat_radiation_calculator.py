#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fireball heat radiation calculator

q(x, t) = E(t) * F(x, t) * tau(x)
  E(t) = epsilon * sigma * T(t)^4
  F(x, t) = 1/4 * ( D(t) / x )^2,  D(t) = 2 * R(t)
  tau(x) = atmospheric transmissivity at range x (dimensionless)

Time-integrated heat radiation (energy density):
  H(x) = ∫_{t=0}^{t=140ms} q(x, t) dt   [J/m^2]

Defaults:
- epsilon = 0.9
- sigma = 5.67e-8  [W/(m^2·K^4)]
- time window: 0–140 ms, 800 points
- temperature model: fireball_temperature_calculator (mode='blend') → T in K
- radius model: fireball_radius_calculator (material='40%Al/Rubber')
- transmissivity: transmissivity_calculator with defaults (Ta=297.15 K, RH=48%, PwSat=2987.87 Pa)

The script plots H(x) for x in [4, 6] m.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from fireball_temperature_calculator import FireballTemperatureCalculator
from fireball_radius_calculator import FireballCalculator
from transmissivity_calculator import TransmissivityParams, transmissivity

# Constants
EPSILON = 0.9
SIGMA = 5.67e-8  # W/(m^2*K^4)


def compute_temperature_profile(t_ms: np.ndarray) -> np.ndarray:
    temp_calc = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
    T_K = temp_calc.temperature_modified(t_ms)
    return T_K


def compute_diameter_profile(t_ms: np.ndarray, material: str = '40%Al/Rubber') -> np.ndarray:
    radius_calc = FireballCalculator()
    t_s = t_ms / 1000.0
    D_m = radius_calc.calculate_diameter(t_s, material)
    return D_m


def compute_heat_flux_over_time(x_m: float, t_ms: np.ndarray, T_K: np.ndarray, D_m: np.ndarray,
                                 trans_params: TransmissivityParams = TransmissivityParams()) -> np.ndarray:
    # E(t)
    E_t = EPSILON * SIGMA * T_K**4  # W/m^2
    # F(x,t)
    F_t = 0.25 * (D_m / x_m)**2
    # tau(x)
    tau_x = transmissivity(x_m, trans_params)  # scalar
    q_t = E_t * F_t * tau_x  # W/m^2
    return q_t


def integrate_heat_radiation(q_t: np.ndarray, t_ms: np.ndarray) -> float:
    t_s = t_ms / 1000.0
    H = float(np.trapz(q_t, t_s))  # J/m^2
    return H


def compute_H_vs_distance(x_min: float = 4.0, x_max: float = 6.0, n_x: int = 200,
                           material: str = '40%Al/Rubber') -> tuple[np.ndarray, np.ndarray]:
    t_ms = np.linspace(0.0, 140.0, 800)
    T_K = compute_temperature_profile(t_ms)
    D_m = compute_diameter_profile(t_ms, material=material)

    params = TransmissivityParams()

    xs = np.linspace(x_min, x_max, n_x)
    Hs = np.zeros_like(xs)
    for i, x in enumerate(xs):
        q_t = compute_heat_flux_over_time(x, t_ms, T_K, D_m, params)
        Hs[i] = integrate_heat_radiation(q_t, t_ms)
    return xs, Hs


def plot_H_vs_distance(xs: np.ndarray, Hs: np.ndarray, material: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(xs, Hs, color='tab:red', linewidth=2)
    plt.xlabel('Distance x (m)')
    plt.ylabel('Heat radiation H (J/m^2)')
    plt.title(f'Time-integrated heat radiation (0–140 ms), material={material}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_heat_flux_vs_time(x_values: list[float] = None, material: str = '40%Al/Rubber',
                           trans_params: TransmissivityParams = TransmissivityParams(),
                           save_path: str = None) -> None:
    """
    Plot heat flux q(x,t) vs time for multiple distance values x.
    
    Args:
        x_values: List of distance values in meters. If None, uses [4.0, 4.5, 5.0, 5.5, 6.0]
        material: Explosive material type
        trans_params: Atmospheric transmissivity parameters
        save_path: Optional path to save the plot as PNG file
    """
    if x_values is None:
        x_values = [4.0, 4.5, 5.0, 5.5, 6.0]
    
    # Time array
    t_ms = np.linspace(0.0, 140.0, 800)
    
    # Compute temperature and diameter profiles
    T_K = compute_temperature_profile(t_ms)
    D_m = compute_diameter_profile(t_ms, material=material)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Colors for different distance curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_values)))
    
    for i, x in enumerate(x_values):
        # Compute heat flux over time for this distance
        q_t = compute_heat_flux_over_time(x, t_ms, T_K, D_m, trans_params)
        
        # Plot the curve
        plt.plot(t_ms, q_t, color=colors[i], linewidth=2, 
                label=f'x = {x:.1f} m')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Heat flux q(x,t) (W/m²)')
    plt.title(f'Heat flux vs time for different distances, material={material}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Heat flux plot saved to: {save_path}')
    
    plt.show()
    
    # Print some statistics
    print(f'\nHeat flux statistics for material={material}:')
    for x in x_values:
        q_t = compute_heat_flux_over_time(x, t_ms, T_K, D_m, trans_params)
        max_flux = np.max(q_t)
        max_time = t_ms[np.argmax(q_t)]
        total_energy = integrate_heat_radiation(q_t, t_ms)
        print(f'  x = {x:.1f} m: max flux = {max_flux:.1f} W/m² at t = {max_time:.1f} ms, '
              f'total energy = {total_energy:.1f} J/m²')


def main():
    material = '40%Al/Rubber'
    
    # Plot 1: Heat flux vs time for multiple distances
    print('Plotting heat flux vs time for different distances...')
    plot_heat_flux_vs_time(material=material)
    
    # Plot 2: Time-integrated heat radiation vs distance
    print('\nComputing time-integrated heat radiation...')
    xs, Hs = compute_H_vs_distance(4.0, 6.0, 200, material=material)

    # Print a few samples
    print('Heat radiation H(x) for 0–140 ms (J/m^2):')
    for x, H in zip(xs[::50], Hs[::50]):
        print(f'  x = {x:.2f} m -> H = {H:.2f} J/m^2')

    print('\nPlotting H(x) for x in [4, 6] m...')
    plot_H_vs_distance(xs, Hs, material)


if __name__ == '__main__':
    main() 