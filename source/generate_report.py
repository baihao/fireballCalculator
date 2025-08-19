#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figures for the fireball report and save them to the images folder.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from fireball_radius_calculator import FireballCalculator
from fireball_temperature_calculator import FireballTemperatureCalculator
from transmissivity_calculator import TransmissivityParams, transmissivity
from fireball_heat_radiation_calculator import compute_H_vs_distance

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')


def ensure_img_dir() -> str:
    path = os.path.abspath(IMG_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def fig_diameter(material: str = '40%Al/Rubber', t_max_ms: float = 140.0) -> str:
    calc = FireballCalculator()
    t_ms = np.linspace(0, t_max_ms, 800)
    t_s = t_ms / 1000.0
    D = calc.calculate_diameter(t_s, material)

    plt.figure(figsize=(7, 5))
    plt.plot(t_ms, D, linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Diameter D (m)')
    plt.title(f'Fireball diameter vs time (material={material})')
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(ensure_img_dir(), 'report_diameter.png')
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def fig_radius_rate(material: str = '40%Al/Rubber', t_max_ms: float = 140.0) -> str:
    calc = FireballCalculator()
    t_ms = np.linspace(0, t_max_ms, 800)
    t_s = t_ms / 1000.0
    V = calc.calculate_expansion_velocity(t_s, material)

    plt.figure(figsize=(7, 5))
    plt.plot(t_ms, V, color='tab:green', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Expansion velocity dR/dt (m/s)')
    plt.title(f'Fireball expansion velocity vs time (material={material})')
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(ensure_img_dir(), 'report_radius_rate.png')
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def fig_temperature(t_max_ms: float = 140.0) -> str:
    temp = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
    t_ms = np.linspace(0, t_max_ms, 800)
    T = temp.temperature_modified(t_ms)

    plt.figure(figsize=(7, 5))
    plt.plot(t_ms, T, color='tab:orange', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Temperature (K)')
    plt.title('Fireball surface temperature vs time')
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(ensure_img_dir(), 'report_temperature.png')
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def fig_temperature_rate(t_max_ms: float = 140.0) -> str:
    temp = FireballTemperatureCalculator(mode='blend', blend_width_ms=12.0)
    t_ms = np.linspace(0, t_max_ms, 800)
    dT = temp.rate_modified(t_ms)

    plt.figure(figsize=(7, 5))
    plt.plot(t_ms, dT, color='tab:blue', linewidth=2)
    plt.axhline(0.0, color='gray', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('dT/dt (K/ms)')
    plt.title('Fireball temperature rate vs time')
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(ensure_img_dir(), 'report_temperature_rate.png')
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def fig_transmissivity() -> str:
    params = TransmissivityParams()
    S = np.linspace(0.5, 150.0, 500)
    tau = transmissivity(S, params)

    plt.figure(figsize=(7, 5))
    plt.plot(S, tau, color='tab:purple', linewidth=2)
    plt.xlabel('Distance S (m)')
    plt.ylabel('Transmissivity τ (dimensionless)')
    plt.title('Atmospheric transmissivity vs distance')
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(ensure_img_dir(), 'report_transmissivity.png')
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def fig_heat_radiation(material: str = '40%Al/Rubber') -> str:
    xs, Hs = compute_H_vs_distance(4.0, 6.0, 200, material=material)
    plt.figure(figsize=(7, 5))
    plt.plot(xs, Hs, color='tab:red', linewidth=2)
    plt.xlabel('Distance x (m)')
    plt.ylabel('Heat radiation H (J/m^2)')
    plt.title(f'Time-integrated heat radiation (0–140 ms), material={material}')
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(ensure_img_dir(), 'report_heat_radiation.png')
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def main():
    ensure_img_dir()
    print('Generating figures...')
    paths = []
    paths.append(fig_diameter())
    paths.append(fig_radius_rate())
    paths.append(fig_temperature())
    paths.append(fig_temperature_rate())
    paths.append(fig_transmissivity())
    paths.append(fig_heat_radiation())
    for p in paths:
        print('  saved:', p)


if __name__ == '__main__':
    main() 