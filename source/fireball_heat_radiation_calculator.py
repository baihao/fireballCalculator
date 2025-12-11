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
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from fireball_temperature_calculator import FireballTemperatureCalculator
from fireball_radius_calculator import FireballCalculator
from transmissivity_calculator import TransmissivityParams, transmissivity

# Constants
EPSILON = 0.9
SIGMA = 5.67e-8  # W/(m^2*K^4)


def load_csv_temperature(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV文件加载温度数据
    
    Args:
        file_path: CSV文件路径，格式应为：时间(ms),温度(K)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (时间数组, 温度数组)
    """
    time_data = []
    temp_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        for row in reader:
            if len(row) < 2:
                continue
            try:
                time_val = float(row[0].strip())
                temp_val = float(row[1].strip())
                if time_val >= 0 and temp_val > 0:
                    time_data.append(time_val)
                    temp_data.append(temp_val)
            except ValueError:
                continue
    
    return np.array(time_data), np.array(temp_data)


def load_csv_diameter(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV文件加载直径数据
    
    Args:
        file_path: CSV文件路径，格式应为：时间(ms),直径(m)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (时间数组, 直径数组)
    """
    time_data = []
    diameter_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        for row in reader:
            if len(row) < 2:
                continue
            try:
                time_val = float(row[0].strip())
                diameter_val = float(row[1].strip())
                if time_val >= 0 and diameter_val > 0:
                    time_data.append(time_val)
                    diameter_data.append(diameter_val)
            except ValueError:
                continue
    
    return np.array(time_data), np.array(diameter_data)


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
    """
    计算热通量随时间的变化
    
    Args:
        x_m: 距离 (m)
        t_ms: 时间数组 (ms)
        T_K: 温度数组 (K)
        D_m: 直径数组 (m)
        trans_params: 大气透射率参数
        
    Returns:
        np.ndarray: 热通量数组 (W/m^2)
    """
    # E(t)
    E_t = EPSILON * SIGMA * T_K**4  # W/m^2
    # F(x,t)
    F_t = 0.25 * (D_m / x_m)**2
    # tau(x)
    tau_x = transmissivity(x_m, trans_params)  # scalar
    q_t = E_t * F_t * tau_x  # W/m^2
    return q_t


def compute_cumulative_heat_radiation(q_t: np.ndarray, t_ms: np.ndarray) -> np.ndarray:
    """
    计算累计热辐射（从t=0到当前时间的积分）
    
    Args:
        q_t: 热通量数组 (W/m^2)
        t_ms: 时间数组 (ms)
        
    Returns:
        np.ndarray: 累计热辐射数组 (J/m^2)
    """
    t_s = t_ms / 1000.0
    cumulative = np.zeros_like(q_t)
    for i in range(1, len(q_t)):
        cumulative[i] = np.trapz(q_t[:i+1], t_s[:i+1])
    return cumulative


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


def simulate_from_csv(diameter_csv: str, temperature_csv: str,
                     x_values: list[float] = None,
                     trans_params: TransmissivityParams = TransmissivityParams(),
                     output_dir: str = None) -> None:
    """
    从CSV文件读取直径和温度数据，进行仿真并输出所有结果图表
    
    Args:
        diameter_csv: 直径CSV文件路径（格式：时间(ms),直径(m)）
        temperature_csv: 温度CSV文件路径（格式：时间(ms),温度(K)）
        x_values: 距离值列表，如果为None则使用默认值
        trans_params: 大气透射率参数
        output_dir: 输出目录，如果为None则不保存文件
    """
    import os
    print("=" * 60)
    print("火球热辐射仿真（CSV数据）")
    print("=" * 60)
    
    # 1. 加载CSV数据
    print(f"\n1. 加载CSV数据...")
    print(f"   直径文件: {diameter_csv}")
    t_dia, D_m = load_csv_diameter(diameter_csv)
    print(f"   ✓ 加载直径数据: {len(t_dia)} 个数据点")
    print(f"   温度文件: {temperature_csv}")
    t_temp, T_K = load_csv_temperature(temperature_csv)
    print(f"   ✓ 加载温度数据: {len(t_temp)} 个数据点")
    
    # 2. 统一时间网格（使用两个时间数组的交集）
    t_min = max(t_dia.min(), t_temp.min())
    t_max = min(t_dia.max(), t_temp.max())
    t_ms = np.linspace(t_min, t_max, min(len(t_dia), len(t_temp)))
    
    # 插值到统一时间网格
    D_interp = np.interp(t_ms, t_dia, D_m)
    T_interp = np.interp(t_ms, t_temp, T_K)
    
    print(f"   ✓ 统一时间网格: {len(t_ms)} 个数据点，范围 {t_min:.1f} - {t_max:.1f} ms")
    
    # 3. 设置距离值
    if x_values is None:
        x_values = [6.0, 7.0, 8.0, 9.0, 10.0]
    
    # 4. 计算热通量和累计热辐射
    print(f"\n2. 计算热通量和累计热辐射...")
    heat_flux_data = {}
    cumulative_radiation_data = {}
    
    for x in x_values:
        q_t = compute_heat_flux_over_time(x, t_ms, T_interp, D_interp, trans_params)
        H_cumulative = compute_cumulative_heat_radiation(q_t, t_ms)
        heat_flux_data[x] = q_t
        cumulative_radiation_data[x] = H_cumulative
    
    print(f"   ✓ 计算完成，距离值: {x_values} m")
    
    # 5. 绘制所有结果图表
    print(f"\n3. 生成结果图表...")
    
    # 创建输出目录（如果指定）
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # 5.1 温度随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(t_ms, T_interp, 'r-', linewidth=2)
    plt.xlabel('时间 (ms)', fontsize=12)
    plt.ylabel('温度 (K)', fontsize=12)
    plt.title('火球表面温度随时间变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        temp_path = os.path.join(output_dir, "temperature_vs_time.png")
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 温度图表已保存: {temp_path}")
    plt.close()
    
    # 5.2 直径随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(t_ms, D_interp, 'b-', linewidth=2)
    plt.xlabel('时间 (ms)', fontsize=12)
    plt.ylabel('直径 (m)', fontsize=12)
    plt.title('火球直径随时间变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        dia_path = os.path.join(output_dir, "diameter_vs_time.png")
        plt.savefig(dia_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 直径图表已保存: {dia_path}")
    plt.close()
    
    # 5.3 热通量随时间变化（多个距离）
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_values)))
    for i, x in enumerate(x_values):
        plt.plot(t_ms, heat_flux_data[x], color=colors[i], linewidth=2,
                label=f'x = {x:.1f} m')
    plt.xlabel('时间 (ms)', fontsize=12)
    plt.ylabel('热通量 q(x,t) (W/m²)', fontsize=12)
    plt.title('热通量随时间变化（不同距离）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        flux_path = os.path.join(output_dir, "heat_flux_vs_time.png")
        plt.savefig(flux_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 热通量图表已保存: {flux_path}")
    plt.close()
    
    # 5.4 累计热辐射随时间变化（多个距离）
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(x_values)))
    for i, x in enumerate(x_values):
        plt.plot(t_ms, cumulative_radiation_data[x], color=colors[i], linewidth=2,
                label=f'x = {x:.1f} m')
    plt.xlabel('时间 (ms)', fontsize=12)
    plt.ylabel('累计热辐射 H(x,t) (J/m²)', fontsize=12)
    plt.title('累计热辐射随时间变化（不同距离）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        cumulative_path = os.path.join(output_dir, "cumulative_heat_radiation_vs_time.png")
        plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 累计热辐射图表已保存: {cumulative_path}")
    plt.close()
    
    # 5.5 综合图表（子图形式）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 温度
    axes[0, 0].plot(t_ms, T_interp, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('时间 (ms)', fontsize=11)
    axes[0, 0].set_ylabel('温度 (K)', fontsize=11)
    axes[0, 0].set_title('温度随时间变化', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 直径
    axes[0, 1].plot(t_ms, D_interp, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('时间 (ms)', fontsize=11)
    axes[0, 1].set_ylabel('直径 (m)', fontsize=11)
    axes[0, 1].set_title('直径随时间变化', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 热通量
    colors_flux = plt.cm.viridis(np.linspace(0, 1, len(x_values)))
    for i, x in enumerate(x_values):
        axes[1, 0].plot(t_ms, heat_flux_data[x], color=colors_flux[i], linewidth=2,
                       label=f'x={x:.1f}m')
    axes[1, 0].set_xlabel('时间 (ms)', fontsize=11)
    axes[1, 0].set_ylabel('热通量 (W/m²)', fontsize=11)
    axes[1, 0].set_title('热通量随时间变化', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 累计热辐射
    colors_cum = plt.cm.plasma(np.linspace(0, 1, len(x_values)))
    for i, x in enumerate(x_values):
        axes[1, 1].plot(t_ms, cumulative_radiation_data[x], color=colors_cum[i], linewidth=2,
                       label=f'x={x:.1f}m')
    axes[1, 1].set_xlabel('时间 (ms)', fontsize=11)
    axes[1, 1].set_ylabel('累计热辐射 (J/m²)', fontsize=11)
    axes[1, 1].set_title('累计热辐射随时间变化', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        combined_path = os.path.join(output_dir, "combined_results.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 综合图表已保存: {combined_path}")
    plt.close()
    
    # 6. 计算并输出累计热辐射随距离变化的表
    print(f"\n4. 计算累计热辐射随距离变化...")
    
    # 生成更密集的距离点用于表格（距离范围：6-10m）
    x_table_min = 6.0
    x_table_max = 10.0
    x_table = np.linspace(x_table_min, x_table_max, 50)  # 50个距离点
    
    H_final_table = []
    for x in x_table:
        q_t = compute_heat_flux_over_time(x, t_ms, T_interp, D_interp, trans_params)
        H_cumulative = compute_cumulative_heat_radiation(q_t, t_ms)
        H_final = H_cumulative[-1]
        H_final_table.append(H_final)
    
    # 输出表格
    print(f"\n   累计热辐射随距离变化表:")
    print(f"   {'距离(m)':<12} {'累计热辐射(J/m²)':<20}")
    print(f"   {'-'*12} {'-'*20}")
    for i in range(0, len(x_table), max(1, len(x_table)//20)):  # 显示约20个点
        print(f"   {x_table[i]:>10.2f}  {H_final_table[i]:>18.2f}")
    
    # 保存CSV文件
    if output_dir:
        csv_path = os.path.join(output_dir, "cumulative_heat_radiation_vs_distance.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['距离(m)', '累计热辐射(J/m²)'])
                for i in range(len(x_table)):
                    writer.writerow([
                        f'{x_table[i]:.6f}',
                        f'{H_final_table[i]:.6f}'
                    ])
            print(f"\n   ✓ 累计热辐射随距离变化表已保存: {csv_path}")
        except Exception as e:
            print(f"\n   ⚠️ 保存CSV文件失败: {str(e)}")
    
    # 绘制累计热辐射随距离变化的图表
    plt.figure(figsize=(10, 6))
    plt.plot(x_table, H_final_table, 'r-', linewidth=2)
    plt.xlabel('距离 (m)', fontsize=12)
    plt.ylabel('累计热辐射 (J/m²)', fontsize=12)
    plt.title('累计热辐射随距离变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        distance_path = os.path.join(output_dir, "cumulative_heat_radiation_vs_distance.png")
        plt.savefig(distance_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ 累计热辐射随距离变化图表已保存: {distance_path}")
    plt.close()
    
    # 7. 输出统计信息
    print(f"\n5. 统计信息:")
    print(f"   时间范围: {t_min:.1f} - {t_max:.1f} ms")
    print(f"   温度范围: {T_interp.min():.1f} - {T_interp.max():.1f} K")
    print(f"   直径范围: {D_interp.min():.3f} - {D_interp.max():.3f} m")
    print(f"\n   各距离的热通量和累计热辐射:")
    for x in x_values:
        q_t = heat_flux_data[x]
        H_final = cumulative_radiation_data[x][-1]
        max_flux = np.max(q_t)
        max_flux_time = t_ms[np.argmax(q_t)]
        print(f"     x = {x:.1f} m:")
        print(f"       最大热通量: {max_flux:.1f} W/m² (t = {max_flux_time:.1f} ms)")
        print(f"       最终累计热辐射: {H_final:.1f} J/m²")
    
    print("\n" + "=" * 60)
    print("仿真完成！")
    print("=" * 60)


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
    import sys
    
    # 检查是否提供了CSV文件参数
    if len(sys.argv) >= 3:
        diameter_csv = sys.argv[1]
        temperature_csv = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        
        # 从CSV文件进行仿真
        simulate_from_csv(diameter_csv, temperature_csv, output_dir=output_dir)
    else:
        # 默认使用模型计算
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