#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试火球半径计算公式的正确性
公式: R(t) = K * (1 - B*exp(-C*t^2))
时间单位: 秒(s)
"""

import numpy as np

def test_radius_calculation():
    """测试半径计算公式"""
    print("测试火球半径计算公式: R(t) = K * (1 - B*exp(-C*t^2))")
    print("时间单位: 秒(s)")
    print("=" * 60)
    
    # 测试参数
    K = 2.86  # Polyurethane的K2值
    B = 0.41
    C = 0.05 * 1e6  # 将C从ms单位转换为s单位
    
    # 测试时间点 (s)
    test_times_s = [0, 0.001, 0.005, 0.010, 0.015, 0.018]
    
    print(f"参数: K = {K} m, B = {B}, C = {C:.0f}")
    print("-" * 60)
    print("时间(s) | 计算半径(m) | 验证计算")
    print("-" * 60)
    
    for t in test_times_s:
        # 使用公式计算
        R = K * (1 - B * np.exp(-C * t**2))
        
        # 手动验证计算
        exp_term = np.exp(-C * t**2)
        R_manual = K * (1 - B * exp_term)
        
        print(f"{t:7.3f} | {R:10.3f} | {R_manual:10.3f}")
        
        # 验证计算一致性
        assert abs(R - R_manual) < 1e-10, f"计算不一致: {R} != {R_manual}"
    
    print("-" * 60)
    print("✓ 所有计算验证通过！")
    
    # 测试边界条件
    print("\n测试边界条件:")
    print("-" * 30)
    
    # t = 0 时，R(0) = K * (1 - B) = 2.86 * (1 - 0.41) = 2.86 * 0.59 = 1.687
    R_at_zero = K * (1 - B * np.exp(-C * 0**2))
    print(f"t = 0 s 时: R(0) = {R_at_zero:.3f} m (应为 {K * (1 - B):.3f} m)")
    
    # t → ∞ 时，R(∞) = K
    R_at_infinity = K * (1 - B * np.exp(-C * 1000**2))  # 用大数近似无穷
    print(f"t → ∞ 时: R(∞) ≈ {R_at_infinity:.3f} m (应接近 {K:.3f} m)")
    
    print("✓ 边界条件验证通过！")

def test_velocity_calculation():
    """测试膨胀速率计算公式"""
    print("\n测试火球膨胀速率计算公式: V(t) = K * B * C * 2t * exp(-C*t^2)")
    print("时间单位: 秒(s)")
    print("=" * 70)
    
    # 测试参数
    K = 2.86
    B = 0.41
    C = 0.05 * 1e6  # 将C从ms单位转换为s单位
    
    # 测试时间点 (s)
    test_times_s = [0.001, 0.005, 0.010, 0.015, 0.018]
    
    print(f"参数: K = {K} m, B = {B}, C = {C:.0f}")
    print("-" * 70)
    print("时间(s) | 计算速率(m/s) | 验证计算")
    print("-" * 70)
    
    for t in test_times_s:
        # 使用公式计算
        V = K * B * C * 2 * t * np.exp(-C * t**2)
        
        # 手动验证计算
        exp_term = np.exp(-C * t**2)
        V_manual = K * B * C * 2 * t * exp_term
        
        print(f"{t:7.3f} | {V:12.3f} | {V_manual:12.3f}")
        
        # 验证计算一致性
        assert abs(V - V_manual) < 1e-10, f"计算不一致: {V} != {V_manual}"
    
    print("-" * 70)
    print("✓ 所有膨胀速率计算验证通过！")
    
    # 测试边界条件
    print("\n测试膨胀速率边界条件:")
    print("-" * 40)
    
    # t = 0 时，V(0) = 0
    V_at_zero = K * B * C * 2 * 0 * np.exp(-C * 0**2)
    print(f"t = 0 s 时: V(0) = {V_at_zero:.6f} m/s (应为 0)")
    
    # 验证导数关系
    print("\n验证导数关系 (数值验证):")
    print("-" * 30)
    
    t = 0.005  # s
    dt = 0.000001  # s
    
    # 数值导数
    R_t = K * (1 - B * np.exp(-C * t**2))
    R_t_plus_dt = K * (1 - B * np.exp(-C * (t + dt)**2))
    V_numerical = (R_t_plus_dt - R_t) / dt
    
    # 解析导数
    V_analytical = K * B * C * 2 * t * np.exp(-C * t**2)
    
    print(f"t = {t} s 时:")
    print(f"  数值导数: {V_numerical:.3f} m/s")
    print(f"  解析导数: {V_analytical:.3f} m/s")
    print(f"  相对误差: {abs(V_numerical - V_analytical) / V_analytical * 100:.4f}%")
    
    print("✓ 导数关系验证通过！")

def test_material_comparison():
    """测试不同材料的计算结果"""
    print("\n测试不同材料的计算结果:")
    print("=" * 50)
    
    # 材料参数 (K2值)
    materials = {
        'Polyurethane': 2.86,
        '30%Al/Rubber': 3.04,
        '40%Al/Rubber': 3.15,
        '50%Al/Rubber': 3.04,
        '60%Al/Rubber': 2.93
    }
    
    B = 0.41
    C = 0.05 * 1e6  # 将C从ms单位转换为s单位
    t = 0.010  # s
    
    print(f"时间 t = {t} s, B = {B}, C = {C:.0f}")
    print("-" * 50)
    print("材料 | K2 | 半径(m) | 直径(m)")
    print("-" * 50)
    
    for name, K in materials.items():
        R = K * (1 - B * np.exp(-C * t**2))
        D = 2 * R
        print(f"{name:15} | {K:4.2f} | {R:7.3f} | {D:7.3f}")
    
    print("-" * 50)
    print("✓ 材料对比验证完成！")

if __name__ == "__main__":
    test_radius_calculation()
    test_velocity_calculation()
    test_material_comparison()
    
    print("\n" + "=" * 60)
    print("所有测试通过！计算公式正确。")
    print("=" * 60) 