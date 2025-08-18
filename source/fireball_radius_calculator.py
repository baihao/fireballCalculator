#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球直径随时间变化计算器

根据指数拖拽模型计算不同非爆轰材料的火球半径和直径随时间的变化
公式: R(t) = K * (1 - B*exp(-C*t^2))
其中: B=0.41, C=0.05, K根据材料类型选择Table 3中的K2参数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class FireballCalculator:
    """火球计算器类"""
    
    def __init__(self):
        """初始化不同材料的参数 (基于Table 3)"""
        # 材料参数表 (Table 3: Explosion cloud expansion radius fitting coefficients)
        self.materials = {
            'Polyurethane': {
                'K1': 2.87,
                'K2': 2.86,
                'B': 0.41,
                'C': 0.05,
                'precision_one_fitting': 94.99,
                'quadratic_fitting': 95.90,
                'description': '聚氨酯'
            },
            '30%Al/Rubber': {
                'K1': 3.03,
                'K2': 3.04,
                'B': 0.41,
                'C': 0.05,
                'precision_one_fitting': 96.44,
                'quadratic_fitting': 96.76,
                'description': '30%铝粉/橡胶'
            },
            '40%Al/Rubber': {
                'K1': 3.13,
                'K2': 3.15,
                'B': 0.41,
                'C': 0.05,
                'precision_one_fitting': 96.70,
                'quadratic_fitting': 96.90,
                'description': '40%铝粉/橡胶'
            },
            '50%Al/Rubber': {
                'K1': 3.07,
                'K2': 3.04,
                'B': 0.41,
                'C': 0.05,
                'precision_one_fitting': 97.95,
                'quadratic_fitting': 97.90,
                'description': '50%铝粉/橡胶'
            },
            '60%Al/Rubber': {
                'K1': 2.92,
                'K2': 2.93,
                'B': 0.41,
                'C': 0.05,
                'precision_one_fitting': 91.77,
                'quadratic_fitting': 93.34,
                'description': '60%铝粉/橡胶'
            }
        }
    
    def calculate_radius(self, t, material_name):
        """
        计算指定材料在时间t的火球半径
        
        参数:
        t: 时间 (s)
        material_name: 材料名称
        
        返回:
        R: 火球半径 (m)
        """
        if material_name not in self.materials:
            raise ValueError(f"未知的材料名称: {material_name}")
        
        params = self.materials[material_name]
        K = params['K2']  # 使用K2参数
        B = params['B']
        C = params['C'] * 1e6  # 将C从ms单位转换为s单位: C_s = C_ms * (1000)^2 = C_ms * 1e6
        
        # 计算半径: R(t) = K * (1 - B*exp(-C*t^2))，其中t单位为s
        R = K * (1 - B * np.exp(-C * t**2))
        return R
    
    def calculate_diameter(self, t, material_name):
        """
        计算指定材料在时间t的火球直径
        
        参数:
        t: 时间 (s)
        material_name: 材料名称
        
        返回:
        D: 火球直径 (m)
        """
        R = self.calculate_radius(t, material_name)
        D = 2 * R
        return D
    
    def calculate_expansion_velocity(self, t, material_name):
        """
        计算指定材料在时间t的火球膨胀速率（半径对时间的导数）
        
        参数:
        t: 时间 (s)
        material_name: 材料名称
        
        返回:
        V: 火球膨胀速率 (m/s)
        """
        if material_name not in self.materials:
            raise ValueError(f"未知的材料名称: {material_name}")
        
        params = self.materials[material_name]
        K = params['K2']
        B = params['B']
        C = params['C'] * 1e6  # 将C从ms单位转换为s单位
        
        # 计算膨胀速率: dR/dt = K * B * C * 2t * exp(-C*t^2)，其中t单位为s
        V = K * B * C * 2 * t * np.exp(-C * t**2)
        return V
    
    def calculate_all_materials(self, t):
        """
        计算所有材料在时间t的火球半径和直径
        
        参数:
        t: 时间 (s)
        
        返回:
        results: 包含所有材料结果的字典
        """
        results = {}
        for name in self.materials.keys():
            results[name] = {
                'radius': self.calculate_radius(t, name),
                'diameter': self.calculate_diameter(t, name),
                'velocity': self.calculate_expansion_velocity(t, name)
            }
        return results
    
    def plot_comparison(self, t_max=140.0, n_points=1000):
        """
        绘制所有材料火球参数对比图
        
        参数:
        t_max: 最大时间 (ms)，默认140ms
        n_points: 时间点数
        """
        t_ms = np.linspace(0, t_max, n_points)  # 时间单位为ms (用于显示)
        t_s = t_ms / 1000  # 转换为秒 (用于计算)
        
        plt.figure(figsize=(12, 5))
        
        # 半径对比
        plt.subplot(1, 2, 1)
        for name, params in self.materials.items():
            R = self.calculate_radius(t_s, name)  # 使用秒进行计算
            plt.plot(t_ms, R, label=f'{name}\n({params["description"]})', linewidth=2)
        
        plt.xlabel('时间 (ms)')
        plt.ylabel('火球半径 (m)')
        plt.title('不同材料火球半径随时间变化对比')
        plt.legend()
        plt.grid(True)
        
        # 膨胀速率对比
        plt.subplot(1, 2, 2)
        for name, params in self.materials.items():
            V = self.calculate_expansion_velocity(t_s, name)  # 使用秒进行计算
            plt.plot(t_ms, V, label=f'{name}\n({params["description"]})', linewidth=2)
        
        plt.xlabel('时间 (ms)')
        plt.ylabel('膨胀速率 (m/s)')
        plt.title('不同材料火球膨胀速率随时间变化对比')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_parameters(self):
        """打印所有材料的参数"""
        print("=" * 100)
        print("不同非爆轰材料的火球半径拟合参数 (Table 3)")
        print("=" * 100)
        print("公式: R(t) = K * (1 - B*exp(-C*t^2))")
        print("固定参数: B = 0.41, C = 0.05")
        print("=" * 100)
        
        for name, params in self.materials.items():
            print(f"\n{name} ({params['description']}):")
            print(f"  K1 = {params['K1']:.2f}")
            print(f"  K2 = {params['K2']:.2f} (使用此参数)")
            print(f"  B  = {params['B']:.2f}")
            print(f"  C  = {params['C']:.2f}")
            print(f"  拟合精度 = {params['precision_one_fitting']:.2f}%")
            print(f"  二次拟合精度 = {params['quadratic_fitting']:.2f}%")
    
    def calculate_at_specific_time(self, t, material_name):
        """
        计算指定材料在特定时间的火球参数
        
        参数:
        t: 时间 (s)
        material_name: 材料名称
        
        返回:
        dict: 包含半径、直径和膨胀速度的字典
        """
        R = self.calculate_radius(t, material_name)
        D = self.calculate_diameter(t, material_name)
        V = self.calculate_expansion_velocity(t, material_name)
        
        return {
            'time': t,
            'radius': R,
            'diameter': D,
            'expansion_velocity': V
        }

def main():
    """主函数"""
    print("火球直径随时间变化计算器")
    print("使用公式: R(t) = K * (1 - B*exp(-C*t^2))")
    print("时间单位: 秒(s) (显示为毫秒)")
    print("=" * 60)
    
    # 创建计算器实例
    calculator = FireballCalculator()
    
    # 打印参数
    calculator.print_parameters()
    
    # 计算特定时间的火球参数
    print("\n" + "=" * 60)
    print("特定时间点的火球参数计算")
    print("=" * 60)
    
    test_times_ms = [1, 5, 10, 50, 100, 140]  # 不同时间点 (ms)
    
    for t_ms in test_times_ms:
        t_s = t_ms / 1000  # 转换为秒
        print(f"\n时间 t = {t_ms} ms ({t_s:.3f} s):")
        print("-" * 50)
        
        for name in calculator.materials.keys():
            result = calculator.calculate_at_specific_time(t_s, name)
            print(f"{name}: 半径={result['radius']:.3f} m, "
                  f"直径={result['diameter']:.3f} m, "
                  f"膨胀速率={result['expansion_velocity']:.3f} m/s")
    
    # 绘制对比图
    print("\n正在生成对比图...")
    calculator.plot_comparison()
    
    print("\n计算完成！")

if __name__ == "__main__":
    main() 