#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球直径拖曳曲线拟合结果绘制模块

提供拖曳曲线拟合结果的可视化功能，包括拟合曲线图和残差分析图。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


class DragFitPlotter:
    """拖曳曲线拟合结果绘制器"""
    
    def __init__(self):
        """初始化绘制器"""
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    @staticmethod
    def drag_function(t: np.ndarray, K: float, B: float, C: float) -> np.ndarray:
        """
        拖曳函数模型
        
        Args:
            t: 时间数组
            K: 最大直径参数
            B: 初始拖曳系数
            C: 拖曳衰减系数
            
        Returns:
            np.ndarray: 计算的直径值
        """
        return K * (1 - B * np.exp(-C * t**2))
    
    def plot_fit_results(self, time_data: List[float], diameter_data: List[float],
                        fit_result: Dict[str, Any], save_path: Optional[str] = None,
                        time_unit: str = 'ms') -> bool:
        """
        绘制拟合结果图（改进版，支持毫秒时间单位）
        
        Args:
            time_data: 原始时间数据（毫秒）
            diameter_data: 原始直径数据（米）
            fit_result: 拟合结果
            save_path: 保存路径（可选）
            time_unit: 时间单位（'ms' 或 's'）
            
        Returns:
            bool: 是否成功绘制
        """
        try:
            if not fit_result.get('success', False):
                print("❌ 无法绘制：拟合失败")
                return False
            
            t = np.array(time_data)
            D = np.array(diameter_data)
            K, B, C = fit_result['K'], fit_result['B'], fit_result['C']
            
            # 时间单位处理
            if time_unit == 's':
                t_display = t / 1000.0  # 转换为秒显示
                time_label = '时间 (s)'
                C_display = C * 1e6  # 转换为 s⁻² 显示
                C_unit = 's⁻²'
            else:
                t_display = t  # 保持毫秒
                time_label = '时间 (ms)'
                C_display = C
                C_unit = 'ms⁻²'
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 主图：拟合曲线
            ax1.scatter(t_display, D, color='red', alpha=0.7, s=50, label='观测数据', zorder=3)
            
            # 检查是否有数据过滤信息，显示截断点
            filtering_info = fit_result.get('data_filtering', {})
            if filtering_info.get('enabled', False) and filtering_info.get('cutoff_time') is not None:
                cutoff_time = filtering_info['cutoff_time']
                if time_unit == 's':
                    cutoff_display = cutoff_time / 1000.0
                else:
                    cutoff_display = cutoff_time
                
                # 标记截断点
                ax1.axvline(x=cutoff_display, color='orange', linestyle='--', linewidth=2, 
                           label=f'数据截断点 ({cutoff_display:.1f}{"s" if time_unit == "s" else "ms"})', zorder=1)
                
                # 标记过滤后的数据范围
                filtered_mask = t <= cutoff_time
                if np.any(filtered_mask):
                    ax1.scatter(t_display[filtered_mask], D[filtered_mask], color='green', alpha=0.8, s=30, 
                              label='过滤后数据', zorder=4)
            
            # 生成平滑的拟合曲线
            t_smooth = np.linspace(t[0], t[-1], 200)
            D_smooth = self.drag_function(t_smooth, K, B, C)
            if time_unit == 's':
                t_smooth_display = t_smooth / 1000.0
            else:
                t_smooth_display = t_smooth
            ax1.plot(t_smooth_display, D_smooth, 'b-', linewidth=2, label='拟合曲线', zorder=2)
            
            # 添加参数信息
            param_text = f'K = {K:.3f} m\nB = {B:.3f}\nC = {C_display:.3e} {C_unit}'
            quality_text = f'R² = {fit_result.get("r_squared", 0):.4f}\nRMSE = {fit_result.get("rmse", 0):.4f} m\nMAE = {fit_result.get("mae", 0):.4f} m'
            
            # 添加数据过滤信息
            filtering_text = ""
            if filtering_info.get('enabled', False):
                filtering_text = f'数据过滤: 启用\n'
                filtering_text += f'保留率: {filtering_info.get("data_retention_rate", 1.0):.1%}\n'
                if filtering_info.get('cutoff_time') is not None:
                    filtering_text += f'截断时间: {filtering_info["cutoff_time"]:.1f}ms'
            else:
                filtering_text = '数据过滤: 禁用'
            
            ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            ax1.text(0.05, 0.75, quality_text, transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            ax1.text(0.05, 0.55, filtering_text, transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            
            ax1.set_xlabel(time_label)
            ax1.set_ylabel('火球直径 (m)')
            ax1.set_title('火球直径拖曳曲线拟合结果（改进算法）')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 残差图
            D_pred = self.drag_function(t, K, B, C)
            residuals = D - D_pred
            
            ax2.scatter(t_display, residuals, color='green', alpha=0.7, s=30)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel(time_label)
            ax2.set_ylabel('残差 (m)')
            ax2.set_title('拟合残差分析')
            ax2.grid(True, alpha=0.3)
            
            # 添加残差统计
            residual_std = np.std(residuals)
            ax2.text(0.05, 0.95, f'残差标准差: {residual_std:.4f} m', 
                    transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            
            plt.tight_layout()
            
            # 保存图片
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 拟合结果图已保存: {save_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 绘制拟合结果失败: {e}")
            return False
    
    def plot_drag_function_curve(self, K: float, B: float, C: float, 
                                time_range: tuple = (0, 150), time_unit: str = 'ms',
                                save_path: Optional[str] = None) -> bool:
        """
        绘制拖曳函数曲线
        
        Args:
            K: 最大直径参数
            B: 初始拖曳系数
            C: 拖曳衰减系数
            time_range: 时间范围 (start, end)
            time_unit: 时间单位（'ms' 或 's'）
            save_path: 保存路径（可选）
            
        Returns:
            bool: 是否成功绘制
        """
        try:
            # 创建时间轴
            t = np.linspace(time_range[0], time_range[1], 1000)
            D = self.drag_function(t, K, B, C)
            
            # 时间单位处理
            if time_unit == 's':
                t_display = t / 1000.0
                time_label = '时间 (s)'
                C_display = C * 1e6
                C_unit = 's⁻²'
            else:
                t_display = t
                time_label = '时间 (ms)'
                C_display = C
                C_unit = 'ms⁻²'
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # 绘制拖曳函数
            ax.plot(t_display, D, 'b-', linewidth=3, label=f'拖曳函数: D(t) = K×(1-B×exp(-C×t²))')
            
            # 标记关键点
            D_0 = K * (1 - B)
            ax.plot(0, D_0, 'ro', markersize=10, label=f'初始值 D(0)={D_0:.1f}m')
            ax.axhline(y=K, color='r', linestyle='--', alpha=0.7, label=f'渐近值 K={K:.1f}m')
            
            # 标记半衰期
            half_life = np.sqrt(np.log(2) / C)
            D_half = self.drag_function(half_life, K, B, C)
            if time_unit == 's':
                half_life_display = half_life / 1000.0
            else:
                half_life_display = half_life
            ax.plot(half_life_display, D_half, 'go', markersize=10, label=f'半衰期 t₁/₂={half_life_display:.1f}{time_unit}')
            
            # 标记95%收敛时间
            t_95 = np.sqrt(-np.log(0.05) / C)
            D_95 = self.drag_function(t_95, K, B, C)
            if time_unit == 's':
                t_95_display = t_95 / 1000.0
            else:
                t_95_display = t_95
            ax.plot(t_95_display, D_95, 'mo', markersize=10, label=f'95%收敛 t={t_95_display:.1f}{time_unit}')
            
            ax.set_xlabel(time_label, fontsize=12)
            ax.set_ylabel('直径 (m)', fontsize=12)
            ax.set_title(f'拖曳函数拟合 (K={K:.1f}, B={B:.4f}, C={C:.4f})', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, time_range[1] / 1000.0 if time_unit == 's' else time_range[1])
            ax.set_ylim(0, K * 1.1)
            
            # 添加文本说明
            textstr = f'''拖曳函数参数:
• 初始值: D(0) = {D_0:.1f}m
• 渐近值: D(∞) = {K:.1f}m
• 半衰期: t₁/₂ = {half_life_display:.1f}{time_unit}
• 95%收敛: t = {t_95_display:.1f}{time_unit}'''
            
            props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # 保存图片
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 拖曳函数图已保存: {save_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 绘制拖曳函数失败: {e}")
            return False


def create_drag_fit_plotter() -> DragFitPlotter:
    """
    创建拖曳曲线拟合结果绘制器的便捷函数
    
    Returns:
        DragFitPlotter: 绘制器实例
    """
    return DragFitPlotter()


def plot_drag_fit_results(time_data: List[float], diameter_data: List[float],
                         fit_result: Dict[str, Any], save_path: Optional[str] = None,
                         time_unit: str = 'ms') -> bool:
    """
    便捷函数：绘制拖曳曲线拟合结果
    
    Args:
        time_data: 原始时间数据（毫秒）
        diameter_data: 原始直径数据（米）
        fit_result: 拟合结果
        save_path: 保存路径（可选）
        time_unit: 时间单位（'ms' 或 's'）
        
    Returns:
        bool: 是否成功绘制
    """
    plotter = create_drag_fit_plotter()
    return plotter.plot_fit_results(time_data, diameter_data, fit_result, save_path, time_unit)


if __name__ == "__main__":
    print("拖曳曲线拟合结果绘制模块")
    print("使用方法:")
    print("  from drag_fit_plotter import plot_drag_fit_results")
    print("  plot_drag_fit_results(time_data, diameter_data, fit_result, 'output.png')")
