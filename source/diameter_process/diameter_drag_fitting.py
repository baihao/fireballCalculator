#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球直径拖曳曲线拟合模块

基于拖曳函数 D(t) = K * (1 - B*exp(-C*t^2)) 对火球直径随时间变化的数据进行参数拟合

拖曳函数参数拟合算法说明：
==========================

1. 拖曳函数理论基础：
   D(t) = K * (1 - B*exp(-C*t^2))
   
   其中：
   - K: 火球最大直径（渐近值）
   - B: 初始拖曳系数（通常接近1）
   - C: 拖曳衰减系数（控制增长速度）
   - t: 时间

2. 物理意义：
   - K 表示火球在无限时间后达到的最大直径
   - B 控制初始时刻的直径（t=0时，D(0) = K*(1-B)）
   - C 控制火球扩张的速度，C越大扩张越快

3. 拟合算法原理：
   a) 初始参数估计：
      - K_init: 使用数据最大值的1.1-1.2倍作为初始估计
      - B_init: 通常设为0.95-0.99（接近1）
      - C_init: 根据数据的增长速度估计
   
   b) 非线性最小二乘拟合：
      - 使用Levenberg-Marquardt算法
      - 最小化目标函数：Σ(D_observed - D_model)²
      - 考虑参数约束：K>0, 0<B<1, C>0
   
   c) 拟合质量评估：
      - R²决定系数
      - 均方根误差(RMSE)
      - 参数置信区间

4. 算法特点：
   - 鲁棒性：对噪声数据有较好的容忍性
   - 物理约束：确保参数符合物理意义
   - 收敛性：通过合理的初始值估计提高收敛概率
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from typing import List, Tuple, Dict, Optional, Any
import warnings


class DiameterDragFitter:
    """火球直径拖曳曲线拟合器"""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-8):
        """
        初始化拟合器
        
        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_results = {}
        
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
    
    def estimate_initial_parameters(self, time_data: np.ndarray, diameter_data: np.ndarray) -> Tuple[float, float, float]:
        """
        估计拟合的初始参数
        
        Args:
            time_data: 时间数据
            diameter_data: 直径数据
            
        Returns:
            Tuple[float, float, float]: (K_init, B_init, C_init)
        """
        try:
            # 1. 估计K（最大直径）
            # 使用数据最大值的1.15倍作为初始估计
            max_diameter = np.max(diameter_data)
            K_init = max_diameter * 1.15
            
            # 2. 估计B（初始拖曳系数）
            # 基于初始直径估计：D(0) = K*(1-B)
            initial_diameter = diameter_data[0] if len(diameter_data) > 0 else 0
            if K_init > 0:
                B_init = max(0.1, min(0.99, 1 - initial_diameter / K_init))
            else:
                B_init = 0.95
            
            # 3. 估计C（拖曳衰减系数）
            # 基于数据增长速度估计
            if len(time_data) > 1 and len(diameter_data) > 1:
                # 计算平均增长率
                time_span = time_data[-1] - time_data[0]
                diameter_change = diameter_data[-1] - diameter_data[0]
                
                if time_span > 0 and diameter_change > 0:
                    # 基于50%增长时间估计C
                    half_growth = diameter_data[0] + diameter_change * 0.5
                    
                    # 找到接近50%增长的时间点
                    half_time_idx = np.argmin(np.abs(diameter_data - half_growth))
                    half_time = time_data[half_time_idx]
                    
                    if half_time > 0:
                        # 根据拖曳函数特性估计C
                        C_init = 2.0 / (half_time**2)
                    else:
                        C_init = 0.1
                else:
                    C_init = 0.1
            else:
                C_init = 0.1
            
            # 确保参数在合理范围内
            K_init = max(max_diameter, K_init)
            B_init = max(0.1, min(0.99, B_init))
            C_init = max(0.01, min(10.0, C_init))
            
            print(f"初始参数估计: K={K_init:.3f}, B={B_init:.3f}, C={C_init:.3f}")
            
            return K_init, B_init, C_init
            
        except Exception as e:
            print(f"⚠️ 参数估计失败，使用默认值: {e}")
            # 默认参数
            max_val = np.max(diameter_data) if len(diameter_data) > 0 else 1.0
            return max_val * 1.2, 0.95, 0.1
    
    def fit_drag_curve(self, time_data: List[float], diameter_data: List[float], 
                       use_robust_fitting: bool = True) -> Dict[str, Any]:
        """
        拟合拖曳曲线参数
        
        Args:
            time_data: 时间数据列表（秒）
            diameter_data: 直径数据列表（米）
            use_robust_fitting: 是否使用鲁棒拟合（全局优化）
            
        Returns:
            Dict[str, Any]: 拟合结果字典，包含参数K、B、C和质量评估
        """
        try:
            # 数据验证
            if len(time_data) != len(diameter_data):
                raise ValueError("时间和直径数据长度不匹配")
            
            if len(time_data) < 4:
                raise ValueError("数据点太少，至少需要4个数据点进行拟合")
            
            # 转换为numpy数组
            t = np.array(time_data, dtype=float)
            D = np.array(diameter_data, dtype=float)
            
            # 数据预处理
            t, D = self._preprocess_data(t, D)
            
            print(f"开始拟合拖曳曲线：{len(t)} 个数据点")
            print(f"时间范围: {t[0]:.3f} - {t[-1]:.3f} 秒")
            print(f"直径范围: {D[0]:.3f} - {D[-1]:.3f} 米")
            
            # 估计初始参数
            K_init, B_init, C_init = self.estimate_initial_parameters(t, D)
            
            if use_robust_fitting:
                # 使用全局优化（更鲁棒但较慢）
                fit_result = self._robust_fit(t, D, K_init, B_init, C_init)
            else:
                # 使用标准非线性最小二乘（更快但可能陷入局部最优）
                fit_result = self._standard_fit(t, D, K_init, B_init, C_init)
            
            # 计算拟合质量
            fit_result.update(self._evaluate_fit_quality(t, D, fit_result))
            
            # 保存拟合结果
            self.fit_results = fit_result
            
            return fit_result
            
        except Exception as e:
            print(f"❌ 拖曳曲线拟合失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'K': 0.0, 'B': 0.0, 'C': 0.0,
                'r_squared': 0.0, 'rmse': float('inf')
            }
    
    def _preprocess_data(self, t: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理：去除异常值，确保时间递增
        
        Args:
            t: 时间数据
            D: 直径数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 预处理后的(时间, 直径)数据
        """
        try:
            # 1. 去除NaN和无穷值
            valid_mask = np.isfinite(t) & np.isfinite(D) & (t >= 0) & (D > 0)
            t_clean = t[valid_mask]
            D_clean = D[valid_mask]
            
            if len(t_clean) == 0:
                raise ValueError("没有有效的数据点")
            
            # 2. 按时间排序
            sort_indices = np.argsort(t_clean)
            t_sorted = t_clean[sort_indices]
            D_sorted = D_clean[sort_indices]
            
            # 3. 去除重复的时间点（保留直径较大的）
            unique_times, unique_indices = np.unique(t_sorted, return_index=True)
            t_unique = unique_times
            D_unique = D_sorted[unique_indices]
            
            # 4. 简单的异常值检测（基于四分位距）
            Q1, Q3 = np.percentile(D_unique, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 保留合理范围内的数据
            outlier_mask = (D_unique >= lower_bound) & (D_unique <= upper_bound)
            t_final = t_unique[outlier_mask]
            D_final = D_unique[outlier_mask]
            
            removed_count = len(t) - len(t_final)
            if removed_count > 0:
                print(f"数据预处理: 移除了 {removed_count} 个异常点")
            
            return t_final, D_final
            
        except Exception as e:
            print(f"⚠️ 数据预处理失败: {e}")
            return t, D
    
    def _standard_fit(self, t: np.ndarray, D: np.ndarray, 
                     K_init: float, B_init: float, C_init: float) -> Dict[str, Any]:
        """
        标准非线性最小二乘拟合
        
        Args:
            t: 时间数据
            D: 直径数据
            K_init, B_init, C_init: 初始参数
            
        Returns:
            Dict[str, Any]: 拟合结果
        """
        try:
            # 设置参数边界
            lower_bounds = [np.max(D), 0.01, 0.001]  # K >= max(D), B >= 0.01, C >= 0.001
            upper_bounds = [np.max(D) * 3, 0.999, 50.0]  # K <= 3*max(D), B <= 0.999, C <= 50
            
            # 执行curve_fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                popt, pcov = curve_fit(
                    self.drag_function,
                    t, D,
                    p0=[K_init, B_init, C_init],
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=self.max_iterations,
                    ftol=self.tolerance,
                    xtol=self.tolerance
                )
            
            K_fit, B_fit, C_fit = popt
            
            # 计算参数不确定性
            param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [0, 0, 0]
            
            print(f"标准拟合结果: K={K_fit:.4f}±{param_errors[0]:.4f}, "
                  f"B={B_fit:.4f}±{param_errors[1]:.4f}, C={C_fit:.4f}±{param_errors[2]:.4f}")
            
            return {
                'success': True,
                'method': 'standard_lsq',
                'K': float(K_fit),
                'B': float(B_fit), 
                'C': float(C_fit),
                'K_error': float(param_errors[0]),
                'B_error': float(param_errors[1]),
                'C_error': float(param_errors[2]),
                'covariance_matrix': pcov.tolist() if pcov is not None else None
            }
            
        except Exception as e:
            print(f"⚠️ 标准拟合失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _robust_fit(self, t: np.ndarray, D: np.ndarray,
                   K_init: float, B_init: float, C_init: float) -> Dict[str, Any]:
        """
        鲁棒全局优化拟合
        
        Args:
            t: 时间数据
            D: 直径数据
            K_init, B_init, C_init: 初始参数
            
        Returns:
            Dict[str, Any]: 拟合结果
        """
        try:
            # 定义目标函数
            def objective(params):
                K, B, C = params
                try:
                    predicted = self.drag_function(t, K, B, C)
                    residuals = D - predicted
                    return np.sum(residuals**2)
                except:
                    return 1e10  # 返回大值表示拟合失败
            
            # 设置参数边界
            max_D = np.max(D)
            bounds = [
                (max_D, max_D * 3),      # K: 最大直径到3倍最大直径
                (0.01, 0.999),           # B: 0.01到0.999
                (0.001, 50.0)            # C: 0.001到50
            ]
            
            # 使用差分进化算法进行全局优化
            result = differential_evolution(
                objective,
                bounds,
                seed=42,  # 固定随机种子确保可重现性
                maxiter=500,
                tol=self.tolerance,
                atol=self.tolerance,
                polish=True  # 使用局部优化进行最终精化
            )
            
            if result.success:
                K_fit, B_fit, C_fit = result.x
                
                print(f"鲁棒拟合结果: K={K_fit:.4f}, B={B_fit:.4f}, C={C_fit:.4f}")
                print(f"优化收敛: {result.success}, 迭代次数: {result.nit}")
                
                return {
                    'success': True,
                    'method': 'robust_global',
                    'K': float(K_fit),
                    'B': float(B_fit),
                    'C': float(C_fit),
                    'optimization_result': {
                        'converged': result.success,
                        'iterations': result.nit,
                        'final_cost': result.fun
                    }
                }
            else:
                print(f"⚠️ 鲁棒拟合未收敛: {result.message}")
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            print(f"⚠️ 鲁棒拟合失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_fit_quality(self, t: np.ndarray, D: np.ndarray, 
                             fit_result: Dict[str, Any]) -> Dict[str, float]:
        """
        评估拟合质量
        
        Args:
            t: 时间数据
            D: 观测直径数据
            fit_result: 拟合结果
            
        Returns:
            Dict[str, float]: 质量评估指标
        """
        try:
            if not fit_result.get('success', False):
                return {'r_squared': 0.0, 'rmse': float('inf'), 'mae': float('inf')}
            
            K, B, C = fit_result['K'], fit_result['B'], fit_result['C']
            
            # 计算预测值
            D_pred = self.drag_function(t, K, B, C)
            
            # 计算R²决定系数
            ss_res = np.sum((D - D_pred)**2)  # 残差平方和
            ss_tot = np.sum((D - np.mean(D))**2)  # 总平方和
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 计算均方根误差 (RMSE)
            rmse = np.sqrt(np.mean((D - D_pred)**2))
            
            # 计算平均绝对误差 (MAE)
            mae = np.mean(np.abs(D - D_pred))
            
            # 计算最大相对误差
            relative_errors = np.abs((D - D_pred) / D)
            max_relative_error = np.max(relative_errors) * 100  # 百分比
            
            print(f"拟合质量: R²={r_squared:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            print(f"最大相对误差: {max_relative_error:.2f}%")
            
            return {
                'r_squared': float(r_squared),
                'rmse': float(rmse),
                'mae': float(mae),
                'max_relative_error': float(max_relative_error)
            }
            
        except Exception as e:
            print(f"⚠️ 质量评估失败: {e}")
            return {'r_squared': 0.0, 'rmse': float('inf'), 'mae': float('inf')}
    
    def predict_diameter(self, time_points: List[float], K: float, B: float, C: float) -> np.ndarray:
        """
        使用拟合参数预测直径
        
        Args:
            time_points: 预测时间点
            K, B, C: 拟合参数
            
        Returns:
            np.ndarray: 预测的直径值
        """
        t = np.array(time_points)
        return self.drag_function(t, K, B, C)
    
    def plot_fit_results(self, time_data: List[float], diameter_data: List[float],
                        fit_result: Dict[str, Any], save_path: Optional[str] = None,
                        show_confidence_bands: bool = True) -> bool:
        """
        绘制拟合结果图
        
        Args:
            time_data: 原始时间数据
            diameter_data: 原始直径数据
            fit_result: 拟合结果
            save_path: 保存路径（可选）
            show_confidence_bands: 是否显示置信带
            
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
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 主图：拟合曲线
            ax1.scatter(t, D, color='red', alpha=0.7, s=50, label='观测数据', zorder=3)
            
            # 生成平滑的拟合曲线
            t_smooth = np.linspace(t[0], t[-1], 200)
            D_smooth = self.drag_function(t_smooth, K, B, C)
            ax1.plot(t_smooth, D_smooth, 'b-', linewidth=2, label='拟合曲线', zorder=2)
            
            # 添加参数信息
            param_text = f'K = {K:.3f} m\nB = {B:.3f}\nC = {C:.3f} s⁻²'
            quality_text = f'R² = {fit_result.get("r_squared", 0):.4f}\nRMSE = {fit_result.get("rmse", 0):.4f} m'
            ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            ax1.text(0.05, 0.75, quality_text, transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('火球直径 (m)')
            ax1.set_title('火球直径拖曳曲线拟合结果')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 残差图
            D_pred = self.drag_function(t, K, B, C)
            residuals = D - D_pred
            
            ax2.scatter(t, residuals, color='green', alpha=0.7, s=30)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('时间 (s)')
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
    
    def get_fit_summary(self) -> Dict[str, Any]:
        """
        获取拟合结果摘要
        
        Returns:
            Dict[str, Any]: 拟合摘要信息
        """
        if not self.fit_results:
            return {"status": "未进行拟合"}
        
        if not self.fit_results.get('success', False):
            return {
                "status": "拟合失败",
                "error": self.fit_results.get('error', '未知错误')
            }
        
        return {
            "status": "拟合成功",
            "parameters": {
                "K": self.fit_results['K'],
                "B": self.fit_results['B'],
                "C": self.fit_results['C']
            },
            "quality": {
                "r_squared": self.fit_results.get('r_squared', 0),
                "rmse": self.fit_results.get('rmse', 0),
                "mae": self.fit_results.get('mae', 0)
            },
            "method": self.fit_results.get('method', 'unknown')
        }


def create_diameter_drag_fitter(max_iterations: int = 1000, tolerance: float = 1e-8) -> DiameterDragFitter:
    """
    创建直径拖曳曲线拟合器的便捷函数
    
    Args:
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        
    Returns:
        DiameterDragFitter: 拟合器实例
    """
    return DiameterDragFitter(max_iterations, tolerance)


def fit_diameter_drag_curve(time_data: List[float], diameter_data: List[float], 
                           use_robust: bool = True) -> Dict[str, Any]:
    """
    便捷函数：拟合直径拖曳曲线
    
    Args:
        time_data: 时间数据列表（秒）
        diameter_data: 直径数据列表（米）
        use_robust: 是否使用鲁棒拟合
        
    Returns:
        Dict[str, Any]: 拟合结果
    """
    fitter = create_diameter_drag_fitter()
    return fitter.fit_drag_curve(time_data, diameter_data, use_robust)


if __name__ == "__main__":
    # 示例用法和测试
    print("火球直径拖曳曲线拟合模块")
    print("=" * 50)
    
    # 创建测试数据
    print("1. 创建测试数据...")
    t_test = np.linspace(0, 0.08, 20)  # 0到80ms，20个点
    K_true, B_true, C_true = 12.0, 0.90, 15.0  # 真实参数
    D_true = DiameterDragFitter.drag_function(t_test, K_true, B_true, C_true)
    
    # 添加适当的噪声
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(D_true))
    D_noisy = D_true + noise
    
    # 确保直径为正值
    D_noisy = np.maximum(D_noisy, 0.1)
    
    print(f"   真实参数: K={K_true}, B={B_true}, C={C_true}")
    print(f"   数据点数: {len(t_test)}")
    
    # 执行拟合
    print("\n2. 执行拟合...")
    fitter = create_diameter_drag_fitter()
    
    # 标准拟合
    result_std = fitter.fit_drag_curve(t_test.tolist(), D_noisy.tolist(), use_robust_fitting=False)
    
    # 鲁棒拟合
    result_robust = fitter.fit_drag_curve(t_test.tolist(), D_noisy.tolist(), use_robust_fitting=True)
    
    # 比较结果
    print("\n3. 结果比较...")
    if result_std.get('success'):
        print(f"标准拟合: K={result_std['K']:.3f}, B={result_std['B']:.3f}, C={result_std['C']:.3f}")
        print(f"           R²={result_std.get('r_squared', 0):.4f}")
    
    if result_robust.get('success'):
        print(f"鲁棒拟合: K={result_robust['K']:.3f}, B={result_robust['B']:.3f}, C={result_robust['C']:.3f}")
        print(f"           R²={result_robust.get('r_squared', 0):.4f}")
    
    # 绘制结果
    print("\n4. 绘制拟合结果...")
    if result_robust.get('success'):
        fitter.plot_fit_results(t_test.tolist(), D_noisy.tolist(), result_robust, 
                               "diameter_drag_fit_example.png")
    
    print("\n✅ 测试完成！")
    print("使用方法:")
    print("  from diameter_drag_fitting import fit_diameter_drag_curve")
    print("  result = fit_diameter_drag_curve(time_list, diameter_list)")
    print("  K, B, C = result['K'], result['B'], result['C']")
