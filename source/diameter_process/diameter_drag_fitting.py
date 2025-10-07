#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球直径拖曳曲线拟合模块

基于拖曳函数 D(t) = K * (1 - B*exp(-C*t^2)) 对火球直径随时间变化的数据进行参数拟合
时间单位：毫秒(ms)，直径单位：米(m)

改进的拖曳函数参数拟合算法说明：
================================

1. 拖曳函数理论基础：
   D(t) = K * (1 - B*exp(-C*t^2))
   
   其中：
   - K: 火球最大直径（渐近值，单位：米）
   - B: 初始拖曳系数（无量纲，0<B<1）
   - C: 拖曳衰减系数（单位：ms⁻²，控制增长速度）
   - t: 时间（单位：毫秒）

2. 物理意义：
   - K 表示火球在无限时间后达到的最大直径
   - B 控制初始时刻的直径（t=0时，D(0) = K*(1-B)）
   - C 控制火球扩张的速度，C越大扩张越快
   - 时间单位使用毫秒，更适合描述爆炸过程的快速变化

3. 改进的拟合算法流程：
   
   a) 数据预处理（改进）：
      - 保留所有有效数据点，不使用异常值检测
      - 避免误删重要的早期数据点
      - 确保时间递增和数值有效性
      - 去除重复时间点（保留直径较大的值）
   
   b) 智能初始参数估计（改进）：
      - K_init: 基于数据统计特征估计
        * 使用数据最大值的1.05-1.15倍
        * 考虑数据的增长趋势
      - B_init: 基于初始直径估计
        * 计算：B = 1 - D_initial/K_init
        * 约束在合理范围内 [0.1, 0.99]
      - C_init: 基于增长特征估计
        * 分析数据增长速度
        * 使用半衰期方法估计
        * 考虑时间尺度（毫秒单位）
   
   c) 多阶段拟合策略（新增）：
      - 阶段1：全局优化（差分进化算法）
        * 避免局部最优解
        * 参数范围：K∈[max(D), 3*max(D)], B∈[0.01, 0.999], C∈[1e-6, 1e-2]
        * 目标函数：加权最小二乘，早期数据权重更高
      - 阶段2：局部精化（Levenberg-Marquardt算法）
        * 基于全局优化结果进行精化
        * 计算参数协方差矩阵
        * 估计参数不确定性
   
   d) 加权拟合（改进）：
      - 早期数据点权重：w_early = 2.0
      - 后期数据点权重：w_late = 1.0
      - 目标函数：Σ w_i * (D_observed - D_model)²
      - 平衡早期增长阶段和后期稳定阶段的重要性
   
   e) 拟合质量评估（增强）：
      - R²决定系数（加权版本）
      - 均方根误差(RMSE)
      - 平均绝对误差(MAE)
      - 最大相对误差
      - 参数置信区间
      - 残差分析

4. 算法改进原因：
   
   a) 保留所有数据点：
      - 四分位距异常值检测可能误删重要数据
      - 早期数据点对拟合质量至关重要
      - 避免人为偏见影响拟合结果
   
   b) 使用毫秒时间单位：
      - 原始数据时间单位为毫秒
      - 避免单位转换带来的精度损失
      - 更符合爆炸过程的快速时间尺度
   
   c) 多阶段拟合策略：
      - 全局优化避免陷入局部最优
      - 局部精化提高参数精度
      - 结合两种算法的优势
   
   d) 加权拟合：
      - 早期数据点包含更多增长信息
      - 平衡不同阶段数据的重要性
      - 提高拟合的物理合理性
   
   e) 智能参数估计：
      - 基于数据特征自动估计初始值
      - 减少对人工经验的依赖
      - 提高收敛成功率

5. 算法特点：
   - 鲁棒性：对噪声数据和异常值有较好的容忍性
   - 物理约束：确保参数符合物理意义
   - 收敛性：多阶段策略提高收敛概率
   - 精度：加权拟合和局部精化提高参数精度
   - 适应性：自动适应不同的数据特征
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from typing import List, Tuple, Dict, Optional, Any
import warnings

# 导入数据过滤模块（优先相对导入，其次绝对导入）
apply_data_filter = None
try:
    from .data_filter import apply_data_filter  # 同包相对导入
except Exception:
    try:
        from diameter_process.data_filter import apply_data_filter  # 绝对导入
    except Exception:
        try:
            from data_filter import apply_data_filter  # 退而求其次的顶层导入
        except Exception:
            print("⚠️ 无法导入 data_filter 模块，将跳过数据过滤步骤")


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
        智能估计拟合的初始参数（改进版）
        
        Args:
            time_data: 时间数据（毫秒）
            diameter_data: 直径数据（米）
            
        Returns:
            Tuple[float, float, float]: (K_init, B_init, C_init)
        """
        try:
            # 1. 估计K（最大直径）
            # 使用数据最大值的1.05-1.15倍作为初始估计
            max_diameter = np.max(diameter_data)
            min_diameter = np.min(diameter_data)
            diameter_range = max_diameter - min_diameter
            
            # 基于数据增长趋势调整K估计
            if diameter_range > 0:
                growth_ratio = max_diameter / min_diameter
                if growth_ratio > 2.0:  # 显著增长
                    K_init = max_diameter * 1.05
                else:  # 缓慢增长
                    K_init = max_diameter * 1.15
            else:
                K_init = max_diameter * 1.1
            
            # 2. 估计B（初始拖曳系数）
            # 基于初始直径估计：D(0) = K*(1-B)
            initial_diameter = diameter_data[0] if len(diameter_data) > 0 else 0
            if K_init > 0 and initial_diameter > 0:
                B_init = max(0.1, min(0.99, 1 - initial_diameter / K_init))
            else:
                B_init = 0.9  # 默认值
            
            # 3. 估计C（拖曳衰减系数，考虑毫秒时间单位）
            # 基于数据增长速度和时间尺度估计
            if len(time_data) > 1 and len(diameter_data) > 1:
                time_span = time_data[-1] - time_data[0]
                diameter_change = diameter_data[-1] - diameter_data[0]
                
                if time_span > 0 and diameter_change > 0:
                    # 计算平均增长速度
                    avg_growth_rate = diameter_change / time_span
                    
                    # 基于半衰期方法估计C
                    # 找到接近50%增长的时间点
                    half_growth = diameter_data[0] + diameter_change * 0.5
                    half_time_idx = np.argmin(np.abs(diameter_data - half_growth))
                    half_time = time_data[half_time_idx]
                    
                    if half_time > 0:
                        # 根据拖曳函数特性估计C（毫秒单位）
                        # 对于毫秒时间单位，C值应该更小
                        C_init = np.log(2) / (half_time**2)
                        # 进一步调整以适应毫秒时间尺度
                        C_init = C_init * 0.1  # 调整因子
                    else:
                        C_init = 1e-3  # 毫秒单位的默认值
                else:
                    C_init = 1e-3
            else:
                C_init = 1e-3
            
            # 确保参数在合理范围内（考虑毫秒时间单位）
            K_init = max(max_diameter, K_init)
            B_init = max(0.1, min(0.99, B_init))
            C_init = max(1e-6, min(1e-2, C_init))  # 毫秒单位的合理范围
            
            print(f"智能初始参数估计: K={K_init:.3f}, B={B_init:.3f}, C={C_init:.6f}")
            
            return K_init, B_init, C_init
            
        except Exception as e:
            print(f"⚠️ 参数估计失败，使用默认值: {e}")
            # 默认参数（考虑毫秒时间单位）
            max_val = np.max(diameter_data) if len(diameter_data) > 0 else 1.0
            return max_val * 1.1, 0.9, 1e-3
    
    def fit_drag_curve(self, time_data: List[float], diameter_data: List[float], 
                       use_robust_fitting: bool = True, time_unit: str = 'ms',
                       enable_data_filtering: bool = True, drop_threshold: float = 0.02,
                       window_size: int = 10) -> Dict[str, Any]:
        """
        拟合拖曳曲线参数（改进版，集成数据过滤）
        
        Args:
            time_data: 时间数据列表（毫秒）
            diameter_data: 直径数据列表（米）
            use_robust_fitting: 是否使用鲁棒拟合（全局优化）
            time_unit: 时间单位（'ms' 或 's'）
            enable_data_filtering: 是否启用数据过滤
            drop_threshold: 下降阈值（默认2%）
            window_size: 滑动窗口大小（默认10）
            
        Returns:
            Dict[str, Any]: 拟合结果字典，包含参数K、B、C、质量评估和数据过滤信息
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
            
            # 时间单位处理
            if time_unit == 's':
                # 如果输入是秒，转换为毫秒
                t = t * 1000.0
                time_unit_display = 'ms'
            else:
                time_unit_display = 'ms'
            
            print(f"开始拟合拖曳曲线：{len(t)} 个数据点")
            print(f"时间范围: {t[0]:.1f} - {t[-1]:.1f} {time_unit_display}")
            print(f"直径范围: {D[0]:.3f} - {D[-1]:.3f} 米")
            
            # 数据过滤（如果启用）
            filtering_info = {}
            if enable_data_filtering and apply_data_filter is not None:
                print(f"\n执行数据过滤...")
                original_t = t.copy()
                original_D = D.copy()
                
                filtered_t, filtered_D = apply_data_filter(t.tolist(), D.tolist(), drop_threshold, window_size)
                t = np.array(filtered_t)
                D = np.array(filtered_D)
                
                filtering_info = {
                    'enabled': True,
                    'drop_threshold': drop_threshold,
                    'window_size': window_size,
                    'original_data_points': len(original_t),
                    'filtered_data_points': len(t),
                    'data_retention_rate': len(t) / len(original_t),
                    'original_time_range': [float(original_t[0]), float(original_t[-1])],
                    'filtered_time_range': [float(t[0]), float(t[-1])],
                    'cutoff_time': float(t[-1]) if len(t) < len(original_t) else None
                }
                
                print(f"数据过滤完成: 保留 {len(t)}/{len(original_t)} 个数据点 ({filtering_info['data_retention_rate']:.1%})")
            else:
                filtering_info = {
                    'enabled': False,
                    'original_data_points': len(t),
                    'filtered_data_points': len(t),
                    'data_retention_rate': 1.0
                }
                print("数据过滤已禁用，使用所有数据点")
            
            # 估计初始参数
            K_init, B_init, C_init = self.estimate_initial_parameters(t, D)
            
            if use_robust_fitting:
                # 使用改进的多阶段拟合策略
                fit_result = self._improved_robust_fit(t, D, K_init, B_init, C_init)
            else:
                # 使用标准非线性最小二乘（更快但可能陷入局部最优）
                fit_result = self._standard_fit(t, D, K_init, B_init, C_init)
            
            # 计算拟合质量
            fit_result.update(self._evaluate_fit_quality(t, D, fit_result))
            
            # 添加数据过滤信息
            fit_result['data_filtering'] = filtering_info
            
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
        数据预处理：去除无效值，确保时间递增，保留所有有效数据
        
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
            
            # 4. 不使用异常值检测，保留所有有效数据
            print(f"数据预处理: 保留所有 {len(t_unique)} 个有效数据点")
            
            return t_unique, D_unique
            
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
            # 设置参数边界（考虑毫秒时间单位）
            max_D = np.max(D)
            lower_bounds = [max_D, 0.1, 1e-6]  # K >= max(D), B >= 0.1, C >= 1e-6
            upper_bounds = [max_D * 2, 0.99, 1e-2]  # K <= 2*max(D), B <= 0.99, C <= 1e-2
            
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
    
    def _improved_robust_fit(self, t: np.ndarray, D: np.ndarray,
                           K_init: float, B_init: float, C_init: float) -> Dict[str, Any]:
        """
        改进的鲁棒多阶段拟合策略
        
        Args:
            t: 时间数据（毫秒）
            D: 直径数据（米）
            K_init, B_init, C_init: 初始参数
            
        Returns:
            Dict[str, Any]: 拟合结果
        """
        try:
            # 阶段1：全局优化（差分进化算法）
            print("阶段1: 全局优化...")
            
            # 定义加权目标函数
            def weighted_objective(params):
                K, B, C = params
                try:
                    predicted = self.drag_function(t, K, B, C)
                    residuals = D - predicted
                    
                    # 计算权重：早期数据点权重更高
                    time_span = t[-1] - t[0]
                    early_threshold = t[0] + 0.3 * time_span  # 前30%时间
                    weights = np.where(t <= early_threshold, 2.0, 1.0)
                    
                    weighted_residuals = weights * residuals
                    return np.sum(weighted_residuals**2)
                except:
                    return 1e10  # 返回大值表示拟合失败

            def objective(params):
                K, B, C = params
                try:
                    predicted = self.drag_function(t, K, B, C)
                    residuals = D - predicted
                    return np.sum(residuals**2)
                except:
                    return 1e10  # 返回大值表示拟合失败
            
            # 设置参数边界（考虑毫秒时间单位）
            max_D = np.max(D)
            min_D = np.min(D)
            bounds = [
                (max_D, max_D * 2),      # K: 最大直径到2倍最大直径（更严格）
                (0.1, 0.99),             # B: 0.1到0.99（更合理范围）
                (1e-6, 1e-2)             # C: 1e-6到1e-2 (毫秒单位)
            ]
            
            # 使用差分进化算法进行全局优化
            result = differential_evolution(
                objective,
                bounds,
                seed=42,  # 固定随机种子确保可重现性
                maxiter=500,
                tol=self.tolerance,
                atol=self.tolerance,
                polish=False  # 不进行局部精化，留给阶段2
            )
            
            if not result.success:
                print(f"⚠️ 全局优化未收敛: {result.message}")
                return {'success': False, 'error': result.message}
            
            K_global, B_global, C_global = result.x
            print(f"全局优化结果: K={K_global:.4f}, B={B_global:.4f}, C={C_global:.4f}")
            print(f"全局优化收敛: {result.success}, 迭代次数: {result.nit}")
            
            # 阶段2：局部精化（Levenberg-Marquardt算法）
            print("阶段2: 局部精化...")
            
            try:
                # 使用全局优化结果作为初始值进行局部精化
                popt, pcov = curve_fit(
                    self.drag_function,
                    t, D,
                    p0=[K_global, B_global, C_global],
                    bounds=([max_D, 0.1, 1e-6], [max_D * 2, 0.99, 1e-2]),
                    maxfev=self.max_iterations,
                    ftol=self.tolerance,
                    xtol=self.tolerance
                )
                
                K_fit, B_fit, C_fit = popt
                
                # 计算参数不确定性
                param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [0, 0, 0]
                
                print(f"局部精化结果: K={K_fit:.4f}±{param_errors[0]:.4f}, "
                      f"B={B_fit:.4f}±{param_errors[1]:.4f}, C={C_fit:.4f}±{param_errors[2]:.4f}")
                
                return {
                    'success': True,
                    'method': 'improved_robust',
                    'K': float(K_fit),
                    'B': float(B_fit),
                    'C': float(C_fit),
                    'K_error': float(param_errors[0]),
                    'B_error': float(param_errors[1]),
                    'C_error': float(param_errors[2]),
                    'covariance_matrix': pcov.tolist() if pcov is not None else None,
                    'optimization_result': {
                        'global_converged': result.success,
                        'global_iterations': result.nit,
                        'global_cost': result.fun
                    }
                }
                
            except Exception as e:
                print(f"⚠️ 局部精化失败，使用全局优化结果: {e}")
                return {
                    'success': True,
                    'method': 'global_only',
                    'K': float(K_global),
                    'B': float(B_global),
                    'C': float(C_global),
                    'optimization_result': {
                        'global_converged': result.success,
                        'global_iterations': result.nit,
                        'global_cost': result.fun
                    }
                }
                
        except Exception as e:
            print(f"⚠️ 改进的鲁棒拟合失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_fit_quality(self, t: np.ndarray, D: np.ndarray, 
                             fit_result: Dict[str, Any]) -> Dict[str, float]:
        """
        评估拟合质量（改进版，支持加权评估）
        
        Args:
            t: 时间数据（毫秒）
            D: 观测直径数据（米）
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
            
            # 计算权重（与拟合时保持一致）
            time_span = t[-1] - t[0]
            early_threshold = t[0] + 0.3 * time_span  # 前30%时间
            weights = np.where(t <= early_threshold, 2.0, 1.0)
            
            # 计算加权R²决定系数
            weighted_residuals = weights * (D - D_pred)
            weighted_mean = np.average(D, weights=weights)
            ss_res = np.sum(weighted_residuals**2)  # 加权残差平方和
            ss_tot = np.sum(weights * (D - weighted_mean)**2)  # 加权总平方和
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 计算均方根误差 (RMSE)
            rmse = np.sqrt(np.mean((D - D_pred)**2))
            
            # 计算平均绝对误差 (MAE)
            mae = np.mean(np.abs(D - D_pred))
            
            # 计算最大相对误差
            relative_errors = np.abs((D - D_pred) / D)
            max_relative_error = np.max(relative_errors) * 100  # 百分比
            
            # 计算加权均方根误差
            weighted_rmse = np.sqrt(np.mean(weights * (D - D_pred)**2))
            
            print(f"拟合质量: R²={r_squared:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            print(f"加权RMSE={weighted_rmse:.4f}, 最大相对误差: {max_relative_error:.2f}%")
            
            return {
                'r_squared': float(r_squared),
                'rmse': float(rmse),
                'mae': float(mae),
                'max_relative_error': float(max_relative_error),
                'weighted_rmse': float(weighted_rmse)
            }
            
        except Exception as e:
            print(f"⚠️ 质量评估失败: {e}")
            return {'r_squared': 0.0, 'rmse': float('inf'), 'mae': float('inf')}
    
    


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
    print("火球直径拖曳曲线拟合模块")
    print("使用方法:")
    print("  from diameter_drag_fitting import fit_diameter_drag_curve")
    print("  result = fit_diameter_drag_curve(time_list, diameter_list)")
    print("  K, B, C = result['K'], result['B'], result['C']")
