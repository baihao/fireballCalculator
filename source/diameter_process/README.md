# 火球直径拖曳曲线拟合模块

## 📖 算法原理

### 拖曳函数理论基础

火球在爆炸过程中的直径变化遵循拖曳函数模型：

```
D(t) = K * (1 - B*exp(-C*t²))
```

#### 参数物理意义

| 参数 | 物理意义 | 典型范围 | 单位 |
|------|----------|----------|------|
| **K** | 火球最大直径（渐近值） | 5-50 | 米 |
| **B** | 初始拖曳系数 | 0.8-0.99 | 无量纲 |
| **C** | 拖曳衰减系数 | 1-100 | s⁻² |

#### 函数特性

- **t=0时**: D(0) = K*(1-B) ≈ 初始直径
- **t→∞时**: D(∞) = K = 最大直径
- **增长速度**: 由参数C控制，C越大增长越快

## 🔬 拟合算法

### 1. 初始参数估计

```python
# K估计：基于数据最大值
K_init = max(diameter_data) * 1.15

# B估计：基于初始直径
B_init = 1 - initial_diameter / K_init

# C估计：基于50%增长时间
C_init = 2.0 / (half_growth_time)²
```

### 2. 非线性拟合方法

#### 标准最小二乘法
- 使用Levenberg-Marquardt算法
- 适用于噪声较小的数据
- 计算速度快

#### 鲁棒全局优化
- 使用差分进化算法
- 适用于噪声较大或复杂的数据
- 避免局部最优解

### 3. 质量评估指标

- **R²决定系数**: 拟合优度，接近1表示拟合良好
- **RMSE**: 均方根误差，越小越好
- **MAE**: 平均绝对误差
- **最大相对误差**: 最大的相对偏差百分比

## 🛠️ 使用方法

### 基本用法

```python
from diameter_drag_fitting import fit_diameter_drag_curve

# 准备数据
time_data = [0.0, 0.01, 0.02, 0.03, ...]  # 时间（秒）
diameter_data = [1.2, 2.5, 4.1, 5.8, ...]  # 直径（米）

# 执行拟合
result = fit_diameter_drag_curve(time_data, diameter_data)

# 获取参数
if result['success']:
    K = result['K']  # 最大直径
    B = result['B']  # 初始拖曳系数
    C = result['C']  # 拖曳衰减系数
    
    print(f"拟合参数: K={K:.3f}, B={B:.3f}, C={C:.3f}")
    print(f"拟合质量: R²={result['r_squared']:.4f}")
```

### 高级用法

```python
from diameter_drag_fitting import create_diameter_drag_fitter

# 创建拟合器
fitter = create_diameter_drag_fitter(max_iterations=2000, tolerance=1e-10)

# 执行拟合
result = fitter.fit_drag_curve(time_data, diameter_data, use_robust_fitting=True)

# 预测新的时间点
new_times = [0.05, 0.06, 0.07]
predicted_diameters = fitter.predict_diameter(new_times, result['K'], result['B'], result['C'])

# 绘制拟合结果
fitter.plot_fit_results(time_data, diameter_data, result, "fit_result.png")

# 获取详细摘要
summary = fitter.get_fit_summary()
```

## 📊 输出结果格式

### 成功拟合

```python
{
    'success': True,
    'method': 'robust_global',  # 或 'standard_lsq'
    'K': 12.345,               # 最大直径
    'B': 0.895,                # 初始拖曳系数
    'C': 15.678,               # 拖曳衰减系数
    'r_squared': 0.9850,       # R²决定系数
    'rmse': 0.0234,            # 均方根误差
    'mae': 0.0187,             # 平均绝对误差
    'max_relative_error': 5.23  # 最大相对误差(%)
}
```

### 拟合失败

```python
{
    'success': False,
    'error': '错误描述',
    'K': 0.0, 'B': 0.0, 'C': 0.0,
    'r_squared': 0.0,
    'rmse': float('inf')
}
```

## 🎯 算法特点

### 优势
- **物理约束**: 确保参数符合物理意义
- **鲁棒性**: 对噪声数据有良好的容忍性
- **多种方法**: 提供标准和鲁棒两种拟合算法
- **质量评估**: 完整的拟合质量评估指标

### 适用场景
- 火球爆炸直径时间序列分析
- 爆炸物理参数反演
- 火球扩张动力学研究
- 实验数据拟合和预测

## 🔧 参数调优建议

### 数据质量要求
- **最少数据点**: 4个（建议≥10个）
- **时间范围**: 覆盖火球主要扩张阶段
- **数据精度**: 直径测量误差<10%

### 拟合参数选择
- **标准拟合**: 数据质量好，噪声小
- **鲁棒拟合**: 数据噪声大，或需要避免局部最优

### 结果验证
- **R² > 0.9**: 拟合质量良好
- **RMSE < 5%**: 相对误差可接受
- **参数合理性**: K>max(D), 0.5<B<0.99, C>0

## 📚 相关文档

- [拖曳模型理论](../../papers/指数拖拽模型文章.pdf)
- [火球扩张动力学](../../papers/fireballexpansion.pdf)
- [实验数据对比](../../papers/TNT和温压炸药的爆炸火球表面温度对比试验研究.pdf)

## 🧪 测试示例

运行测试：
```bash
cd source/diameter_process
python diameter_drag_fitting.py
```

预期输出：
- 拟合参数和质量评估
- 拟合结果图 `diameter_drag_fit_example.png`
- R²通常>0.9表示拟合成功
