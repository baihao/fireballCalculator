# 火球半径计算器

## 文件说明

- **`fireball_radius_calculator.py`** - 火球半径和直径计算器主程序
- **`test_calculations.py`** - 计算公式验证测试程序
- **`requirements.txt`** - Python依赖包列表

## 快速开始

### 1. 自动环境设置（推荐）

使用提供的设置脚本自动创建虚拟环境：

```bash
# 运行设置脚本
./setup.sh
```

脚本将自动：
- 检查Python环境
- 创建虚拟环境
- 安装所需依赖
- 验证安装

### 2. 手动环境设置

如果您想手动设置环境：

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 激活虚拟环境

每次使用前都需要激活虚拟环境：

```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

激活成功后，命令行前面会显示 `(venv)` 前缀。

### 4. 运行计算器

```bash
python fireball_radius_calculator.py
```

### 5. 运行测试

验证计算公式的正确性：

```bash
python test_calculations.py
```

### 6. 退出虚拟环境

使用完毕后退出虚拟环境：

```bash
deactivate
```

## 功能特性

- 支持五种非爆轰材料：Polyurethane, 30%Al/Rubber, 40%Al/Rubber, 50%Al/Rubber, 60%Al/Rubber
- 计算火球半径和直径随时间的变化
- 计算火球膨胀速率
- 自动生成对比图表
- 支持中文显示
- 基于Table 3的实验数据拟合参数

## 计算公式

### 火球半径
```
R(t) = K * (1 - B*exp(-C*t^2))
```

### 火球直径
```
D(t) = 2 × R(t)
```

### 火球膨胀速率
```
V(t) = dR/dt = K * B * C * 2t * exp(-C*t^2)
```

其中：
- `K`: 火球最终稳定半径 (m)，使用Table 3中的K2参数
- `B`: 拟合系数，固定值0.41
- `C`: 拟合系数，固定值50000 (已从ms单位转换为s单位)
- `t`: 时间 (s)

**注意**: 公式中的时间单位为秒(s)，但为了便于分析，通常显示和分析使用毫秒(ms)单位。

## 材料参数 (Table 3)

| 非爆轰材料 | K2 (m) | B | C | 拟合精度 (%) |
|------------|--------|---|---|-------------|
| Polyurethane | 2.86 | 0.41 | 50000 | 94.99 |
| 30%Al/Rubber | 3.04 | 0.41 | 50000 | 96.44 |
| 40%Al/Rubber | 3.15 | 0.41 | 50000 | 96.70 |
| 50%Al/Rubber | 3.04 | 0.41 | 50000 | 97.95 |
| 60%Al/Rubber | 2.93 | 0.41 | 50000 | 91.77 |

## 输出结果

程序会生成包含以下内容的图表：
1. 不同材料火球半径随时间变化对比
2. 不同材料火球膨胀速率随时间变化对比

## 物理特性分析

### 火球发展过程
1. **初始阶段 (t ≈ 0)**: 火球半径接近0，开始快速膨胀
2. **膨胀阶段**: 火球半径快速增长，膨胀速率达到最大值后开始下降
3. **稳定阶段 (t → ∞)**: 火球半径趋近于K值，膨胀速率趋近于0

### 材料特性
- **40%Al/Rubber**: 最终半径最大(3.15m)，铝粉含量适中，效果最佳
- **30%Al/Rubber & 50%Al/Rubber**: 最终半径中等(3.04m)
- **60%Al/Rubber**: 最终半径较小(2.93m)，铝粉含量过高可能影响效果
- **Polyurethane**: 最终半径最小(2.86m)，膨胀相对温和

## 环境管理

### 虚拟环境优势
- 避免与系统Python包冲突
- 确保依赖版本一致性
- 便于项目管理和部署

### 常用命令
```bash
# 查看当前Python环境
which python
python --version

# 查看已安装的包
pip list

# 更新依赖包
pip install --upgrade -r requirements.txt

# 删除虚拟环境（重新开始）
rm -rf venv
```

## 注意事项

- 确保系统安装了中文字体
- 时间单位：秒(s) (公式中)，毫秒(ms) (显示和分析)
- 长度单位：米(m)
- 建议时间范围：0-18ms
- 使用Table 3中的K2参数作为K值
- B和C为固定参数：B=0.41, C=50000 (已从ms单位转换为s单位)
- **重要**: 每次使用前都要激活虚拟环境 