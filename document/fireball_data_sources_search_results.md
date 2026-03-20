# 爆炸火球数据源搜索结果汇总

本文档汇总了网上搜索到的可用于火球直径/半径研究的文献与数据源。

---

## 一、英文文献与数据源

### 1. 贝鲁特爆炸火球演化数据（强烈推荐）

**文献**：Beirut explosion: TNT equivalence from the fireball evolution in the first 170 milliseconds  
**期刊**：Shock Waves, 2021, 31: 813–827  
**链接**：https://link.springer.com/article/10.1007/s00193-021-01031-9  
**开放获取**：是（可免费下载 PDF）

**数据特点**：
- 使用业余视频提取火球演化数据
- 39 帧，时间间隔 16.66–33.33 ms
- 火球半径追踪至约 170 ms，距离约 128 m
- 使用 Sedov-Taylor 模型和拖曳模型拟合
- 论文中有 **Table 2: Data extracted from videos**，包含时间-半径数据点
- 可反推 TNT 当量约 200±80 吨

**用途**：可直接从论文表格或图中提取 (t, R) 数据，用于验证拖曳函数拟合。

---

### 2. Gordon 等化学爆炸火球实验数据

**引用**：Beirut 论文引用 Gordon et al. 的化学爆炸火球演化实验观测  
**用途**：作为 Sedov-Taylor 模型验证的基准数据

---

### 3. 燃料-空气爆炸火球

**文献**：Fireballs from deflagration and detonation of heterogeneous fuel-rich clouds  
**来源**：ScienceDirect / Academia.edu  
**数据**：汽油、煤油、柴油，燃料质量 0.1–100 吨  
**内容**：最大火球半径、热效应持续时间、总辐射能与燃料质量的关系

---

### 4. 氢气储罐破裂火球

**文献**：Hydrogen Tank Rupture in Fire in the Open Atmosphere  
**内容**：火球尺寸工程关联式，区分火球直径与最大水平尺寸

---

### 5. Caltech 爆轰数据库

**名称**：GALCIT Explosion Dynamics Laboratory Detonation Database  
**链接**：https://shepherd.caltech.edu/detn_db/html/db.html  
**内容**：气相爆轰实验数据（胞格宽度、临界管径等），非火球直径，但可作背景参考

---

## 二、中文文献与数据源

### 1. 强爆炸火球热辐射尺度效应（爆炸与冲击，2024）

**文献**：李康等. 强爆炸火球热辐射尺度效应理论和数值研究  
**期刊**：爆炸与冲击, 2024, 44(10): 102101  
**链接**：https://pubs.cstam.org.cn/article/doi/10.11883/bzycj-2023-0199  
**DOI**：10.11883/bzycj-2023-0199

**数据特点**：
- 图 1、图 2：1 kt 当量、海平面爆炸火球有效半径和有效温度随时间变化
- 图 4–7：不同爆炸高度、不同当量下的热辐射功率和有效半径
- 表 1–3：尺度效应参数（含 LFB 火球特征尺度）
- 可数字化提取半径-时间曲线

---

### 2. 温压炸药爆炸性能实验研究（爆炸与冲击，2016）

**文献**：黄亚峰等. 温压炸药爆炸性能实验研究  
**期刊**：爆炸与冲击, 2016, 36(4): 573-576  
**链接**：https://www.bzycj.cn/cn/article/doi/10.11883/1001-1455(2016)04-0573-04

**数据特点**：
- 25 g 温压炸药，5.8 L 密闭爆炸罐
- 图 2：爆炸压力-时间曲线
- 图 3：爆炸温度-时间曲线
- 主要为压力、温度，火球直径需从其他文献补充

---

### 3. 温压炸药爆炸火球特征（火炸药学报，2007）

**文献**：阚金玲等. 温压炸药爆炸火球的特征  
**期刊**：火炸药学报, 2007, 30(2): 55-58  
**DOI**：10.3969/j.issn.1007-7812.2007.02.015

**数据特点**（根据检索摘要）：
- 66 ms 时半径约 6 m
- 后燃阶段可燃云团最终抛撒半径约 5.6 m
- 后燃持续时间约 86 ms
- 白光持续时间约 10 ms

---

### 4. 30 kg 温压炸药火球数据（项目 papers 目录）

**文献**：30kg温压炸药表面平均温度.pdf  
**数据**：30 kg 试验弹，最大直径约 17.4 m，为装药直径的 75.65 倍

---

## 三、数据提取建议

### 1. 优先提取

| 优先级 | 文献 | 提取内容 | 工具 |
|--------|------|----------|------|
| 高 | Beirut 爆炸论文 | Table 2 时间-半径数据 | 直接抄表或 CSV |
| 高 | 强爆炸火球尺度效应 | 图 1、图 4–7 半径-时间曲线 | WebPlotDigitizer |
| 中 | 温压炸药火球特征 | 66 ms 半径 6 m 等离散点 | 手动记录 |
| 中 | 30 kg 温压炸药 | 最大直径 17.4 m | 补充到数据集 |

### 2. 数字化工具

- **WebPlotDigitizer**：https://automeris.io/WebPlotDigitizer/
- **Engauge Digitizer**：开源桌面软件

### 3. 数据格式

提取后建议整理为：
```csv
source,equivalent_kg,material,time_ms,radius_m,notes
Beirut_2021,200000,AN,0,0,extracted from Table 2
Beirut_2021,200000,AN,16.66,45,...
```

---

## 四、检索关键词备忘

**英文**：fireball diameter explosion, TNT equivalent fireball, Sedov-Taylor explosion, shock wave radius time  
**中文**：火球直径 爆炸, 温压炸药 火球, 爆炸火球 膨胀, 强爆炸 火球

---

## 五、参考文献

1. Beirut explosion: TNT equivalence from the fireball evolution in the first 170 milliseconds. Shock Waves (2021). https://doi.org/10.1007/s00193-021-01031-9
2. 李康等. 强爆炸火球热辐射尺度效应理论和数值研究. 爆炸与冲击, 2024.
3. 黄亚峰等. 温压炸药爆炸性能实验研究. 爆炸与冲击, 2016.
4. 阚金玲等. 温压炸药爆炸火球的特征. 火炸药学报, 2007.
