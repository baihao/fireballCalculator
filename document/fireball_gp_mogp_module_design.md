# 火球数据高斯过程（**K / C 分模型**）训练与推理模块设计说明

## 1. 文档目的与范围

本文档描述基于**高斯过程回归（Gaussian Process Regression）**，在 **\(K\)** 与 **\(C\)** **各自独立**的单输出 GP 框架下，对火球直径拖曳拟合数据进行**超参数训练**与**新点推理**的软件模块划分、数据流与命令行接口；**直径–时间曲线绘制**为独立模块 **`curve_plot`**（逻辑与依赖可单测），**CLI 的 `predict` 默认在推理完成后调用该模块**，并将图像保存到**与本次预测结果相同的输出目录**；独立 `plot` 子命令仍用于仅根据已有 `kbc.json` 补图或重画。

**推荐实现栈（训练与推理）**：**Python 3.10+**，**PyTorch**（张量与自动微分），**GPyTorch**（**单任务** `ExactGP`、精确边际似然、后验）。曲线绘制使用 **Matplotlib**（或 **Plotly**，二选一）。备选：纯 **NumPy/SciPy** 自写核矩阵与 Cholesky（工作量大，仅作对照或小规模原型）。

**不在本文范围**：直径–时间原始曲线的物理建模、拖曳拟合算法本身（假定每条训练 JSON 已含 `drag_fit` 结果）；**不**在 GP 内对整条 `diameter_over_time` 曲线建模。

**默认建模策略**（与 `gp_fireball_kc_lmc_strategy.md` 一致）：**两个独立单输出 GP**，分别拟合 **\(K\)** 与 **\(C\)**（**不**采用单一 MOGP 前向一次性输出 \((K,C)\)）；**\(\bar B\)** 为训练集所有有效样本 **\(B\)** 的**算术均值**，推理时对任意新输入恒用 **\(\hat B=\bar B\)**，**不**经 GP。输入 **\(X\)** 保持物理单位、**不做**逐维仿射归一化；**每个** GP 的输入核为 **RBF + ARD（\(\ell_0\) 对应当量、\(\ell_1\) 对应含铝量）**，**两模型超参不共享**。**不对 \(K,C\)** 做输出标准化。**历史**联合 MOGP（LMC/ICM）仅作**对照**时在策略文档中说明，**非**本设计默认路径。

---

## 2. 总体架构

系统划分为四个功能模块与统一入口 **CLI**：

| 模块 | 职责摘要 |
|------|----------|
| **data_input** | 扫描训练数据目录，解析单条 JSON，抽取 `X`、`Y_kc`（\(K,C\) 两列）及 **`b_mean`**（训练集 **\(B\)** 均值），构成 `Dataset`。可从同一条 JSON 的 `drag_fit` 同时读出 **\(B\)** 以汇总均值。 |
| **train_infer** | **训练**：在相同 **`X`** 上分别优化 **`(X, K 列)`** 与 **`(X, C 列)`** 两个单输出 GP；**推理**：对 **`X_*` 分别**前向得到 \(\hat K\)、\(\hat C\) 的边缘高斯预测，再与 **`b_mean`** 拼装为 **`curve_plot`** 所需的 \((K,B,C)\)。**不**以单次多任务前向张量输出 \((K,C)\)。 |
| **curve_plot** | **独立模块**：根据一组 `K,B,C` 与 `plot` 配置，按拖曳公式计算 \(D(t)\) 并输出图像；**不**加载 GP 模型、**不**调用 `train_infer`。 |
| **main（CLI）** | 解析子命令：`train` / `predict` / `plot`；`predict` **默认**编排 `train_infer` 后调用 **`curve_plot`**（可用 `--no-plot` 关闭）。 |

依赖关系：

- `main` → `data_input`、`train_infer`、`curve_plot`。  
  - **`predict`**：先 **`train_infer.predict`**：**\(K\)** 模型与 **\(C\)** 模型**各自**对 **`X_*`** 推理，将 **`b_mean`** 注入为 **\(B\)**，拼装 **\((K,B,C)\)** 后**默认**交给 `curve_plot`；图像写入 **`--out-dir`**（与预测 JSON 同目录）。  
  - **`plot`**：仅调用 `curve_plot`，用于离线补图或手工 `kbc.json`。
- `train_infer` 仅依赖已结构化的 **`Dataset`（`X`, `Y_kc`, `b_mean`）**，不直接读原始目录。
- **`curve_plot` 仅依赖** `(K,B,C)` 数值、可选工况标注（用于图题）、以及 **`plot` 段配置**（时间轴）；可与推理**进程内**串联（`predict` 默认）或**离线**串联（`plot` 子命令）。

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  CLI main   │────▶│  data_input  │────▶│ Dataset (X,Y_kc,b_mean) │
└──────┬──────┘     └──────────────┘     └────────┬──────────┘
       │                                           │
       │              ┌──────────────┐             │
       └─────────────▶│ train_infer  │◀────────────┘
                      │  train /     │
                      │  predict     │
                      └──────┬───────┘
                      │ 分别预测 K、C（B=训练均值）
                             ▼
┌────────────────────────────────────────────────────────────┐
│  predict 结果目录（--out-dir）                              │
│  ├── predict_result.json（或约定文件名）                  │
│  └── diameter_vs_time.png（默认；多点时可 diameter_i.png） │
└────────────────────────────────────────────────────────────┘
         ▲ 默认调用 curve_plot，图像与 JSON 同目录
```

独立 **`plot`** 子命令仍可从任意 `kbc.json` 读入，与目录约定无关。

---

## 3. 模块一：`data_input`

### 3.1 输入

- **路径**：包含若干训练样本文件的**目录**（例如 `training_data/`）。
- **文件约定**：每个可解析文件为 **JSON**（扩展名建议 `.json`；实现可配置是否递归子目录、是否忽略非 JSON）。

### 3.2 单文件解析（以 `fireball_diameter_fit_1_40.json` 为例）

每条样本对应一个 JSON 文件，与业务相关的字段结构如下（节选）：

- **`parameters`**（工况参数）
  - `equivalent`：当量（示例中为字符串 `"1"`，解析为实数特征）。
  - `al_percent`：铝粉百分比（示例中为字符串 `"40"`，解析为实数特征）。
  - 其他字段（如 `material_type`、`explosion_duration`、`pixel_length`）是否进入 `X` 由产品需求决定；**本设计默认仅将 `equivalent`、`al_percent` 作为 `X` 的二维分量**，以便与现有描述一致。

- **`drag_fit`**（拖曳函数拟合结果）
  - `success`：布尔值；为 `false` 时该样本应**跳过**或按策略处理（见 3.4）。
  - `K`、`B`、`C`：拖曳形式 `D(t) = K * (1 - B * exp(-C * t^2))` 中的标量参数（示例中为浮点数）。
  - `expression` 等字段仅作审计，不参与张量构造。

- **`diameter_over_time`**：时间序列数据；**构造 `(X, Y)` 时默认不读入**（若未来要用曲线核或双阶段模型，可扩展，不在本文默认路径内）。

### 3.3 输出：数据集 `Dataset`

**与策略文档一致**时，`Dataset` 含：

- **`X`**：形状 `n × 2`，列顺序固定为：
  - 第 0 列：`equivalent`（数值化）
  - 第 1 列：`al_percent`（数值化）
  - **不在此阶段对 `X` 做标准化**；训练与推理始终在**相同物理单位**下将 `X` / `X_*` 送入核（**RBF，ARD，`ard_num_dims=2`**，对应 \(\ell_0,\ell_1\)）。

- **`Y`**（记号 **`Y_kc`**）：形状 **`n × 2`**，列为 **\(K\)** 与 **\(C\)** 的监督标签（**训练时拆为两路**单输出 GP），顺序固定为：
  - 第 0 列：`K`（来自 `drag_fit.K`）
  - 第 1 列：`C`（来自 `drag_fit.C`）

- **`b_mean`**：`float`，在**当前训练集**上对所有有效样本的 **`drag_fit.B`** 求算术平均：
  \[
  \bar B = \frac{1}{n}\sum_{i=1}^n B^{(i)}.
  \]
  训练落盘时写入 **`trained_hyperparams.json`**（如键名 `b_mean`），**推理**时对每个新点取 \(\hat B \equiv \bar B\)。

- **元数据（建议一并输出）**：文件名、原始字符串、是否被过滤、`drag_fit.success` 等，供日志与调试。

**实现注意**：单文件 JSON 仍含完整 **\(K,B,C\)**；装载器在构建 `Dataset` 时**拆分**：两列进入 **`Y`**，**\(B\)** 参与聚合 **`b_mean`**。对外接口类型若暂保留 `Y: (n,3)` 仅兼容旧代码，须在 `train` 路径明确**仅使用 \(K,C\) 与 `b_mean`**，与本文及策略文档不一致的用法视为待迁移遗留。

### 3.4 校验与错误策略

- JSON 无法解析、缺少 `parameters` / `drag_fit`、或 `drag_fit.success != true`：**默认跳过并记录警告**（可配置为严格模式：遇错即中止）。
- `K`、`B`、`C` 缺失或非数值：**跳过或报错**（与 `success` 语义一致）。
- 空目录或过滤后无有效样本：**向调用方返回明确错误**，不进入训练。

### 3.5 对外接口（概念）

- `load_training_dir(path: Path) -> Dataset`  
  其中 `Dataset` 包含 **`X: Float[n,2]`**、**`Y: Float[n,2]`**（**K, C**）、**`b_mean: float`** 及可选 **`meta`** 列表。  
  （实现可选用 `dataclass` 字段名 `Y_kc` / `b_mean` 等，语义与上等价即可。）

---

## 4. 模块二：`train_infer`（训练与推理）

### 4.1 模型假设

- **`Y` 的两列**（**\(K,C\)**）在**软件层**仍可由同一 `Dataset` 承载，但**训练与推理**拆为 **两个**单输出模型：
  - **`model_K`**：`train_y` 为 **`(n,1)`** 或 **`(n,)`** 的 **\(K\)**；**`GaussianLikelihood`**（标量噪声）。
  - **`model_C`**：同上，监督目标为 **\(C\)**。
- **\(B\)** **不**进入任一侧似然，由 **`b_mean`** 在推理时拼入下游 \((K,B,C)\)。
- **训练目标**：对每个模型 **边际似然最大化**（型-II ML），见策略文档 §5.2（向量形式退化为长度 **\(n\)**）。
- **输入核（各模型一份）**：**RBF**，**ARD**，二维长度尺度 \((\ell_{m,0},\ell_{m,1})\) 与信号方差 \(\sigma_m^2\)（\(m\in\{K,C\}\)）。
- **输出**：**不对 \(K,C\)** 做标准化；预测均值与方差直接为物理量纲。
- **不确定性**：**默认** **\(K\)**、**\(C\)** 边缘预测**独立**拼装；**不**要求实现层维护 **\(2\times 2\)** **\(K\)–\(C\)** 联合协方差块（若 `curve_plot` 蒙特卡洛需「一致」\((K,C)\) 样本，见策略文档 §5.3：独立采样近似）。

### 4.1.1 库选型说明（训练与预测）

| 组件 | 库 | 作用 |
|------|-----|------|
| 数组与张量 | **PyTorch** | `FloatTensor` 存放 `X,Y,X_*`；与 GPyTorch 一致。 |
| GP 模型与似然 | **GPyTorch** | **两个** `ExactGP` 族实例 + **标量** `GaussianLikelihood`；**非** `MultitaskGaussianLikelihood`（默认路径）。 |
| 目标函数 | **GPyTorch** | **各模型**一个 `ExactMarginalLogLikelihood`：\(\log p(\mathbf{y}_m\mid \mathbf{X},\boldsymbol{\phi}_m)\)，\(\mathbf{y}_m\in\mathbb{R}^{n}\)。 |
| 优化 | **PyTorch**（`torch.optim`） | 常用 **Adam**（或 L-BFGS）**分别**优化两套可训练超参。 |
| 推理后验 | **GPyTorch** | `model_K.eval()` / `model_C.eval()` + 各自 `likelihood` 得到**标量**高斯后验。 |
| 绘图 | **Matplotlib**（推荐） | `D(t)` 曲线、坐标轴、可选保存 PNG/SVG。 |

**为何不默认 sklearn 多输出**：本设计**刻意**拆成两回归；`GaussianProcessRegressor` 多输出模式与「**独立**任务」接近，但 **GPyTorch** 对 **可学习核参数 + 精确 MLL** 更一致。

### 4.1.2 基于 GPyTorch 的计算过程（训练与预测）

以下按实现顺序描述**一次完整训练**与**一次预测**（**双模型**）。**产品级策略**（独立 \(K\) GP / 独立 \(C\) GP、\(B=b\_\mathrm{mean}\)、原坐标 ARD、无 \(Y\) 标准化）以 `gp_fireball_kc_lmc_strategy.md` 为准。

**（1）数据进入张量**

- **`X`**：`(n, d)`，`d=2`，**原始物理单位**，**不**标准化。
- **`Y_K`**、**`Y_C`**：由 **`Y_kc`** 拆出两列，形状 **`(n,)`**（或 `(n,1)`）。

**（2）定义似然与模型（×2）**

- **`likelihood_K = GaussianLikelihood()`**，**`model_K`**：`ConstantMean` + **`ScaleKernel(RBFKernel(ard_num_dims=2))`**（或等价写法）；`ExactGP(train_x, train_y_K, likelihood_K)`。
- **`likelihood_C`**、**`model_C`**：**结构同**上，**参数不绑定**。

**（3）训练：边际似然与反向传播**

- **`mll_K = ExactMarginalLogLikelihood(likelihood_K, model_K)`**，`loss_K = -mll_K(...)`；**同理** **`mll_C`**。迭代次数、学习率、重启策略可**共享配置**或为**每任务**单独配置（产品决定）。

**（4）预测：后验高斯（×2）**

- **`pred_K = likelihood_K(model_K(x_star))`**：均值 **`(m,)`** 的 \(\hat K\)；**`pred_C`**：\(\hat C\)。
- **`\(B\)`**：无 GP 方差（常数 **`b_mean`**）。

**（5）与 `b_mean` 拼装**

- **`curve_plot` / 预测 JSON**：**\(B \leftarrow\)** 产物字段 **`b_mean`**，与 **`pred_K`、`pred_C`** 拼成 \((K,B,C)\)。

**（6）与闭式公式的对应关系**

- **训练**：两次标量输出 GP 的 \(\log p(\mathbf{y}\mid \mathbf{X},\boldsymbol{\phi})\)。  
- **预测**：两次一元高斯条件后验。

### 4.2 协方差构造（**每任务独立**）

- **默认**：**每个** GP **各自**一个 **ARD RBF**（或可选 Matérn）**输入核**，**不**通过 `MultitaskKernel` / ICM / LMC 与另一任务耦合。
- **对照实验**（可选）：若需与历史基线对比，可在单独配置中启用 **MOGP（LMC \(Q=2\)）**，此时超参 JSON 结构与产物 `torch` 字段与**双模型**方案**不兼容**，须在文档与加载器中显式区分 **`schema_version`** 或 **`model_family`**（由实现约定）。

### 4.3 训练

- **输入**：`Dataset` 的 **`X`**、**`Y`**（**`n×2` 的 K,C**）、**`b_mean`**（落盘写入 JSON，**不**进 GP 似然），以及超参数初值（§6.3）。
- **过程**：**分别**在 **\(K\)** 与 **\(C\)** 上优化对数边际似然，得 \(\hat{\boldsymbol{\phi}}_K,\hat{\boldsymbol{\phi}}_C\)。
- **输出（落盘约定）**：**`trained_hyperparams.json`**：含 **两套** \(\hat{\boldsymbol{\phi}}\)（或两套 `state_dict`）、**`b_mean`**、训练快照 **\((X,Y)\)**（**`Y`** 仍可为 **`n×2`** **便于追溯**）；**不包含**对 **\(B\)** 的 GP 超参。

### 4.4 推理

- **输入**：新输入 **`X_*`**（`m×2`，**物理单位与训练一致**）；**训练产物 JSON**（含两套 \(\hat{\boldsymbol{\phi}}\)、**`b_mean`**、训练缓存）。
- **输出**：
  - **GP 部分**：**\(\hat K\)** 与 **\(\hat C\)** **各自**的（标量）高斯预测——均值 **`(m,)`**（或拼成 **`(m,2)`** **仅 Convenience**），方差为 **两条边际**；**无**默认的 **\(2\times 2\)** **块协方差**。
  - **完整拖曳三元组**：**\(\hat B \equiv \texttt{b_mean}\)**（**无** GP 不确定度）；对外 **`predict` JSON / `kbc`** 仍为 **`K,B,C`**，其中 **`B`** 恒为训练集均值。
- 若需 **\(D(t)\)** 蒙特卡洛云图：**默认**对 **\(K,C\)** **独立**采样（各自边缘）；**不要**假设存在已估计的 **\(2\times 2\)** 联合协方差（除非启用可选 MOGP 分支）。

### 4.5 对外接口（概念）

- `train(dataset: Dataset, hyper_config: HyperConfig) -> TrainedModel`（内存态含 **`model_K`,`model_C`**）；**CLI 落盘主产物**为 **训练好的超参数 JSON**（默认路径见 §6.2），供 `predict` 加载。  
- `predict(trained_artifact: TrainedModel | HyperparamsJSON, X_star: Float[m,2]) -> PredictiveDistribution`：**两次**单任务前向再拼装 **K,C**（实现可从 JSON 反序列化）。

由预测的 \((K,B,C)\)（**其中 \(B=\texttt{b_mean}\)**）生成直径–时间曲线见 **§5**；`train_infer` **不包含**绘图逻辑。

---

## 5. 模块三：`curve_plot`（直径–时间曲线）

本模块在**代码与依赖**上与 `train_infer` 分离（便于单测与复用）；在 **CLI 上**，**`predict` 默认在推理成功后调用本模块**生成 \(D(t)\) 图，**输出路径**与本次 **`predict` 写入的预测结果目录一致**（见 **§6.2**）。仅当用户需要**仅推理不画图**或**对已有 `kbc.json` 补图**时，分别使用 **`predict --no-plot`** 或独立 **`plot`** 子命令。

### 5.1 公式与输入输出

训练标签中的拖曳形式与单条训练 JSON 中 `drag_fit.expression` 一致（示例）：

\[
D(t) = K \bigl(1 - B\,\exp(-C\,t^2)\bigr)
\]

其中 \(t\) 与训练数据 `diameter_over_time[].time_ms` 的单位一致，为**毫秒（ms）**；\(D\) 为火球直径（与数据一致，一般为米 **m**）。\((K,B,C)\) 来自 **`predict`**：**\(\hat K,\hat C\)** 为**两路 GP** 后验均值（或用户指定），**\(B\)** 为训练产物 **`b_mean`**；亦可由用户手工写入 **`kbc.json`**。

**计算步骤**

1. **输入**：一组 \(\hat K,\hat B,\hat C\)（可广播）；可选：**各自**标量预测方差或 **`K,C`** **独立**蒙特卡洛；**\(B\)** 固定为 **`b_mean`**，用于不确定性带。
2. **时间轴**：在 \([t_{\min}, t_{\max}]\) 上取均匀或其它网格 `t_i`（步数 `N` 由配置决定）。时间范围来自 **超参数 JSON 的 `plot` 段**（见 §6.3）或 `plot` 子命令 CLI 覆盖。
3. **逐点求值**：\(D_i = K \bigl(1 - B\,\exp(-C\,t_i^2)\bigr)\)，用 **NumPy** 或 **PyTorch** 向量化；**禁止**在 \(C<0\) 或导致数值溢出时静默画图——应对输入做物理/先验校验（产品层决策）。
4. **出图**：**Matplotlib** 绘制 `D`–`t` 曲线；横轴 `时间 t (ms)`，纵轴 `直径 D (m)`；标题或图例中标注可选的 `equivalent`、`al_percent` 与 \((K,B,C)\)。
5. **可选：不确定性可视化**：对 **\(K\)、\(C\)** **各自**从边缘分布抽取，或独立联合抽取 \(S\) 组，**\(B\)** 固定为 **`b_mean`**，画 \(D(t)\) 半透明曲线或分位点带；**非**基于 **\(2\times 2\)** 联合 MOGP 协方差的默认路径。

### 5.2 模块边界与对外接口

- **不依赖** `train_infer` / GPyTorch；仅依赖数值 \((K,B,C)\)、`PlotConfig`（时间轴、输出路径、可选随机种子）。
- **对外接口（概念）**：`plot_diameter_curve(kbc: Vector3, meta: PlotMeta | None, plot_cfg: PlotConfig, path_out: Path | None) -> Figure`。

### 5.3 `plot` 子命令（CLI，离线补图）

与 **`predict` 并列**，在**不重新跑 GP** 时单独调用 `curve_plot`（例如复用已有 `kbc.json`）：

```text
fireball-gp plot --hyperparams <trained_hyperparams.json> --kbc-json <kbc.json> --out <diameter_vs_time.png>
```

- **`--hyperparams`**：与 **`predict`** 相同，指向 **训练产出的超参数 JSON**，至少读取其中 **`plot` 段**（`time_ms_min` / `time_ms_max` / `num_points` 等）；实现可另允许「仅含 `plot` 的瘦 JSON」作为等价输入。
- **`--kbc-json`**：见下文 **`kbc.json` 示例**。
- **`--out`**：输出图像路径（PNG/SVG 等）。

可选：`--time-ms-max` 等参数**覆盖**配置中的 `plot` 字段，便于快速出图。

**与 `predict` 默认画图的区别**：`predict` 不写 `--out` 到 `plot` 子命令，而是由 **`--out-dir`** 统一指定目录，图像**默认**落盘到该目录下（见 §6.2）；`plot` 子命令则显式指定单文件 `--out`。

#### `kbc.json` 示例（`plot --kbc-json`）

**单组参数**（与一次 `predict` 单点输出对应）：

```json
{
  "K": 4.725336052929695,
  "B": 0.5616611224348907,
  "C": 0.0021707653600251987,
  "equivalent": 1.0,
  "al_percent": 40.0,
  "B_source": "train_mean"
}
```

- `K`、`B`、`C`：**必填**（浮点）。若 **`B`** 来自训练均值策略，建议附带 **`"B_source": "train_mean"`**（可选，仅追溯）。  
- `equivalent`、`al_percent`：**可选**，仅用于图题/图例标注；不参与 \(D(t)\) 计算。

实现可将 **`predict` 的标准输出**或**专用预测结果 JSON** 设计成与 `kbc.json` **兼容**，便于 `predict > kbc.json` 后接 `plot`（字段名需统一）。

---

## 6. 模块四：主程序（CLI）

### 6.1 职责

- 解析命令与子命令、必选/可选参数。
- 加载**超参数 JSON**：**训练**时 `--config` 可选（省略则用内置默认，见 §6.3）；**推理**时加载 **训练阶段产出的「训练好的超参数 JSON」**（见 §6.2），其中含 \(\hat{\boldsymbol{\phi}}\)、`plot`/`io` 等；**`predict` 默认画图**时消费其中的 **`plot` 段**与可选 **`io` 中图像文件名约定**。
- **训练命令**：指定 `--data-dir`；可选 `--config`；训练结束后**默认**将 **训练好的超参数 JSON** 写入 **`--data-dir` 目录**下约定文件名（见 §6.2），无需单独 `--out-model` 二进制产物。
- **推理命令**：指定 **`--hyperparams`** 指向上述训练产出 JSON、`X_*`、`--out-dir` → `train_infer.predict` → **默认**调用 **`curve_plot`**，将图像写入 **`--out-dir`**（与预测结果 JSON 同目录）。**`--no-plot`** 可关闭画图。
- **独立绘图命令**：见 **§5.3**（`plot` 子命令），在不重跑推理时根据 **`kbc.json`** 出图。

### 6.2 建议命令形态（示例，非最终实现）

```text
# 训练：扫描目录，分别拟合 K / C 两路 GP；默认在 data_dir 下写出「训练好的超参数 JSON」（--config 可省略则用内置默认）
fireball-gp train --data-dir <training_data_dir> [--config <hyperparams.json>] \
  [--out-hyperparams <trained_hyperparams.json>]

# 推理（默认）：仅传入训练产出的超参数 JSON + X_*；预测结果与曲线图写入 out-dir
fireball-gp predict --hyperparams <trained_hyperparams.json> --out-dir <predict_result_dir> \
  --equivalent <float> --al-percent <float>
# 或批量 X_*
fireball-gp predict --hyperparams <trained_hyperparams.json> --out-dir <predict_result_dir> \
  --x-json <x_star.json>

# 仅推理、不画图
fireball-gp predict --hyperparams <trained_hyperparams.json> --out-dir <predict_result_dir> \
  --x-json <x_star.json> --no-plot

# 离线补图：不跑 GP，仅根据 kbc.json 与超参数 JSON 中的 plot 段出图
fireball-gp plot --hyperparams <trained_hyperparams.json> --kbc-json <kbc.json> --out <diameter_vs_time.png>
```

- **`train --out-hyperparams`**：**可选**。未指定时，**默认**写入 **`<training_data_dir>/trained_hyperparams.json`**（文件名可配置，需在实现中固定并写入本文或 `--help`）。文件内容为训练后的完整超参与推理所需元数据（见 §6.3 说明及扩展字段）。
- **`predict --hyperparams`**：**必选**，指向 **`train` 产出的 JSON**（或内容等价、可被同一加载器解析的文件）。
- **`--out-dir`**：**推理必选**（或提供与之一致的默认目录策略）；其下写入预测 JSON（如 `predict_result.json`）及**默认生成的曲线图**（与 JSON 同级）。  
- **`--no-plot`**：关闭默认画图。  

#### `x_star.json` 示例（`predict --x-json`）

推理输入与训练侧 `X` 的列顺序一致：**先 `equivalent`，后 `al_percent`**，值为数值（整数或浮点均可，解析为 `float64`）。

**单点**（一个 `X_*`）：

```json
{
  "equivalent": 1.0,
  "al_percent": 40.0
}
```

**多点**（批量推理，推荐显式数组字段 `points`，按数组顺序构成 `m×2` 的 `X_*`）：

```json
{
  "points": [
    { "equivalent": 1.0, "al_percent": 40.0 },
    { "equivalent": 2.0, "al_percent": 30.0 }
  ]
}
```

实现可二选一支持：仅识别根级 `points`；或同时允许根级直接为 `[{...},{...}]` 的 JSON 数组（与 `{"points":[...]}` 等价语义）。键名建议固定为 `equivalent`、`al_percent`，与 `data_input` 中从 `parameters` 抽取的语义一致。

- **`--config`（超参数 JSON）**：**仅训练**使用且**可省略**，省略则使用**内置默认**（与 §6.3 一致）；提供则作为初值/边界并与默认合并。**推理不再使用 `--config`**，改为 **`--hyperparams`** 指向训练产出 JSON；**`plot`** 子命令用 **`--hyperparams`** 读取其中 `plot` 段（或与训练 JSON 同结构的瘦文件，由实现约定）。
- **训练**：必须提供 `--data-dir`；**推理**：必须提供 **`--hyperparams`**、`X_*`（命令行标量或 JSON 数组）及 **`--out-dir`**（除非实现提供可接受的默认预测结果目录）。

### 6.3 超参数 JSON（逻辑结构建议）

以下为**逻辑字段**示例，实际键名可由实现统一命名。

- **训练入口（`train --config`）**：**未提供 `--config` 时**，使用与下表一致的**内置默认值**；提供 `--config` 则覆盖或合并。
- **训练出口（`train` 默认写入 `data_dir/trained_hyperparams.json`）**：在下列结构基础上写入 **两套** \(\hat{\boldsymbol{\phi}}\)（如 **`torch`** 下 **`model_K_state_dict` / `model_C_state_dict`** 等）、**`b_mean`**、训练快照 **`X`** 与 **`Y_kc`（\(n\times 2\)）**；建议 **`model_family": "dual_single_output_gp"`** 或与历史 **MOGP** 可区分的 **`schema_version`**。**默认不写** `input_standardization` / `output_standardization`（与策略文档一致）。若实现保留可选标准化，须与策略冲突时在 JSON 中显式标记并默认关闭。

```json
{
  "b_mean": 0.5616611224348907,
  "model_family": "dual_single_output_gp",
  "covariance_K": {
    "input_kernel": "RBF_ARD",
    "ard_num_dims": 2
  },
  "covariance_C": {
    "input_kernel": "RBF_ARD",
    "ard_num_dims": 2
  },
  "optimization": {
    "max_iter": 200,
    "learning_rate": 0.05
  },
  "torch": {
    "model_K_state_dict": {},
    "likelihood_K_state_dict": {},
    "model_C_state_dict": {},
    "likelihood_C_state_dict": {}
  },
  "io": {
    "strict_drag_fit_success": true,
    "predict_plot_filename": "diameter_vs_time.png",
    "predict_point_plot_pattern": "diameter_{index}.png"
  },
  "plot": {
    "time_ms_min": 0.0,
    "time_ms_max": 75.0,
    "num_points": 300
  }
}
```

- `b_mean`：训练集 **\(B\)** 算术均值，**推理**时 **\(\hat B\)** 恒取之。  
- **`covariance_K` / `covariance_C`**：**各自**任务的核与噪声超参（键名实现可略有出入）。  
- **输入核**：须 **`RBF` + ARD（2 维）**，**不在训练前对 `X` 做标准化**。  
- 训练结束后将优化后的超参写入同一 JSON（如各段 `optimized` 标记），与 §6.2 约定一致。
- `plot.*`：供 **`curve_plot`**、`predict`（默认画图）与 **`plot` 子命令**使用，控制 `D(t)` 曲线的时间范围与网格密度；`time_ms_max` 可与常见实验 `explosion_duration`（ms）同量级。
- `io.predict_plot_filename`：单点或合并输出时，**默认**曲线图文件名（位于 `--out-dir` 下）。  
- `io.predict_point_plot_pattern`：批量多点时，可按 `diameter_0.png`、`diameter_1.png` 等与 `predict_point_plot_pattern` 对齐；或使用单图多曲线（产品二选一，需在实现中固定）。

---

## 7. 数据与文件约定汇总

| 项目 | 约定 |
|------|------|
| 训练样本文件 | JSON，含 `parameters.equivalent`、`parameters.al_percent`、`drag_fit.{success,K,B,C}` |
| `trained_hyperparams.json`（`train` 默认产出） | 位于 **`--data-dir`**（或 `--out-hyperparams`）；含 **两套** \(\hat{\boldsymbol{\phi}}\) 或 **`model_K`/`model_C` 的 `state_dict`**、**`b_mean`**、`X` 与 **`Y`（\(n\times 2\)，仅 K/C）** 快照或等价路径，供 **`predict` / `plot --hyperparams`** |
| **`b_mean`** | 浮点，训练集 **\(B\)** 均值；**推理**时 \(\hat B\equiv\texttt{b_mean}\) |
| `x_star.json`（`predict`） | 推理输入 `X_*`，见 **§6.2** |
| `predict` 结果目录（`--out-dir`） | 默认写入预测 JSON 与 **同目录下** 的直径–时间曲线图；见 **§6.2** |
| `kbc.json`（`plot`） | 含 `K,B,C` 及可选 `equivalent`,`al_percent`（图题），见 **§5.3**（离线补图） |
| `X` 维数 | 2（`equivalent`, `al_percent`），解析为浮点 |
| `Y` 维数（`Dataset`） | **2**（`K`, `C`）；**训练时拆成两路**单输出 GP |
| GP 输出 | **两次**边际预测后拼装 **\(\hat K,\hat C\)**；**非**单次 MOGP 向量前向 |

---

## 8. 扩展与风险（简）

- **特征扩展**：若将 `explosion_duration` 等纳入 `X`，需同步修改 `d`、数据模块与 CLI（`predict` 的 `--x-json` schema）；`plot` 模块仅消费 `kbc.json`，一般无需改。
- **失败样本**：`drag_fit.success == false` 的样本默认丢弃；若保留需定义伪标签策略（不推荐在未定义时静默使用）。
- **数值尺度**：当采用 **`gp_fireball_kc_lmc_strategy.md`** 时，输入保持**物理单位**、**不做** \(X\) 的仿射归一化，在核中使用 **ARD** 各维长度尺度；若实现仍保留可选 `input_standardization`，须与策略文档约定一致并在产物中显式记录。

---

## 9. 参考文档

- 项目内：`document/gp_fireball_kc_lmc_strategy.md`（**算法与建模约定**的权威说明：**\(K\)**、**\(C\)** **分模型**单输出 GP、\(B\) 训练均值、原坐标 ARD-RBF、**无** \(Y\) 标准化等；文件名保留历史）。
- 训练数据示例：`training_data/fireball_diameter_fit_1_40.json`。

**说明**：本文档描述**模块边界、数据形态与 CLI**；**与策略不一致时以 `gp_fireball_kc_lmc_strategy.md` 为准**。
