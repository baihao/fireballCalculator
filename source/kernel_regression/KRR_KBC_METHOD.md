# 核岭回归拟合 \(K,B,C\) 方法说明

本目录实现对拖曳系数 **\(K,B,C\)** 的独立 **Kernel Ridge（RBF 核）** 拟合；高斯核 **长度尺度 \(\sigma\)**（与 LOOCV 网格）由 **留一交叉验证（LOOCV）** 择优；**\[实现时见代码\]** 将拆分为：

- **`train_kbc_kernel_ridge.py`**：仅承载 **可被 `training_tab` / 同包逻辑直接调用的训练与推理接口**（接收 `TrainingDatasetModel`、落盘路径、返回结构化误差等），**不包含**命令行解析与磁盘训练目录加载。
- **`run.py`**：命令行入口；负责 **`--data-dir`** 等与训练 JSON 文件夹相关的导入，再调用 `train_kbc_kernel_ridge.py` 中的训练/预测能力。

数学与网格约定与下文一致。

---

## 1. 数据与对齐

每条样本与同仓库 **`desktop/training_tab/training_dataset_model.TrainingExperimentRecord`** 对齐：

- `equivalent_kg_tnt`：炸药当量（kg TNT）
- `al_percent`：含铝量（%）
- `K, B, C`：`drag_fit` 数值

**Python 训练接口**从 **`TrainingDatasetModel`**（已填充的 `records`）读取样本；模型中的其它字段（如 `data_folder`）可选写入 **`manifest`** 以供追溯。**命令行**路径下由 **`run.py`** 使用与桌面 **`dataset_io.import_training_folder`** 相同的目录扫描规则载入数据，构造或传入与上述记录等价的数据集合后再训练。

---

## 2. 特征构造

记 \(\mathrm{al_\%}\) 为**含铝量的百分数取值**（如 UI/JSON 中 `30` 表示 **30 %**，先化为小数再参与运算）。第二维为先乘质量分数后与当量的乘积，在 \(\mathrm{equiv}\) 与装药总质量 (kg TNT 当量口径) **一致** 的约定下，即为**炸药配方中铝粉的质量 (kg)**：

\[
X = (\,x_1,\, x_2\,) =
\bigl(\mathrm{equiv},\;\; \mathrm{equiv}\times (\mathrm{al_\%}/100)\bigr).
\]

**推理**：输入标量当量与含铝百分比后，同上式得到 \(X\)，分别送入 \(K,B,C\) 三个 `KernelRidge` 模型。

**兼容**：若以旧公式 \(\mathrm{equiv}\times\mathrm{al_\%}\)（百分数未除 100）训练的 artifact 仍存在，其与当前实现 **不兼容**；请用本约定 **重新训练** 后再推理。

---

## 3. `KernelRidge` 与正则

**核（数学约定）**采用高斯径向形式，长度尺度记为 **\(\sigma>0\)**：

\[
k(x,x')=\exp\!\left(-\frac{\|x-x'\|^2}{2\sigma^2}\right).
\]

**`sklearn.kernel_ridge.KernelRidge`** 的 ``kernel='rbf'`` 使用等价写法

\[
k(x,x')=\exp\!\left(-\gamma_{\mathrm{sk}}\|x-x'\|^2\right),
\qquad
\gamma_{\mathrm{sk}}=\frac{1}{2\sigma^2}.
\]

实现中 **网格上扫描的是 \(\sigma\)**，再换算为 ``KernelRidge(..., gamma=1/(2*sigma**2))``。

- **`alpha`**（岭正则）：**训练接口参数 `alpha`** 约定为 **`None` 时使用数值 `10^{-3}`**；若传入浮点数则覆盖（与 sklearn ``KernelRidge`` 构造函数默认 `alpha=1.0` **不同**，本仓库 deliberate 压低默认正则）。
- **\(\sigma\)**：由下节网格 + LOOCV 选取；非 RBF 相关参数不参与。

对每个目标 **\(K\)、\(B\)、\(C\)** 各拟合一台模型。

---

## 4. \(\sigma\) 网格与 LOOCV

令训练集当量为 \(e_{\min}, e_{\max}\)，

\[
\mathrm{stride} = \frac{e_{\max} - e_{\min}}{30}.
\]

退化情形（极差 \(\approx 0\)）下文略，由实现中取 **`stride = 1`** 等容错。

\[
\sigma_i = 1 + i\cdot \mathrm{stride},\quad i=0,1,\ldots,30 \quad\text{（共 31 个候选）}.
\]

（**与旧版实现中误把 ``sklearn`` 的 ``gamma`` 当作超参时，网格上使用的数值相同**；现改正为 **\(\sigma\)**，并据上式换算 ``gamma``。）

对每个 \(\sigma_i\) 与每个目标 \(y\)：

1. LOO：**\(n\)** 折，每折在 **\(n-1\)** 条上训练、在留出的一条上得分。
2. **测试侧 LOOCV MSE**：\(\mathrm{MSE}_{\mathrm{test}}(\sigma)=\frac{1}{n}\sum_j (y_j-\hat y_j^{(-j)})^2\)。
3. **训练侧 LOOCV MSE（折内）**：每折仅在 **\(n-1\)** 条训练点上算拟合误差并平均，再在折上对外层平均——记为 \(\mathrm{MSE}_{\mathrm{train}}(\sigma)\)。

### 4.1 \(\sigma\) 下限（相对旧版的改动）

**旧策略（2026-09 之前 artefact）**：在全部 31 个候选 \(\sigma_i\) 上取

\[
\sigma^\* = \arg\min_{\sigma_i}\ \mathrm{MSE}_{\mathrm{test}}(\sigma_i).
\]

**现策略**：在比较 LOOCV 测试误差前，先算特征空间几何尺度，并禁止过小的 \(\sigma\) 参与择优。

记训练特征矩阵 \(X\in\mathbb{R}^{n\times 2}\)（见第 2 节），所有样本对的欧氏距离为 \(\{d_{ab}\}\)，

\[
d_{\mathrm{med}} = \mathrm{median}_{a<b}(d_{ab}),\qquad
\sigma_{\mathrm{floor}} = \max\bigl(1,\; 0.25\, d_{\mathrm{med}}\bigr).
\]

对每个候选 \(\sigma_i\)：

\[
\widetilde{\mathrm{MSE}}_{\mathrm{test}}(\sigma_i)=
\begin{cases}
\mathrm{MSE}_{\mathrm{test}}(\sigma_i), & \sigma_i \ge \sigma_{\mathrm{floor}},\\[4pt]
+\infty, & \sigma_i < \sigma_{\mathrm{floor}}.
\end{cases}
\]

\[
\sigma^\* = \arg\min_{\sigma_i}\ \widetilde{\mathrm{MSE}}_{\mathrm{test}}(\sigma_i).
\]

**要点**：

- **K、B、C 共用同一 \(\sigma_{\mathrm{floor}}\)**（同一套 \(X\)），但 **各自独立** 选 \(\sigma^\*\)（三个目标的最优 \(\sigma\) 可以不同）。
- **候选网格公式未改**（仍只由当量 \(e_{\min},e_{\max}\) 生成）；改的是 **谁有资格成为 \(\sigma^\*\)**。
- \(\sigma_{\mathrm{floor}}\)、\(d_{\mathrm{med}}\)、最近邻距离 \(d_{\min}\) 写入 **`manifest.json`**，便于排查。

**动机（C 预测坍缩为 0）**：当 \(\sigma\) 远小于样本间距时，查询点处 RBF 权重 \(\approx 0\)，`KernelRidge` 外推 \(\hat C\approx 0\)。拖曳式 \(D(t)=K(1-B e^{-C_{\mathrm{eff}} t^2})\) 在 **\(C=0\)** 时退化为常数 **\(D\equiv K(1-B)\)**，与渐近 **\(K\)** 不一致。旧 LOOCV 对 **量级 \(\sim 10^{-4}\)** 的 **\(C\)** 常误选 **\(\sigma=1\)**（留一法上“预测成 0”的 MSE 仍很小）。

**示例（4 条训练、当量约 475–2000 kg）**：\(d_{\mathrm{med}}\approx 796\) → \(\sigma_{\mathrm{floor}}\approx 199\)；**C** 的 \(\sigma^\*\) 由 **1** 变为 **约 200** 量级，1000 kg 插值处 **\(\hat C\)** 恢复为 **\(10^{-4}\)** 量级而非 0。

### 4.2 **\(C\)** 过拟合抑制（相对 **\(K,B\)** 的额外约定）

**\(C\)** 量级小（\(10^{-5}\sim10^{-3}\)）、样本少时，LOOCV 在 **\(\sigma\)** 网格上仍可能选出 **过局部** 的核，训练 Tab 在当量 100 点网格上画 **\(C\)** 曲线时会出现 **训练点之间的深谷/尖峰**（过拟合），而非物理上平滑的随当量变化。

实现上对 **仅目标 C** 增加：

| 项 | **\(K,B\)** | **\(C\)** |
|----|------------|-----------|
| 岭正则 **`alpha`** | 默认 **`10^{-3}`** | **`10^{-2}`** |
| **\(\sigma\)** 下限 | **`max(1, 0.25\, d_{\mathrm{med}})`** | **`max(上述, 0.5\, d_{\mathrm{med}}, \sigma^\*_K)`**（先训 **\(K\)** 再训 **\(C\)**） |
| 拟合域 | 原值 | **`y = log10(max(C, \varepsilon))`**，预测 **`10^{\hat y}`** |

**`joblib`** 中 **\(C\)** bundle 含 **`y_transform: "log10"`**；旧 artefact 无此字段则按线性 **\(C\)** 解读。

### 4.3 择优与落盘

以 **\(\widetilde{\mathrm{MSE}}_{\mathrm{test}}\)** 最小对应的 \(\sigma^\*\) 为最终 \(\sigma\)（平局取实现规定的第一个）；落盘时在 ``manifest`` 中同时写明 **``sklearn_rbf_gamma = 1/(2(\sigma^\*)^2)``** 以便核对。**\(C\)** 另记 **`alpha_C`**、**`C_y_transform`**。

---

## 5. 落盘目录约定

给定调用方传入的 **`model_path`**（表示**父目录**或「模型输出根路径」语义），单次训练创建的**artifact 根目录**为：

\[
\texttt{\{model\_path\}/kernel\_regression\_\{timestamp\}}
\]

其中 **`\{timestamp\}`** 为该次训练生成的唯一时间戳字符串（格式由实现确定，便于避免覆盖）。

**该子目录内需包含**：

- **`K` / `B` / `C`** 三套持久化模型（如 `joblib`），文件名由实现约定（与旧版脚本中 `kbc_krr_K.joblib` 等可对齐或可重命名，以代码为准）。
- 各目标的 **LOOCV 误差表**（CSV 首列为 **`sigma`**、折内平均训练 MSE、折外平均测试 MSE），用于描 **\(\sigma\)**–误差曲线（**旧 artefact** 可能仍为列名 **`gamma`**，实为历史误用下的 ``sklearn`` ``gamma``，作图脚本可兼容读取）。
- **`manifest.json`**（或其它摘要文件）：\(n\) 样本、选用的 **`alpha`**、**`sigmas`** 候选列表、**`rbf_parameterization`** 说明、**`min_pairwise_feature_distance`**、**`median_pairwise_feature_distance`**、**`sigma_loocv_floor`**、每个目标 **`best_sigma`** / **`sklearn_rbf_gamma`**、最佳 LOOCV 测试误差、相对路径文件名等。
- 各目标 **`joblib` bundle**（新 artefact）：除 `model` 外可含 **`X_train`**、**`y_train`**，供 **\(C\)** 预测兜底（见第 6.2 节）。**旧 artefact** 无此字段时仅依赖仿真侧回退。

**读取已训练结果进行预测时**，应使用指向 **上述 `kernel_regression_{timestamp}` 目录**的路径（或与实现一致的「模型 bundle 路径」）。

---

## 6. Python 接口约定（供 `training_tab` / `kernel_regression` 调用）

以下描述 **`train_kbc_kernel_ridge` 模块**对外暴露的稳定契约（具体函数名以实现为准）。

### 6.1 训练

**入参**

| 名称 | 说明 |
|------|------|
| **`training_model`** | `TrainingDatasetModel` 实例：`records` 非空，`TrainingExperimentRecord` 字段齐备。 |
| **`alpha`** | `float \| None`。`None` 表示使用 **`10^{-3}`**。 |
| **`model_path`** | 字符串或 **`pathlib.Path`**：父目录路径；其子目录 **`kernel_regression_{timestamp}`** 由实现创建并写入三套模型及误差/manifest。 |

**返回（二元组或可命名元组）**

1. **`saved_root`（`Path` / `str`）**  
   实际写入的根目录：**`\{model_path\}/kernel_regression_{\{timestamp\}}`** 的解析路径，供桌面侧保存「最近一次核回归会话」之用。

2. **`errors_by_target`**  
   **结构化**的各个模型、各个 **\(\sigma\)** 上的误差，便于 UI 或直接绑图。**建议最小结构**示例（实现可等价使用 `dataclass` / `TypedDict`）：

```text
{
  "K": { "sigma": float[], "train_mse": float[], "test_mse": float[] },
  "B": { "sigma": ..., "train_mse": ..., "test_mse": ... },
  "C": { "sigma": ..., "train_mse": ..., "test_mse": ... }
}
```

三者 **`sigma`** 序列相同（由同一 \(\sigma\) 网格决定）；向量长度均为候选个数（31）。

### 6.2 预测

**入参**

| 名称 | 说明 |
|------|------|
| **`model_path`** | 指向**某次训练产出的 **`kernel_regression_{timestamp}` 目录**（含 `K,B,C` 模型文件）；非父级 `model_path`。 |
| **当量、含铝百分比** | 标量：**当量** (kg TNT)、**含铝量**数值为百分数刻度（30 表示 30 %）；拼成的 \(x_2=\mathrm{equiv}\times(\mathrm{al}/100)\)。 |

**返回**

- **`K, B, C`**：三个浮点预测。**\(K,B\)** 直接来自对应 `KernelRidge.predict`。**\(C\)** 见下节兜底。

**\(C\) 预测兜底（`predict_kernel_regression_kbc`）**

1. 先用 **\(C\)** 模型 **`predict`** 得 \(\hat C\)。
2. 若 **\(\hat C > 0\)**：直接返回。
3. 若 **\(\hat C \le 0\)** 且 bundle 含 **`X_train`/`y_train`**：用与 RBF 一致的高斯核 **逆距离加权（IDW）** 再估一次，核宽度  
   \(\sigma_{\mathrm{fb}} = \max(\sigma^\*,\; 0.25\, d_{\mathrm{med}},\; 1)\)（\(\sigma^\*\) 为该 **\(C\)** 模型 LOOCV 选中的 \(\sigma\)）。
4. 仍 **\(\le 0\)** 或无训练缓存：**原样返回**（由仿真层处理，见第 8 节）。

---

## 7. 命令行（`run.py`）

CLI **仅**存在于 **`kernel_regression/run.py`**：**解析参数、加载 `--data-dir`、拼装 `TrainingDatasetModel` 或等价记录列表**，再调用 **`train_kbc_kernel_ridge` 的实现**。

在 **`source` 目录**下（与工作目录惯例一致），约定为：

```bash
python kernel_regression/run.py train \
  --data-dir /path/to/training_json_dir \
  --out-dir ./krr_outputs

# 训练结束时顺带生成 σ～LOOCV 误差图（与 graph.py loocv 等价；可用 -graph）
python kernel_regression/run.py train \
  --data-dir /path/to/training_json_dir \
  --out-dir ./krr_outputs \
  --graph

python kernel_regression/run.py predict \
  --model-dir ./krr_outputs/kernel_regression_<timestamp> \
  --equiv 10 \
  --al-percent 30
```

说明：

- **`train`**：`--out-dir` 语义上对应 **`model_path`**（父目录）；**stdout** 仅一行输出 **`saved_root`**（`kernel_regression_<timestamp>`），便于脚本解析； **`--graph` / `-graph`**：成功后调用 **`plot_loocv_gamma_curves`**（文件名未改），横轴为 **\(\sigma\)**（新 CSV）或 **`gamma`** 列（旧 artefact）；PNG 绝对路径另以 **`[graph] <path>`** 写入 **stderr**。
- **`predict`**：`--model-dir` 必须为**含三套模型 artefact** 的那一层目录。

---

## 8. 仿真侧兜底（`model_tab` / `ModelController`）

当量仿真若已加载 KRR artefact，直径走 **`diameter_drag_series(K,B,C)`**（显式拖曳）。若 **`predict_kernel_regression_kbc`** 返回 **\(C\le 0\)**（常见于 **未重训的旧 artefact**、**\(C\)** 模型 **\(\sigma^\*=1\)**）：

- **`ModelController.resolve_kbc_for_simulation`** 保留 KRR 的 **\(K,B\)**，将 **\(C\)** 替换为 **`FireballCalculator` 材料标准 `C` 按当量比 \(M\)** 缩放：\(C_{\mathrm{eff}} = C_{\mathrm{std}}/M\)（与非 KRR 计算器路径一致）。
- 控制台打印警告，便于与 **manifest** 中 **`targets.C.best_sigma`** 对照。

**建议**：修改 \(\sigma\) 策略或更换训练集后 **重新训练** 并导入新 **`kernel_regression_{timestamp}`**；仿真兜底仅保证直径随时间变化，**不保证**与 LOOCV 最优 **\(C\)** 一致。

---

## 9. 依赖

需 **`numpy`、`matplotlib`、`joblib`、`scikit-learn`**（参见仓库 **`requirements.txt`**）。

---

## 10. 小样本说明

\(n&lt;2\) 时无法进行 LOOCV，实现应报错并提示。**极少样本下「最优 \(\sigma\)」不确定性大**，仅作工程上的折中选取；增量数据后可再次训练更新时间戳目录。

---

## 11. 绘图（`graph.py`）

脚本 **`kernel_regression/graph.py`** 提供两组能力（亦可在 Python 中 `import kernel_regression.graph` 调用同名函数）：

### 10.1 训练后与 \(\sigma\) 有关的误差曲线

单次训练 artefact 目录内已写有 **`kbc_krr_loocv_K.csv` / `B` / `C`**。**新版**列为 **`sigma,train_mse_loocv_mean,test_mse_loocv_mean`**（**`sigma`** 即核长度尺度）。**旧版**列为 **`gamma`**（值为当时误用的 ``sklearn`` RBF ``gamma``）。作图方式为 **对每个目标一行子图**，同一子图叠加 **训练 MSE、测试 MSE** ～ 横轴超参。**未指定 `--out` 时**，PNG **`kbc_gamma_loocv_curves.png`** 与 **`kbc_krr_*.joblib`/CSV 同目录**；**`--out` 为相对路径**则相对于 **`--model-dir`**；**绝对路径**则按其保存。

在 **`source` 目录**下示例：

```bash
python kernel_regression/graph.py loocv \
  --model-dir ./krr_outputs/kernel_regression_<timestamp>

# 相对文件名（写入该 artefact 目录下）
python kernel_regression/graph.py loocv \
  --model-dir ./krr_outputs/kernel_regression_<timestamp> \
  --out my_loocv.png
```

**注意**：本条与「训练中自动作图」的衔接：使用 **`python kernel_regression/run.py train … --graph`**（或 `-graph`）可在训练收尾时直接生成与同目录 **`kbc_gamma_loocv_curves.png`**。否则训练逻辑本身只写 CSV/Manifest；在训练流水线或 Jupyter 内也可在完成 `train_kernel_regression_kbc` 后调用 **`plot_loocv_gamma_curves(saved_root)`**，或另跑 **`graph.py loocv`**。

---

## 12. 批量预测与 \(K,B,C\)～当量 曲线（`run_test.py`）

脚本 **`kernel_regression/run_test.py`** 对已落盘的模型：**固定含铝（默认 \(30\\%\)**）**，令当量依次为 **\(1,2,\ldots,150\)（可调范围）**，逐点调用 **`predict_kernel_regression_kbc`**，并将 **\(K,B,C\) 随当量** 绘制为三张纵向子图。**默认 PNG 与模型同目录**：**`kbc_vs_equivalent_al30.png`**；**`--out` 逻辑与 §10 相同**（相对路径相对 `--model-dir`）。

示例（文件名以 **`run_test.py`** 为准；勿与 `run_text` 混淆）：

```bash
python kernel_regression/run_test.py \
  --model-dir ./krr_outputs/kernel_regression_<timestamp>
```

若在 **`kernel_regression/`** 子目录内启动，可自行调整相对路径，例如 **`--model-dir ../krr_outputs/kernel_regression_<timestamp>`**。

可选：`--equiv-min`、`--equiv-max`、`--al-percent`、`--out`（不写则仍为 artefact 内默认文件名；相对 `--model-dir`）。

---

## 13. `matplotlib`

作图脚本使用 **`Agg`** 后端，无需显示设备即可写 PNG。
