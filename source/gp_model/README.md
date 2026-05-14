# `gp_model`：火球 K/C 双单任务 GP 训练 / 推理 / 曲线绘制

实现说明与契约见项目内设计文档：[`document/fireball_gp_mogp_module_design.md`](../../document/fireball_gp_mogp_module_design.md)。**默认建模策略**（**\(K\)**、**\(C\)** **各一**独立单输出 GP、\(B\) 为训练集均值、原坐标 ARD-RBF）：[`document/gp_fireball_kc_lmc_strategy.md`](../../document/gp_fireball_kc_lmc_strategy.md)。训练产物 **`schema_version: 3`**；旧版 **MOGP**（单一 `model_state_dict`）须重新 `train`。

## 依赖环境

- **Python 3.10+**，**PyTorch**、**GPyTorch**、**Matplotlib**、**NumPy** 等（与 `source/requirements.txt` 一致）。
- 建议在仓库中执行：

```bash
bash source/setup.sh
conda activate fireball_calculator
```

## 命令行入口（CLI）

**统一使用包内脚本 `source/gp_model/gp_cli.py`**：启动前会把 `source/` 加入 `sys.path`，**无需**手动设置 `PYTHONPATH`，也**不必**先 `cd` 到 `source`。

下文命令中 **`python source/gp_model/gp_cli.py`** 均相对于**仓库根目录**（`fireball_calculator/`）。若使用绝对路径，将 `source/gp_model/gp_cli.py` 换为  
`/path/to/fireball_calculator/source/gp_model/gp_cli.py` 即可。

**等价写法**（当前目录在 `source/gp_model/` 时）：

```bash
python gp_cli.py train --data-dir ../../training_data
```

**常见错误**：在 `source/gp_model/` 下执行 `python -m gp_model.cli` 会报 `No module named 'gp_model'`（默认 `sys.path` 不含 `source/`）。请改用上面的 **`gp_cli.py`**，或先 `cd` 到 `source` 再执行 `python -m gp_model.cli`，或设置 `PYTHONPATH=.../source`。

**可选**：`bash source/gp_model/run_cli.sh ...` 与 `gp_cli.py` 等价（内部为 `python -m gp_model.cli`）。

## 子命令说明

子命令：`train` / `predict` / `plot` / `sweep-k`。命令形式均为：

```text
python source/gp_model/gp_cli.py <子命令> ...
```

### `train`：扫描训练 JSON，分别训练 **K-GP** 与 **C-GP**，写出超参数快照（schema 3）

- **输入**：`--data-dir` 指向含 `*.json` 的目录（每条 JSON 需含 `parameters.equivalent`、`parameters.al_percent` 与 `drag_fit` 中 `success=true` 时的 `K,B,C`）。
- **可选**：`--config` 超参数 JSON（省略则用内置默认：含 **`covariance_K`** / **`covariance_C`**；仅有 legacy **`covariance`** 时会复制到二者）。
- **输出**：默认写入 **`<data-dir>/trained_hyperparams.json`**；也可用 `--out-hyperparams` 指定路径。
- **建模约定**（`gp_fireball_kc_lmc_strategy.md`）：**\(K\)**、**\(C\)** **分别**为单任务 `ExactGP` + `GaussianLikelihood`；**\(B\)** 在产物中记 **`b_mean`**。**\(X\)、\(K,C\)** 均**不做**仿射标准化；核为 **RBF + ARD**（各任务一套超参）。
- **`covariance_K` / `covariance_C` 与观测噪声**：训练前由 `build_covariance_kc_from_dataset` 根据 **X 各维跨度**与 **K、C 列方差**补全（含 `length_scale_init_span_floor`，并写入训练产物）。默认 **`covariance_autoscale`**：各任务 `σ²_max = 0.5×var(标签)`、`σ²_init = 0.1×var`（`"covariance_autoscale": {"enabled": false}` 关闭）。另可手写 `observation_noise_*` 或 `observation_noise_*_times_training_{k,c}_var`。

```bash
# 在仓库根目录执行
python source/gp_model/gp_cli.py train --data-dir training_data
python source/gp_model/gp_cli.py train --data-dir training_data --config my_hyperparams.json
python source/gp_model/gp_cli.py train --data-dir training_data --out-hyperparams training_output/trained_hyperparams.json
```

### `predict`：加载训练产物，对新 `X_*` 预测 **K、C**（**B** 恒为 **`b_mean`**），并默认画直径–时间曲线

- **必选**：`--hyperparams`（`train` 生成的 JSON）。
- **`--out-dir`**：写入 `predict_result.json` 与同目录下的 PNG；**默认 `training_output/`**（相对当前工作目录）。
- **输入 X**：二选一  
  - `--equivalent` 与 `--al-percent`（单点），或  
  - `--x-json` 指向 `x_star.json`（单点、或 `{ "points": [ {...}, ... ] }`）。
- **可选**：`--no-plot` 只写 JSON、不画图。

```bash
# 默认写入 training_output/
python source/gp_model/gp_cli.py predict \
  --hyperparams training_data/trained_hyperparams.json \
  --equivalent 1 --al-percent 40

python source/gp_model/gp_cli.py predict \
  --hyperparams training_data/trained_hyperparams.json \
  --out-dir training_output \
  --x-json x_star.json

python source/gp_model/gp_cli.py predict ... --no-plot
```

### `plot`：不跑 GP，仅根据已有 K、B、C 与 `plot` 配置出图

- **必选**：`--hyperparams`（含 `plot` 段）、`--kbc-json`、`--out`（输出 PNG 路径）。

```bash
python source/gp_model/gp_cli.py plot \
  --hyperparams training_data/trained_hyperparams.json \
  --kbc-json kbc.json \
  --out training_output/diameter_vs_time.png
```

`kbc.json` 至少含 `K`、`B`、`C`；可选 `equivalent`、`al_percent` 用于图题（见设计文档 §5.3）。

### `sweep-k`：固定含铝量，沿当量扫掠 **K、C**（及常数 **B=b_mean**）并出图

在 **含铝量** 固定（默认 **40%**）时，对 **当量** 在 `[equivalent-min, equivalent-max]` 内均匀取 `num-points` 个点，用训练产物做**两路** GP 推理。**K、C** 各有后验均值与不确定性（**边缘独立**）；**B** 恒为 **`b_mean`**。

- **输出目录** `--out-dir`（**默认 `training_output/`**）下生成 **`kbc_vs_equivalent.csv` / `.json` / `.png`**（三行子图 **K、B、C**；**B** 为水平线，±2σ 为 0）。

```bash
# 含铝量 40%，当量 1～160 kg（默认 160 个采样点）；默认输出到 training_output/
python source/gp_model/gp_cli.py sweep-k \
  --hyperparams training_data/trained_hyperparams.json
```

---

## Python API（库方式调用）

不依赖 GPyTorch 的导入（仅数据与配置）：

```python
from gp_model import (
    load_training_dir,
    default_hyperparams,
    load_hyperparams_json,
    merge_hyperparams,
    DEFAULT_TRAINED_FILENAME,
)
```

训练 / 推理在 **`gp_model.train_infer`**（需已安装 `gpytorch`）：

```python
from gp_model.data_input import load_training_dir
from gp_model.config import merge_hyperparams, load_hyperparams_json
from gp_model.train_infer import train_mogp, predict_mogp
from gp_model.config import load_trained_artifact

ds = load_training_dir("training_data")
hp = merge_hyperparams(load_hyperparams_json("my.json"))  # 或 default_hyperparams()

trained = train_mogp(ds, hp)  # 等价于 train_dual_gp；先后优化 K 与 C
# 落盘请用 train_infer.run_train_cli（见 cli.py）或自行调用 save_trained_artifact（四套 state_dict）

art = load_trained_artifact("training_data/trained_hyperparams.json")
import numpy as np
X_star = np.array([[1.0, 40.0]], dtype=np.float64)
pred = predict_mogp(art, X_star)  # K、C 为独立边缘；B 列均值= b_mean，方差=0
```

直径曲线在 **`gp_model.curve_plot`**：

```python
from gp_model.curve_plot import PlotConfig, plot_diameter_curve, save_figure
from pathlib import Path

cfg = PlotConfig(time_ms_min=0.0, time_ms_max=75.0, num_points=300)
fig = plot_diameter_curve(4.7, 0.56, 0.002, cfg, equivalent=1.0, al_percent=40.0)
save_figure(fig, Path("training_output/diameter_example.png"))
```

---

## 包内文件一览

| 模块 | 作用 |
|------|------|
| `data_input.py` | 读训练目录、`x_star.json` |
| `config.py` | 默认/合并超参、训练产物 JSON 读写 |
| `train_infer.py` | GPyTorch：K-GP / C-GP 训练与预测 |
| `curve_plot.py` | \(D(t)=K(1-B e^{-C t^2})\) 绘图 |
| `cli.py` / `__main__.py` | 命令行实现（由 `gp_cli.py` 调用） |
| `k_sweep.py` | 当量扫掠 K/B/C 的预测、CSV/JSON、单图三子图 |
| `gp_cli.py` | **推荐 CLI 入口**：自动设置 `sys.path`，可在仓库根用 `python source/gp_model/gp_cli.py` |

---

## 常见问题

- **ImportError: No module named 'gpytorch'**：先完成 `source/setup.sh` 或 `pip install -r source/requirements.txt`。
- **`No module named 'gp_model'`**：不要用 `python -m gp_model.cli`（除非已 `cd source` 或设置 `PYTHONPATH`）。请使用 **`python source/gp_model/gp_cli.py`**（仓库根）或 **`python gp_cli.py`**（在 `source/gp_model/` 下）。
- **训练样本过少**：精确 GP 建议 **n≥2**；单条样本可跑但数值可能不稳定。
- **旧版 MOGP 产物无法 `predict`**：`load_trained_artifact` 若见单一 `model_state_dict` 会报错，请重新 `train`。
- **`sweep-k` 中 K 几乎为常数**：多与训练点稀疏、长度尺度相对当量过小有关；已默认按数据跨度初始化 \(\ell\)；仍建议补数据或调 `covariance_K` / `covariance_C`。

更细的 JSON 字段与文件命名，以 **`fireball_gp_mogp_module_design.md`** 为准。
