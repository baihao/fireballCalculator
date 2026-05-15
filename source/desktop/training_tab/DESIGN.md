# 模型训练标签页（`training_tab`）软件设计

本文档描述在 `source/desktop/training_tab` 中实现「模型训练」桌面 UI 的**架构与职责划分**，对齐 `document/machine_vision_ui_prototype.html` 中 **「模型训练」** 标签页的布局与交互意图；**软件组织形式参考** `source/desktop/extract_tab`。  
**范围说明**：以**当前代码状态**为准做设计与验收说明；与 HTML 原型的差异见 §2 备注。

---

## 1. 背景与目标

- **产品来源**：HTML 原型将「模型训练」作为独立顶栏标签，左侧为操作与参数，右侧为图表区 + 训练日志（原型曾为四图网格；**当前实现为三张散点图一行排列**，无内嵌「训练曲线」图）。
- **桌面目标**：在 PySide6 应用中提供等价信息架构与控件集合，便于后续接入真实数据加载、划分（**留一交叉验证**）、训练与图表刷新。
- **约束**：首版实现可保留「占位 + 示意数据」路径，但目录与类职责应按本设计拆开，避免在单文件中堆叠 UI 与业务逻辑。

---

## 2. 原型对齐清单

以下与 `machine_vision_ui_prototype.html` 中 `pane-train-aside` / `pane-train-content` 一一对应，作为验收检查表。

### 2.1 左侧栏（`aside`）

| 分组（原型 legend） | 控件 | 行为要点（**当前实现**） |
|---------------------|------|--------------------------|
| **输入数据** | 主按钮「输入数据」+ 目录选择 | 选择**包含多组实验 JSON** 的文件夹（解析约定见 `utils/dataset_io.py`）；状态行显示已载入样本数与目录名。说明文案：**数据需大于 5 条，否则难以获得较好训练效果**。 |
| **模型训练** | 算法下拉 | 选项：`核回归`、`高斯过程`；更新模型与顶部「算法：…」状态、侧栏概要中的模型名称（**已无第四张训练曲线图**，无需切换曲线类型）。 |
| **划分与执行** | 「划分策略」下拉 | **当前唯一选项**：`留一交叉验证`（`TrainingDatasetModel.split_strategy = loocv`）。已摒弃原型的「测试集比例 10%～40%」。 |
| **划分与执行** | 按钮「开始训练」 | **核回归**时：调用 `utils/krr_workflow.run_train_and_predict` → `kernel_regression.train_kernel_regression_kbc`（LOOCV）；模型 artefact 写入与训练数据目录**同级**的 `kr_model/kernel_regression_<timestamp>/`。成功后按当量范围 100 点、含铝范围 3 档预测 K/B/C，在三张散点上**叠加三条彩色曲线**（每档含铝一色）。高斯过程未接入时弹窗提示。 |
| **训练数据集信息** | 只读多行文本（概要） | 展示：训练模型、数据目录、总样本数、**划分策略（LOOCV）**及一句 LOOCV 说明；若样本数 ≤5 则追加**提示**；含 B 均值辅助信息与样本明细（截取）。**不再**展示按比例划分的训练/测试条数。 |

### 2.2 右侧主区（`content`）

| 区域 | 内容 |
|------|------|
| 标题栏 | 「模型训练视图」+ 当前算法状态（如「算法：核回归」）。 |
| 说明文案 | 短 hint：三张散点含义、点大小与含铝量、划分策略为 LOOCV 等。 |
| 图表区（**1×3**） | ① K–当量 ② B–当量 ③ C–当量；`FireballTrainingScatterChart`。导入后仅散点；**核回归训练完成后**在同图叠加 3 条预测曲线（含铝 min～max 三等分，当量 min～max 百等分；颜色固定橙/紫/绿）。 |
| 训练日志 | 只读文本区，追加「输入数据」「开始训练」等事件。 |

### 2.3 与已有 `chart_widgets` 的关系

- **三张散点**：`FireballTrainingScatterChart`，`update_data(..., curves=[(x,y,color),…])` 可选叠加折线；训练成功后由 `TrainingChartController.redraw_scatters_with_prediction_curves` 写入核回归预测曲线。
- **训练曲线部件**：包内仍保留 `KernelRegressionTrainingCurveChart`，**本标签页不再挂载**；需要时由其它入口或后续版本使用。

---

## 3. 参考模块：`extract_tab` 组织方式

`extract_tab` 的惯例（`training_tab` 应与之同构）：

| 部分 | 职责 |
|------|------|
| **`extract_tab.py`** | 继承 `QWidget` 的标签页入口；装配 `*TabUI`、模型/管理器、各 `Controller`；连接信号槽、初始化图表、对外暴露少量兼容属性。 |
| **`ui_widgets/extract_tab_ui.py`** | `ExtractTabUI`：纯 UI 搭建（布局、GroupBox、控件创建），通过 `get_ui_components()` 返回 **字典** 供控制器与主类引用。 |
| **`controllers/`** | 侧重 **与 UI 强相关** 的交互：如图表刷新、控件状态联动、可见性/重绘等；**不**承载「读盘写盘、调用训练后端」这类与界面弱耦合的能力。 |
| **`utils/`** | 无界面工具与 **应用–数据/模型** 的中间层：序列管理、导入导出、与训练模型通信等（与 `extract_tab` 一致，能下沉则下沉）。 |
| **领域模型**（如 `sequence_model.py`） | 与 UI 解耦的状态与操作封装；`TrainingTab` 与 `utils`、`controllers` 共享。 |

`training_tab` 采用相同分层：**一个主 Tab 类 + 一个 UI 构建器 +（少量）UI 向控制器 + utils +（推荐）轻量模型类**。其中 **训练数据导入/导出**、**应用与训练 model 之间的连接** 放在 `utils`，**不**单独设 `training_data_controller` / `training_run_controller`。

---

## 4. 建议目录结构

```
source/desktop/training_tab/
├── DESIGN.md                 # 本设计文档
├── __init__.py               # from .training_tab import TrainingTab; __all__
├── training_tab.py           # TrainingTab(QWidget)：装配、信号槽、调用 utils 与 ChartController
├── training_dataset_model.py # 会话状态：含 `last_krr_artifact_root`（最近核回归 artefact 路径）等
├── ui_widgets/
│   ├── __init__.py
│   └── training_tab_ui.py    # TrainingTabUI：create_sidebar_widget + 标签页正文（图表/日志）；与 ExtractTabUI 侧边栏拆分方式一致
├── controllers/
│   ├── __init__.py
│   └── training_chart_controller.py   # 三张散点 refresh；`redraw_scatters_with_prediction_curves` 叠加 KRR 曲线
└── utils/
    ├── __init__.py
    ├── dataset_io.py         # 训练数据导入
    ├── krr_workflow.py       # 核岭回归：`kr_model` 路径、`train_kernel_regression_kbc`、预测网格 → K/B/C 矩阵
    └── training_bridge.py    # 预留：其它后端/GP 等
```

说明：

- **`training_dataset_model.py`**：建议保留，对齐 `extract_tab` 中 `sequence_model` 与界面之间的缓冲层。
- **`utils/krr_workflow.py`**：**无 Qt**。将 `TrainingDatasetModel` 传入 `kernel_regression.train_kbc_kernel_ridge.train_kernel_regression_kbc`；`model_path` 为 ``<训练数据父目录>/kr_model``；训练后在 **[当量_min, 当量_max]** 上 **100** 点、**[含铝_min, 含铝_max]** 上 **3** 档调用 `predict_kernel_regression_kbc` 填满 K/B/C 矩阵。
- **`utils/training_bridge.py`**：预留；核回归主路径已走 `krr_workflow`。

---

## 5. UI 架构（Qt 布局）

**与机器视觉模块一致**：**全局左侧窄栏（`SidebarWidget`）**挂载操作面板；**标签页正文**仅占中间区域图表与日志。**不**在 `TrainingTab` 内再放操作区 `QSplitter` 左列。

### 5.1 全局左侧栏（≈260～300px，`framework` 已与顶栏并排）

- 由 `TrainingTab.get_sidebar_widget()` 返回的根控件（通常为 `TrainingTabUI.create_sidebar_widget()` 产出的外层 `QGroupBox`）。
- **内容**：纵向 `QScrollArea`（可选）+ 多个 `QGroupBox`，分组与原型一致：**输入数据**、**模型训练**（算法）、**划分与执行**、**训练数据集信息**。
- **切换**：主窗口 `QTabWidget` 切至「模型训练」时，`set_sidebar_content(self.training_tab.get_sidebar_widget())`，与 `ExtractTab.get_sidebar_widget()` 用法对齐。

### 5.2 标签页正文（`TrainingTab`）

- **仅**纵向布局：**工具栏行**（「模型训练视图」+ `train-model-status` 等价 QLabel）、**hint `QLabel`**、**1×3 `QGridLayout`**（三张散点）、**训练日志** `QPlainTextEdit`。

### 5.3 样式

复用桌面应用全局 QSS / 调色板（深蓝底、accent 青色标题），不必逐像素照搬 HTML CSS。

---

## 6. 控制器职责（仅图表）

### 6.1 `TrainingChartController`

- 持有三张散点引用。
- **`redraw_scatters_from_training_model`**：仅散点。
- **`redraw_scatters_with_prediction_curves`**：散点 + 传入的 `(equiv_grid, al_levels, K/B/C 矩阵)` 生成 `curves` 三元组列表调用 `update_data`。

---

## 7. `utils` 职责（数据与训练后端）

### 7.1 `dataset_io`

- **导入**：给定路径 → 解析为规整表结构（或与 `training_dataset_model` 约定的一致结构）；错误码或异常语义清晰，便于 Tab 弹出提示与写日志。
- **导出**：将当前数据集或预处理结果写出（路径、格式与产品约定），**不涉及** QLabel/QMessageBox（由 Tab 决定是否弹窗）。
- 列映射、编码、缺省字段（当量 / 最大直径 / 初始直径 / 时间常数 / 含铝量）等实现细节都放在此模块。

### 7.2 `training_bridge`

- 封装「应用 ↔ 训练 model」：**传参（数据划分、算法类型、超参）**、**启动训练**、**读取指标与曲线数据**（及后续扩展：进度、取消）。
- 可为同步 API + 可由 `TrainingTab` 外包 `QThread`/`QRunnable`；**核心业务不依赖 Qt Widgets**。
- 与 `gp_model` CLI、Python API 或独立进程的细节隔离在本模块内部，便于单测替换。

---

## 8. `TrainingTab` 主类职责（装配 + 界面事件）

以下内容**不**再放独立 Controller，而是由 **`TrainingTab` 内槽函数**（或极少数私有助手）串联 **model + utils + `TrainingChartController`**：

| 原型交互 | Tab 槽函数职责 |
|----------|----------------|
| 「输入数据」 | `QFileDialog` 选目录 → `dataset_io.import_training_folder` → 更新 model → 刷新侧栏状态 / 概要 / `TrainingChartController.redraw_scatters_from_training_model`。 |
| 「划分策略」变化 | 更新 `model.split_strategy`（当前仅 loocv）→ 刷新概要文本。 |
| 「开始训练」 | 选「核回归」且样本 ≥2：`krr_workflow.run_train_and_predict` → 更新 `model.last_krr_artifact_root`、概要 → `redraw_scatters_with_prediction_curves`。选「高斯过程」时提示未接入。 |
| 算法下拉切换 | 更新 model 与「算法：…」文案、概要；**不再**调用控制器切换曲线视图。 |

- 实例化：`TrainingTabUI`（内含 `create_sidebar_widget` + `create_main_layout`）、`TrainingDatasetModel`、`TrainingChartController`；按需 import `dataset_io`、`training_bridge`。
- **`get_sidebar_widget()`**：提供给主窗口挂载至**全局左侧边栏**，与 `ExtractTab` 一致。
- 可选：`get_ui_components()` 供调试或外层访问。
- 避免在 Tab 内实现解析与后端协议细节——**委托给 `utils`**。

---

## 9. 数据流（概念）

```
[目录] ──► dataset_io（导入）──► TrainingDatasetModel
                                       │
TrainingTab ◄──── 用户修改划分策略 ────┤→ 概要 UI 刷新
                                       │
TrainingTab ──► krr_workflow（核回归 train + predict）──► artifact …/kr_model/kernel_regression_<ts>/
                                       │
                                       └──► TrainingChartController（散点 + 预测曲线）

导出请求 ─────► dataset_io（导出）──── （数据来自 Model）
```

---

## 10. 主程序集成（预留）

- 在承载顶栏的主窗口（与「机器视觉」`ExtractTab` 同级）增加 **「模型训练」** 页，嵌入 `TrainingTab` 实例。
- **侧边栏**：`on_tab_changed` 在切换到「模型训练」时，`set_sidebar_content(self.training_tab.get_sidebar_widget())`；实现上可与「机器视觉」「机器学习」侧边栏一并预载入同一 `sidebar_container`，互斥显示。
- 不在本文档约定主窗口文件名；实现阶段与现有 `framework`/`main` 路由对齐。

---

## 11. 非目标与后续迭代

| 非本次设计强制内容 | 说明 |
|-------------------|------|
| 真实 GP 训练 | 未接线；核回归已走 `krr_workflow` + `source/kernel_regression`。 |
| XLSX 解析 | 可依赖 pandas/openpyxl，放在 `utils/dataset_io.py`。 |
| 单元测试 | 建议对 `TrainingDatasetModel`、`dataset_io`、`training_bridge` mock；`TrainingChartController` 可做最小 smoke。 |
| 与 `gp_model` CLI 对齐 | 收口在 **`utils/training_bridge.py`**（或同级子模块），不在 `controllers/` 重复实现。 |

---

## 12. 文档修订

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.1 | 2026-05-14 | 初稿：目录、`extract_tab` 对齐、原型映射、控制器拆分 |
| 0.2 | 2026-05-14 | 取消 `training_data_controller` / `training_run_controller`；数据导入导出与训练连接层归入 `utils`；图表保留 `training_chart_controller` |
| 0.3 | 2026-05-14 | 操作面板迁至**全局左侧边栏**（`get_sidebar_widget`），与 ExtractTab；标签页仅存图表区 + 日志；主窗口预载入三页侧边栏并互斥显示 |
| 0.4 | 2026-05-15 | 去掉第四张训练曲线图（1×3 散点）；「测试集比例」改为「划分策略」且仅 **留一交叉验证**；侧栏与 `TrainingDatasetModel` 概要对齐 LOOCV、样本数 >5 提示；`TrainingChartController` 仅维护三张散点 |
| 0.5 | 2026-05-15 | 「开始训练」接入 `krr_workflow` + `kernel_regression`；artefact 目录 ``<数据父目录>/kr_model``；训练后当量 100 点 × 含铝 3 档预测并在散点叠加曲线；`krr_workflow.py` |
