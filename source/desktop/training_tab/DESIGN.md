# 模型训练标签页（`training_tab`）软件设计

本文档描述在 `source/desktop/training_tab` 中实现「模型训练」桌面 UI 的**架构与职责划分**，对齐 `document/machine_vision_ui_prototype.html` 中 **「模型训练」** 标签页的布局与交互意图；**软件组织形式参考** `source/desktop/extract_tab`。  
**范围说明**：仅设计文档，不含实现代码。

---

## 1. 背景与目标

- **产品来源**：HTML 原型将「模型训练」作为独立顶栏标签，左侧为操作与参数，右侧为四图网格 + 训练日志。
- **桌面目标**：在 PySide6 应用中提供等价信息架构与控件集合，便于后续接入真实数据加载、划分、训练与图表刷新。
- **约束**：首版实现可保留「占位 + 示意数据」路径，但目录与类职责应按本设计拆开，避免在单文件中堆叠 UI 与业务逻辑。

---

## 2. 原型对齐清单

以下与 `machine_vision_ui_prototype.html` 中 `pane-train-aside` / `pane-train-content` 一一对应，作为验收检查表。

### 2.1 左侧栏（`aside`）

| 分组（原型 legend） | 控件 | 行为要点 |
|---------------------|------|----------|
| **输入数据** | 主按钮「输入数据」+ 文件选择 | 选择实验数据表（CSV / JSON / TXT / XLSX 等，具体格式由后续数据层定义）；状态行显示是否已加载。 |
| **模型训练** | 算法下拉 | 选项：`核回归`、`高斯过程`；切换后更新右侧第四图类型/标题及状态文案。 |
| **划分与执行** | 「测试集比例」下拉 | 仅 **10%～40%**（步长 5%），与原型一致。 |
| **划分与执行** | 按钮「开始训练」 | 触发训练流程（或首版：写日志 + 刷新示意曲线）。 |
| **训练数据集信息** | 只读多行文本（概要） | 展示：训练模型名称、总样本数、训练集数量、测试集数量、当前测试集比例；样式与意图对齐「机器学习」侧「已选参考点」式只读信息区。 |

### 2.2 右侧主区（`content`）

| 区域 | 内容 |
|------|------|
| 标题栏 | 「模型训练视图」+ 当前算法状态（如「算法：核回归」）。 |
| 说明文案 | 短 hint：散点含义、点大小与含铝量关系等（可与原型一致或略缩）。 |
| 图表网格（2×2） | ① 最大直径–当量散点 ② 初始直径–当量散点 ③ 时间常数–当量散点 ④ 训练曲线（核回归：σ–MSE；高斯过程：示意超参曲线，与原型一致）。 |
| 训练日志 | 只读文本区，追加「输入数据」「开始训练」等事件。 |

### 2.3 与已有 `chart_widgets` 的关系

- 三张散点：使用 `FireballTrainingScatterChart` 三个实例（`for_max_diameter` / `for_initial_diameter` / `for_time_constant`），`update_data(当量, y, 含铝量或约定 size 列)`。
- 第四张训练曲线：核回归使用 `KernelRegressionTrainingCurveChart`；高斯过程可预留第二套更新接口或独立小部件（设计阶段允许「控制器内分支」）。

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
├── training_dataset_model.py # （推荐）会话状态：路径、表格、划分计数、当前算法等（供 Tab / utils / 控制器读）
├── ui_widgets/
│   ├── __init__.py
│   └── training_tab_ui.py    # TrainingTabUI：create_sidebar_widget + 标签页正文（图表/日志）；与 ExtractTabUI 侧边栏拆分方式一致
├── controllers/
│   ├── __init__.py
│   └── training_chart_controller.py   # 仅 UI 密切相关：四图 update_data / reset / 算法切换视图
└── utils/
    ├── __init__.py
    ├── dataset_io.py         # 训练数据导入 / 导出（路径、格式、解析结果 → 可被 model 吸收）
    └── training_bridge.py    # App 与训练 model（核回归/GP CLI、进程、服务）的连接层：发起训练、取结果 tensor/曲线等
```

说明：

- **`training_dataset_model.py`**：建议保留，对齐 `extract_tab` 中 `sequence_model` 与界面之间的缓冲层。
- **`utils/dataset_io.py`**：**与 UI 无关** 的读写与解析约定（CSV/JSON/XLSX、列映射、导出中间结果）；`TrainingTab` 只在槽函数里调用（如 `QFileDialog` 选路径 → 调用 `dataset_io.import_...` → 写回 model → 刷新 UI）。
- **`utils/training_bridge.py`**：**与 UI 无关** 的训练执行与结果拉回（同步/异步、`QThread` 可在 Tab 层包一层薄壳，但 **业务编排与后端 API** 实现在此模块或其子模块）；不向 Qt 控件直接耦合。

---

## 5. UI 架构（Qt 布局）

**与机器视觉模块一致**：**全局左侧窄栏（`SidebarWidget`）**挂载操作面板；**标签页正文**仅占中间区域图表与日志。**不**在 `TrainingTab` 内再放操作区 `QSplitter` 左列。

### 5.1 全局左侧栏（≈260～300px，`framework` 已与顶栏并排）

- 由 `TrainingTab.get_sidebar_widget()` 返回的根控件（通常为 `TrainingTabUI.create_sidebar_widget()` 产出的外层 `QGroupBox`）。
- **内容**：纵向 `QScrollArea`（可选）+ 多个 `QGroupBox`，分组与原型一致：**输入数据**、**模型训练**（算法）、**划分与执行**、**训练数据集信息**。
- **切换**：主窗口 `QTabWidget` 切至「模型训练」时，`set_sidebar_content(self.training_tab.get_sidebar_widget())`，与 `ExtractTab.get_sidebar_widget()` 用法对齐。

### 5.2 标签页正文（`TrainingTab`）

- **仅**纵向布局：**工具栏行**（「模型训练视图」+ `train-model-status` 等价 QLabel）、**hint `QLabel`**、**2×2 `QGridLayout`**（四图）、**训练日志** `QPlainTextEdit`。

### 5.3 样式

复用桌面应用全局 QSS / 调色板（深蓝底、accent 青色标题），不必逐像素照搬 HTML CSS。

---

## 6. 控制器职责（仅图表）

### 6.1 `TrainingChartController`

- 持有四个图表部件引用。
- **算法切换**：更新第四图数据源与标题、`train-model-status` 等价 QLabel；若高斯过程无真实曲线，可走占位绘制（与原型一致）。
- **可见性**：若标签页采用懒加载，首次显示时需触发 `canvas.draw`/resize（语义对齐 HTML 中 `train-charts-layout`，在 Qt 中用 `showEvent` 或 `resizeEvent` 防抖刷新）。
- **数据入口**：对外提供形如「用当前 model 中的列刷新三张散点」「用给定 σ/MSE 曲线刷新第四图」等方法，由 `TrainingTab` 在适当时机调用（数据来自 model，聚合前可能已走 `dataset_io` / `training_bridge`）。

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
| 「输入数据」 | `QFileDialog` → `dataset_io` 导入 → 更新 model → 刷新**侧边栏**状态 / 概要 / 调用 `TrainingChartController` 更新散点。 |
| 「测试集比例」变化 | 更新 model 划分计数 → 仅刷新概要文本（不必重训）。 |
| 「开始训练」 | 校验数据已加载；追加训练日志；调用 `training_bridge`（可在另一线程）；完成后根据结果刷新第四图与日志、`TrainingChartController`。 |
| 算法下拉切换 | 更新 model 与界面文案 → `TrainingChartController` 切换第四图视图。 |

- 实例化：`TrainingTabUI`（内含 `create_sidebar_widget` + `create_main_layout`）、`TrainingDatasetModel`、`TrainingChartController`；按需 import `dataset_io`、`training_bridge`。
- **`get_sidebar_widget()`**：提供给主窗口挂载至**全局左侧边栏**，与 `ExtractTab` 一致。
- 可选：`get_ui_components()` 供调试或外层访问。
- 避免在 Tab 内实现解析与后端协议细节——**委托给 `utils`**。

---

## 9. 数据流（概念）

```
[文件] ──► dataset_io（导入）──► TrainingDatasetModel
                                       │
TrainingTab ◄──── 用户修改比例 ────────┤→ 概要 UI 刷新
                                       │
TrainingTab ──► training_bridge（训练）──► 指标/曲线 ──► TrainingChartController（第四图）+ 日志
                                       │
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
| 真实训练后端 | 可先日志 + 示意 `KernelRegressionTrainingCurveChart.update_data()` 无参；真训练走 `training_bridge` 逐步实现。 |
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
