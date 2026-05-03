# 机器视觉 Tab：参数面板与 `MvParametersController` 规划

> 本文档合并两部分内容：**(A)** 已落地的 Controller 抽取原则；**(B)** 待做的「图片参数 / 炸药参数」**语义与 UI 改版**及对应**代码改动清单**。  
> 实现时请以 **§B（需求语义）、§C（代码清单）** 为准推进。

---

## A. 已实现：Controller 抽取（回顾）

### A.1 原则

- **View**：`extract_tab_ui.py` 创建控件并放入 `ui_components['mv_*']`。
- **控制**：`controllers/mv_parameters_controller.py` 的 `MvParametersController` 私有持有控件引用；**`ExtractTab` 仅调用** `sync_model_from_ui`、`sync_ui_from_model`、`parameter_values_for_sequence_creation`、`set_enabled`。
- **模型**：`sequence_model.apply_parameters_from_ui` 等仍在 `SequenceModel`，由 Controller 调用。

### A.2 公开 API（保持不变）

| 方法 | 作用 |
|------|------|
| `sync_model_from_ui()` | 根据当前 UI 写入 `SequenceModel` / 序列 JSON 所需字段 |
| `sync_ui_from_model()` | 从模型回填 UI |
| `parameter_values_for_sequence_creation()` | 供 `create_work_sequence_from_image_folder` 等使用的元组 |
| `set_enabled(bool)` | 分割锁等场景下统一启停参数控件 |

---

## B. 待做：UI 与语义改版（需求）

### B.1 界面文案与结构

| 现状态 | 目标 |
|--------|------|
| 分组标题「爆炸信息」 | **「图片参数」** |
| 标签「爆炸时长（ms）」 | **「帧率（fps）」**；用户输入 **帧率**，**不再直接编辑爆炸时长** |
| 标签「单位像素实际长度(m)」 | **「视场范围(m)」**；用户输入 **视场总宽度（米）** |
| 「炸药参数」分组内「炸药类别」`QComboBox` | **删除**；界面仅保留 **当量**、**含铝量**（分组标题可仍为「炸药参数」或后续再改名） |

**控件默认值（实现时与新建序列一致）**：

| 控件 | 默认值 |
|------|--------|
| 帧率（fps） | **1000** |
| 视场范围（m） | **60** |

新建工作序列、`QLineEdit` 初始占位、以及 JSON 缺少 `frame_rate_fps` / `field_of_view_m` 时的兜底（若采用）均建议与上表一致。

### B.2 派生量定义（内部时间单位一律为毫秒 ms）

1. **爆炸时长 `explosion_duration`（存库与计算均为毫秒）**  
   - UI 输入为 **帧率 $\mathrm{fps}$（帧/秒）** 与序列张数 $N$；**不**在界面直接编辑毫秒。  
   - **由帧率得到总时长（秒）** 的常规定义为 $T_\mathrm{s} = N / \mathrm{fps}$ 或 $T_\mathrm{s} = (N - 1) / \mathrm{fps}$（与「首尾帧跨度」是否含单帧有关）。**写入模型与 JSON 的 `explosion_duration` 一律为毫秒**：  
     $$
     T_\mathrm{ms} = T_\mathrm{s} \times 1000 .
     $$  
   - **实现时择一并与时间轴统一**：若时间轴仍为  
     `time_ms = (index / (total_frames - 1)) * explosion_duration_ms`，  
     则通常取 **$T_\mathrm{ms} = \dfrac{N - 1}{\mathrm{fps}} \times 1000$**，使末帧对应时间为全长；若业务上希望末帧对应 $N/\mathrm{fps}$，则需同步调整时间轴公式（见 §C.6）。  
   - **结论**：对脚本、直径曲线、时间轴而言，**内部只认 `explosion_duration` 为 ms**；由 fps 与 $N$ 在同步时算出该毫秒值。

2. **单位像素实际长度（m/px）**  
   - **公式**：  
     $$
     \texttt{pixel\_length} = \frac{\text{视场范围(m)}}{\text{首张图像像素宽度(px)}} .
     $$  
   - **首张图像**：取当前序列 `image_paths[0]`（或模型中第一张路径），需用 **PIL / QImage / OpenCV** 等读取 **宽度（像素）**；读取失败时要有降级策略（提示用户、保持旧值或禁用写盘）。

### B.3 JSON / `parameters` 与 UI 一致性（已定稿）

- **`explosion_duration`（ms）**、**`pixel_length`（m/px）**：仍为下游（分割、时间轴、直径等）使用的**派生权威字段**，由帧率 + $N$、视场 ÷ 首图宽度计算得到后写入。  
- **必须在 `parameters` 中额外持久化**（键名实现时统一即可，例如）：  
  - **`frame_rate_fps`**（实现时键名可约定为 `fps` 等，全项目统一即可）：与 UI「帧率」一致；  
  - **`field_of_view_m`**：与 UI「视场范围(m)」一致。  
  以便 **`sync_ui_from_model` 直接反填**，保证 **JSON 与界面一致**；避免仅靠 $T_\mathrm{ms}$、$N$ 反推 fps 带来的歧义（$N$ 与 $N-1$）。  
- **炸药类别**：UI 删除后，**代码路径统一传固定默认** `material_type`（如 `"温压弹"`）写入 JSON，满足 `SequenceManager` 等校验；不在界面选择。若将来要从 JSON 移除该键，另开任务改 `sequence_manager` 与校验。

---

## C. 代码改动清单（按模块）

以下为落实 **§B** 时建议修改的文件与要点（**不含**具体行号，以当前仓库为准）。

### C.1 `source/desktop/extract_tab/ui_widgets/extract_tab_ui.py`

- 将 **「爆炸信息」** `QGroupBox` 标题改为 **「图片参数」**。
- **「爆炸时长（ms）」** 标签改为 **「帧率（fps）」**；`QLineEdit` **默认文本 `1000`**（与 §B.1 默认值表一致）；`ui_components` 键名可改为 `mv_frame_rate_fps`（若改名，需全链路替换）。
- **「单位像素实际长度」** 标签改为 **「视场范围(m)」**；`QLineEdit` **默认文本 `60`**；键名可考虑 `mv_field_of_view_m`。
- **删除** `mv_explosive_type` 的 `QComboBox` 及其标签；布局只保留当量、含铝量（仍在「炸药参数」组内）。

### C.2 `source/desktop/extract_tab/controllers/mv_parameters_controller.py`

- **控件集合**：由「5 个」变为 **4 个输入**（帧率、视场、当量、含铝）；删除对 `mv_explosive_type` 的引用与信号连接。
- **`sync_model_from_ui()`**：  
  - 读取 fps、视场 $W_\mathrm{m}$；从 `sequence_model.image_paths`（或父级）取 **首张图宽度**；计算 `pixel_length`、`explosion_duration_ms`（按 §B.2 最终选定公式与 $N=\len(\text{image\_paths})$）。  
  - 调用 `sequence_model.apply_parameters_from_ui`：需同步调整 **方法签名**（见 C.4），**材料类型**传常量。
- **`sync_ui_from_model()`**：  
  - **优先**从 `parameters` 读取 **`frame_rate_fps`、`field_of_view_m`** 填入 UI（与 §B.3 一致）。  
  - **旧 JSON 无上述键时**：可降级为由 `explosion_duration`、`N`、首图宽度与 `pixel_length` **反推** fps/视场（与 B.2 选定公式一致），或提示用户重新保存一次以写入新键。  
  - 不再操作炸药类别控件。
- **`parameter_values_for_sequence_creation()`**：  
  - 返回元组需与 `SequenceManager.create_work_sequence_from_image_folder` **签名一致**；若该 API 仍要 `explosive_type`，则 **硬编码默认字符串** 占位。
- **`set_enabled(bool)`**：只对现存 4 个输入控件生效。

### C.3 `source/desktop/extract_tab/sequence_model.py`

- **`apply_parameters_from_ui`**：去掉 UI 传入的 `material_type`，内部**写死默认**（如 `"温压弹"`）；同时写入 **`frame_rate_fps`、`field_of_view_m`** 与派生的 **`explosion_duration`（ms）、`pixel_length`**。  
- 确保 **`explosion_duration`、`pixel_length`** 及 **B.3 新增键** 随 flush 进入 JSON，且 `image_sequence.duration` 等与现逻辑一致。

### C.4 `source/desktop/extract_tab/utils/sequence_manager.py`

- **`create_work_sequence_from_image_folder`**：若签名含 `explosive_type`，可改为 **默认参数** 或 **内部写死**，调用方少传一维。
- **`required_param_keys`、校验**：若曾强制 `material_type` 来自 UI，改为 **默认通过** 或从生成数据写入默认类型。
- 文档字符串与 `_create_parameters_data`：与新产品语义对齐（时长由 fps+N 推导、pixel_length 由视场/宽度推导）。

### C.5 首张图宽度读取

- **新增小工具**（择一位置即可）：  
  - `extract_tab/utils/image_geometry.py` 或放在 `MvParametersController` 私有方法内：  
    `get_image_width_pixels(path: str) -> int | None`  
  - 实现可用 **Pillow**、**PySide6 QImageReader** 等；注意 **异常与超大图**。

### C.6 `source/desktop/extract_tab/controllers/sequence_display_controller.py`

- 时间轴公式若仍用 `explosion_duration_ms`，则 **只要模型中时长与 B.2 公式一致**即可，**一般不必改**；若产品改用「每帧间隔 = 1/fps」而总长为 $(N-1)/\mathrm{fps}$，则需 **同时** 改此处与 `segment_utils`，与 B.2 定案一致。

### C.7 `source/desktop/extract_tab/utils/segment_utils.py`

- `build_time_diameter_series` 使用 `explosion_duration_ms` 与 `pixel_length`：**无需改函数签名**，前提是 **写入模型的两个量已按新定义计算正确**。

### C.8 `source/desktop/extract_tab/extract_tab.py`

- 所有依赖 `parameter_values_for_sequence_creation()` 解包处：核对 **元组长度与顺序** 是否与 `SequenceManager` 一致。
- 状态栏/日志中「时长: xxx ms」可与 **计算后的** `explosion_duration` 一致；若有「帧率」展示需求可追加文案。

### C.9 其它引用

- 全文搜索 **`material_type`、`mv_explosive_type`、`炸药类别`、`温压弹`（UI）** 与 **`explosion_duration` 直接编辑** 的假设，更新文案与测试说明（如 `info_builder`、README、`document` 中用户指引）。

---

## D. 风险与测试要点

- **公式确认**：固定 $T_\mathrm{ms}$ 与 $N$、$N-1$ 的关系，并与 `sequence_display_controller` / `segment_utils` 时间轴一致。  
- **无路径 / 无图像 / 首图损坏**：`sync_model_from_ui` 时 $N$ 或宽度不可用时的行为（跳过写 duration、弹窗、禁止 flush 等）。  
- **fps ≤ 0、视场 ≤ 0、宽度为 0**：校验与错误提示。  
- **旧 JSON 加载**：缺少 `frame_rate_fps` / `field_of_view_m` 时的 **降级反推或引导重新保存**，加载新保存文件后 UI 与 JSON 应一致。  
- **回归**：导入序列、图像文件夹创建工作序列、特征提取、时间轴拖动、直径曲线、温度图（若仍绑定时长）。

---

## E. 小结

| 项目 | 说明 |
|------|------|
| Controller | 已存在 `MvParametersController`；改版时在其内集中 **fps / 视场 → duration / pixel_length** 的推导与同步。 |
| UI | `extract_tab_ui.py`：分组与标签改名、删炸药类别；帧率默认 **1000** fps、视场默认 **60** m。 |
| 模型/API | `material_type` **固定默认**；`parameters` **必须**含 `frame_rate_fps`、`field_of_view_m`，并继续写入派生的 `explosion_duration`（ms）、`pixel_length`。 |
| 新依赖 | 读取首图宽度的小工具函数 + 异常处理。 |
| 文档 | 本文件随需求迭代；实现后可在 PR 中引用路径 `document/explosion_explosive_params_controller_plan.md`。 |
