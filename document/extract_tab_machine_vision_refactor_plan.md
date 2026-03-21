# extract_tab「机器视觉」改造方案（已定稿）

本文档依据 `document/machine_vision_ui_prototype.html` 与评审结论，对 `source/desktop/extract_tab` 的改造范围、数据流与实现约束进行说明，**作为实施依据**。

---

## 一、现状摘要（与改造相关）

| 模块 | 职责 |
|------|------|
| `ExtractTab` | 仅「选择火球爆炸序列文件」→ `load_sequence_file` → `_apply_sequence_data` |
| `SequenceModel` | 内存态：`sequence_data`、`image_paths`、`parameters`、prompt、分割结果等 |
| `SequenceManager` | 读写 JSON、校验 `_validate_sequence_format`、各类 `save_*_to_sequence` / `clear_*` |
| `SequencyDisplayController` | 预览、时间轴、按模型状态切换显示（含分割叠加等） |
| `PromptController` | 参考点、起爆点；持久化多依赖 **磁盘上的 `current_path`** |

**缺口**：无「仅选图像序列目录」入口；工作 JSON 路径未与「图像文件夹同级命名」约定对齐；参数与导入 UI 原在 `input_tab.py`，需在机器视觉中统一并写回 JSON。

---

## 二、需求与实现要点（已定稿）

### 1. 导入火球图像序列 → 工作 JSON 路径与内容

**行为对齐** `source/desktop/input_tab.py` 的 `select_image_sequence`：

1. 使用 `QFileDialog.getExistingDirectory` **只选择图像序列所在文件夹**（不支持多目录、不支持跨目录多选文件；**当前版本不支持视频**）。
2. 在该文件夹内用 `glob` 收集 `*.png, *.jpg, *.jpeg, *.bmp, *.tiff`（与 `input_tab` 一致），排序后作为 `image_paths`（建议存**绝对路径**）。
3. **工作 JSON 不落盘在图像文件夹内部**，而与该文件夹**同级**：
   - 记图像文件夹为 `seq_dir`，其父目录为 `parent = dirname(seq_dir)`，文件夹名为 `name = basename(seq_dir)`（无尾部分隔符）。
   - 工作文件路径：  
     `join(parent, f"{name}_fireball_sequence.json")`
4. 生成**最小合法** `sequence_data`，通过 `load_sequence_file` 校验，字段与现有 `export_sequence_data` / `_validate_sequence_format` 一致，至少包含：
   - `metadata`（可最小占位）
   - `image_sequence.image_paths`：与排序后列表一致
   - `parameters`：**默认值与 `input_tab.py` 保持一致**（当前 UI 默认：`explosion_duration=140`，`pixel_length=0.01`，`material_type` 对应「温压弹」，`equivalent=1`，`al_percent=30`；若导出到 JSON 的键名与 Combo 文案需映射，与现有 `SequenceManager.export_sequence_data` 一致）。
5. **不包含**：`image_sequence.prompt_data`、`target_center`、`image_sequence_segmentation`；`temperature` 仅在单独导入温度序列后写入（见下文）。
6. 写盘成功后：`SequenceModel.apply_sequence_dict(data, file_path=上述 json 路径)`，`current_path` 指向该文件。

**分割结果序列文件名**（与现有逻辑一致，仅随工作文件 stem 变化）：

- 当前代码：`segmented_path = original_path.with_name(f"{original_path.stem}_segmented{original_path.suffix}")`（见 `extract_tab.py`）。
- 工作文件为 `{name}_fireball_sequence.json` 时，分割导出为 **`{name}_fireball_sequence_segmented.json`**，无需再硬编码 `fireball_sequence_segmented.json`。

**涉及文件**：`extract_tab.py`、`sequence_manager.py`（如 `create_work_sequence_for_image_folder(...)`）、必要时微调校验说明。

---

### 2. 修改炸药参数与爆炸参数 → 写回工作 JSON

**目标**：内存与磁盘一致；**不采用防抖定时器**。

**策略：在「用户去做其他操作之前」先写盘**（保证 JSON 反映最新状态），包括但不限于：

- 点击**任意按钮**（导入、保存、开始特征提取、重新提取、清除参考点、侧栏其他按钮等）
- **开始选择 / 修改特征点、起爆点**（在真正写入 prompt 或改变交互状态前，先 flush 参数）
- **开始图像分割**（进入异步分割前）
- **关闭程序**（`closeEvent` 或应用退出前）

**实现要点**：

1. 参数控件布局与分组见 **第六节 UI**；值变化时**立即更新** `SequenceModel` 内存中的 `_parameters` / `_sequence_data['parameters']`（以及模型内 `explosion_duration_ms`、`pixel_length` 等派生字段，保持与现有一致）。
2. **`flush_parameters_to_disk()`**（或合并为通用 `flush_sequence_to_disk()`）：仅当 `current_path` 有效时，将当前内存中的 `parameters`（及需同步的字段）写回 `current_path` 指向的 JSON。
3. 各上述入口在业务逻辑开头调用一次 flush（注意避免递归；分割进行中见 **2.5** 不调参数）。

**Input 废弃后**：所有参数以当前序列 JSON / `SequenceModel` 为唯一数据源，不再从 `input_tab` 读取。

**涉及文件**：`extract_tab.py`、`extract_tab_ui.py`、`sequence_model.py`、`sequence_manager.py`、`prompt_controller.py`（选点前 flush）、主窗口（关闭前 flush）。

---

### 3. 爆炸序列可带特征点 + 分割结果 → 预览须继续展示

保持现有 `_apply_sequence_data`、`display_controller`、`prompt_controller` 路径；从磁盘加载的 `prompt_data`、`target_center`、`image_sequence_segmentation` 继续在预览中体现。

写盘时注意：**参数 flush** 与 **prompt/分割保存** 应对同一 JSON **读-改-写** 时避免覆盖未同步字段——推荐统一「先读入最新 `sequence_data` 再合并内存再写回」或单线程串行化（分割进行中禁止改参，可降低并发写风险）。

---

### 4. 仅导入火球图像序列（无 prompt、无分割）→ 只显示火球图像

- 预览仅原始帧，**不绘制** mask / 质心 / 参考点等叠加。
- 图表：无分割则直径/速率为空或占位；温度仅在有 `temperature` 或已导入温度文件后有数据。

实现上核对 `SequencyDisplayController` / `interactive_image_widget` 在「无分割且无参考点」时的分支。

---

### 5. 先导入爆炸序列，再导入火球图像序列 → 等价 reset

与现有一次导入前 `_reset_state_before_import()` 一致，再扫描新目录、生成/覆盖**新的** `{文件夹名}_fireball_sequence.json` 并加载。

**不要求**二次确认对话框（`QMessageBox`）。

---

### 6. 温度序列导入

- 导入**成功即写盘**：合并写入当前 `current_path` 对应 JSON 的 `temperature` 数组（或项目约定字段），并刷新内存与温度曲线。
- 写盘策略**不按参数 flush 规则防抖**；一次导入一次完整保存即可。

---

### 7. 与 `input_tab` 的关系

- **`input_tab` 废弃**：主界面移除「输入」Tab 或不再挂载该模块；原导入图像 / 温度、参数默认值与写 JSON 职责并入机器视觉（`extract_tab` + UI）。

---

### 8. 视频与并发

- **当前版本不支持视频**（仅图像扩展名目录扫描，与 `input_tab` 一致）。
- **分割进行中**：**禁止修改参数**；除**图片预览**（时间轴切换帧等仅浏览）外，**其余交互禁用**（与当前「分割运行中」逻辑一致）。因此不存在「分割异步写盘与用户改参写盘」并发冲突；仍建议在实现上保持「写 JSON 单路径串行」以防后续扩展。

---

## 三、已定稿项汇总（原「待确认」表）

| 主题 | 结论 |
|------|------|
| 图像导入方式 | 仅**文件夹**（`getExistingDirectory`），规则同 `input_tab.py` |
| 工作 JSON 位置与命名 | 与图像文件夹**同级**：`{图像文件夹名}_fireball_sequence.json` |
| 分割后序列命名 | `{同名 stem}_segmented.json`，即 `{文件夹名}_fireball_sequence_segmented.json` |
| 默认参数 | 与 `input_tab.py` 一致 |
| 参数写盘 | **非防抖**；在点击按钮、选点、开始分割、关程序等**其他操作之前** flush |
| 再导图像确认框 | **不需要** |
| 温度导入 | **成功即写盘** |
| `input_tab` | **废弃** |
| 视频 | **本期不支持** |
| 分割进行中 | **禁改参数**；除预览外**禁用其他操作** |

**仍建议在实现阶段处理**：工作 JSON 父目录**不可写**时的错误提示；是否 `.bak` 备份由实现酌情（本方案不强制）。

---

## 四、推荐实施顺序

1. **数据层**  
   - `SequenceManager`：由「图像文件夹路径 + 默认 parameters」生成合法 dict 并写入 `{name}_fireball_sequence.json`；统一路径计算工具函数（避免散落字符串拼接）。  
   - 确认 `extract_tab` 内分割完成后的 `_segmented` 路径与上述 stem 规则一致（通常仅需保证 `current_path` 已是新命名）。

2. **ExtractTab 导入**  
   - `select_image_sequence_folder()`：选目录 → 扫图 → 写工作 JSON → `_reset_state_before_import` → `load` / `_apply_sequence_data`。  
   - 「导入爆炸序列文件」：仍可打开任意路径 JSON；若用户再导图像目录，走 reset + 新工作文件。

3. **UI（对齐 `machine_vision_ui_prototype.html`）**  
   - **侧栏**：数据源（两按钮：爆炸序列 JSON、火球图像序列目录）+ 可选温度导入；参考点与分割；输出。  
   - **主区左列**：预览 + 时间轴 + **爆炸信息 / 炸药参数** 两分组（组内纵向）+ 运行日志（不占右列图表区）。  
   - 在列出的所有「其他操作」入口挂接 **参数 flush**。

4. **主程序**  
   - 移除或停用 `InputTab`；关闭应用前对当前 `ExtractTab` 调用 flush。

5. **联调**  
   - 仅图像 / 带分割 / 带 prompt / 温度导入 / 关程序 / 分割中禁用 等场景下 JSON 与 UI 一致性测试。

---

## 五、主要将修改的文件清单（预估）

- `extract_tab/extract_tab.py`  
- `extract_tab/ui_widgets/extract_tab_ui.py`  
- `extract_tab/sequence_model.py`  
- `extract_tab/utils/sequence_manager.py`  
- `extract_tab/controllers/prompt_controller.py`、`sequence_display_controller.py`（flush 钩子、分割中禁用范围）  
- 主窗口入口（移除 `input_tab`、Tab 名称改为「机器视觉」、关闭事件）

---

## 六、小结

- **统一入口**：爆炸序列 JSON **或** 图像文件夹 → 均落到 **`sequence_data` + `current_path`**；图像入口在**文件夹同级**生成 `{文件夹名}_fireball_sequence.json`。  
- **参数持久化**：内存即时更新；在**任何其他操作前**将参数写入当前工作 JSON；温度导入**成功即写盘**。  
- **分割期**：仅允许预览浏览，**禁止改参**与其余操作，与现有一致。  
- **UI**：文件导入区、参数区、日志区与 **`machine_vision_ui_prototype.html`** 一致；**`input_tab` 废弃**。
