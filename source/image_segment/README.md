# 迭代掩码传播图像序列分割模块

基于SAM的智能化图像序列分割系统，支持部分图片提供prompt点，通过迭代掩码传播自动处理整个序列。

## 🚀 核心特性

- ✅ **智能化迭代传播**: 从少量prompt图片开始，自动传播到整个序列
- ✅ **基于RGB相似性的点生成**: 智能生成高质量的正负prompt点
- ✅ **优化的采样策略**: 正点优先mask中心，负点远离边缘
- ✅ **模块化架构**: 清晰的职责分离，易于维护和扩展
- ✅ **完整的可视化**: 6子图详细展示传播过程
- ✅ **鲁棒的错误处理**: 智能跳过失败图片，继续传播

## 📁 模块结构

```
source/image_segment/
├── iterative_mask_propagation.py    # 核心分割模块
├── prompt_generation.py             # 智能点生成模块
├── mask_utils.py                    # 掩码分析工具
├── adjacent_group_finder.py         # 相邻图片组查找
├── test_complete_propagation.py     # 完整测试程序
└── README.md                        # 本文档
```

## 🎯 设计原理

### 1. 迭代传播策略

```
步骤1: 处理有prompt点的图片
[图片1*] → [图片3*] → 生成初始掩码

步骤2: 第一次迭代传播
[图片1] → [图片2]    # 向前传播
[图片3] → [图片4]    # 向后传播

步骤3: 继续迭代传播
[图片2] → [图片5]    # 继续向后传播
... 直到所有图片处理完成
```

### 2. 智能点生成算法

#### 正点采样 (mask中心优先)
- 使用距离变换计算到mask边缘的距离
- 优先选择距离边缘最远的点（中心区域）
- 选择前80%高权重点作为候选池
- 网格化均匀采样确保空间分布

#### 负点采样 (远离mask边缘)
- 创建动态缓冲区排除边缘附近区域
- 计算到mask的距离，优先选择最远点
- 选择前70%距离最远的点
- 备选图像边缘区域采样

### 3. RGB相似性筛选

#### 正点筛选条件
```python
# 至少与参考图片的2个正点"非常相似"
if similar_count >= 2:
    valid_positive_points.append(point)
```

#### 负点筛选条件
```python
# 不能与任何正点"非常相似"，最多与1个正点"相似"
if very_similar_count == 0 and similar_count <= 1:
    valid_negative_points.append(point)
```

## 🛠️ 使用方法

### 1. 基本使用

```python
from image_segment.iterative_mask_propagation import create_iterative_segmenter

# 创建分割器
segmenter = create_iterative_segmenter()

# 准备图像序列
image_paths = [
    "sequence/image_01.jpg",
    "sequence/image_02.jpg", 
    "sequence/image_03.jpg",
    "sequence/image_04.jpg"
]

# 设置prompt数据（只需要部分图片有prompt）
prompt_data = {
    0: {  # 第1张图片
        'points': [(400, 300), (390, 300), (410, 300)],  # 正点：目标中心
        'labels': [1, 1, 1]
    },
    2: {  # 第3张图片  
        'points': [(420, 320), (410, 320), (430, 320)],  # 正点：目标移动后位置
        'labels': [1, 1, 1]
    }
}

# 执行迭代掩码传播分割
masks = segmenter.segment_sequence_with_iterative_propagation(
    image_paths=image_paths,
    prompt_data=prompt_data,
    output_dir="output",
    save_masks=True,
    save_visualization=False  # 使用自定义可视化
)
```

### 2. 高级配置

```python
# 自定义模型和设备
segmenter = create_iterative_segmenter(
    model_type="vit_h",        # 使用更大的模型
    device="cuda",             # 指定GPU设备
    checkpoint_path="path/to/model.pth"  # 自定义模型路径
)

# 自定义RGB相似性阈值
from image_segment.prompt_generation import create_prompt_generator
prompt_generator = create_prompt_generator(
    very_similar_threshold=6.0,  # 更严格的"非常相似"阈值
    similar_threshold=10.0       # 更严格的"相似"阈值
)
```

## 🧪 测试方法

### 运行完整测试

```bash
cd source/image_segment
conda activate fireball_calculator
python test_complete_propagation.py
```

### 测试输出

测试会生成以下文件：

```
test_output/
├── test_image_00.png              # 测试图片序列
├── test_image_01.png
├── ...
├── masks/                          # 分割掩码
│   ├── test_image_00_prompted_mask.png
│   ├── test_image_01_propagated_mask.png
│   └── ...
└── visualization/                  # 完整可视化
    ├── test_image_00_merged_debug.png    # 1x3布局：prompt图片
    ├── test_image_01_merged_debug.png    # 2x3布局：传播图片
    └── ...
```

### 可视化说明

#### Prompt图片可视化 (1x3布局)
1. **Prompt Points**: 显示用户提供的正负点
2. **Segmentation Result**: 分割结果和质量分数
3. **Next Iteration Sampling**: 为下次迭代准备的采样点

#### 传播图片可视化 (2x3布局)
1. **Reference Segmentation**: 参考图片的分割结果
2. **Reference Points**: 参考图片的采样点
3. **Mapped Points**: 映射到目标图片的点
4. **Filtered Points**: RGB筛选后的有效点
5. **Segmentation Result**: 最终分割结果
6. **Debug Information**: 详细的统计信息

## 📊 性能指标

### 测试结果示例
```
============================================================
分割完成统计
============================================================
总图片数: 5
已处理图片数: 5
处理失败图片数: 0
成功分割图片数: 5
处理成功率: 100.0%
平均质量分数: 0.782
============================================================
```

### 质量评估标准

- **质量分数**: 0-1范围，综合考虑面积比例和形状紧凑性
- **面积验证**: 掩码面积应在图像总面积的1%-90%之间
- **形状分析**: 基于轮廓的紧凑性评估

## 🔧 API 参考

### IterativeMaskPropagationSegmenter

#### 初始化
```python
segmenter = IterativeMaskPropagationSegmenter(
    model_type="vit_l",           # SAM模型类型: "vit_b", "vit_l", "vit_h"
    checkpoint_path=None,         # 模型检查点路径（可选）
    device="auto"                 # 设备: "auto", "cuda", "mps", "cpu"
)
```

#### 主要方法

##### segment_sequence_with_iterative_propagation()
执行迭代掩码传播分割

**参数:**
- `image_paths`: List[str] - 图像文件路径列表
- `prompt_data`: Dict[int, Dict[str, Any]] - prompt数据字典
  ```python
  {
      image_index: {
          'points': [(x, y), ...],  # 点坐标列表
          'labels': [1, 0, ...],    # 点标签 (1=正点, 0=负点)
          'boxes': [(x, y, w, h)]   # 可选：矩形prompt
      }
  }
  ```
- `output_dir`: Optional[str] - 输出目录
- `save_masks`: bool - 是否保存掩码文件
- `save_visualization`: bool - 是否保存内置可视化

**返回:**
- `List[Optional[np.ndarray]]` - 每张图片的分割掩码，None表示失败

## 🔍 算法详解

### 1. 相邻图片组查找算法

```python
# 示例：已处理图片 [1, 4]，未处理图片 [0, 2, 3, 5]
# 第一次迭代创建组：
{processed: 1, unprocessed: [0, 2]}  # 图片1的相邻未处理图片
{processed: 4, unprocessed: [3, 5]}  # 图片4的相邻未处理图片
```

**特点:**
- 每个未处理图片只被一个组包含，避免重复处理
- 智能跳过失败图片，寻找更远的相邻图片
- 支持双向传播（前向和后向）

### 2. RGB相似性算法

#### 距离计算
```python
def rgb_distance(rgb1, rgb2):
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2))

# 默认阈值
very_similar_threshold = 8.0   # 非常相似
similar_threshold = 12.0       # 相似
```

#### 筛选逻辑
- **正点**: 必须与≥2个参考正点"非常相似"
- **负点**: 不能与任何参考正点"非常相似"，最多与1个"相似"

### 3. 失败处理机制

#### 失败检测
- 掩码为空或质量分数过低
- 面积比例异常（<10%或>500%）
- RGB相似性不足
- 图像差异过大

#### 失败恢复
- 标记失败图片，不参与后续传播
- 寻找更远的相邻图片继续传播
- 详细的失败原因分析和日志

## 🎨 自定义选点策略

### 修改采样参数

```python
# 在prompt_generation.py中调整
def _sample_positive_points(self, mask, num_points):
    # 调整中心区域比例
    top_ratio = 0.8  # 选择前80%高权重区域
    
def _sample_negative_points(self, mask, num_points):
    # 调整边缘缓冲区大小
    edge_buffer = max(10, int(min(h, w) * 0.05))
    # 调整远离区域比例
    top_ratio = 0.7  # 选择前70%距离最远区域
```

### 修改筛选阈值

```python
# 创建自定义prompt生成器
prompt_generator = create_prompt_generator(
    very_similar_threshold=6.0,  # 更严格的筛选
    similar_threshold=10.0
)
```

## 🔧 故障排除

### 1. SAM未安装
```bash
# 运行安装脚本
cd source
./setup.sh
```

### 2. 内存不足
```python
# 使用CPU模式
segmenter = create_iterative_segmenter(device="cpu")
```

### 3. 分割质量不佳
- 调整prompt点位置，确保在目标中心
- 增加prompt图片数量
- 调整RGB相似性阈值
- 检查图像序列的连续性

### 4. 传播失败
- 查看详细的失败分析日志
- 调整采样策略参数
- 增加负点数量
- 检查图像差异度

## 📈 性能优化建议

### 1. 硬件优化
```python
# 使用GPU加速
segmenter = create_iterative_segmenter(device="cuda")  # NVIDIA GPU
# 或
segmenter = create_iterative_segmenter(device="mps")   # Apple Silicon
```

### 2. 参数调优
```python
# 调整采样点数量
positive_points = 15  # 增加正点数量
negative_points = 8   # 增加负点数量

# 调整质量阈值
min_area_ratio = 0.005  # 降低最小面积要求
max_area_ratio = 0.95   # 提高最大面积限制
```

### 3. 策略选择
```python
# 使用不同的掩码选择策略
best_mask = mask_analyzer.select_best_mask(masks, strategy="best_quality")
# 可选策略: "largest_area", "best_quality", "most_compact"
```

## 📝 开发指南

### 1. 添加新的采样策略

在 `prompt_generation.py` 中扩展 `_sample_positive_points` 或 `_sample_negative_points` 方法。

### 2. 添加新的质量评估指标

在 `mask_utils.py` 中扩展 `MaskAnalyzer.calculate_mask_quality` 方法。

### 3. 添加新的失败分析规则

在 `mask_utils.py` 中扩展 `PropagationFailureAnalyzer` 类。

### 4. 自定义相邻图片查找

在 `adjacent_group_finder.py` 中修改 `AdjacentGroupFinder` 类。

## 🧪 测试用例

### 基础测试
```bash
# 运行标准测试
python test_complete_propagation.py
```

### 自定义测试
```python
# 创建自定义测试
def custom_test():
    segmenter = create_iterative_segmenter()
    
    # 自定义图像序列和prompt
    image_paths = ["your_image_01.jpg", "your_image_02.jpg"]
    prompt_data = {
        0: {
            'points': [(x, y), ...],  # 你的prompt点
            'labels': [1, 1, 0, 0]    # 对应的标签
        }
    }
    
    # 执行分割
    masks = segmenter.segment_sequence_with_iterative_propagation(
        image_paths, prompt_data, "custom_output"
    )
    
    return masks
```

## 📊 输出文件说明

### 掩码文件 (`output/masks/`)
- `{basename}_prompted_mask.png` - 有prompt点的图片掩码
- `{basename}_propagated_mask.png` - 传播生成的图片掩码
- 格式：8位灰度PNG，255表示前景，0表示背景

### 可视化文件 (`output/visualization/`)
- `{basename}_merged_debug.png` - 完整的debug可视化
- 包含传播过程的所有关键信息
- 适合分析和调试使用

## 🔬 技术细节

### 1. 采样点生成流程

```
参考掩码 → 距离变换 → 权重排序 → 候选池选择 → 网格采样 → 最终点集
```

### 2. 传播流程

```
参考点采样 → 点映射 → RGB筛选 → SAM分割 → 质量验证 → 结果保存
```

### 3. 迭代控制

```
初始化 → prompt处理 → 相邻组查找 → 批量传播 → 质量检查 → 下次迭代
```

## 🎯 最佳实践

### 1. Prompt点设置
- 在目标的中心区域设置正点
- 在目标外围设置负点
- 避免在边缘模糊区域设置点
- 每张prompt图片至少3-5个点

### 2. 图像序列要求
- 相邻帧之间变化不宜过大
- 目标在序列中应保持相对连续
- 图像质量要求清晰，避免过度模糊

### 3. 参数调优
- 根据目标大小调整采样点数量
- 根据图像特点调整RGB阈值
- 根据序列长度调整迭代策略

## 📚 相关文档

- [USAGE.md](USAGE.md) - 详细使用指南
- [POINT_PROMPT_GUIDE.md](POINT_PROMPT_GUIDE.md) - 点prompt设置指南

## 🤝 贡献指南

1. Fork本项目
2. 创建功能分支
3. 提交代码改进
4. 创建Pull Request

## 📄 许可证

本项目遵循 [MIT License](../../LICENSE)

---

**注意**: 首次运行需要下载SAM模型文件，请确保网络连接正常。建议使用GPU加速以获得最佳性能。
