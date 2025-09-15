# 图像序列分割模块

基于SAM2的图像序列分割功能，支持在图像序列的第一张图片上设置矩形prompt，然后对整个序列进行分割。

## 功能特性

- ✅ 支持图像序列分割
- ✅ 支持矩形prompt输入
- ✅ 自动保存分割掩码和可视化结果
- ✅ 支持多种图像格式 (JPG, PNG, BMP等)
- ✅ 自动设备检测 (CUDA, MPS, CPU)

## 文件说明

- `sam2_sequence_segmentation.py` - 主要的分割模块
- `test_fireball_segmentation.py` - 完整的测试程序
- `simple_test.py` - 简化的测试程序
- `README.md` - 使用说明

## 使用方法

### 1. 环境准备

确保已安装SAM2和相关依赖：

```bash
cd source
./setup.sh
```

### 2. 基本使用

```python
from image_segment.sam2_sequence_segmentation import create_segmenter

# 创建分割器
segmenter = create_segmenter()

# 设置图像路径和prompt矩形
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
prompt_rectangles = [(350, 270, 100, 60)]  # (x, y, w, h)

# 执行分割
masks = segmenter.segment_sequence(
    image_paths=image_paths,
    prompt_rectangles=prompt_rectangles,
    output_dir="output",
    save_masks=True,
    save_visualization=True
)
```

### 3. 运行测试

#### 简化测试（推荐）
```bash
cd source/image_segment
python simple_test.py
```

#### 完整测试
```bash
cd source/image_segment
python test_fireball_segmentation.py
```

## API 参考

### SAM2SequenceSegmenter

#### 初始化
```python
segmenter = SAM2SequenceSegmenter(
    model_cfg="sam2_hiera_l.yaml",  # 模型配置
    checkpoint_path=None,            # 检查点路径（可选）
    device="auto"                   # 设备类型
)
```

#### 主要方法

##### segment_sequence()
对图像序列进行分割

**参数:**
- `image_paths`: List[str] - 图像文件路径列表
- `prompt_rectangles`: List[Tuple[int, int, int, int]] - 矩形prompt列表 [(x, y, w, h), ...]
- `output_dir`: Optional[str] - 输出目录
- `save_masks`: bool - 是否保存分割掩码
- `save_visualization`: bool - 是否保存可视化结果

**返回:**
- `List[np.ndarray]` - 每张图片的分割掩码列表

##### segment_single_image()
对单张图片进行分割

**参数:**
- `image_path`: str - 图像文件路径
- `prompt_rectangles`: List[Tuple[int, int, int, int]] - 矩形prompt列表

**返回:**
- `Tuple[np.ndarray, np.ndarray, np.ndarray]` - (masks, scores, logits)

## 输出结果

分割完成后，会在输出目录中生成：

```
output/
├── masks/                    # 分割掩码
│   ├── image1_mask_0.png
│   ├── image2_mask_0.png
│   └── ...
└── visualization/            # 可视化结果
    ├── image1_segmentation.png
    ├── image2_segmentation.png
    └── ...
```

## 示例：火球序列分割

使用指定的prompt矩形 (350, 270, 100, 60) 对火球序列进行分割：

```python
# 火球序列图像路径
fireball_images = [
    "images/fireball_sequence/1.jpg",
    "images/fireball_sequence/2.jpg", 
    "images/fireball_sequence/3.jpg",
    "images/fireball_sequence/4.jpg"
]

# 设置prompt矩形
prompt_rect = [(350, 270, 100, 60)]

# 创建分割器并执行分割
segmenter = create_segmenter()
masks = segmenter.segment_sequence(
    image_paths=fireball_images,
    prompt_rectangles=prompt_rect,
    output_dir="fireball_segmentation_output"
)
```

## 故障排除

### 1. SAM2未安装
```
ImportError: SAM2未安装
```
**解决方案:** 运行 `cd source && ./setup.sh`

### 2. 模型检查点未找到
```
FileNotFoundError: 检查点文件不存在
```
**解决方案:** 确保SAM2模型文件已下载，或手动指定检查点路径

### 3. 内存不足
```
CUDA out of memory
```
**解决方案:** 使用CPU模式或减小图像尺寸

### 4. 图像读取失败
```
ValueError: 无法读取图片
```
**解决方案:** 检查图像文件路径和格式

## 性能优化

1. **使用GPU加速**: 确保CUDA或MPS可用
2. **批量处理**: 一次处理多张图像
3. **图像预处理**: 适当调整图像尺寸
4. **模型选择**: 根据需要选择不同大小的模型

## 注意事项

1. 首次运行会下载模型文件，需要网络连接
2. GPU内存需求较大，建议使用8GB以上显存
3. 分割质量取决于prompt矩形的位置和大小
4. 建议在图像的主要目标区域设置prompt
