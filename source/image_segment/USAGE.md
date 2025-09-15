# 图像序列分割使用说明

## 功能概述

本项目提供了基于SAM（Segment Anything Model）的图像序列分割功能，支持在图像序列的第一张图片上设置矩形prompt，然后对整个序列进行分割。

## 文件结构

```
image_segment/
├── sam_sequence_segmentation.py    # SAM分割模块（推荐使用）
├── sam2_sequence_segmentation.py   # SAM2分割模块（实验性）
├── test_sam_segmentation.py        # SAM测试程序
├── test_fireball_segmentation.py   # 完整测试程序
├── simple_test.py                  # 简化测试程序
├── README.md                       # 详细说明文档
├── USAGE.md                        # 使用说明（本文件）
└── output_sam/                     # 输出结果目录
    ├── masks/                      # 分割掩码
    └── visualization/              # 可视化结果
```

## 快速开始

### 1. 环境准备

确保已安装SAM和相关依赖：

```bash
cd source
./setup.sh
```

### 2. 运行测试

使用指定的prompt矩形 (350, 270, 100, 60) 对火球序列进行分割：

```bash
# 使用SAM进行分割（推荐）
KMP_DUPLICATE_LIB_OK=TRUE ./python310 image_segment/test_sam_segmentation.py
```

### 3. 查看结果

分割完成后，结果保存在 `image_segment/output_sam/` 目录中：

- `masks/`: 每张图片的分割掩码（PNG格式）
- `visualization/`: 可视化结果，显示原图和分割结果

## 使用方法

### 基本用法

```python
from image_segment.sam_sequence_segmentation import create_segmenter

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

### 单张图片分割

```python
# 对单张图片进行分割
masks, scores, logits = segmenter.segment_single_image(
    image_path="single_image.jpg",
    prompt_rectangles=[(350, 270, 100, 60)]
)
```

## 参数说明

### prompt_rectangles
矩形prompt列表，格式为 `[(x, y, w, h), ...]`：
- `x, y`: 矩形左上角坐标
- `w, h`: 矩形的宽度和高度

### 输出选项
- `save_masks`: 是否保存分割掩码（PNG格式）
- `save_visualization`: 是否保存可视化结果
- `output_dir`: 输出目录路径

## 测试结果

使用火球序列图片和prompt矩形 (350, 270, 100, 60) 的测试结果：

- ✅ 成功处理了4张火球序列图片
- ✅ 每张图片都生成了1个分割区域
- ✅ 生成了分割掩码和可视化结果
- ✅ 结果保存在 `output_sam/` 目录中

## 故障排除

### 1. 模型文件未找到
```
FileNotFoundError: 检查点文件不存在
```
**解决方案:** 确保SAM模型文件已下载到 `third_party/segment-anything/checkpoints/` 目录

### 2. OpenMP库冲突
```
OMP: Error #15: Initializing libomp.dylib
```
**解决方案:** 设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`

### 3. 中文字体警告
```
Glyph missing from font(s) DejaVu Sans
```
**解决方案:** 这是matplotlib的中文字体警告，不影响功能，可以忽略

### 4. 内存不足
```
CUDA out of memory
```
**解决方案:** 使用CPU模式或减小图像尺寸

## 性能说明

- **设备支持**: 自动检测CUDA、MPS、CPU
- **模型大小**: SAM ViT-L模型约1.2GB
- **处理速度**: 在MPS设备上每张图片约1-2秒
- **内存需求**: 建议8GB以上内存

## 注意事项

1. 首次运行会下载模型文件，需要网络连接
2. 分割质量取决于prompt矩形的位置和大小
3. 建议在图像的主要目标区域设置prompt
4. 支持多种图像格式（JPG、PNG、BMP等）

## 扩展功能

### 自定义模型
```python
# 使用不同的SAM模型
segmenter = create_segmenter(
    model_type="vit_b",  # 或 "vit_h", "vit_l"
    checkpoint_path="path/to/checkpoint.pth"
)
```

### 批量处理
```python
# 处理多个图像序列
for sequence_dir in sequence_dirs:
    image_paths = get_image_paths(sequence_dir)
    masks = segmenter.segment_sequence(image_paths, prompt_rectangles)
```

## 技术支持

如有问题，请检查：
1. 环境是否正确安装
2. 模型文件是否存在
3. 图像路径是否正确
4. prompt矩形是否合理
