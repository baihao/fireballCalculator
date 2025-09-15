# 火球分割优化指南

## 问题分析

原始的分割结果只覆盖了prompt矩形区域的一小部分，主要原因：
1. **矩形prompt太小**：原始矩形 (350, 270, 100, 60) 可能没有完全覆盖火球区域
2. **缺乏点prompt引导**：仅使用矩形prompt，SAM难以理解目标边界
3. **火球边缘模糊**：火球边缘渐变，需要更精确的prompt策略

## 改进方案

### 1. 基础改进
- ✅ **添加中心点prompt**：在矩形中心添加正点
- ✅ **使用多掩码输出**：让SAM生成多个候选，选择最佳结果
- ✅ **扩大矩形范围**：将矩形扩大1.5-2倍以覆盖更多火球区域

### 2. 高级优化
- ✅ **多点prompt策略**：在矩形中心和四个角添加点prompt
- ✅ **多矩形策略**：使用多个不同大小的矩形
- ✅ **自适应矩形**：根据火球大小动态调整矩形

## 使用方法

### 方法1: 使用优化方法（推荐）

```python
from image_segment.sam_sequence_segmentation import create_segmenter

# 创建分割器
segmenter = create_segmenter()

# 使用火球分割优化方法
masks = segmenter.segment_fireball_optimized(
    image_paths=image_paths,
    prompt_rectangles=[(350, 270, 100, 60)],
    output_dir="output",
    save_masks=True,
    save_visualization=True
)
```

### 方法2: 使用改进的基础方法

```python
# 使用改进的基础方法
masks = segmenter.segment_sequence(
    image_paths=image_paths,
    prompt_rectangles=[(350, 270, 100, 60)],
    output_dir="output",
    save_masks=True,
    save_visualization=True,
    use_multimask=True,        # 使用多掩码输出
    add_center_points=True     # 添加中心点prompt
)
```

### 方法3: 使用更大的矩形

```python
# 使用扩大的矩形
masks = segmenter.segment_fireball_optimized(
    image_paths=image_paths,
    prompt_rectangles=[(350, 270, 150, 90)],  # 1.5倍大小
    output_dir="output"
)
```

## 测试结果

运行了5种不同的prompt策略：

1. **原始矩形** (350, 270, 100, 60)
2. **扩大矩形 (1.5x)** (350, 270, 150, 90)
3. **扩大矩形 (2x)** (350, 270, 200, 120)
4. **中心偏移矩形** (325, 255, 150, 90)
5. **多个矩形** 组合使用

所有策略都成功生成了分割结果，保存在 `output_advanced/` 目录中。

## 建议的最佳实践

### 1. 矩形大小调整
```python
# 根据火球大小调整矩形
# 如果火球较大，使用更大的矩形
prompt_rectangles = [(350, 270, 150, 90)]  # 1.5倍
# 或者
prompt_rectangles = [(350, 270, 200, 120)]  # 2倍
```

### 2. 多矩形策略
```python
# 使用多个矩形覆盖不同区域
prompt_rectangles = [
    (350, 270, 100, 60),   # 原始矩形
    (340, 260, 120, 80),   # 左上偏移
    (360, 280, 80, 40)     # 右下偏移
]
```

### 3. 点prompt优化
```python
# 在关键位置添加点prompt
# 火球中心、边缘等关键位置
```

## 运行测试

### 基础测试
```bash
cd source
KMP_DUPLICATE_LIB_OK=TRUE ./python310 image_segment/test_sam_segmentation.py
```

### 优化测试
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./python310 image_segment/test_fireball_optimized.py
```

### 高级测试
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./python310 image_segment/test_fireball_advanced.py
```

## 输出结果

所有测试结果保存在以下目录：
- `output_sam/` - 基础SAM分割结果
- `output_optimized/` - 优化方法对比结果
- `output_advanced/` - 多种策略测试结果

每个目录包含：
- `masks/` - 分割掩码（PNG格式）
- `visualization/` - 可视化结果（原图+分割结果）

## 进一步优化建议

### 1. 手动调整矩形
根据火球的实际位置和大小，手动调整prompt矩形：
```python
# 观察火球位置，调整矩形
prompt_rectangles = [(x, y, w, h)]  # 根据实际情况调整
```

### 2. 使用点prompt
在火球的关键位置添加点prompt：
```python
# 在火球中心、边缘等位置添加点
point_coords = np.array([[center_x, center_y]])
point_labels = np.array([1])  # 正点
```

### 3. 后处理优化
对分割结果进行后处理：
- 形态学操作（开闭运算）
- 连通域分析
- 边缘平滑

### 4. 多尺度分割
使用不同尺度的矩形进行分割，然后融合结果。

## 故障排除

### 1. 分割结果不理想
- 尝试更大的矩形
- 添加点prompt
- 使用多矩形策略

### 2. 分割区域过小
- 检查矩形是否覆盖火球
- 调整矩形大小和位置
- 使用多点prompt

### 3. 分割区域过大
- 缩小矩形范围
- 添加负点prompt
- 调整阈值参数

## 总结

通过以上优化方案，火球分割效果应该会有显著改善。建议：

1. **首先尝试** `segment_fireball_optimized` 方法
2. **调整矩形大小** 以更好地覆盖火球区域
3. **查看可视化结果** 选择最佳策略
4. **根据实际效果** 进一步微调参数

记住：SAM的分割质量很大程度上取决于prompt的质量，合适的prompt是获得良好分割结果的关键。
