# GPU未检测到问题诊断

根据您的终端输出，SAM显示"未检测到GPU支持"，可能的原因如下：

## 可能原因分析

### 1. PyTorch是CPU版本（最常见）⭐
**症状**: `torch.cuda.is_available()` 返回 `False`

**检查方法**:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)  # 如果是None，说明是CPU版本
```

**解决方案**:
```bash
# 1. 卸载当前PyTorch
pip uninstall torch torchvision torchaudio

# 2. 安装支持CUDA的版本（以CUDA 11.8为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者访问 https://pytorch.org/get-started/locally/ 选择适合您系统的版本
```

### 2. 未安装NVIDIA GPU驱动
**检查方法**:
```bash
nvidia-smi
```
如果命令不存在或报错，说明未安装驱动。

**解决方案**:
- 访问 https://www.nvidia.com/drivers 下载并安装驱动

### 3. CUDA版本不匹配
**症状**: 有GPU硬件和驱动，但PyTorch仍检测不到

**检查方法**:
```bash
nvidia-smi  # 查看驱动支持的CUDA版本
python -c "import torch; print(torch.version.cuda)"  # 查看PyTorch的CUDA版本
```

**解决方案**: 确保PyTorch的CUDA版本与驱动兼容

## 快速诊断步骤

1. **检查PyTorch版本**:
   ```python
   import torch
   print("PyTorch版本:", torch.__version__)
   print("CUDA可用:", torch.cuda.is_available())
   print("CUDA版本:", torch.version.cuda)
   ```

2. **检查GPU硬件**:
   ```bash
   nvidia-smi
   ```

3. **根据结果采取行动**:
   - 如果 `torch.version.cuda` 是 `None` → 安装CUDA版本的PyTorch
   - 如果 `nvidia-smi` 失败 → 安装NVIDIA驱动
   - 如果两者都有但 `torch.cuda.is_available()` 仍为 `False` → 版本不匹配，重新安装匹配的PyTorch

## 推荐操作

运行以下Python代码进行完整诊断：

```python
import torch
import subprocess

print("=" * 60)
print("GPU诊断")
print("=" * 60)

# 1. PyTorch信息
print(f"\n1. PyTorch版本: {torch.__version__}")
cuda_compiled = hasattr(torch.version, 'cuda') and torch.version.cuda
if cuda_compiled:
    print(f"   CUDA版本: {torch.version.cuda}")
    print("   ✅ PyTorch支持CUDA")
else:
    print("   ❌ PyTorch是CPU版本（未编译CUDA支持）")
    print("   💡 需要重新安装支持CUDA的PyTorch")

# 2. CUDA运行时
print(f"\n2. CUDA运行时: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU数量: {torch.cuda.device_count()}")
else:
    print("   ❌ CUDA不可用")

# 3. 检查驱动
print("\n3. NVIDIA驱动:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=3)
    if result.returncode == 0:
        print("   ✅ 驱动已安装")
        print("   输出前3行:")
        for line in result.stdout.split('\n')[:3]:
            print(f"   {line}")
    else:
        print("   ⚠️ nvidia-smi执行失败")
except FileNotFoundError:
    print("   ❌ 未找到nvidia-smi（未安装驱动）")
except Exception as e:
    print(f"   ⚠️ 检查失败: {e}")

print("\n" + "=" * 60)
```

