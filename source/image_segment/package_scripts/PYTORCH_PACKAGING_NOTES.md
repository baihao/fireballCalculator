# PyTorch 应用打包注意事项

## 关键问题与解决方案

### 1. PyTorch 必须安装在 Conda 环境中

**问题**：如果 PyTorch 安装在用户目录（`~/Library/Python/`）而非 Conda 环境，PyInstaller 无法找到它。

**解决方案**：
```bash
conda activate fireball_calculator
pip install torch torchvision  # 确保安装到 conda 环境
```

**验证**：
```bash
python -c "import torch; print(torch.__file__)"
# 应该输出类似：/Users/xxx/miniconda3/envs/fireball_calculator/lib/python3.9/site-packages/torch/__init__.py
```

### 2. OpenMP 库冲突

**错误信息**：
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**原因**：PyTorch 和其他库（如 numpy, scipy）可能都包含 OpenMP 库，导致冲突。

**解决方案**：在打包脚本中设置环境变量
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 3. 使用 --onedir 而非 --onefile

**原因**：
- PyTorch 包含大量动态库（.dylib, .so, .dll）
- `--onefile` 模式下这些库的加载可能失败
- `--onedir` 模式将所有依赖放在目录中，更可靠

**权衡**：
- `--onefile`：单个文件，方便分发，但可能失败
- `--onedir`：一个目录，需要打包整个目录，但更可靠

### 4. 显式声明依赖

PyInstaller 无法自动识别 PyTorch 的所有依赖，需要显式声明：

```bash
--hidden-import torch
--hidden-import torch.nn
--hidden-import torch.nn.functional
--hidden-import segment_anything
--hidden-import segment_anything.modeling
--hidden-import segment_anything.predictor
--hidden-import segment_anything.utils
--collect-all torch
--collect-all segment_anything
```

### 5. 文件大小

**预期大小**：
- 打包后的应用程序通常会达到 **500MB - 2GB**
- PyTorch 本身就有 ~500MB
- 加上所有依赖，最终大小会很大

**优化建议**：
- 如果只使用 CPU，可以安装 CPU 版本的 PyTorch（更小）
- 考虑使用 Docker 容器化部署，而非打包成可执行文件

## 打包流程

### macOS
```bash
cd /path/to/fireball_calculator
./source/image_segment/package_scripts/build_mac.sh
```

### Windows
```powershell
cd C:\path\to\fireball_calculator
powershell -ExecutionPolicy Bypass -File source\image_segment\package_scripts\build_windows.ps1
```

## 常见错误及解决

### ModuleNotFoundError: No module named 'torch'

**可能原因**：
1. PyTorch 未安装在 conda 环境中
2. PyInstaller 未能收集 torch 模块

**解决步骤**：
1. 确认 torch 安装位置：
   ```bash
   python -c "import torch; print(torch.__file__)"
   ```
2. 如果不在 conda 环境，重新安装：
   ```bash
   conda activate fireball_calculator
   pip install torch torchvision
   ```
3. 检查打包警告：
   ```bash
   cat build/image_segment_propagation/warn-*.txt | grep torch
   ```

### SubprocessDiedError during collect_all('torch')

**原因**：OpenMP 库冲突

**解决**：确保打包脚本中有：
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 打包后文件过大

**预期**：500MB - 2GB 是正常的

**优化**：
- 使用 CPU 版本的 PyTorch
- 移除不必要的依赖
- 考虑替代方案（见下文）

## 替代方案

对于包含 PyTorch 的应用，以下方案可能更合适：

### 方案 1：Conda 包
创建 conda 包，用户安装后直接运行 Python 脚本。

### 方案 2：Docker 容器
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "source/image_segment/test_complete_propagation.py"]
```

### 方案 3：虚拟环境 + 启动脚本
提供环境文件和启动脚本，用户创建环境后运行：
```bash
conda env create -f environment.yml
conda activate fireball_calculator
python source/image_segment/test_complete_propagation.py
```

## 总结

PyTorch 应用打包虽然可行，但面临以下挑战：
- ✅ 技术上可行
- ⚠️  文件大小极大（0.5-2GB）
- ⚠️  打包时间长（5-10分钟）
- ⚠️  容易遇到库冲突
- ⚠️  跨平台打包复杂

**建议**：除非必须提供单个可执行文件，否则优先考虑虚拟环境或容器化方案。

