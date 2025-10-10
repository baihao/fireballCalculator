# PyInstaller 打包优化指南

## 问题分析

### 1. 执行速度慢的原因

**PyInstaller 打包的程序通常比 Python 脚本慢，主要原因：**

1. **冷启动开销**
   - 打包的程序需要先解压内部资源到临时目录
   - 第一次加载动态库需要验证签名
   - Python 解释器初始化

2. **import 机制**
   - PyInstaller 使用自定义的 import hook
   - 从打包文件中加载模块比从文件系统加载慢

3. **没有使用编译优化**
   - PyInstaller 打包的是 `.pyc` 字节码，不是原生代码
   - 没有进行 JIT 编译优化

**解决方案：**
- 使用 `--onedir` 而非 `--onefile`（已采用）
- 排除不必要的模块（见下文）
- 考虑使用 Nuitka（将 Python 编译为 C++）

### 2. 体积过大的原因

**当前打包体积分析（优化前）：**

| 组件 | 大小 | 是否必需 | 优化建议 |
|------|------|---------|----------|
| third_party/checkpoints | 1.5GB | 部分 | **只保留一个模型文件** |
| torch | 339MB | 是 | 使用 `--collect-submodules` 替代 `--collect-all` |
| cv2 | 89MB | 是 | 无法优化，OpenCV 本身较大 |
| scipy | 38MB | 否 | **可排除** |
| pandas | 19MB | 否 | **可排除** |
| matplotlib | 11MB | 是 | 排除不必要的后端 |

**优化策略：**

#### A. 只打包一个 SAM 模型（减少 ~1.1GB）

```bash
# 只复制 vit_b 模型（375MB），不复制 vit_l 和 vit_h
cp checkpoints/sam_vit_b*.pth temp_dir/
```

#### B. 排除不使用的模块（减少 ~50-100MB）

```bash
--exclude-module pandas \
--exclude-module scipy.stats \
--exclude-module tkinter \
--exclude-module unittest \
--exclude-module test \
--exclude-module IPython \
--exclude-module jupyter
```

#### C. 使用 `--collect-submodules` 替代 `--collect-all`

```bash
# 原来：打包所有 torch 相关文件（包括文档、示例）
--collect-all torch

# 优化：只收集 Python 模块
--collect-submodules torch
```

#### D. 清理 .pyc 和 __pycache__

```bash
find dist/ -name "*.pyc" -delete
find dist/ -type d -name "__pycache__" -exec rm -rf {} +
```

### 3. 确保只打包需要的库

**策略 1：使用 spec 文件精确控制**

PyInstaller 生成的 `.spec` 文件可以精确控制打包内容。

**策略 2：使用 `--exclude-module` 显式排除**

分析导入：
```bash
python -m PyInstaller --log-level DEBUG your_script.py 2>&1 | grep "import"
```

**策略 3：事后分析和清理**

```bash
# 找出最大的文件
du -sh dist/app/_internal/* | sort -h

# 检查是否使用（运行程序，删除文件，测试是否报错）
```

## 优化版打包脚本

### 使用优化版脚本

```bash
cd /path/to/fireball_calculator
./source/image_segment/package_scripts/build_mac_optimized.sh
```

### 优化效果对比

| 版本 | 大小 | 启动时间 | 说明 |
|------|------|----------|------|
| 原始版 | ~2.0GB | 3-5秒 | 包含所有 SAM 模型和不必要的库 |
| 优化版 | ~800MB | 2-3秒 | 只包含一个模型，排除不必要的库 |
| Python脚本 | N/A | 1-2秒 | 直接运行，无打包开销 |

### 优化版包含的改进

1. **只打包单个 SAM 模型**
   - 默认使用 `vit_b`（375MB）
   - 可根据需要选择 `vit_l` 或 `vit_h`

2. **排除不必要的模块**
   ```bash
   --exclude-module pandas
   --exclude-module scipy.stats
   --exclude-module tkinter
   --exclude-module unittest
   --exclude-module setuptools
   ```

3. **使用更精确的收集策略**
   ```bash
   --collect-submodules torch  # 替代 --collect-all torch
   ```

4. **自动清理**
   - 删除 `.pyc` 文件
   - 删除 `__pycache__` 目录
   - 删除测试文件

## 进一步优化选项

### 选项 1：使用 PyInstaller + UPX 压缩

```bash
# 安装 UPX
brew install upx

# 打包时启用压缩
pyinstaller --onedir --upx-dir /usr/local/bin your_script.py
```

**效果**：可减少 30-50% 体积，但会增加启动时间。

### 选项 2：使用 Nuitka（推荐用于性能敏感场景）

```bash
# 安装 Nuitka
pip install nuitka

# 编译
python -m nuitka \
  --standalone \
  --onefile \
  --plugin-enable=numpy \
  --plugin-enable=torch \
  your_script.py
```

**优点**：
- 真正的原生代码编译
- 执行速度接近原生 Python
- 体积通常比 PyInstaller 小

**缺点**：
- 编译时间更长（10-30分钟）
- 对某些库的支持不如 PyInstaller 完善

### 选项 3：Docker 容器化（推荐用于服务器部署）

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY source/ ./source/
COPY test_data/ ./test_data/
CMD ["python", "source/image_segment/test_complete_propagation.py"]
```

**优点**：
- 体积小（分层缓存）
- 跨平台一致
- 易于更新和维护

**缺点**：
- 需要 Docker 环境
- 不适合桌面应用分发

## 测试和验证

### 1. 功能测试

```bash
# 测试基本功能
./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json --no-viz

# 测试完整功能
./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json
```

### 2. 性能测试

```bash
# 测试启动时间
time ./dist/image_segment_propagation/image_segment_propagation --help

# 测试运行时间
time ./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json --no-viz
```

### 3. 缺失依赖检测

运行程序，检查是否有 `ModuleNotFoundError`：

```bash
./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json --no-viz 2>&1 | grep "ModuleNotFoundError"
```

## 最佳实践建议

### 对于不同使用场景的建议

**场景 1：内部使用/开发**
- **推荐**：直接使用 Python 脚本 + conda 环境
- **原因**：最快，最灵活，体积最小

**场景 2：给无技术背景用户**
- **推荐**：PyInstaller 优化版打包
- **原因**：无需安装 Python，双击运行

**场景 3：服务器部署**
- **推荐**：Docker 容器化
- **原因**：便于管理、更新、扩展

**场景 4：性能要求极高**
- **推荐**：Nuitka 编译
- **原因**：接近原生性能

### 权衡表

| 方案 | 体积 | 速度 | 易用性 | 维护性 |
|------|------|------|--------|--------|
| Python脚本 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| PyInstaller 原始 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| PyInstaller 优化 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Nuitka | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Docker | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 总结

1. **体积优化**：从 2GB → 800MB
2. **速度优化**：启动时间从 5秒 → 3秒
3. **只打包必要库**：通过 `--exclude-module` 和精确的模型选择

**推荐使用优化版脚本**：`build_mac_optimized.sh`

