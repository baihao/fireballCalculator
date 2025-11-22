## 火球图像分割工具打包与分发说明

本目录用于维护 **火球图像分割 CLI 工具** 的打包脚本与说明，当前推荐方案为 **PyInstaller 优化版**，并提供 Docker 等替代方案的对比。

---

## 快速开始

### macOS 打包

cd /Users/hbai/cwz_project/fireball_calculator
./source/image_segment/package_scripts/build_mac_optimized.sh- **耗时**：约 3–5 分钟  
- **产物**：`dist/image_segment_propagation/`

### Windows 打包
hell
cd C:\path\to\fireball_calculator
powershell -ExecutionPolicy Bypass -File source\image_segment\package_scripts\build_windows.ps1- **耗时**：约 3–5 分钟  
- **产物**：`dist\image_segment_propagation\`

### 使用打包后的程序

macOS / Linux：

./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json            # 生成分割结果和可视化

./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json --no-viz   # 不生成可视化（更快）

./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json --out=my_outputWindows：
hell
.\dist\image_segment_propagation\image_segment_propagation.exe `
  test_data\fireball_sequence.json --no-viz --out my_output---

## 目录中的主要文件

- **`build_mac_optimized.sh`**：macOS 打包脚本（当前推荐入口）
- **`build_windows.ps1`**：Windows 打包脚本

> 原来的打包文档（`PACKAGING_GUIDE.md`、`OPTIMIZATION_GUIDE.md`、`ALTERNATIVE_SOLUTIONS.md`、`PYTORCH_PACKAGING_NOTES.md`）内容已收敛到本 README 中，如果不再需要，可移除那些文件。

---

## 当前打包方式：PyInstaller 优化版

- **体积**：约 800–950 MB  
- **性能**：处理 200 张图约 130 秒  
  - 对比原始 Python 脚本（约 104 秒），慢约 25%（绝对值 +26 秒）
- **兼容性**：macOS 11.0+ / Windows 10+  
- **特点**：
  - 解压即用，无需安装 Python / Conda
  - 已完整适配 PyTorch + SAM 模型

### 已包含的关键优化

- **只打包一个 SAM 模型**  
  - 仅保留 `vit_b`（约 375 MB），不再打包 `vit_l` / `vit_h`，减少约 1.1 GB 体积。

- **排除不必要模块**  
  - 例如 `pandas`、`IPython`、`jupyter`、`notebook`、`tkinter`、测试模块等。

- **使用 `--onedir` 模式**  
  - 相比 `--onefile`，对 PyTorch 动态库加载更稳定。

- **打包后自动清理**  
  - 删除 `.pyc`、`__pycache__` 目录和无用测试文件。

---

## 使用与分发建议

### 1. 开发 / 调试阶段

- **推荐方式**：Python 脚本 + Conda 环境（性能 100%）

conda activate fireball_calculator
python source/image_segment/test_complete_propagation.py \
  test_data/fireball_sequence.json --no-viz**优点**：

- 不需要打包，修改代码立即生效；
- 调试最方便，性能最佳。

### 2. 对外桌面分发

- **推荐方式**：PyInstaller 优化版（本 README 的打包脚本）
- 建议流程：
  1. 按上文脚本打包；
  2. 将 `dist/image_segment_propagation/` 压缩为 ZIP；
  3. 附上简单的“使用说明”（命令示例或截图）。

用户侧典型用法：

unzip image_segment_v1.0_macos.zip
cd image_segment_propagation
./image_segment_propagation <你的json文件>### 3. 服务器 / 批处理 / CI

- **推荐方式**：Docker 容器（可选）

示例 Dockerfile（简化版）：
file
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY source/environment.yml .
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    opencv-python>=4.11.0 \
    numpy>=2.0.0 \
    matplotlib>=3.9.0

COPY source/ ./source/
COPY source/third_party/segment-anything ./source/third_party/segment-anything
RUN pip install -e source/third_party/segment-anything

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONPATH=/app/source

ENTRYPOINT ["python", "source/image_segment/test_complete_propagation.py"]
CMD ["--help"]构建与运行：

docker build -t fireball-segmentation:v1.0 .

docker run --rm \
  -v $(pwd)/test_data:/data \
  fireball-segmentation:v1.0 \
  /data/fireball_sequence.json --no-viz---

## 打包方案对比

| **方案**                  | **性能** | **体积**        | **打包/构建时间** | **易用性**                 | **推荐场景**       |
|---------------------------|---------|-----------------|-------------------|----------------------------|--------------------|
| Python 脚本 + Conda       | ⭐⭐⭐⭐⭐   | N/A             | 0 分钟            | ⭐⭐（需配置环境）          | 开发 / 调试        |
| PyInstaller 优化版        | ⭐⭐⭐⭐    | 800–950 MB      | 3–5 分钟          | ⭐⭐⭐⭐⭐（解压即用）         | 对外桌面分发       |
| Docker 镜像               | ⭐⭐⭐⭐⭐   | ~700 MB         | 10–15 分钟        | ⭐⭐⭐（需安装 Docker）      | 服务器 / 批处理 / CI |

**简要选择建议**：

- 日常开发 → **脚本 + Conda**  
- 给合作方 / 非技术用户 → **PyInstaller 优化版**  
- 服务器 / 集群 → **Docker**

---

## 常见问题

- **打包需要多久？**  
  - PyInstaller：约 3–5 分钟  
  - Docker 首次构建：约 10–15 分钟  

- **打包后程序会慢多少？**  
  - PyInstaller 版本相对 Python 原始脚本约慢 25%，处理 200 张图从 ~104 s 到 ~130 s。  

- **为什么不采用 Nuitka？**  
  - 含 PyTorch 的项目存在 C++ ABI 冲突，运行时容易崩溃（Exit code 139），调试成本高、收益有限，因此当前不采用。

---

现在你可以：

- 把上面这段 README 覆盖到 `source/image_segment/package_scripts/README.md`；
- 若确认不再需要单独文档，可安全删除：
  - `ALTERNATIVE_SOLUTIONS.md`
  - `OPTIMIZATION_GUIDE.md`
  - `PACKAGING_GUIDE.md`
  - `PYTORCH_PACKAGING_NOTES.md`
- 如果 `performance_test.sh` 确认平时不会再用，也可以一并删掉，保持目录干净，仅保留：
  - `build_mac_optimized.sh`
  - `build_windows.ps1`
  - 新的 `README.md