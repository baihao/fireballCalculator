# 火球计算器 - 环境构建与运行（单文档版）

本文件已合并原 `CONDA_SETUP.md`、`DEPENDENCY_MANAGEMENT.md`、`README.md`（环境相关部分）与旧版 `QUICK_START.md` 的有效内容。

## 1. 准备工作
- 安装 Miniconda/Anaconda。
- 首次使用运行 `conda init` 并重新打开终端。

## 2. 创建/复用环境
```bash
cd source
./setup.sh
```
说明：
- 如果 `fireball_calculator` 环境不存在将自动创建；已存在则直接复用。
- 依赖以 `environment.yml` 为唯一事实来源：
  - conda: python、numpy、matplotlib、pandas、pillow、scipy
  - pip:   PySide6、opencv-python（在 `environment.yml` 的 `pip:` 部分）
- 若存在本地子模块 `third_party/segment-anything`，脚本会尝试以可编辑模式安装（失败不影响主流程）。

## 3. 激活环境与运行
```bash
conda activate fireball_calculator
python source/desktop/app.py
```

## 4. 常用维护命令
```bash
# 查看环境
conda env list

# 退出环境
conda deactivate

# 删除并重新创建
conda env remove -n fireball_calculator
cd source && ./setup.sh
```

## 5. 依赖维护（仅当需要新增依赖时）
- 统一修改 `environment.yml`：
```yaml
dependencies:
  - package_from_conda>=version
  - pip:
    - package_from_pip>=version
```
- 然后删除并重建环境，以确保一致性。

## 6. 常见问题
- 建议先更新 conda：`conda update -n base -c defaults conda`
- GUI 无法启动：确认已激活环境；如仍有问题，删除并重建环境。
- 终端未识别 conda：确保已执行 `conda init` 并重开终端。

## 7. 目录索引
- `source/environment.yml`：依赖清单（唯一事实来源）
- `source/setup.sh`：环境创建/复用脚本（不自动激活）
- `source/desktop/app.py`：桌面应用入口
