# Conda 环境设置指南

本项目现在使用 Conda 来管理 Python 环境和依赖包，确保 SAM2 等需要 Python 3.10+ 的包能正常工作。

## 快速开始

### 1. 安装 Conda（如果未安装）

```bash
# 运行自动安装脚本
./install_conda.sh

# 或者手动安装
# macOS (Apple Silicon)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# macOS (Intel)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. 初始化 Conda

```bash
# 初始化 conda
conda init

# 重新打开终端或运行
source ~/.bashrc
# 或
source ~/.zshrc
```

### 3. 创建项目环境

```bash
# 进入项目目录
cd source

# 运行环境设置脚本
./setup.sh
```

## 环境管理

### 激活环境
```bash
conda activate fireball_calculator
```

### 退出环境
```bash
conda deactivate
```

### 查看所有环境
```bash
conda env list
```

### 删除环境
```bash
conda env remove -n fireball_calculator
```

### 导出环境配置
```bash
conda env export > environment.yml
```

## 文件说明

- `setup.sh`: 主环境设置脚本（使用 conda）
- `environment.yml`: Conda 环境配置文件（基础依赖）
- `install_conda.sh`: Conda 自动安装脚本
- `requirements.txt`: 传统 pip 依赖文件（备用）

## 环境特性

- **Python 版本**: 3.10（满足 SAM2 要求）
- **包管理**: Conda + pip
- **子模块支持**: 
  - Segment Anything (SAM) - 通过本地子模块安装
  - SAM2 (需要 Python 3.10+) - 通过本地子模块安装
- **自动版本检查**: 脚本会自动检查 Python 版本并跳过不兼容的包

## 子模块管理

### 添加子模块
```bash
# 添加 SAM
git submodule add https://github.com/facebookresearch/segment-anything.git source/third_party/segment-anything

# 添加 SAM2
git submodule add git@github.com:facebookresearch/sam2.git source/third_party/sam2

# 更新子模块
git submodule update --init --recursive
```

### 安装子模块
```bash
# 激活环境
conda activate fireball_calculator

# 安装 SAM（如果子模块存在）
pip install -e third_party/segment-anything

# 安装 SAM2（如果子模块存在且 Python >= 3.10）
pip install -e third_party/sam2
```

## 设计说明

### 为什么不在 environment.yml 中包含子模块？

1. **版本控制**: 子模块的版本由 Git 管理，不是由 conda 管理
2. **灵活性**: 用户可以选择是否安装子模块
3. **本地开发**: 支持本地修改和调试子模块代码
4. **可选依赖**: SAM/SAM2 不是必需的，用户可以选择性安装

### 安装流程

1. `conda env create -f environment.yml` - 创建基础环境
2. `setup.sh` 检测本地子模块并安装
3. 用户可以选择是否添加子模块

## 故障排除

### 问题 1: Conda 命令未找到
```bash
# 检查 conda 是否在 PATH 中
which conda

# 如果未找到，使用完整路径
~/miniconda3/bin/conda --version
```

### 问题 2: 环境激活失败
```bash
# 重新初始化 conda
conda init

# 重新打开终端
```

### 问题 3: SAM2 安装失败
```bash
# 检查 Python 版本
python --version

# 确保版本 >= 3.10
```

### 问题 4: 子模块未安装
```bash
# 检查子模块是否存在
ls -la third_party/

# 如果不存在，添加子模块
git submodule add https://github.com/facebookresearch/segment-anything.git source/third_party/segment-anything
git submodule add git@github.com:facebookresearch/sam2.git source/third_party/sam2
git submodule update --init --recursive
```

## 开发工作流

1. **激活环境**: `conda activate fireball_calculator`
2. **开发代码**: 正常开发
3. **安装新包**: `conda install package_name` 或 `pip install package_name`
4. **更新子模块**: `git submodule update --remote`
5. **导出环境**: `conda env export > environment.yml`
6. **退出环境**: `conda deactivate`

## 优势

- ✅ 支持多个 Python 版本
- ✅ 自动解决依赖冲突
- ✅ 环境隔离
- ✅ 支持 SAM2 (Python 3.10+)
- ✅ 跨平台兼容
- ✅ 子模块版本控制
- ✅ 可选依赖管理