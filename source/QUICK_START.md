# 火球计算器快速使用指南

## 环境设置

### 1. 创建环境（首次使用）
```bash
cd source
./setup.sh
```

### 2. 激活环境（每次使用）
```bash
# 方法1: 使用conda激活环境
conda activate fireball_calculator

# 方法2: 使用一键激活脚本（推荐）
source ./activate_env.sh
```

## 使用方法

### 方法1: 使用包装脚本（推荐）
```bash
# 检查Python版本
./python310 --version

# 查看已安装的包
./pip310 list

# 运行应用程序
./python310 app.py

# 安装新包
./pip310 install package_name
```

### 方法2: 使用别名（需要先设置）
```bash
# 设置别名
source ./setup_aliases.sh

# 检查Python版本
python310 --version

# 查看已安装的包
pip310 list

# 运行应用程序
python310 app.py

# 安装新包
pip310 install package_name
```

### 环境管理
```bash
# 查看所有conda环境
conda env list

# 退出环境
conda deactivate

# 删除环境（如果需要重新创建）
conda env remove -n fireball_calculator

# 重新创建环境
./setup.sh
```

## 别名说明

- `python310`: 指向环境中的Python 3.10.18
- `pip310`: 指向环境中的pip

这些别名只在当前终端会话中有效，每次打开新终端都需要重新设置。

## 永久设置别名

如果您希望别名永久有效，请将以下行添加到 `~/.zshrc` 或 `~/.bashrc`：

```bash
alias python310='/Users/hbai/miniconda3/envs/fireball_calculator/bin/python'
alias pip310='/Users/hbai/miniconda3/envs/fireball_calculator/bin/pip'
```

然后运行：
```bash
source ~/.zshrc  # 或 source ~/.bashrc
```

## 故障排除

### 问题1: 命令未找到
```bash
# 确保已激活环境
conda activate fireball_calculator

# 重新设置别名
source ./setup_aliases.sh
```

### 问题2: Python版本错误
```bash
# 使用完整路径
/Users/hbai/miniconda3/envs/fireball_calculator/bin/python --version
```

### 问题3: Qt库冲突（GUI应用无法启动）
```bash
# 自动检测和修复Qt冲突
./check_qt_conflict.sh

# 或者手动修复
./fix_qt_conflict.sh
```

### 问题4: 环境损坏
```bash
# 删除并重新创建环境
conda env remove -n fireball_calculator
./setup.sh
```

### 问题5: 重新创建环境后仍有Qt冲突
```bash
# setup.sh 现在会自动检测和修复Qt冲突
# 如果仍有问题，手动运行：
./fix_qt_conflict.sh
```
