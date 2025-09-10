# 依赖管理说明

## 文件作用

### `environment.yml` - 主要依赖管理
- **用途**: conda环境创建和基础依赖管理
- **内容**: Python版本、conda包、核心pip包
- **使用**: `conda env create -f environment.yml`

**包含的包**:
- Python 3.10
- numpy, matplotlib, pandas, pillow, scipy (conda安装)
- PySide6, opencv-python (pip安装)

### `requirements.txt` - 额外依赖管理
- **用途**: 额外的、可选的依赖包
- **内容**: 开发和调试工具、可选功能包
- **使用**: `pip install -r requirements.txt`

**设计原则**:
- 不包含 `environment.yml` 中已有的包
- 主要用于可选功能和开发工具
- 所有包都注释掉，需要时手动启用

## 安装流程

### 1. 环境创建
```bash
# 使用 environment.yml 创建基础环境
conda env create -f environment.yml
```

### 2. 额外依赖安装
```bash
# 激活环境
conda activate fireball_calculator

# 安装额外依赖（如果需要）
pip install -r requirements.txt
```

### 3. 自动安装（推荐）
```bash
# setup.sh 会自动处理所有依赖
./setup.sh
```

## 最佳实践

### 添加新依赖

#### 核心依赖（必须）
```yaml
# 添加到 environment.yml
dependencies:
  - package_name>=version
```

#### 可选依赖
```bash
# 添加到 requirements.txt（取消注释）
package_name>=version
```

### 版本管理

#### 固定版本
```yaml
# environment.yml - 固定核心依赖版本
- numpy=1.24.0
- matplotlib=3.7.0
```

#### 灵活版本
```bash
# requirements.txt - 允许版本范围
package_name>=1.0.0,<2.0.0
```

## 故障排除

### 依赖冲突
```bash
# 检查已安装的包
conda list
pip list

# 重新创建环境
conda env remove -n fireball_calculator
./setup.sh
```

### 版本问题
```bash
# 更新特定包
pip install --upgrade package_name

# 降级特定包
pip install package_name==specific_version
```

## 文件维护

### 定期更新
1. 检查 `environment.yml` 中的包版本
2. 更新 `requirements.txt` 中的可选依赖
3. 测试环境创建流程

### 清理无用依赖
```bash
# 查看未使用的包
pip-autoremove

# 手动清理
pip uninstall package_name
```

## 总结

- **environment.yml**: 核心依赖，必须维护
- **requirements.txt**: 可选依赖，按需使用
- **setup.sh**: 自动安装，推荐使用
- **原则**: 避免重复，明确分工
