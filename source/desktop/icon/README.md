# 图标资源目录

本目录包含爆炸火球分析系统的所有图标文件和生成脚本。

## 📁 文件说明

### 推荐使用图标
- **`fireball_app_icon.ico`** - 主要应用图标（推荐用于打包）
- **`fireball_app_icon.png`** - PNG格式版本

### 其他风格图标
- **`fireball_professional_icon.ico`** - 专业风格图标
- **`fireball_modern_icon.ico`** - 现代风格图标
- **`fireball_minimal_icon.ico`** - 简约风格图标

### 生成脚本
- **`generate_icon.py`** - 基础图标生成脚本
- **`create_professional_icon.py`** - 专业图标生成脚本

### 文档
- **`图标使用说明.md`** - 详细的使用说明文档

## 🚀 快速使用

### 生成图标
```bash
cd icon
python create_professional_icon.py
```

### 在打包中使用
```bash
cd ..
pyinstaller --icon=icon/fireball_app_icon.ico --onefile --windowed app.py
```

## 🎨 图标特点

- **火球设计**: 金色渐变核心，多层火焰效果
- **分析元素**: 蓝色数据点和连接线
- **专业背景**: 深蓝色渐变背景
- **多尺寸支持**: ICO格式包含16x16到256x256多种尺寸

## 📋 文件清单

```
icon/
├── fireball_app_icon.ico          # 推荐使用的主图标
├── fireball_app_icon.png          # PNG格式主图标
├── fireball_professional_icon.ico # 专业风格图标
├── fireball_modern_icon.ico       # 现代风格图标
├── fireball_minimal_icon.ico      # 简约风格图标
├── generate_icon.py               # 基础图标生成脚本
├── create_professional_icon.py    # 专业图标生成脚本
├── 图标使用说明.md                # 详细使用说明
└── README.md                      # 本文件
```

---

**注意**: 如需修改图标设计，请运行相应的生成脚本。
