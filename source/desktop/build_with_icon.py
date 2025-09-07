#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用图标构建可执行文件的脚本
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_with_icon():
    """使用图标构建可执行文件"""
    
    # 获取项目路径
    project_root = Path(__file__).parent.parent.parent
    source_dir = project_root / 'source'
    desktop_dir = source_dir / 'desktop'
    
    print("🚀 开始构建带图标的爆炸火球分析系统...")
    
    # 检查图标文件是否存在
    icon_file = desktop_dir / 'icon' / 'fireball_app_icon.ico'
    if not icon_file.exists():
        print("❌ 图标文件不存在，请先运行 icon/create_professional_icon.py")
        return False
    
    # 切换到desktop目录
    os.chdir(desktop_dir)
    
    # 构建命令（使用图标）
    build_cmd = [
        'pyinstaller',
        '--onefile',  # 打包成单个文件
        '--windowed',  # 不显示控制台窗口
        '--name=FireballAnalysis',
        f'--icon=icon/fireball_app_icon.ico',  # 使用图标
        '--add-data=../fireball_radius_calculator.py:.',
        '--add-data=../fireball_temperature_calculator.py:.',
        '--add-data=../transmissivity_calculator.py:.',
        '--add-data=../fireball_heat_radiation_calculator.py:.',
        '--add-data=../../images:images',
        '--add-data=../../document:document',
        '--add-data=../../papers:papers',
        '--add-data=../../temp.csv:.',
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtWidgets',
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=matplotlib.backends.backend_qt5agg',
        '--hidden-import=matplotlib.backends.backend_agg',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=cv2',
        '--hidden-import=PIL',
        '--clean',  # 清理之前的构建
        'app.py'
    ]
    
    try:
        print("📦 执行打包命令...")
        print(f"🎨 使用图标: {icon_file}")
        result = subprocess.run(build_cmd, check=True, capture_output=True, text=True)
        print("✅ 打包成功！")
        
        # 检查输出文件
        exe_path = desktop_dir / 'dist' / 'FireballAnalysis.exe'  # Windows
        if not exe_path.exists():
            exe_path = desktop_dir / 'dist' / 'FireballAnalysis'  # macOS/Linux
            
        if exe_path.exists():
            print(f"🎉 可执行文件已生成: {exe_path}")
            print(f"📁 文件大小: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            
            # 创建发布目录
            release_dir = project_root / 'release'
            release_dir.mkdir(exist_ok=True)
            
            # 复制可执行文件到发布目录
            release_exe = release_dir / exe_path.name
            shutil.copy2(exe_path, release_exe)
            print(f"📋 已复制到发布目录: {release_exe}")
            
        else:
            print("❌ 未找到生成的可执行文件")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 打包失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

def create_release_package():
    """创建发布包"""
    
    project_root = Path(__file__).parent.parent.parent
    release_dir = project_root / 'release'
    release_dir.mkdir(exist_ok=True)
    
    print("📦 创建发布包...")
    
    # 复制必要文件到发布目录
    files_to_copy = [
        ('README.md', 'README.md'),
        ('source/requirements.txt', 'requirements.txt'),
        ('temp.csv', 'temp.csv'),
        ('images', 'images'),
        ('document', 'document'),
    ]
    
    for src, dst in files_to_copy:
        src_path = project_root / src
        dst_path = release_dir / dst
        
        if src_path.exists():
            if src_path.is_dir():
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            print(f"✅ 已复制: {src} -> {dst}")
    
    # 创建使用说明
    usage_file = release_dir / '使用说明.txt'
    with open(usage_file, 'w', encoding='utf-8') as f:
        f.write("""爆炸火球分析系统 - 使用说明

1. 运行应用程序
   - Windows: 双击 FireballAnalysis.exe
   - macOS/Linux: 运行 ./FireballAnalysis

2. 系统要求
   - Windows 10/11, macOS 10.14+, 或 Linux
   - 至少 4GB 内存
   - 至少 500MB 可用磁盘空间

3. 功能模块
   - 输入模块: 导入火球图像序列和温度数据
   - 特征提取: 自动分析火球特征
   - 建模预测: 生成预测曲线和结果

4. 文件说明
   - images/: 图像资源目录
   - document/: 技术文档
   - temp.csv: 温度数据示例

5. 技术支持
   - 项目地址: https://github.com/baihao/fireballCalculator
   - 问题反馈: 请使用 GitHub Issues

版本: 1.0
构建日期: """ + str(Path(__file__).stat().st_mtime))
    
    print(f"📋 已创建使用说明: {usage_file}")
    print(f"🎉 发布包创建完成: {release_dir}")

if __name__ == "__main__":
    success = build_with_icon()
    if success:
        create_release_package()
        print("\n🎉 构建完成！")
        print("📁 可执行文件位于: source/desktop/dist/")
        print("📦 发布包位于: release/")
    else:
        print("\n💥 构建失败，请检查错误信息。")
