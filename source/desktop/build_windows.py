#!/usr/bin/env python3
"""
Windows平台打包脚本 - 火球分析器
适用于Windows系统
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("=" * 50)
    print("火球分析器 - Windows打包工具")
    print("=" * 50)
    
    # 切换到desktop目录
    desktop_dir = Path(__file__).parent
    os.chdir(desktop_dir)
    print(f"工作目录: {desktop_dir}")
    
    # 检查必要文件
    if not Path("app.py").exists():
        print("❌ 未找到 app.py 文件")
        return False
    
    if not Path("icon/fireball_app_icon.ico").exists():
        print("❌ 未找到图标文件")
        return False
    
    print("✅ 必要文件检查通过")
    
    # 清理之前的构建
    print("\n🧹 清理之前的构建文件...")
    for dir_name in ['dist', 'build']:
        dir_path = Path(dir_name)
        if dir_path.exists():
            import shutil
            shutil.rmtree(dir_path)
            print(f"✅ 清理 {dir_name} 目录")
    
    # 构建命令 - Windows版本
    print("\n🔨 开始构建Windows应用程序...")
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',                    # 单文件模式
        '--windowed',                   # 无控制台窗口
        '--name=FireballAnalysis',      # 应用名称
        '--icon=icon/fireball_app_icon.ico',  # 图标
        '--add-data=icon/fireball_app_icon.ico;icon',  # Windows使用分号分隔
        '--add-data=../fireball_radius_calculator.py;.',  # 添加计算器
        '--add-data=../fireball_temperature_calculator.py;.',
        '--add-data=../fireball_heat_radiation_calculator.py;.',
        '--add-data=../transmissivity_calculator.py;.',
        '--add-data=../../temp.csv;.',  # 添加温度数据
        '--add-data=../../images;images',  # 添加图片资源
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtWidgets', 
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=matplotlib.backends.backend_qt5agg',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=cv2',
        '--hidden-import=PIL',
        '--hidden-import=scipy',
        '--hidden-import=scipy.integrate',
        '--hidden-import=scipy.interpolate',
        'app.py'
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("这可能需要几分钟时间，请耐心等待...")
    
    try:
        result = subprocess.run(cmd, cwd=desktop_dir)
        
        if result.returncode == 0:
            print("✅ 构建成功!")
            
            # 检查输出文件
            exe_path = Path("dist/FireballAnalysis.exe")
            if exe_path.exists():
                file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
                print(f"📦 可执行文件: {exe_path}")
                print(f"📏 文件大小: {file_size:.1f} MB")
                
                print("\n" + "=" * 50)
                print("🎉 Windows打包完成!")
                print("=" * 50)
                print(f"可执行文件: {exe_path.absolute()}")
                print(f"大小: {file_size:.1f} MB")
                print("\n使用方法:")
                print(f"1. 双击运行: {exe_path}")
                print("2. 或在命令提示符中运行:")
                print(f"   {exe_path.absolute()}")
                print("\n注意:")
                print("- 首次运行可能需要较长时间")
                print("- 确保Windows系统已安装必要的运行时库")
                print("- 如果遇到安全警告，请选择'允许'")
                
                return True
            else:
                print("❌ 未找到可执行文件")
                return False
        else:
            print("❌ 构建失败!")
            return False
            
    except Exception as e:
        print(f"❌ 构建过程中出现异常: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
