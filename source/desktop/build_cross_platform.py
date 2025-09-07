#!/usr/bin/env python3
"""
跨平台打包脚本 - 火球分析器
支持 macOS 和 Windows 系统
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def get_platform_info():
    """获取当前平台信息"""
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    if system == "darwin":
        return "macos", arch
    elif system == "windows":
        return "windows", arch
    elif system == "linux":
        return "linux", arch
    else:
        return "unknown", arch

def create_spec_file(platform_name, arch):
    """创建PyInstaller spec文件"""
    desktop_dir = Path(__file__).parent
    icon_file = desktop_dir / 'icon' / 'fireball_app_icon.ico'
    
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['{desktop_dir}'],
    binaries=[],
    datas=[
        ('icon/fireball_app_icon.ico', 'icon'),
        ('../fireball_radius_calculator.py', '.'),
        ('../fireball_temperature_calculator.py', '.'),
        ('../fireball_heat_radiation_calculator.py', '.'),
        ('../transmissivity_calculator.py', '.'),
        ('../../temp.csv', '.'),
        ('../../images', 'images'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
        'numpy',
        'pandas',
        'cv2',
        'PIL',
        'scipy',
        'scipy.integrate',
        'scipy.interpolate',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FireballAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{icon_file}',
)
'''
    
    spec_file = desktop_dir / f'fireball_analysis_{platform_name}_{arch}.spec'
    with open(spec_file, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    return spec_file

def build_application():
    """构建应用程序"""
    print("=" * 60)
    print("火球分析器 - 跨平台打包工具")
    print("=" * 60)
    
    # 获取平台信息
    platform_name, arch = get_platform_info()
    print(f"检测到平台: {platform_name} ({arch})")
    
    # 检查当前目录
    desktop_dir = Path(__file__).parent
    os.chdir(desktop_dir)
    print(f"工作目录: {desktop_dir}")
    
    # 检查必要文件
    required_files = [
        'app.py',
        'framework.py',
        'input_tab.py',
        'extract_tab.py',
        'model_tab.py',
        'icon/fireball_app_icon.ico',
        '../fireball_radius_calculator.py',
        '../fireball_temperature_calculator.py',
        '../fireball_heat_radiation_calculator.py',
        '../transmissivity_calculator.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ 所有必要文件检查通过")
    
    # 创建spec文件
    print("\n📝 创建PyInstaller配置文件...")
    spec_file = create_spec_file(platform_name, arch)
    print(f"✅ 配置文件创建完成: {spec_file.name}")
    
    # 清理之前的构建
    print("\n🧹 清理之前的构建文件...")
    dist_dir = desktop_dir / 'dist'
    build_dir = desktop_dir / 'build'
    
    if dist_dir.exists():
        import shutil
        shutil.rmtree(dist_dir)
        print("✅ 清理 dist 目录")
    
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print("✅ 清理 build 目录")
    
    # 开始构建
    print(f"\n🔨 开始构建应用程序 ({platform_name})...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    try:
        # 使用spec文件构建
        cmd = [sys.executable, '-m', 'PyInstaller', str(spec_file)]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=desktop_dir)
        
        if result.returncode == 0:
            print("✅ 构建成功!")
            
            # 检查输出文件
            exe_name = "FireballAnalysis"
            if platform_name == "windows":
                exe_name += ".exe"
            
            exe_path = dist_dir / exe_name
            if exe_path.exists():
                file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
                print(f"📦 可执行文件: {exe_path}")
                print(f"📏 文件大小: {file_size:.1f} MB")
                
                # 显示使用说明
                print("\n" + "=" * 60)
                print("🎉 打包完成!")
                print("=" * 60)
                print(f"可执行文件位置: {exe_path}")
                print(f"平台: {platform_name} ({arch})")
                print(f"大小: {file_size:.1f} MB")
                print("\n使用方法:")
                print(f"1. 双击运行: {exe_name}")
                print("2. 或在终端中运行:")
                print(f"   {exe_path}")
                print("\n注意:")
                print("- 首次运行可能需要较长时间")
                print("- 确保目标系统已安装必要的运行时库")
                print("- 如果遇到问题，请检查系统兼容性")
                
                return True
            else:
                print(f"❌ 未找到预期的可执行文件: {exe_path}")
                return False
        else:
            print("❌ 构建失败!")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 构建过程中出现异常: {e}")
        return False

def main():
    """主函数"""
    try:
        success = build_application()
        if success:
            print("\n🎊 所有操作完成!")
            sys.exit(0)
        else:
            print("\n💥 构建失败，请检查错误信息")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 未预期的错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
