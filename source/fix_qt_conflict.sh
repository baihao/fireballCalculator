#!/bin/bash

# Qt库冲突修复脚本
# 此脚本尝试解决PySide6和conda Qt库的冲突问题

echo "=========================================="
echo "Qt库冲突修复脚本"
echo "=========================================="

ENV_NAME="fireball_calculator"
ENV_PATH="/Users/hbai/miniconda3/envs/${ENV_NAME}"

# 检查环境是否存在
if [ ! -d "${ENV_PATH}" ]; then
    echo "错误: 环境 ${ENV_NAME} 不存在"
    exit 1
fi

echo "✓ 检测到环境: ${ENV_NAME}"

# 方法1: 尝试移除conda的Qt库，只使用PySide6的Qt
echo ""
echo "方法1: 移除conda的Qt库，只使用PySide6的Qt..."
conda remove qt6-main -y

if [ $? -eq 0 ]; then
    echo "✓ 成功移除conda的Qt库"
else
    echo "⚠️ 移除conda Qt库失败，尝试其他方法"
fi

# 方法2: 设置环境变量来避免冲突
echo ""
echo "方法2: 设置环境变量来避免Qt冲突..."

# 创建环境变量设置脚本
cat > "${ENV_PATH}/etc/conda/activate.d/qt_fix.sh" << 'EOF'
#!/bin/bash
# Qt库冲突修复环境变量

# 设置Qt插件路径，优先使用PySide6的插件
export QT_PLUGIN_PATH="/Users/hbai/miniconda3/envs/fireball_calculator/lib/python3.10/site-packages/PySide6/Qt/plugins"

# 设置Qt库路径
export QT_QPA_PLATFORM_PLUGIN_PATH="/Users/hbai/miniconda3/envs/fireball_calculator/lib/python3.10/site-packages/PySide6/Qt/plugins/platforms"

# 禁用Qt的自动库加载
export QT_AUTO_SCREEN_SCALE_FACTOR=0

# 设置Qt平台
export QT_QPA_PLATFORM=cocoa

echo "✓ Qt环境变量已设置"
EOF

# 创建停用脚本
cat > "${ENV_PATH}/etc/conda/deactivate.d/qt_fix.sh" << 'EOF'
#!/bin/bash
# 清理Qt环境变量

unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH
unset QT_AUTO_SCREEN_SCALE_FACTOR
unset QT_QPA_PLATFORM

echo "✓ Qt环境变量已清理"
EOF

# 设置脚本权限
chmod +x "${ENV_PATH}/etc/conda/activate.d/qt_fix.sh"
chmod +x "${ENV_PATH}/etc/conda/deactivate.d/qt_fix.sh"

echo "✓ 环境变量脚本已创建"

# 方法3: 创建Qt库符号链接修复
echo ""
echo "方法3: 创建Qt库符号链接修复..."

# 备份原始库
if [ -f "${ENV_PATH}/lib/libQt6Core.6.9.2.dylib" ]; then
    mv "${ENV_PATH}/lib/libQt6Core.6.9.2.dylib" "${ENV_PATH}/lib/libQt6Core.6.9.2.dylib.backup"
    echo "✓ 已备份原始Qt库"
fi

# 创建符号链接指向PySide6的Qt库
if [ -f "${ENV_PATH}/lib/python3.10/site-packages/PySide6/Qt/lib/QtCore.framework/Versions/A/QtCore" ]; then
    ln -sf "${ENV_PATH}/lib/python3.10/site-packages/PySide6/Qt/lib/QtCore.framework/Versions/A/QtCore" "${ENV_PATH}/lib/libQt6Core.6.9.2.dylib"
    echo "✓ 已创建Qt库符号链接"
fi

echo ""
echo "=========================================="
echo "Qt冲突修复完成！"
echo "=========================================="
echo ""
echo "请尝试以下步骤："
echo "1. 重新激活环境: conda deactivate && conda activate ${ENV_NAME}"
echo "2. 测试应用程序: ./python310 app.py"
echo ""
echo "如果仍有问题，可以尝试："
echo "1. 重新安装PySide6: pip310 uninstall PySide6 && pip310 install PySide6"
echo "2. 或者使用无GUI模式运行应用程序"
echo ""
