# 打包脚本目录

本目录包含火球图像分割工具的打包脚本和相关文档。

## 🚀 快速开始

### macOS 打包
```bash
./build_mac_optimized.sh
```

### Windows 打包
```powershell
.\build_windows.ps1
```

### 使用打包程序
```bash
./dist/image_segment_propagation/image_segment_propagation \
  test_data/fireball_sequence.json --no-viz
```

---

## 📁 文件说明

### 脚本文件
- **`build_mac_optimized.sh`** ⭐ - macOS 打包脚本（推荐）
- **`build_windows.ps1`** - Windows 打包脚本
- **`performance_test.sh`** - 性能对比测试脚本

### 文档文件
- **`PACKAGING_GUIDE.md`** ⭐ - **完整打包指南**（主要文档，包含所有使用说明）
- **`OPTIMIZATION_GUIDE.md`** - 详细优化技术参考
- **`PYTORCH_PACKAGING_NOTES.md`** - PyTorch 打包技术细节
- **`ALTERNATIVE_SOLUTIONS.md`** - Docker、Conda Constructor 等替代方案

---

## 📖 推荐阅读顺序

1. **首次使用**：先读 `PACKAGING_GUIDE.md` 的"快速开始"章节
2. **遇到问题**：查看 `PACKAGING_GUIDE.md` 的"故障排除"章节
3. **深入优化**：参考 `OPTIMIZATION_GUIDE.md`
4. **替代方案**：考虑 Docker 等方案时阅读 `ALTERNATIVE_SOLUTIONS.md`

---

## 🎯 打包方案对比

| 方案 | 性能 | 体积 | 打包时间 | 推荐场景 |
|------|------|------|---------|---------|
| **PyInstaller** | 80% | 952MB | 3-5分钟 | ⭐ 用户分发 |
| **Docker** | 100% | 700MB | 10-15分钟 | ⭐ 服务器部署 |
| **Python脚本** | 100% | N/A | 0分钟 | ⭐ 开发调试 |

---

## ✅ 最终推荐

**当前最佳方案**：PyInstaller 优化版
- 已验证可用
- 打包快速（3-5分钟）
- 性能可接受（处理200张图约130秒，比Python脚本慢26秒）

**详细说明**：见 `PACKAGING_GUIDE.md`

---

## 💡 快速答疑

**Q: 打包需要多久？**  
A: 3-5 分钟

**Q: 打包后的程序慢多少？**  
A: 慢约 25%（绝对值 +26 秒，在 2 分钟级别任务中可接受）

**Q: 如何提升性能？**  
A: 使用 Docker 容器可达到 100% 性能（见 `ALTERNATIVE_SOLUTIONS.md`）

**Q: Windows 版本如何打包？**  
A: 在 Windows 机器上运行 `build_windows.ps1`

**Q: 为什么不用 Nuitka？**  
A: PyTorch 项目 C++ ABI 不兼容，运行时崩溃，调试成本高

---

**开始打包你的火球分割工具吧！** 🔥

详细文档：[PACKAGING_GUIDE.md](PACKAGING_GUIDE.md)

