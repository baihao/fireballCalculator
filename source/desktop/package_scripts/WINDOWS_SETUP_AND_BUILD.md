# Windows 环境搭建与桌面应用打包指南

本文档说明在 **Windows 10/11** 上如何获取源码、配置 Conda 环境、运行开发版，以及使用 `build_desktop_app_windows.ps1` 打包独立桌面程序。文中专门列出 Windows 上常见的**工具缺失**与**权限/安全策略**问题及应对办法。

---

## 1. 概述

| 阶段 | 脚本/命令 | 产出 |
|------|-----------|------|
| 环境安装 | `source\setup.ps1` | Conda 环境 `fireball_calculator`（Python 3.10 + pip 依赖 + PyTorch CUDA 12.8） |
| 开发运行 | `python source\desktop\app.py` | 源码方式启动 GUI |
| 打包发布 | `source\desktop\package_scripts\build_desktop_app_windows.ps1` | `dist\FireballAnalysisApp\FireballAnalysisApp.exe` |

**建议硬件与磁盘**

- 系统盘或工作盘 **至少 20 GB 可用空间**（Conda 环境 + PyTorch + PyInstaller 产物）
- 内存 **16 GB 及以上**（打包时 PyInstaller 收集 `torch` 较耗内存）
- 可选：**NVIDIA GPU + 驱动**（开发阶段 PyTorch CUDA；打包后 exe 在无 GPU 机器上仍可运行，但分割等 GPU 加速不可用）

---

## 2. 获取源代码

### 2.1 安装 Git

若未安装 Git，从 [https://git-scm.com/download/win](https://git-scm.com/download/win) 下载并安装。安装时建议勾选 **「Git from the command line and also from 3rd-party software」**，以便在 PowerShell 中使用 `git`。

### 2.2 克隆仓库

在 PowerShell 中执行（路径请按实际修改，**尽量避免过深目录与中文路径**）：

```powershell
cd C:\Dev
git clone https://github.com/baihao/fireballCalculator.git
cd fireballCalculator
```

若使用 SSH：

```powershell
git clone git@github.com:baihao/fireballCalculator.git
```

### 2.3 初始化子模块（分割功能必需）

项目依赖 `segment-anything` 子模块：

```powershell
git submodule update --init --recursive
```

验证目录存在：

```powershell
Test-Path source\third_party\segment-anything\segment_anything\__init__.py
```

应返回 `True`。

### 2.4 下载 SAM 权重（分割功能必需）

打包脚本只会打入 **`sam_vit_b*.pth`**（体积较小）。请手动下载 ViT-B 权重：

1. 打开 [Segment Anything 模型页](https://github.com/facebookresearch/segment-anything#model-checkpoints)
2. 下载 **ViT-B**：`sam_vit_b_01ec64.pth`
3. 放到：

```
fireballCalculator\source\third_party\segment-anything\checkpoints\sam_vit_b_01ec64.pth
```

未放置权重时：**环境可装好、程序可打包**，但特征提取中的 SAM 分割会不可用，打包日志会出现 `[Warn] No SAM checkpoints found`。

---

## 3. 安装 Conda（Python 环境管理）

### 3.1 安装 Miniconda 或 Anaconda

推荐 [Miniconda Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html)。

安装注意：

- 可选「为所有用户安装」——若公司策略禁止写 `ProgramData`，选**仅当前用户**
- **不必**强行勾选「Add Miniconda3 to PATH」（官方不推荐）；后续用 **`conda init`** 更稳妥

### 3.2 初始化 PowerShell

**以普通用户**打开 **PowerShell**（不必管理员），执行：

```powershell
conda init powershell
```

关闭并**重新打开** PowerShell，确认：

```powershell
conda --version
```

若提示 `conda : 无法将“conda”项识别为 cmdlet`：

- 开始菜单打开 **「Anaconda Prompt」** 或 **「Miniconda Prompt」** 再执行后续步骤；或
- 手动将 `%USERPROFILE%\miniconda3\Scripts` 与 `%USERPROFILE%\miniconda3\condabin` 加入用户 PATH（需重开终端）

---

## 4. PowerShell 执行策略（常见“闪退/无法运行脚本”）

Windows 默认可能禁止 `.ps1` 脚本，表现为双击脚本窗口一闪而过，或提示 **running scripts is disabled**。

**推荐做法**：不要双击 `.ps1`，在 PowerShell 中显式加 `-ExecutionPolicy Bypass`：

```powershell
cd C:\Dev\fireballCalculator\source
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

若需当前用户永久放宽（自行承担安全责任）：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

公司域策略若强制 `AllSigned`/`Restricted`，需联系 IT 放行，或使用 **Anaconda Prompt + `conda run`** 绕过部分限制。

---

## 5. 运行 setup.ps1 配置环境

### 5.1 执行

```powershell
cd C:\Dev\fireballCalculator\source
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

脚本会：

1. 检查 Conda 与 `requirements.txt`
2. 接受 Anaconda 官方 channel 的 ToS（首次可能需联网）
3. 创建或复用环境 **`fireball_calculator`**（Python 3.10）
4. 若依赖**已满足**则**跳过** pip 重装；否则安装 PyTorch（**CUDA 12.8**）及 `requirements.txt`
5. 可选安装 `segment-anything` 可编辑包
6. 写入 Conda activate 钩子（`KMP_DUPLICATE_LIB_OK` 等）

### 5.2 激活环境

```powershell
conda activate fireball_calculator
python --version    # 应为 3.10.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 5.3 setup 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `conda not found` | 未 init 或未开新终端 | `conda init powershell` 后重开 |
| pip / PyTorch 下载极慢或超时 | 网络或代理 | 配置 pip 镜像；或使用 VPN；多次重试 setup |
| PyTorch 体积约 2–3 GB | 正常 | 确保磁盘与流量充足 |
| `Accept Terms of Service` 失败 |  conda 24+ 新 channel 策略 | 按脚本提示对 defaults channel 执行 `conda tos accept` |
| 重复运行 setup 仍重装 | 依赖版本不满足 | 查看脚本输出中哪一项检测未通过；修复后 rerun |

---

## 6. 开发模式运行（打包前验证）

在项目**根目录** `fireballCalculator` 下：

```powershell
conda activate fireball_calculator
cd C:\Dev\fireballCalculator
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python source\desktop\app.py
```

若 import 失败，可临时设置：

```powershell
$env:PYTHONPATH = "C:\Dev\fireballCalculator\source"
python source\desktop\app.py
```

确认 GUI 能打开、各 Tab 无报错后再打包。

---

## 7. 使用 build_desktop_app_windows.ps1 打包

### 7.1 执行

```powershell
conda activate fireball_calculator
cd C:\Dev\fireballCalculator
powershell -ExecutionPolicy Bypass -File source\desktop\package_scripts\build_desktop_app_windows.ps1
```

脚本会：

1. 激活 `fireball_calculator`（若尚未激活）
2. 安装/升级 **PyInstaller**
3. 清理旧的 `dist`、`build`、`FireballAnalysisApp.spec`
4. 调用 PyInstaller **`--onedir --windowed`**，收集 PySide6、torch、segment_anything 等
5. 打入 `desktop\resources`、`desktop\icon`、精简版 SAM 与 vit_b 权重
6. 写入 Windows 用 **MPS stub**（避免 macOS 后端 import 报错）
7. 清理部分测试目录以减小体积

### 7.2 打包产物

```
fireballCalculator\dist\FireballAnalysisApp\
├── FireballAnalysisApp.exe      ← 主程序
└── _internal\                   ← 依赖库（勿删）
```

运行：

```powershell
cd C:\Dev\fireballCalculator\dist\FireballAnalysisApp
.\FireballAnalysisApp.exe
```

**分发**：将整个 **`FireballAnalysisApp` 文件夹**压缩为 zip 拷贝到其他 Windows 机器解压运行，**不要只拷贝 exe**。

体积通常为 **数 GB**（含 PyTorch），属正常现象。

### 7.3 打包前置检查清单

- [ ] 已 `git submodule update --init --recursive`
- [ ] `checkpoints\sam_vit_b_01ec64.pth` 已就位（需要分割时）
- [ ] `conda activate fireball_calculator` 后 `python source\desktop\app.py` 正常
- [ ] 磁盘剩余空间 > 10 GB
- [ ] 杀毒软件可能拦截 PyInstaller 写入 `dist`——见下文

---

## 8. Windows 权限与安全软件问题

### 8.1 无需管理员的情况

- `git clone`、`conda create`、`pip install`、`pyinstaller` 在**用户目录**下通常**不需要**管理员权限
- 建议项目放在：`C:\Dev\fireballCalculator` 或 `%USERPROFILE%\Projects\...`

### 8.2 可能需要管理员或 IT 协助的情况

| 场景 | 说明 |
|------|------|
| 安装 Miniconda「为所有用户」 | 需写 `C:\ProgramData` |
| 域策略禁止脚本 | `ExecutionPolicy` 无法改为 RemoteSigned |
| 禁止运行未签名 exe | 打包后的 `FireballAnalysisApp.exe` 无代码签名，可能被 SmartScreen 拦截 |
| 禁止下载 .pth / 大文件 | SAM 权重、PyTorch wheel 无法下载 |

**SmartScreen「Windows 已保护你的电脑」**：点「更多信息」→「仍要运行」，或由单位对 exe 做代码签名。

### 8.3 Windows Defender / 第三方杀毒

打包时大量写入 `dist\_internal\*.dll`，可能触发**误报**或**实时扫描卡顿**：

- 临时将项目目录 `fireballCalculator` 加入**排除项**（Defender → 病毒和威胁防护 → 管理设置 → 排除项）
- 打包完成后再扫描一次 dist 目录

若 exe 启动即被删除，多为杀毒隔离——恢复并加白名单。

### 8.4 路径与长路径

- 避免路径含**中文、空格、特殊符号**（如 `C:\Users\Think\Desktop\...` 可用但深路径+中文偶发工具兼容问题）
- 启用长路径（Win10 1703+）：组策略或注册表 `LongPathsEnabled=1`，否则 PyInstaller 深层目录可能失败

### 8.5 只读 / 同步盘

OneDrive、企业网盘同步目录下的 `dist` 可能被锁定。将仓库放在**非同步本地目录**再打包。

### 8.6 防火墙与代理

Conda/pip/PyTorch 下载需访问外网；公司代理需配置：

```powershell
$env:HTTP_PROXY = "http://proxy.company:8080"
$env:HTTPS_PROXY = "http://proxy.company:8080"
```

---

## 9. 打包失败排查

| 错误/现象 | 排查方向 |
|-----------|----------|
| `Conda is not available` | 安装 Miniconda 并 `conda init powershell` |
| `Failed to activate env fireball_calculator` | 先运行 `setup.ps1` 创建环境 |
| `conda-hook.ps1 not found` | 用 **Anaconda PowerShell Prompt** 或重装 conda |
| PyInstaller `ModuleNotFoundError` | 在已激活环境中 `pip install` 缺失包后重试 |
| 内存不足 / 页面文件 | 关闭其它程序；增大虚拟内存；不要同时开多个打包任务 |
| `Permission denied` 删 dist | 关闭正在运行的 exe；杀毒释放文件锁；以资源管理器手动删 `dist` |
| exe 启动闪退 | 在 cmd 中运行 exe 看 stderr；查 `%TEMP%` 或程序同目录是否生成 `app.log` |
| 分割不可用 | 确认打包前 checkpoints 存在且日志有 `Included vit_b checkpoints` |

**查看运行日志**：打包版与开发版均可能在工作目录或程序目录写入 `log\` / `app.log`（取决于启动方式），闪退时优先保留该文件。

---

## 10. 推荐完整流程（从零到 exe）

```powershell
# 1. 克隆
cd C:\Dev
git clone https://github.com/baihao/fireballCalculator.git
cd fireballCalculator
git submodule update --init --recursive

# 2. SAM 权重（浏览器下载后复制到 checkpoints 目录）
#    source\third_party\segment-anything\checkpoints\sam_vit_b_01ec64.pth

# 3. 环境（首次约 10–30 分钟，视网络而定）
cd source
powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 4. 验证
conda activate fireball_calculator
cd ..
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python source\desktop\app.py

# 5. 打包（约 10–20 分钟）
powershell -ExecutionPolicy Bypass -File source\desktop\package_scripts\build_desktop_app_windows.ps1

# 6. 运行
.\dist\FireballAnalysisApp\FireballAnalysisApp.exe
```

---

## 11. 相关文件

| 文件 | 说明 |
|------|------|
| `source\setup.ps1` | Windows 环境安装（依赖已满足则跳过重装） |
| `source\requirements.txt` | pip 依赖清单 |
| `source\desktop\package_scripts\build_desktop_app_windows.ps1` | Windows 桌面打包 |
| `source\desktop\app.py` | 应用入口 |
| `source\image_segment\USAGE.md` | 分割模块与 SAM 说明 |

---

## 12. 版本说明

- 环境：**Python 3.10**，Conda 环境名 **`fireball_calculator`**
- Windows PyTorch：**CUDA 12.8**（由 `setup.ps1` 安装）
- 打包工具：**PyInstaller**，产物为 **onedir** 目录分发

若 conda / PyTorch / PyInstaller 大版本升级导致脚本失效，请以仓库内最新 `setup.ps1` 与 `build_desktop_app_windows.ps1` 为准。
