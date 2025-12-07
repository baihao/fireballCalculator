#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU诊断脚本
用于检查PyTorch和SAM的GPU支持情况
"""

import torch
import subprocess

print("=" * 60)
print("GPU诊断工具")
print("=" * 60)

# 1. PyTorch版本
print(f"\n1. PyTorch版本: {torch.__version__}")

# 2. PyTorch CUDA编译信息
cuda_compiled = hasattr(torch.version, 'cuda') and torch.version.cuda
if cuda_compiled:
    print(f"2. PyTorch编译时CUDA版本: {torch.version.cuda}")
else:
    print("2. ⚠️ PyTorch未编译CUDA支持（CPU版本）")

# 3. CUDA运行时和GPU信息
print(f"\n3. CUDA运行时: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✅ 检测到 {torch.cuda.device_count()} 个GPU")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        # 显示驱动支持的CUDA版本
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        driver_cuda = line.split('CUDA Version:')[1].strip().split()[0]
                        print(f"      驱动支持CUDA版本: {driver_cuda}")
                        break
        except:
            pass
else:
    print("   ❌ CUDA运行时不可用")
    # 检查驱动
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            print("   ⚠️ 有驱动但CUDA不可用（可能是PyTorch版本问题）")
        else:
            print("   ⚠️ 未检测到NVIDIA驱动")
    except FileNotFoundError:
        print("   ⚠️ 未安装NVIDIA驱动")

print("\n" + "=" * 60)
