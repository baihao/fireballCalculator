#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备选择模块
用于检查GPU兼容性并选择适合的设备（CUDA/CPU/MPS）
"""

import torch
import warnings
from typing import Optional


def get_compatible_device(device: str = "auto") -> str:
    """
    获取兼容的设备
    
    检查GPU兼容性，如果不兼容则自动退回到CPU模式
    
    Args:
        device: 设备类型 ("cpu", "cuda", "mps", "auto")
        
    Returns:
        str: 实际使用的设备
    """
    if device != "auto":
        return device
    
    if torch.cuda.is_available():
        # 显示必要的GPU信息
        print(f"🚀 检测到CUDA支持")
        print(f"   PyTorch版本: {torch.__version__}")
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"   PyTorch CUDA版本: {torch.version.cuda}")
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"   GPU: {device_name}")
        
        # 检查GPU计算能力兼容性
        if not _check_gpu_compatibility():
            print(f"   💡 自动退回到CPU模式（如需使用GPU，请安装PyTorch CUDA 12.4+）")
            return "cpu"
        
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 检测到MPS支持")
        return "mps"
    else:
        print("💻 使用CPU")
        print(f"   PyTorch版本: {torch.__version__}")
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"   PyTorch CUDA版本: {torch.version.cuda} (但CUDA运行时不可用)")
        else:
            print("   ⚠️ PyTorch未编译CUDA支持")
        return "cpu"


def _check_gpu_compatibility() -> bool:
    """
    检查GPU与当前PyTorch版本的兼容性
    
    Returns:
        bool: True表示兼容，False表示不兼容
    """
    try:
        # 获取GPU计算能力
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        capability_str = f"{capability[0]}.{capability[1]}"
        
        # 获取PyTorch CUDA版本
        cuda_version = None
        cuda_major = 0
        cuda_minor = 0
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            cuda_version = torch.version.cuda
            try:
                version_parts = cuda_version.split('.')
                cuda_major = int(version_parts[0])
                cuda_minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            except:
                pass
        
        # 检查兼容性：sm_120需要CUDA 12.4+
        is_incompatible = False
        if capability[0] >= 12:
            # sm_120 (Blackwell架构)需要CUDA 12.4+
            if not cuda_version or cuda_major < 12 or (cuda_major == 12 and cuda_minor < 4):
                is_incompatible = True
        
        # 尝试创建tensor并捕获警告来验证兼容性
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                
                # 检查是否有兼容性警告
                for warning in w:
                    warning_msg = str(warning.message)
                    if "not compatible" in warning_msg or "CUDA capability" in warning_msg:
                        is_incompatible = True
                        break
            except Exception as e:
                error_msg = str(e)
                if "not compatible" in error_msg or "CUDA capability" in error_msg:
                    is_incompatible = True
        
        # 如果不兼容，显示详细信息
        if is_incompatible:
            print(f"   ⚠️ GPU计算能力 {capability_str} (sm_{capability[0]}{capability[1]}) 与当前PyTorch不兼容")
            print(f"   当前PyTorch CUDA版本: {cuda_version or '未知'}")
            return False
        
        return True
    except Exception as e:
        # 如果检查过程中出错，记录警告但继续尝试使用CUDA
        print(f"   ⚠️ GPU兼容性检查时出错: {e}")
        print(f"   💡 将尝试使用CUDA，如果失败请手动切换到CPU模式")
        return True


def get_device_info() -> dict:
    """
    获取设备信息
    
    Returns:
        dict: 包含设备信息的字典
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "mps_available": False,
        "recommended_device": "cpu"
    }
    
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        info["cuda_version"] = torch.version.cuda
    
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            info["gpu_names"].append(torch.cuda.get_device_name(i))
        info["recommended_device"] = "cuda"
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["mps_available"] = True
        if not info["cuda_available"]:
            info["recommended_device"] = "mps"
    
    return info

