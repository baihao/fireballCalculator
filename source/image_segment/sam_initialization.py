#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM模型初始化模块
包含SAM模型的加载、设备检测和检查点管理功能
"""

import torch
from pathlib import Path
from typing import Optional

# 尝试导入SAM相关模块
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("⚠️ Segment Anything未安装")
    sam_model_registry = None
    SamPredictor = None


class SAMModelManager:
    """SAM模型管理器"""
    
    def __init__(self, model_type: str = "vit_l", 
                 checkpoint_path: Optional[str] = None,
                 device: str = "auto"):
        """
        初始化SAM模型管理器
        
        Args:
            model_type: SAM模型类型 ("vit_b", "vit_l", "vit_h")
            checkpoint_path: 模型检查点路径
            device: 设备类型 ("cpu", "cuda", "mps", "auto")
        """
        if not SAM_AVAILABLE:
            raise ImportError("Segment Anything未安装，请先运行 setup.sh 安装SAM")
        
        self.model_type = model_type
        self.device = self._get_device(device)
        self.checkpoint_path = checkpoint_path or self._get_default_checkpoint_path()
        self.predictor = None
        
        print(f"📋 SAM配置: 模型={self.model_type}, 设备={self.device}")
        print(f"📁 检查点路径: {self.checkpoint_path}")
    
    def _get_device(self, device: str) -> str:
        """
        获取可用设备
        
        Args:
            device: 设备类型
            
        Returns:
            str: 实际使用的设备
        """
        if device == "auto":
            if torch.cuda.is_available():
                print("🚀 检测到CUDA支持")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🍎 检测到MPS支持")
                return "mps"
            else:
                print("💻 使用CPU")
                return "cpu"
        return device
    
    def _get_default_checkpoint_path(self) -> str:
        """
        获取默认检查点路径
        
        Returns:
            str: 检查点文件路径
        """
        sam_dir = Path(__file__).parent.parent / "third_party" / "segment-anything"
        checkpoint_dir = sam_dir / "checkpoints"
        
        # 按优先级查找检查点文件
        checkpoint_files = [
            f"sam_{self.model_type}_0b3195.pth",
            f"sam_{self.model_type}.pth",
            f"sam_{self.model_type}_01ec64.pth"
        ]
        
        for checkpoint_file in checkpoint_files:
            checkpoint_path = checkpoint_dir / checkpoint_file
            if checkpoint_path.exists():
                print(f"✓ 找到检查点文件: {checkpoint_file}")
                return str(checkpoint_path)
        
        # 如果没找到，返回默认路径
        default_path = checkpoint_dir / f"sam_{self.model_type}.pth"
        print(f"⚠️ 未找到现有检查点，使用默认路径: {default_path}")
        return str(default_path)
    
    def load_model(self):
        """
        加载SAM模型
        
        Raises:
            Exception: 模型加载失败时抛出异常
        """
        try:
            print(f"🔄 正在加载SAM模型: {self.model_type}")
            print(f"🖥️  目标设备: {self.device}")
            
            # 检查检查点文件是否存在
            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")
            
            # 加载模型
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            
            print("✅ SAM模型加载成功")
            
        except Exception as e:
            print(f"❌ SAM模型加载失败: {e}")
            print("💡 请检查:")
            print("   1. 检查点文件是否存在")
            print("   2. 模型类型是否正确")
            print("   3. 设备是否可用")
            print("   4. 是否正确安装了segment-anything")
            raise
    
    def get_predictor(self):
        """
        获取SAM预测器
        
        Returns:
            SamPredictor: SAM预测器实例
        """
        if self.predictor is None:
            self.load_model()
        return self.predictor
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型是否已加载
        """
        return self.predictor is not None
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型配置信息
        """
        return {
            "model_type": self.model_type,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "is_loaded": self.is_loaded(),
            "sam_available": SAM_AVAILABLE
        }


def create_sam_manager(model_type: str = "vit_l", 
                      checkpoint_path: Optional[str] = None,
                      device: str = "auto") -> SAMModelManager:
    """
    创建SAM模型管理器的便捷函数
    
    Args:
        model_type: SAM模型类型
        checkpoint_path: 模型检查点路径
        device: 设备类型
        
    Returns:
        SAMModelManager: SAM模型管理器实例
    """
    return SAMModelManager(model_type, checkpoint_path, device)


def check_sam_availability() -> bool:
    """
    检查SAM是否可用
    
    Returns:
        bool: SAM是否可用
    """
    return SAM_AVAILABLE


def list_available_devices() -> list:
    """
    列出可用设备
    
    Returns:
        list: 可用设备列表
    """
    devices = ["cpu"]
    
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"🚀 CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
        print("🍎 MPS (Metal Performance Shaders) 可用")
    
    return devices


if __name__ == "__main__":
    # 测试模块功能
    print("SAM初始化模块测试")
    print("=" * 40)
    
    print("1. 检查SAM可用性:")
    print(f"   SAM可用: {check_sam_availability()}")
    
    print("\n2. 列出可用设备:")
    devices = list_available_devices()
    print(f"   可用设备: {devices}")
    
    if check_sam_availability():
        print("\n3. 创建SAM管理器:")
        try:
            manager = create_sam_manager()
            info = manager.get_model_info()
            print("   配置信息:")
            for key, value in info.items():
                print(f"     {key}: {value}")
        except Exception as e:
            print(f"   创建失败: {e}")
    
    print("\n测试完成!")
