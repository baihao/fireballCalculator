#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM模型初始化模块
包含SAM模型的加载、设备检测和检查点管理功能
"""

import torch
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error
import sys

# 导入设备选择模块
try:
    from .device_selector import get_compatible_device
except ImportError:
    from device_selector import get_compatible_device

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
    
    def __init__(self, model_type: str = "vit_b", 
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
        return get_compatible_device(device)
    
    def _get_checkpoint_url(self, model_type: str) -> Optional[str]:
        """
        获取检查点文件的下载URL（仅支持 vit_b 自动下载）
        
        Args:
            model_type: 模型类型
            
        Returns:
            str: 下载URL，仅支持 vit_b，其他类型返回None
        """
        # 只支持 vit_b 的自动下载
        if model_type == "vit_b":
            return "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        return None
    
    def _download_checkpoint(self, url: str, output_path: Path) -> bool:
        """
        下载检查点文件
        
        Args:
            url: 下载URL
            output_path: 保存路径
            
        Returns:
            bool: 是否下载成功
        """
        try:
            print(f"📥 开始下载检查点文件...")
            print(f"   来源: {url}")
            print(f"   保存到: {output_path}")
            
            # 确保目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 显示下载进度的回调函数
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
                downloaded_mb = downloaded / (1024 * 1024)
                
                # 显示进度条
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = "=" * filled + "-" * (bar_length - filled)
                sys.stdout.write(f"\r   进度: [{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)")
                sys.stdout.flush()
            
            # 下载文件
            urllib.request.urlretrieve(url, str(output_path), show_progress)
            print()  # 换行
            print(f"✓ 检查点文件下载成功")
            return True
            
        except urllib.error.URLError as e:
            print(f"\n❌ 下载失败: {e}")
            print("💡 请检查网络连接或手动下载检查点文件")
            return False
        except Exception as e:
            print(f"\n❌ 下载过程中出错: {e}")
            return False
    
    def _get_default_checkpoint_path(self) -> str:
        """
        获取默认检查点路径，如果不存在则尝试自动下载
        
        Returns:
            str: 检查点文件路径
        """
        sam_dir = Path(__file__).parent.parent / "third_party" / "segment-anything"
        checkpoint_dir = sam_dir / "checkpoints"
        
        # 模型类型到文件名的映射
        checkpoint_mapping = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        # 按优先级查找检查点文件
        checkpoint_files = [
            checkpoint_mapping.get(self.model_type),  # 官方文件名
            f"sam_{self.model_type}_0b3195.pth",      # 可能的变体
            f"sam_{self.model_type}.pth",              # 简化文件名
            f"sam_{self.model_type}_01ec64.pth"        # 其他变体
        ]
        
        # 移除None值
        checkpoint_files = [f for f in checkpoint_files if f is not None]
        
        for checkpoint_file in checkpoint_files:
            checkpoint_path = checkpoint_dir / checkpoint_file
            if checkpoint_path.exists():
                print(f"✓ 找到检查点文件: {checkpoint_file}")
                return str(checkpoint_path)
        
        # 如果没找到，返回默认路径（下载逻辑在 load_model 中处理）
        default_filename = checkpoint_mapping.get(self.model_type, f"sam_{self.model_type}.pth")
        default_path = checkpoint_dir / default_filename
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
            
            # 检查检查点文件是否存在，如果不存在则尝试下载
            checkpoint_path_obj = Path(self.checkpoint_path)
            if not checkpoint_path_obj.exists():
                print(f"⚠️ 检查点文件不存在，尝试自动下载...")
                url = self._get_checkpoint_url(self.model_type)
                if url:
                    if self._download_checkpoint(url, checkpoint_path_obj):
                        print(f"✓ 检查点文件已下载，继续加载模型...")
                    else:
                        raise FileNotFoundError(
                            f"检查点文件不存在且自动下载失败: {self.checkpoint_path}\n"
                            f"请手动下载检查点文件或检查网络连接。"
                        )
                else:
                    raise FileNotFoundError(
                        f"检查点文件不存在: {self.checkpoint_path}\n"
                        f"当前模型类型 '{self.model_type}' 不支持自动下载\n"
                        f"仅支持 vit_b 模型的自动下载，其他模型需要手动下载检查点文件"
                    )
            
            # 加载模型
            # 注意：sam_model_registry内部加载检查点时可能没有指定设备
            # 我们需要确保检查点被加载到正确的设备上
            print(f"📦 正在从检查点加载模型...")
            
            # 加载模型（检查点会在加载时自动映射到当前设备）
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            
            # 将模型移动到目标设备
            print(f"📤 将模型移动到设备: {self.device}")
            sam.to(device=self.device)
            
            # 验证模型实际所在的设备
            # 检查模型参数所在的设备
            first_param_device = next(sam.parameters()).device
            print(f"✅ 模型已加载，实际设备: {first_param_device}")
            
            if self.device.startswith("cuda") and str(first_param_device) != self.device:
                print(f"⚠️  警告: 模型设备 ({first_param_device}) 与目标设备 ({self.device}) 不匹配")
                print(f"   尝试强制移动到 {self.device}...")
                sam.to(device=self.device)
                first_param_device = next(sam.parameters()).device
                print(f"   当前设备: {first_param_device}")
            
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


def create_sam_manager(model_type: str = "vit_b", 
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
    
    print(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        devices.append("cuda")
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"PyTorch CUDA版本: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU数量: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {device_name}")
    else:
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"PyTorch CUDA版本: {torch.version.cuda} (但CUDA运行时不可用)")
        else:
            print("PyTorch未编译CUDA支持")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    
    return devices


def diagnose_gpu_setup() -> dict:
    """
    诊断GPU设置，返回诊断信息
    
    Returns:
        dict: 包含诊断信息的字典
    """
    diagnosis = {
        "pytorch_version": torch.__version__,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "recommended_device": "cpu"
    }
    
    # 检查CUDA
    if torch.cuda.is_available():
        diagnosis["cuda_available"] = True
        diagnosis["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') and torch.version.cuda else None
        diagnosis["gpu_count"] = torch.cuda.device_count()
        diagnosis["recommended_device"] = "cuda"
    
    # 检查MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if not diagnosis["cuda_available"]:
            diagnosis["recommended_device"] = "mps"
    
    return diagnosis


if __name__ == "__main__":
    # 测试模块功能
    print("SAM初始化模块测试")
    print("=" * 40)
    
    print("1. 检查SAM可用性:")
    print(f"   SAM可用: {check_sam_availability()}")
    
    print("\n2. GPU诊断:")
    diagnosis = diagnose_gpu_setup()
    print(f"   PyTorch版本: {diagnosis['pytorch_version']}")
    if diagnosis['cuda_available']:
        print(f"   CUDA可用: ✅")
        if diagnosis['cuda_version']:
            print(f"   PyTorch CUDA版本: {diagnosis['cuda_version']}")
        print(f"   GPU数量: {diagnosis['gpu_count']}")
        for i in range(diagnosis['gpu_count']):
            device_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {device_name}")
    else:
        print(f"   CUDA可用: ❌")
        if diagnosis['cuda_version']:
            print(f"   PyTorch CUDA版本: {diagnosis['cuda_version']} (但CUDA运行时不可用)")
        else:
            print("   PyTorch未编译CUDA支持")
    
    print(f"   推荐设备: {diagnosis['recommended_device']}")
    
    print("\n3. 列出可用设备:")
    devices = list_available_devices()
    print(f"   可用设备: {devices}")
    
    if check_sam_availability():
        print("\n4. 创建SAM管理器:")
        try:
            manager = create_sam_manager()
            info = manager.get_model_info()
            print("   配置信息:")
            for key, value in info.items():
                print(f"     {key}: {value}")
            
            # 尝试加载模型以验证设备
            print("\n5. 测试模型加载:")
            manager.load_model()
            predictor = manager.get_predictor()
            print("   ✅ 模型加载成功")
            
        except Exception as e:
            print(f"   ❌ 创建/加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n测试完成!")
