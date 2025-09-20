#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列文件管理模块
负责火球分析序列数据的导入和导出
"""

import json
import numpy as np
import os
from typing import List, Optional, Dict, Any, Tuple, Union


class SequenceManager:
    """序列数据管理器"""
    
    def __init__(self):
        """初始化管理器"""
        pass
    
    def load_sequence_file(self, file_path: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        加载序列JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            Tuple[bool, Dict[str, Any], str]: (是否成功, 序列数据, 错误信息)
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False, {}, f"文件不存在: {file_path}"
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                sequence_data = json.load(f)
            
            # 验证JSON文件格式
            is_valid, error_msg = self._validate_sequence_format(sequence_data)
            if not is_valid:
                return False, {}, f"文件格式错误: {error_msg}"
            
            return True, sequence_data, "加载成功"
            
        except json.JSONDecodeError as e:
            return False, {}, f"JSON格式错误: {str(e)}"
        except Exception as e:
            return False, {}, f"读取文件失败: {str(e)}"
    
    def get_image_paths_from_sequence(self, sequence_data: Dict[str, Any]) -> List[str]:
        """
        从序列数据中提取图像路径列表
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            List[str]: 图像路径列表
        """
        try:
            return sequence_data.get('image_sequence', {}).get('image_paths', [])
        except Exception as e:
            print(f"提取图像路径失败: {e}")
            return []
    
    def get_parameters_from_sequence(self, sequence_data: Dict[str, Any]) -> Dict[str, str]:
        """
        从序列数据中提取参数信息
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Dict[str, str]: 参数字典
        """
        try:
            return sequence_data.get('parameters', {})
        except Exception as e:
            print(f"提取参数信息失败: {e}")
            return {}
    
    def get_temperature_data_from_sequence(self, sequence_data: Dict[str, Any]) -> Tuple[List[float], List[float]]:
        """
        从序列数据中提取温度数据
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Tuple[List[float], List[float]]: (时间数据, 温度数据)
        """
        try:
            temperature_data = sequence_data.get('temperature', [])
            if not temperature_data:
                return [], []
            
            # 分离时间和温度数据
            time_data = []
            temp_data = []
            
            for time_temp_pair in temperature_data:
                if len(time_temp_pair) >= 2:
                    time_data.append(float(time_temp_pair[0]))
                    temp_data.append(float(time_temp_pair[1]))
            
            return time_data, temp_data
            
        except Exception as e:
            print(f"提取温度数据失败: {e}")
            return [], []
    
    def get_prompt_data_from_sequence(self, sequence_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        从序列数据中提取prompt数据
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Dict[int, Dict[str, Any]]: prompt数据字典 {image_index: {"points": [[x,y], ...], "labels": [1,0,1,...]}}
        """
        try:
            # 从image_sequence中获取prompt_data
            image_sequence = sequence_data.get('image_sequence', {})
            prompt_data_raw = image_sequence.get('prompt_data', {})
            
            # 转换键为整数
            prompt_data = {}
            for key, value in prompt_data_raw.items():
                try:
                    image_index = int(key)
                    prompt_data[image_index] = value
                except ValueError:
                    print(f"无效的图像索引: {key}")
                    continue
            
            return prompt_data
            
        except Exception as e:
            print(f"提取prompt数据失败: {e}")
            return {}
    
    def save_prompt_data_to_sequence(self, file_path: str, prompt_data: Dict[int, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        将prompt数据保存到序列文件中
        
        Args:
            file_path: 序列文件路径
            prompt_data: prompt数据字典
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息或成功信息)
        """
        try:
            # 读取现有的序列文件
            success, sequence_data, message = self.load_sequence_file(file_path)
            if not success:
                return False, f"无法读取序列文件: {message}"
            
            # 转换prompt_data的键为字符串（JSON要求）
            prompt_data_str_keys = {str(k): v for k, v in prompt_data.items()}
            
            # 将prompt_data添加到image_sequence中
            if 'image_sequence' not in sequence_data:
                sequence_data['image_sequence'] = {}
            
            sequence_data['image_sequence']['prompt_data'] = prompt_data_str_keys
            
            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=4)
            
            # 统计信息
            total_images_with_prompts = len(prompt_data)
            total_points = sum(len(data["points"]) for data in prompt_data.values())
            
            success_msg = f"prompt数据保存成功！包含 {total_images_with_prompts} 张图像的 {total_points} 个prompt点"
            return True, success_msg
            
        except Exception as e:
            error_msg = f"保存prompt数据失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def get_sequence_summary(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取序列数据摘要
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Dict[str, Any]: 摘要信息
        """
        try:
            image_paths = self.get_image_paths_from_sequence(sequence_data)
            time_data, temp_data = self.get_temperature_data_from_sequence(sequence_data)
            parameters = self.get_parameters_from_sequence(sequence_data)
            prompt_data = self.get_prompt_data_from_sequence(sequence_data)
            
            summary = {
                'image_count': len(image_paths),
                'has_temperature_data': len(temp_data) > 0,
                'temperature_points': len(temp_data),
                'has_prompt_data': len(prompt_data) > 0,
                'prompt_images_count': len(prompt_data),
                'total_prompt_points': sum(len(data["points"]) for data in prompt_data.values()) if prompt_data else 0,
                'explosion_duration': parameters.get('explosion_duration', '未知'),
                'material_type': parameters.get('material_type', '未知'),
                'metadata': sequence_data.get('metadata', {})
            }
            
            if summary['has_temperature_data']:
                summary['temperature_range'] = {
                    'time_min': min(time_data),
                    'time_max': max(time_data),
                    'temp_min': min(temp_data),
                    'temp_max': max(temp_data)
                }
            
            return summary
            
        except Exception as e:
            print(f"生成序列摘要失败: {e}")
            return {}
    
    def _validate_sequence_format(self, sequence_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证序列文件格式
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            # 检查必需的顶级键
            required_keys = ['metadata', 'image_sequence', 'parameters']
            for key in required_keys:
                if key not in sequence_data:
                    return False, f"缺少必需字段: {key}"
            
            # 检查image_sequence结构
            image_seq = sequence_data.get('image_sequence', {})
            if 'image_paths' not in image_seq:
                return False, "image_sequence中缺少image_paths字段"
            
            # 检查parameters结构
            params = sequence_data.get('parameters', {})
            required_param_keys = ['material_type', 'explosion_duration']
            for key in required_param_keys:
                if key not in params:
                    return False, f"parameters中缺少必需字段: {key}"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证格式时出错: {str(e)}"
    
    def export_sequence_data(
        self,
        file_path: str,
        image_files: List[str],
        explosive_type: str,
        equivalent: str,
        al_percent: str,
        explosion_duration: str,
        pixel_length: str,
        imported_time_data: Optional[Union[List[float], np.ndarray]] = None,
        imported_temp_data: Optional[Union[List[float], np.ndarray]] = None
    ) -> Tuple[bool, str]:
        """
        导出序列数据到JSON文件（包含数据验证）
        
        Args:
            file_path: 导出文件路径
            image_files: 图像文件路径列表
            explosive_type: 炸药类型
            equivalent: 当量
            al_percent: 含铝量百分比
            explosion_duration: 爆炸时长
            pixel_length: 像素长度
            imported_time_data: 导入的时间数据
            imported_temp_data: 导入的温度数据
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息或成功信息)
        """
        try:
            # 首先验证数据完整性
            is_valid, error_msg = self.validate_export_data(
                image_files, explosive_type, equivalent, al_percent, explosion_duration, pixel_length
            )
            
            if not is_valid:
                return False, f"数据验证失败: {error_msg}"
            
            # 准备保存的数据
            sequence_data = {
                "metadata": self._create_metadata(),
                "image_sequence": self._create_image_sequence_data(image_files, explosion_duration),
                "parameters": self._create_parameters_data(
                    explosive_type, equivalent, al_percent, explosion_duration, pixel_length
                ),
                "temperature": self._create_temperature_data(imported_time_data, imported_temp_data)
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=2)
            
            # 获取导出摘要
            summary = self.get_export_summary(image_files, imported_time_data, imported_temp_data)
            success_msg = f"导出成功！包含 {summary['image_count']} 张图像"
            if summary['has_temperature_data']:
                success_msg += f"，{summary['temperature_points']} 个温度数据点"
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"导出序列数据失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def _create_metadata(self) -> Dict[str, str]:
        """
        创建元数据
        
        Returns:
            Dict[str, str]: 元数据字典
        """
        return {
            "description": "爆炸火球分析采样序列",
            "version": "1.0",
            "created_at": str(np.datetime64('now'))
        }
    
    def _create_image_sequence_data(self, image_files: List[str], explosion_duration: str) -> Dict[str, Any]:
        """
        创建图像序列数据
        
        Args:
            image_files: 图像文件路径列表
            explosion_duration: 爆炸时长
            
        Returns:
            Dict[str, Any]: 图像序列数据
        """
        return {
            "image_paths": image_files if image_files else [],
            "duration": explosion_duration
        }
    
    def _create_parameters_data(
        self,
        explosive_type: str,
        equivalent: str,
        al_percent: str,
        explosion_duration: str,
        pixel_length: str
    ) -> Dict[str, str]:
        """
        创建参数数据
        
        Args:
            explosive_type: 炸药类型
            equivalent: 当量
            al_percent: 含铝量百分比
            explosion_duration: 爆炸时长
            pixel_length: 像素长度
            
        Returns:
            Dict[str, str]: 参数数据
        """
        return {
            "material_type": explosive_type,
            "equivalent": equivalent,
            "al_percent": al_percent,
            "explosion_duration": explosion_duration,
            "pixel_length": pixel_length
        }
    
    def _create_temperature_data(
        self,
        imported_time_data: Optional[Union[List[float], np.ndarray]],
        imported_temp_data: Optional[Union[List[float], np.ndarray]]
    ) -> List[List[float]]:
        """
        创建温度数据（时间-温度二元组）
        
        Args:
            imported_time_data: 导入的时间数据
            imported_temp_data: 导入的温度数据
            
        Returns:
            List[List[float]]: 时间-温度二元组列表
        """
        try:
            # 检查数据是否存在（兼容numpy数组和Python列表）
            has_time_data = imported_time_data is not None and len(imported_time_data) > 0
            has_temp_data = imported_temp_data is not None and len(imported_temp_data) > 0
            
            # 如果有导入的温度数据，使用导入的数据
            if has_time_data and has_temp_data:
                # 将时间和温度数据组合成二元组列表
                time_temp_pairs = []
                for time_val, temp_val in zip(imported_time_data, imported_temp_data):
                    time_temp_pairs.append([float(time_val), float(temp_val)])
                return time_temp_pairs
            
            # 如果没有导入温度数据，返回空列表
            return []
            
        except Exception as e:
            print(f"创建温度数据失败: {e}")
            # 返回空列表作为备用
            return []
    
    def validate_export_data(
        self,
        image_files: List[str],
        explosive_type: str,
        equivalent: str,
        al_percent: str,
        explosion_duration: str,
        pixel_length: str
    ) -> Tuple[bool, str]:
        """
        验证导出数据的完整性
        
        Args:
            image_files: 图像文件路径列表
            explosive_type: 炸药类型
            equivalent: 当量
            al_percent: 含铝量百分比
            explosion_duration: 爆炸时长
            pixel_length: 像素长度
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            # 检查图像文件
            if not image_files:
                return False, "没有选择图像文件"
            
            # 检查基本参数
            if not explosive_type:
                return False, "炸药类型不能为空"
            
            if not equivalent:
                return False, "当量不能为空"
            
            if not al_percent:
                return False, "含铝量不能为空"
            
            if not explosion_duration:
                return False, "爆炸时长不能为空"
            
            if not pixel_length:
                return False, "像素长度不能为空"
            
            # 检查数值参数格式
            try:
                float(equivalent)
            except ValueError:
                return False, "当量必须是数字"
            
            try:
                float(al_percent)
            except ValueError:
                return False, "含铝量必须是数字"
            
            try:
                float(explosion_duration)
            except ValueError:
                return False, "爆炸时长必须是数字"
            
            try:
                float(pixel_length)
            except ValueError:
                return False, "像素长度必须是数字"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证数据时出错: {str(e)}"
    
    def get_export_summary(
        self,
        image_files: List[str],
        imported_time_data: Optional[Union[List[float], np.ndarray]] = None,
        imported_temp_data: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        获取导出数据摘要
        
        Args:
            image_files: 图像文件路径列表
            imported_time_data: 导入的时间数据
            imported_temp_data: 导入的温度数据
            
        Returns:
            Dict[str, Any]: 导出数据摘要
        """
        # 检查数据是否存在（兼容numpy数组和Python列表）
        has_time_data = imported_time_data is not None and len(imported_time_data) > 0
        has_temp_data = imported_temp_data is not None and len(imported_temp_data) > 0
        has_temperature_data = has_time_data and has_temp_data
        
        summary = {
            "image_count": len(image_files) if image_files else 0,
            "has_temperature_data": has_temperature_data,
            "temperature_points": len(imported_temp_data) if has_temp_data else 0
        }
        
        if has_temperature_data:
            summary["temperature_range"] = {
                "time_min": float(min(imported_time_data)),
                "time_max": float(max(imported_time_data)),
                "temp_min": float(min(imported_temp_data)),
                "temp_max": float(max(imported_temp_data))
            }
        
        return summary
