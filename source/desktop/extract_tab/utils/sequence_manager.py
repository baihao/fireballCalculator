#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列文件管理模块
负责火球分析序列数据的导入和导出
"""

import glob
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
        从序列数据中提取图像路径列表（支持多种格式）
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            List[str]: 图像路径列表
        """
        try:
            # 优先检查简单格式
            if 'image_paths' in sequence_data:
                return sequence_data.get('image_paths', [])
            
            # 检查image_sequence格式
            if 'image_sequence' in sequence_data:
                return sequence_data.get('image_sequence', {}).get('image_paths', [])
            
            return []
            
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
    
    def get_ignition_point_from_sequence(self, sequence_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        从序列数据中提取起爆点
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Optional[Tuple[int, int]]: 起爆点坐标 (x, y) 或 None
        """
        try:
            # 从image_sequence中获取target_center
            image_sequence = sequence_data.get('image_sequence', {})
            target_center = image_sequence.get('target_center', None)
            
            if target_center and len(target_center) >= 2:
                return tuple(target_center[:2])  # 确保返回 (x, y) 元组
            
            return None
            
        except Exception as e:
            print(f"提取起爆点失败: {e}")
            return None
    
    def get_segmentation_results_from_sequence(self, sequence_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        从序列数据中提取分割结果
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Optional[List[Dict[str, Any]]]: 分割结果列表，每个元素包含轮廓、质心、最大半径信息
        """
        try:
            segmentation_results = sequence_data.get('image_sequence_segmentation', None)
            
            if segmentation_results is None:
                print("未找到分割结果数据")
                return None
            
            if not isinstance(segmentation_results, list):
                print("分割结果数据格式错误，应为数组")
                return None
            
            print(f"✓ 找到 {len(segmentation_results)} 张图片的分割结果")
            
            # 统计成功的分割数量
            successful_count = sum(1 for result in segmentation_results if result.get('success', False))
            print(f"✓ 其中 {successful_count} 张图片分割成功")
            
            return segmentation_results
            
        except Exception as e:
            print(f"提取分割结果失败: {e}")
            return None
    
    def has_segmentation_results(self, sequence_data: Dict[str, Any]) -> bool:
        """
        检查序列数据是否包含分割结果
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            bool: 是否包含分割结果
        """
        try:
            segmentation_results = sequence_data.get('image_sequence_segmentation', None)
            return segmentation_results is not None and len(segmentation_results) > 0
        except Exception:
            return False
    
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
    
    def save_prompt_and_ignition_data_to_sequence(
        self, 
        file_path: str, 
        prompt_data: Dict[int, Dict[str, Any]], 
        ignition_point: Optional[Tuple[int, int]]
    ) -> Tuple[bool, str]:
        """
        将prompt数据和起爆点保存到序列文件中
        
        Args:
            file_path: 序列文件路径
            prompt_data: prompt数据字典
            ignition_point: 起爆点坐标 (x, y) 或 None
            
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
            
            # 将prompt_data和target_center添加到image_sequence中
            if 'image_sequence' not in sequence_data:
                sequence_data['image_sequence'] = {}
            
            sequence_data['image_sequence']['prompt_data'] = prompt_data_str_keys
            
            # 保存起爆点
            if ignition_point:
                sequence_data['image_sequence']['target_center'] = list(ignition_point)
            elif 'target_center' in sequence_data['image_sequence']:
                # 如果没有起爆点但之前有，则删除
                del sequence_data['image_sequence']['target_center']
            
            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=4)
            
            # 统计信息
            total_images_with_prompts = len(prompt_data)
            total_points = sum(len(data["points"]) for data in prompt_data.values())
            
            success_msg = f"参考点数据保存成功！包含 {total_images_with_prompts} 张图像的 {total_points} 个参考点"
            if ignition_point:
                success_msg += f"，起爆点: ({ignition_point[0]}, {ignition_point[1]})"
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"保存参考点数据失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def clear_prompt_data_from_sequence(self, file_path: str) -> Tuple[bool, str]:
        """
        从序列文件中清除特征点数据
        
        Args:
            file_path: 序列文件路径
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息或成功信息)
        """
        try:
            # 读取现有的序列文件
            success, sequence_data, message = self.load_sequence_file(file_path)
            if not success:
                return False, f"无法读取序列文件: {message}"
            
            # 清除prompt_data
            if 'image_sequence' in sequence_data and 'prompt_data' in sequence_data['image_sequence']:
                del sequence_data['image_sequence']['prompt_data']
                print("✅ 已清除特征点数据")
            
            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=4)
            
            return True, "特征点数据清除成功"
            
        except Exception as e:
            error_msg = f"清除特征点数据失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def clear_ignition_point_from_sequence(self, file_path: str) -> Tuple[bool, str]:
        """
        从序列文件中清除爆心数据
        
        Args:
            file_path: 序列文件路径
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息或成功信息)
        """
        try:
            # 读取现有的序列文件
            success, sequence_data, message = self.load_sequence_file(file_path)
            if not success:
                return False, f"无法读取序列文件: {message}"
            
            # 清除target_center(爆心)
            if 'image_sequence' in sequence_data and 'target_center' in sequence_data['image_sequence']:
                del sequence_data['image_sequence']['target_center']
                print("✅ 已清除爆心数据")
            
            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=4)
            
            return True, "爆心数据清除成功"
            
        except Exception as e:
            error_msg = f"清除爆心数据失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def clear_segmentation_results_from_sequence(self, file_path: str) -> Tuple[bool, str]:
        """
        从序列文件中清除分割结果
        
        Args:
            file_path: 序列文件路径
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息或成功信息)
        """
        try:
            # 读取现有的序列文件
            success, sequence_data, message = self.load_sequence_file(file_path)
            if not success:
                return False, f"无法读取序列文件: {message}"
            
            # 清除image_sequence_segmentation
            if 'image_sequence_segmentation' in sequence_data:
                del sequence_data['image_sequence_segmentation']
                print("✅ 已清除分割结果")
            
            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=4)
            
            return True, "分割结果清除成功"
            
        except Exception as e:
            error_msg = f"清除分割结果失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def clear_all_analysis_data_from_sequence(self, file_path: str) -> Tuple[bool, str]:
        """
        从序列文件中清除所有分析数据（特征点、爆心、分割结果）
        
        Args:
            file_path: 序列文件路径
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误信息或成功信息)
        """
        try:
            # 读取现有的序列文件
            success, sequence_data, message = self.load_sequence_file(file_path)
            if not success:
                return False, f"无法读取序列文件: {message}"
            
            cleared_items = []
            
            # 清除prompt_data
            if 'image_sequence' in sequence_data:
                if 'prompt_data' in sequence_data['image_sequence']:
                    del sequence_data['image_sequence']['prompt_data']
                    cleared_items.append("特征点数据")
                
                # 清除target_center(爆心)
                if 'target_center' in sequence_data['image_sequence']:
                    del sequence_data['image_sequence']['target_center']
                    cleared_items.append("爆心数据")
            
            # 清除分割结果
            if 'image_sequence_segmentation' in sequence_data:
                del sequence_data['image_sequence_segmentation']
                cleared_items.append("分割结果")
            
            if not cleared_items:
                return True, "没有找到需要清除的分析数据"
            
            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, ensure_ascii=False, indent=4)
            
            cleared_text = "、".join(cleared_items)
            print(f"✅ 已清除: {cleared_text}")
            
            return True, f"所有分析数据清除成功: {cleared_text}"
            
        except Exception as e:
            error_msg = f"清除分析数据失败: {str(e)}"
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
            ignition_point = self.get_ignition_point_from_sequence(sequence_data)
            
            summary = {
                'image_count': len(image_paths),
                'has_temperature_data': len(temp_data) > 0,
                'temperature_points': len(temp_data),
                'has_prompt_data': len(prompt_data) > 0,
                'prompt_images_count': len(prompt_data),
                'total_prompt_points': sum(len(data["points"]) for data in prompt_data.values()) if prompt_data else 0,
                'has_ignition_point': ignition_point is not None,
                'ignition_point': ignition_point,
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
        验证序列文件格式（支持包含分割结果的格式）
        
        Args:
            sequence_data: 序列数据字典
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            # 检查是否包含分割结果的格式（我们的导出格式）
            if 'image_sequence_segmentation' in sequence_data:
                # 包含分割结果的格式，需要检查基本结构
                if 'image_paths' in sequence_data:
                    # 简单格式 + 分割结果
                    return True, ""
                elif 'image_sequence' in sequence_data:
                    # image_sequence格式 + 分割结果
                    image_seq = sequence_data.get('image_sequence', {})
                    if 'image_paths' not in image_seq:
                        return False, "image_sequence中缺少image_paths字段"
                    return True, ""
            
            # 检查标准的完整格式
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

    # ------------------------------------------------------------------ #
    # 机器视觉：图像文件夹 → 同级工作 JSON
    # ------------------------------------------------------------------ #
    def work_sequence_json_path_for_image_folder(self, folder_path: str) -> str:
        """与图像序列文件夹同级：{文件夹名}_fireball_sequence.json"""
        folder_path = os.path.abspath(folder_path)
        parent = os.path.dirname(folder_path)
        name = os.path.basename(folder_path.rstrip(os.sep))
        return os.path.join(parent, f"{name}_fireball_sequence.json")

    def collect_image_paths_in_folder(self, folder_path: str) -> List[str]:
        """与 input_tab 一致：仅当前目录下常见图像扩展名，排序。"""
        folder_path = os.path.abspath(folder_path)
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        image_files: List[str] = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, ext)
            image_files.extend(glob.glob(pattern))
        image_files = sorted(set(image_files))
        return [os.path.abspath(p) for p in image_files]

    def create_work_sequence_from_image_folder(
        self,
        folder_path: str,
        material_type: str = "温压弹",
        equivalent: str = "1",
        al_percent: str = "30",
        explosion_duration: str = "140",
        pixel_length: str = "0.01",
    ) -> Tuple[bool, str, Optional[str]]:
        """
        扫描文件夹内图像，在与文件夹同级路径写入 {文件夹名}_fireball_sequence.json。
        返回 (成功, 消息, 工作文件绝对路径)。
        """
        abs_paths = self.collect_image_paths_in_folder(folder_path)
        if not abs_paths:
            return False, "所选文件夹中没有支持的图像文件", None
        work_path = self.work_sequence_json_path_for_image_folder(folder_path)
        ok, msg = self.export_sequence_data(
            work_path,
            abs_paths,
            material_type,
            equivalent,
            al_percent,
            explosion_duration,
            pixel_length,
            None,
            None,
        )
        if not ok:
            return False, msg, None
        return True, msg, work_path

    def load_temperature_data_file(self, file_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        从 CSV / JSON / 文本读取温度序列（与 input_tab.load_temperature_data 行为对齐）。
        """
        try:
            if file_path.lower().endswith(".csv"):
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    time_col = None
                    temp_col = None
                    for col in df.columns:
                        cl = col.lower()
                        if "time" in cl or "时间" in cl or "ms" in cl:
                            time_col = col
                        elif "temp" in cl or "温度" in cl or "k" in cl:
                            temp_col = col
                    if time_col is None or temp_col is None:
                        time_col = df.columns[0]
                        temp_col = df.columns[1]
                    t = [float(x) for x in df[time_col].values]
                    T = [float(x) for x in df[temp_col].values]
                    return t, T
                except ImportError:
                    return self._read_temperature_csv_manual(file_path)
            if file_path.lower().endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "time" in data and "temperature" in data:
                    return [float(x) for x in data["time"]], [float(x) for x in data["temperature"]]
                return None, None
            time_data: List[float] = []
            temp_data: List[float] = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "," in line:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                time_data.append(float(parts[0]))
                                temp_data.append(float(parts[1]))
                            except ValueError:
                                continue
            if time_data and temp_data:
                return time_data, temp_data
            return None, None
        except Exception as e:
            print(f"加载温度数据失败: {e}")
            return None, None

    def _read_temperature_csv_manual(self, file_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        time_data: List[float] = []
        temp_data: List[float] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if line and "," in line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            time_data.append(float(parts[0]))
                            temp_data.append(float(parts[1]))
                        except ValueError:
                            continue
            if time_data and temp_data:
                return time_data, temp_data
        except Exception as e:
            print(f"手动读 CSV 失败: {e}")
        return None, None


if __name__ == "__main__":
    pass
