#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分析脚本
统计训练集、验证集、测试集中每个类别的数量
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.dataloaders import LoadImagesAndLabels
from utils.general import check_dataset, yaml_load


def analyze_dataset(data_yaml):
    """分析数据集，统计每个集合中每个类别的数量"""
    
    # 加载数据集配置
    data_dict = check_dataset(data_yaml)
    nc = data_dict['nc']  # 类别数量
    names = data_dict['names']  # 类别名称
    
    # 如果是列表格式，转换为字典
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    
    print("=" * 80)
    print("数据集分析报告")
    print("=" * 80)
    print(f"\n数据集配置文件: {data_yaml}")
    print(f"类别数量: {nc}")
    print(f"类别名称: {names}")
    print("\n" + "=" * 80)
    
    results = {}
    
    # 分析每个数据集划分
    for split in ['train', 'val', 'test']:
        if split not in data_dict or data_dict[split] is None:
            print(f"\n{split.upper()} 集: 未配置")
            results[split] = None
            continue
        
        print(f"\n{split.upper()} 集路径: {data_dict[split]}")
        
        try:
            # 加载数据集
            dataset = LoadImagesAndLabels(
                path=data_dict[split],
                img_size=640,
                batch_size=1,
                augment=False,
                cache_images=False,
                single_cls=False,
                prefix=f'{split}: '
            )
            
            # 统计每个类别的实例数量
            all_labels = np.concatenate(dataset.labels, 0) if len(dataset.labels) > 0 else np.array([])
            
            if len(all_labels) > 0:
                class_counts = np.bincount(all_labels[:, 0].astype(int), minlength=nc)
            else:
                class_counts = np.zeros(nc, dtype=int)
            
            # 统计包含每个类别的图像数量
            images_with_class = np.zeros(nc, dtype=int)
            for label in dataset.labels:
                if len(label) > 0:
                    classes_in_image = np.unique(label[:, 0].astype(int))
                    for cls in classes_in_image:
                        if 0 <= cls < nc:
                            images_with_class[cls] += 1
            
            # 保存结果
            results[split] = {
                'total_images': dataset.n,
                'total_instances': int(class_counts.sum()),
                'per_class_instances': class_counts.tolist(),
                'per_class_images': images_with_class.tolist()
            }
            
            # 打印结果
            print(f"总图像数: {dataset.n}")
            print(f"总实例数: {int(class_counts.sum())}")
            print("\n每个类别的统计:")
            print(f"{'类别':<15} {'实例数':<12} {'图像数':<12} {'平均每图':<12}")
            print("-" * 60)
            
            for i in range(nc):
                class_name = names.get(i, f'class_{i}')
                instances = int(class_counts[i])
                images = int(images_with_class[i])
                avg_per_image = instances / images if images > 0 else 0
                print(f"{class_name:<15} {instances:<12} {images:<12} {avg_per_image:<12.2f}")
            
        except Exception as e:
            print(f"错误: 无法加载 {split} 集 - {e}")
            results[split] = None
    
    # 打印汇总表
    print("\n" + "=" * 80)
    print("汇总表")
    print("=" * 80)
    
    # 表头
    header = f"{'类别':<15}"
    for split in ['train', 'val', 'test']:
        if results.get(split) is not None:
            header += f" {split.upper():<20}"
    print(header)
    print("-" * 80)
    
    # 每个类别的统计
    for i in range(nc):
        class_name = names.get(i, f'class_{i}')
        row = f"{class_name:<15}"
        for split in ['train', 'val', 'test']:
            if results.get(split) is not None:
                instances = results[split]['per_class_instances'][i]
                images = results[split]['per_class_images'][i]
                row += f" 实例:{instances:<6} 图像:{images:<6}"
        print(row)
    
    # 总计
    print("-" * 80)
    total_row = f"{'总计':<15}"
    for split in ['train', 'val', 'test']:
        if results.get(split) is not None:
            total_instances = results[split]['total_instances']
            total_images = results[split]['total_images']
            total_row += f" 实例:{total_instances:<6} 图像:{total_images:<6}"
    print(total_row)
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='分析数据集统计信息')
    parser.add_argument('--data', type=str, default='data/SnowCCTSDB.yaml', help='数据集配置文件路径')
    args = parser.parse_args()
    
    analyze_dataset(args.data)

