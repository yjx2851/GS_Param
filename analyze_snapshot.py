#!/usr/bin/env python3
import torch
import sys
import os
import numpy as np

def analyze_snapshot(filename):
    """分析snapshot_bw.dump文件"""
    print(f"正在分析文件: {filename}")
    print(f"文件大小: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    try:
        # 加载快照
        snapshot = torch.load(filename)
        print(f"\n快照类型: {type(snapshot)}")
        
        if isinstance(snapshot, tuple):
            print(f"快照包含 {len(snapshot)} 个元素")
            return snapshot
        else:
            print(f"快照内容: {snapshot}")
            return None
            
    except Exception as e:
        print(f"加载快照时出错: {e}")
        return None

def check_tensor_details(snapshot):
    """详细检查张量信息"""
    print("\n=== 张量详细信息 ===")
    
    if not isinstance(snapshot, tuple):
        return
    
    for i, item in enumerate(snapshot):
        print(f"\n元素 {i}:")
        if isinstance(item, torch.Tensor):
            print(f"  类型: torch.Tensor")
            print(f"  形状: {item.shape}")
            print(f"  数据类型: {item.dtype}")
            print(f"  设备: {item.device}")
            print(f"  元素数量: {item.numel()}")
            
            if item.numel() > 0:
                print(f"  最小值: {item.min().item()}")
                print(f"  最大值: {item.max().item()}")
                
                # 检查数值异常
                if item.dtype in [torch.float16, torch.float32, torch.float64]:
                    print(f"  是否包含NaN: {torch.isnan(item).any()}")
                    print(f"  是否包含Inf: {torch.isinf(item).any()}")
                    if torch.isnan(item).any():
                        nan_count = torch.isnan(item).sum().item()
                        print(f"  NaN数量: {nan_count}")
                    if torch.isinf(item).any():
                        inf_count = torch.isinf(item).sum().item()
                        print(f"  Inf数量: {inf_count}")
                    
                    # 计算统计信息
                    try:
                        mean_val = item.mean().item()
                        std_val = item.std().item()
                        print(f"  均值: {mean_val}")
                        print(f"  标准差: {std_val}")
                    except:
                        print(f"  统计信息: 无法计算")
                else:
                    print(f"  统计信息: 整数类型")
        else:
            print(f"  类型: {type(item)}")
            print(f"  值: {item}")

def check_gradients(snapshot):
    """专门检查梯度张量"""
    print("\n=== 梯度张量检查 ===")
    
    if not isinstance(snapshot, tuple) or len(snapshot) < 15:
        return
    
    # 梯度张量的索引
    grad_indices = {
        12: "grad_out_color",
        13: "grad_out_segment", 
        14: "grad_depth"
    }
    
    for idx, name in grad_indices.items():
        if idx < len(snapshot):
            item = snapshot[idx]
            if isinstance(item, torch.Tensor):
                print(f"\n{name}:")
                print(f"  形状: {item.shape}")
                print(f"  数据类型: {item.dtype}")
                print(f"  设备: {item.device}")
                
                if item.numel() > 0:
                    print(f"  最小值: {item.min().item()}")
                    print(f"  最大值: {item.max().item()}")
                    
                    if item.dtype in [torch.float16, torch.float32, torch.float64]:
                        print(f"  是否包含NaN: {torch.isnan(item).any()}")
                        print(f"  是否包含Inf: {torch.isinf(item).any()}")
                        
                        if torch.isnan(item).any():
                            nan_count = torch.isnan(item).sum().item()
                            print(f"  NaN数量: {nan_count}")
                            print(f"  NaN比例: {nan_count/item.numel()*100:.2f}%")
                        
                        if torch.isinf(item).any():
                            inf_count = torch.isinf(item).sum().item()
                            print(f"  Inf数量: {inf_count}")
                            print(f"  Inf比例: {inf_count/item.numel()*100:.2f}%")
                        
                        # 检查梯度是否全零
                        zero_count = (item == 0).sum().item()
                        print(f"  零值数量: {zero_count}")
                        print(f"  零值比例: {zero_count/item.numel()*100:.2f}%")

def check_buffers(snapshot):
    """检查缓冲区"""
    print("\n=== 缓冲区检查 ===")
    
    if not isinstance(snapshot, tuple) or len(snapshot) < 22:
        return
    
    buffer_indices = {
        18: "geomBuffer",
        20: "binningBuffer", 
        21: "imageBuffer"
    }
    
    for idx, name in buffer_indices.items():
        if idx < len(snapshot):
            item = snapshot[idx]
            if isinstance(item, torch.Tensor):
                print(f"\n{name}:")
                print(f"  形状: {item.shape}")
                print(f"  数据类型: {item.dtype}")
                print(f"  设备: {item.device}")
                print(f"  元素数量: {item.numel()}")
                
                if item.numel() > 0:
                    print(f"  最小值: {item.min().item()}")
                    print(f"  最大值: {item.max().item()}")
                    
                    # 检查缓冲区是否为空
                    if item.numel() == 0:
                        print(f"  警告: 缓冲区为空!")
                    elif item.numel() < 100:
                        print(f"  警告: 缓冲区大小异常小: {item.numel()}")

def check_matrices(snapshot):
    """检查矩阵"""
    print("\n=== 矩阵检查 ===")
    
    if not isinstance(snapshot, tuple) or len(snapshot) < 10:
        return
    
    matrix_indices = {
        8: "viewmatrix",
        9: "projmatrix"
    }
    
    for idx, name in matrix_indices.items():
        if idx < len(snapshot):
            item = snapshot[idx]
            if isinstance(item, torch.Tensor):
                print(f"\n{name}:")
                print(f"  形状: {item.shape}")
                print(f"  数据类型: {item.dtype}")
                print(f"  设备: {item.device}")
                
                if item.numel() > 0:
                    print(f"  最小值: {item.min().item()}")
                    print(f"  最大值: {item.max().item()}")
                    
                    # 检查矩阵是否为单位矩阵或接近单位矩阵
                    if item.shape == (4, 4):
                        print(f"  矩阵内容:")
                        print(f"  {item}")
                        
                        # 检查行列式
                        try:
                            det = torch.det(item).item()
                            print(f"  行列式: {det}")
                            if abs(det) < 1e-6:
                                print(f"  警告: 矩阵接近奇异!")
                        except:
                            print(f"  行列式: 无法计算")

def check_critical_values(snapshot):
    """检查关键数值"""
    print("\n=== 关键数值检查 ===")
    
    if not isinstance(snapshot, tuple) or len(snapshot) < 23:
        return
    
    # 检查关键参数
    critical_indices = {
        10: "tanfovx",
        11: "tanfovy",
        16: "sh_degree",
        19: "num_rendered"
    }
    
    for idx, name in critical_indices.items():
        if idx < len(snapshot):
            item = snapshot[idx]
            if isinstance(item, torch.Tensor):
                print(f"\n{name}:")
                print(f"  形状: {item.shape}")
                print(f"  值: {item}")
            else:
                print(f"\n{name}: {item}")

def identify_problems(snapshot):
    """识别潜在问题"""
    print("\n=== 问题识别 ===")
    
    problems = []
    
    if not isinstance(snapshot, tuple):
        return problems
    
    for i, item in enumerate(snapshot):
        if isinstance(item, torch.Tensor):
            # 检查NaN
            if item.dtype in [torch.float16, torch.float32, torch.float64]:
                if torch.isnan(item).any():
                    problems.append(f"元素{i}: 包含NaN值")
                
                if torch.isinf(item).any():
                    problems.append(f"元素{i}: 包含Inf值")
            
            # 检查异常大的值
            if item.numel() > 0:
                max_val = item.max().item()
                min_val = item.min().item()
                if abs(max_val) > 1e6 or abs(min_val) > 1e6:
                    problems.append(f"元素{i}: 包含异常大的值 ({min_val} 到 {max_val})")
    
    if problems:
        print("发现的问题:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("未发现明显问题")

if __name__ == "__main__":
    filename = "snapshot_bw.dump"
    
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在")
        sys.exit(1)
    
    # 分析快照
    snapshot = analyze_snapshot(filename)
    
    if snapshot is not None:
        # 详细检查
        check_tensor_details(snapshot)
        check_gradients(snapshot)
        check_buffers(snapshot)
        check_matrices(snapshot)
        check_critical_values(snapshot)
        identify_problems(snapshot)
    else:
        print("无法加载快照文件") 