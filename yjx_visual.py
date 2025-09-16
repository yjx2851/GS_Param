import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import os
import colorsys

def hsv_to_rgb(h, s, v):
    """
    将HSV颜色转换为RGB颜色
    h: 色相 (0-360)
    s: 饱和度 (0-1)
    v: 亮度 (0-1)
    返回: RGB值 (0-255)
    """
    # 将色相从0-360转换为0-1
    h = h / 360.0
    # 使用colorsys转换
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # 转换为0-255范围
    return [int(r * 255), int(g * 255), int(b * 255)]

def analyze_ply_rgb_distribution(ply_path):
    """
    读取PLY文件并分析RGB数据分布
    """
    print(f"正在读取文件: {ply_path}")
    
    # 检查文件是否存在
    if not os.path.exists(ply_path):
        print(f"错误: 文件不存在 - {ply_path}")
        return
    
    try:
        # 读取PLY文件
        plydata = PlyData.read(ply_path)
        print(f"成功读取PLY文件")
        
        # 获取顶点数据
        vertices = plydata['vertex']
        print(f"顶点数量: {len(vertices)}")
        
        # 获取XYZ坐标
        xyz = np.stack((np.asarray(vertices.data['x']),
                       np.asarray(vertices.data['y']),
                       np.asarray(vertices.data['z'])), axis=1)
        
        # 获取segment值（RGB通道相同）
        r_values = np.asarray(vertices.data['r'])
        g_values = np.asarray(vertices.data['g'])
        b_values = np.asarray(vertices.data['b'])
        
        print(f"原始RGB值范围: R[{r_values.min()}-{r_values.max()}], G[{g_values.min()}-{g_values.max()}], B[{b_values.min()}-{b_values.max()}]")
        
        # 检查RGB是否相同
        rgb_same = np.allclose(r_values, g_values) and np.allclose(g_values, b_values)
        print(f"RGB通道是否相同: {rgb_same}")
        
        if rgb_same:
            # 使用R通道作为segment值
            segment_values = r_values.astype(int)
            print(f"Segment值范围: {segment_values.min()} - {segment_values.max()}")
            
            # 获取唯一的segment值
            unique_segments = np.unique(segment_values)
            print(f"唯一segment值数量: {len(unique_segments)}")
            print(f"唯一segment值: {unique_segments}")
            
            # 为每个唯一的segment值分配相近但可区分的颜色
            color_map = {}
            
            # 按segment值排序
            sorted_segments = np.sort(unique_segments)
            print(f"排序后的segment值: {sorted_segments}")
            
            # 使用HSV颜色空间生成相近但可区分的颜色
            num_segments = len(sorted_segments)
            
            if num_segments <= 1:
                # 只有一个segment值，使用固定颜色
                color_map[sorted_segments[0]] = [255, 0, 0]  # 红色
            else:
                # 生成颜色渐变 - 相近的segment值使用相近的颜色
                for i, seg_val in enumerate(sorted_segments):
                    # 根据segment值在排序后的位置分配色相
                    # 使用连续的色相值，确保相近的segment值有相近的颜色
                    hue = (i * 360.0 / num_segments) % 360  # 色相在0-360度之间均匀分布
                    
                    # 根据segment值的相对位置调整饱和度和亮度
                    # 让相近的值在颜色上更接近
                    relative_pos = i / (num_segments - 1) if num_segments > 1 else 0
                    
                    # 饱和度：相近的值使用相近的饱和度
                    saturation = 0.7 + 0.2 * (0.5 + 0.5 * np.sin(relative_pos * np.pi))  # 0.7-0.9之间变化
                    
                    # 亮度：相近的值使用相近的亮度
                    value = 0.8 + 0.15 * (0.5 + 0.5 * np.cos(relative_pos * np.pi))  # 0.8-0.95之间变化
                    
                    # 转换HSV到RGB
                    rgb = hsv_to_rgb(hue, saturation, value)
                    color_map[seg_val] = rgb
                    print(f"Segment {seg_val} -> HSV({hue:.1f}, {saturation:.2f}, {value:.2f}) -> RGB {rgb}")
            
            # 为每个点分配颜色
            new_colors = np.zeros((len(segment_values), 3), dtype=np.uint8)
            for i, seg_val in enumerate(segment_values):
                new_colors[i] = color_map[seg_val]
            
            # 保存新的PLY文件
            output_path = '/data1/yjx/for_cvpr_2026/GS_Param/GS/colored_segments.ply'
            save_colored_ply(xyz, new_colors, output_path)
            print(f"彩色segment PLY文件已保存到: {output_path}")
            
        else:
            print("RGB通道不相同，无法处理")
            
    except Exception as e:
        print(f"读取PLY文件时出错: {e}")

def save_colored_ply(xyz, colors, output_path):
    """
    保存带颜色的点云PLY文件
    """
    from plyfile import PlyData, PlyElement
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建PLY数据
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1] 
    elements['z'] = xyz[:, 2]
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]
    
    # 写入PLY文件
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"成功保存 {len(xyz)} 个点到 {output_path}")


if __name__ == "__main__":
    ply_path = "/data1/yjx/research_data/ABC-NEF-2DGS/00000006_surface/point_cloud/iteration_30000/point_cloud_segment.ply"
    analyze_ply_rgb_distribution(ply_path)
