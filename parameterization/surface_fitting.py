import json
import os
import numpy as np
from itertools import combinations
from collections import defaultdict
import trimesh
from scipy.spatial import ConvexHull
from plyfile import PlyData

def read_line_points(file_path):
    """读取JSON文件中的线段端点数据"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['curves_ctl_pts']

def calculate_plane_from_three_points(p1, p2, p3):
    """
    通过三个点计算平面方程 ax + by + cz + d = 0
    
    Args:
        p1, p2, p3: 三个3D点坐标
        
    Returns:
        tuple: (a, b, c, d) 平面方程系数
    """
    # 计算两个向量
    v1 = p2 - p1
    v2 = p3 - p1
    
    # 计算法向量（叉积）
    normal = np.cross(v1, v2)
    
    # 归一化法向量
    norm = np.linalg.norm(normal)
    if norm < 1e-10:  # 三点共线的情况
        return None
    
    normal = normal / norm
    
    # 计算d值
    d = -np.dot(normal, p1)
    
    return (normal[0], normal[1], normal[2], d)

def point_to_plane_distance(point, plane_coeffs):
    """
    计算点到平面的距离
    
    Args:
        point: 3D点坐标
        plane_coeffs: 平面方程系数 (a, b, c, d)
        
    Returns:
        float: 点到平面的距离
    """
    a, b, c, d = plane_coeffs
    x, y, z = point
    
    # 点到平面距离公式: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    numerator = abs(a * x + b * y + c * z + d)
    denominator = np.sqrt(a * a + b * b + c * c)
    
    return numerator / denominator

def lines_are_coplanar(line1, line2, line3, threshold=0.01):
    """
    判断三条线段是否共面
    
    Args:
        line1, line2, line3: 三条线段，每条线段包含两个端点
        threshold: 共面性阈值
        
    Returns:
        bool: 是否共面
    """
    # 提取三个点（每条线段的第一个端点）
    p1 = np.array(line1[0])
    p2 = np.array(line2[0])
    p3 = np.array(line3[0])
    
    # 计算平面方程
    plane_coeffs = calculate_plane_from_three_points(p1, p2, p3)
    if plane_coeffs is None:
        return False
    
    # 检查所有端点是否都在该平面附近
    all_points = []
    for line in [line1, line2, line3]:
        all_points.extend(line)
    
    for point in all_points:
        distance = point_to_plane_distance(np.array(point), plane_coeffs)
        if distance > threshold:
            return False
    
    return True

def analyze_plane_groups(line_points, coplanar_groups):
    """
    分析平面组的详细信息
    
    Args:
        line_points: 线段端点数组
        coplanar_groups: 共面线段组列表
        
    Returns:
        dict: 分析结果
    """
    analysis = {
        'total_planes': len(coplanar_groups),
        'plane_details': [],
        'statistics': {}
    }
    
    total_lines_in_groups = 0
    
    for i, group in enumerate(coplanar_groups):
        group_lines = [line_points[idx] for idx in group]
        
        # 计算组内所有端点的中心
        all_points = []
        for line in group_lines:
            all_points.extend(line)
        
        all_points = np.array(all_points)
        center = np.mean(all_points, axis=0)
        
        # 计算组内线段的平均长度
        lengths = []
        for line in group_lines:
            length = np.linalg.norm(np.array(line[1]) - np.array(line[0]))
            lengths.append(length)
        
        avg_length = np.mean(lengths)
        total_length = np.sum(lengths)
        
        # 计算平面方程（使用前三个点）
        if len(group_lines) >= 3:
            p1 = np.array(group_lines[0][0])
            p2 = np.array(group_lines[1][0])
            p3 = np.array(group_lines[2][0])
            plane_coeffs = calculate_plane_from_three_points(p1, p2, p3)
        else:
            plane_coeffs = None
        
        plane_info = {
            'plane_id': i + 1,
            'num_lines': len(group),
            'line_indices': group,
            'center': center.tolist(),
            'average_length': avg_length,
            'total_length': total_length,
            'plane_equation': plane_coeffs
        }
        
        analysis['plane_details'].append(plane_info)
        total_lines_in_groups += len(group)
    
    # 统计信息
    analysis['statistics'] = {
        'total_lines': len(line_points),
        'lines_in_groups': total_lines_in_groups,
        'lines_not_grouped': len(line_points) - total_lines_in_groups,
        'grouping_rate': total_lines_in_groups / len(line_points) if len(line_points) > 0 else 0
    }
    
    return analysis


def find_coplanar_groups(line_points, threshold=0.01, min_group_size=3):
    """
    找到所有共面的线段组，允许一条线段属于多个平面
    
    Args:
        line_points: 线段端点数组
        threshold: 共面性阈值
        min_group_size: 最小组大小
        
    Returns:
        list: 共面线段组列表，每个组是一个线段索引列表
    """
    n_lines = len(line_points)
    print(f"总共有 {n_lines} 条线段")
    
    # 存储所有共面组
    coplanar_groups = []
    
    # 记录每条线段属于多少个组（用于统计，不限制归属）
    line_group_count = [0] * n_lines
    
    # 遍历所有可能的三条线段组合
    for i, j, k in combinations(range(n_lines), 3):
        # 检查三条线段是否共面
        if lines_are_coplanar(line_points[i], line_points[j], line_points[k], threshold):
            # 创建新的共面组
            current_group = [i, j, k]
            
            # 尝试添加更多共面的线段
            for m in range(n_lines):
                # 跳过已经在当前组中的线段
                if m in current_group:
                    continue
                
                # 检查当前组中的所有线段与新线段是否共面
                group_lines = [line_points[idx] for idx in current_group]
                all_coplanar = True
                
                # 随机选择组内的三条线段与新线段一起检查
                for _ in range(min(3, len(current_group))):
                    sample_indices = np.random.choice(current_group, size=3, replace=False)
                    sample_lines = [line_points[idx] for idx in sample_indices]
                    
                    if not lines_are_coplanar(sample_lines[0], sample_lines[1], sample_lines[2], threshold):
                        all_coplanar = False
                        break
                
                if all_coplanar:
                    # 进一步验证：检查新线段是否与组内所有线段共面
                    new_line = line_points[m]
                    for group_line in group_lines:
                        # 选择组内两条线段与新线段一起验证
                        if len(current_group) >= 2:
                            idx1, idx2 = np.random.choice(current_group, size=2, replace=False)
                            if not lines_are_coplanar(line_points[idx1], line_points[idx2], new_line, threshold):
                                all_coplanar = False
                                break
                    
                    if all_coplanar:
                        current_group.append(m)
            
            # 如果组大小满足要求，添加到结果中
            if len(current_group) >= min_group_size:
                coplanar_groups.append(current_group)
                
                # 更新线段归属统计
                for line_idx in current_group:
                    line_group_count[line_idx] += 1
                
    
    return coplanar_groups

def merge_similar_planes(coplanar_groups, line_points, center_threshold=0.1, normal_threshold=0.1):
    """
    合并相似的平面（中心点和法向量接近的平面）
    
    Args:
        coplanar_groups: 共面线段组列表
        line_points: 线段端点数组
        center_threshold: 中心点距离阈值
        normal_threshold: 法向量夹角阈值（弧度）
        
    Returns:
        list: 去重后的共面线段组列表
    """
    if len(coplanar_groups) <= 1:
        return coplanar_groups
    
    print(f"\n=== 平面去重 ===")
    print(f"原始平面数量: {len(coplanar_groups)}")
    
    # 计算每个平面的中心点和法向量
    plane_info = []
    for group_idx, group in enumerate(coplanar_groups):
        group_lines = [line_points[idx] for idx in group]
        
        # 计算中心点
        all_points = []
        for line in group_lines:
            all_points.extend(line)
        all_points = np.array(all_points)
        center = np.mean(all_points, axis=0)
        
        # 计算法向量（使用前三个点）
        if len(group_lines) >= 3:
            p1 = np.array(group_lines[0][0])
            p2 = np.array(group_lines[1][0])
            p3 = np.array(group_lines[2][0])
            normal = calculate_plane_from_three_points(p1, p2, p3)
            
            if normal is not None:
                normal_vector = np.array(normal[:3])  # 取前三个分量作为法向量
                # 确保法向量指向同一方向（统一到上半空间）
                if normal_vector[2] < 0:
                    normal_vector = -normal_vector
            else:
                normal_vector = np.array([0, 0, 1])  # 默认法向量
        else:
            normal_vector = np.array([0, 0, 1])  # 默认法向量
        
        plane_info.append({
            'group_idx': group_idx,
            'center': center,
            'normal': normal_vector,
            'lines': group,
            'merged': False
        })
    
    # 合并相似的平面
    merged_groups = []
    merged_indices = set()
    
    for i in range(len(plane_info)):
        if i in merged_indices:
            continue
            
        current_group = plane_info[i]['lines'].copy()
        current_center = plane_info[i]['center']
        current_normal = plane_info[i]['normal']
        
        # 标记当前平面已处理
        merged_indices.add(i)
        plane_info[i]['merged'] = True
        
        # 查找相似的平面
        for j in range(i + 1, len(plane_info)):
            if j in merged_indices:
                continue
                
            other_center = plane_info[j]['center']
            other_normal = plane_info[j]['normal']
            
            # 计算中心点距离
            center_distance = np.linalg.norm(current_center - other_center)
            
            # 计算法向量夹角
            cos_angle = np.dot(current_normal, other_normal)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
            angle = np.arccos(abs(cos_angle))  # 取绝对值，因为法向量方向可能相反
            
            # 判断是否相似
            if center_distance < center_threshold and angle < normal_threshold:
                # 合并平面
                current_group.extend(plane_info[j]['lines'])
                merged_indices.add(j)
                plane_info[j]['merged'] = True
                
                print(f"合并平面 {i+1} 和平面 {j+1}")
                print(f"  中心点距离: {center_distance:.4f} (阈值: {center_threshold})")
                print(f"  法向量夹角: {np.degrees(angle):.2f}° (阈值: {np.degrees(normal_threshold):.2f}°)")
        
        # 去重线段索引
        current_group = list(dict.fromkeys(current_group))
        
        if len(current_group) >= 3:  # 保持最小组大小要求
            merged_groups.append(current_group)
            print(f"合并后平面 {len(merged_groups)}: {len(current_group)} 条线段")
    
    print(f"去重后平面数量: {len(merged_groups)}")
    print(f"减少了 {len(coplanar_groups) - len(merged_groups)} 个平面")
    
    return merged_groups

def surface_fitting(line_points, threshold=0.01, min_group_size=3):
    """
    主要的表面拟合函数
    
    Args:
        line_points: 线段端点数组
        threshold: 共面性阈值
        min_group_size: 最小组大小
        
    Returns:
        dict: 拟合结果
    """
    print(f"开始表面拟合，共面性阈值: {threshold}")
    print(f"最小组大小: {min_group_size}")
    
    # 找到共面组
    coplanar_groups = find_coplanar_groups(line_points, threshold, min_group_size)

    merged_groups = merge_similar_planes(
    coplanar_groups, 
    line_points, 
    center_threshold=0.3,   # 更宽松的中心点要求
    normal_threshold=0.3    # 更宽松的法向量要求
)
    return merged_groups


def param_and_save(plane_groups, object_path, line_points):
    """
    计算平面参数并保存为网格文件
    
    Args:
        plane_groups: 平面组列表
        object_path: 对象路径
        line_points: 线段端点数组
    """
    print(f"\n=== 计算平面参数并保存网格 ===")
    
    # 存储所有平面的参数和网格

    whole_points_path = os.path.join(object_path, "point_cloud", "iteration_30000", "point_cloud_segment.ply")
    plydata = PlyData.read(whole_points_path)
    whole_points = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

    all_planes_data = []
    
    for plane_idx, plane_group in enumerate(plane_groups):
        print(f"\n处理平面 {plane_idx + 1}: {len(plane_group)} 条线段")
        
        # 提取当前平面的线段
        plane_lines = [line_points[idx] for idx in plane_group]
        
        # 计算平面参数
        plane_params = calculate_plane_parameters(plane_lines, plane_idx)
        all_planes_data.append(plane_params)
        
        # 生成平面网格
        plane_mesh = generate_convex_plane_mesh(plane_params, whole_points)
        
        color_palette = [
        [255, 0, 0, 255],      # 红色
        [0, 255, 0, 255],      # 绿色
        [0, 0, 255, 255],      # 蓝色
        [255, 255, 0, 255],    # 黄色
        [255, 0, 255, 255],    # 洋红
        [0, 255, 255, 255],    # 青色
        [255, 128, 0, 255],    # 橙色
        [128, 0, 255, 255],    # 紫色
        [0, 128, 255, 255],    # 天蓝
        [255, 128, 128, 255],  # 粉红
        [128, 255, 0, 255],    # 青绿
        [255, 0, 128, 255],    # 玫瑰
        [128, 128, 255, 255],  # 淡蓝
        [255, 255, 128, 255],  # 淡黄
        [128, 255, 128, 255],  # 淡绿
        [255, 128, 255, 255],  # 淡紫
        ]
        if plane_mesh is not None:
            # 保存单个平面网格
            plane_mesh_file = os.path.join(object_path, f"plane_{plane_idx + 1:03d}.obj")
            plane_mesh.visual.face_colors = np.array(color_palette[plane_idx % len(color_palette)])

            plane_mesh.export(plane_mesh_file, file_type='obj')
    
    # 保存所有平面参数
    params_file = os.path.join(object_path, "plane_parameters.json")
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(all_planes_data, f, indent=2, ensure_ascii=False)
    

def calculate_plane_parameters(plane_lines, plane_idx):
    """
    计算单个平面的参数
    
    Args:
        plane_lines: 平面内的线段列表
        plane_idx: 平面索引
        
    Returns:
        dict: 平面参数
    """
    # 收集所有端点
    all_points = []
    for line in plane_lines:
        all_points.extend(line)
    
    all_points = np.array(all_points)
    
    # 计算中心点
    center = np.mean(all_points, axis=0)
    
    # 计算法向量和平面方程
    if len(plane_lines) >= 3:
        p1 = np.array(plane_lines[0][0])
        p2 = np.array(plane_lines[1][0])
        p3 = np.array(plane_lines[2][0])
        plane_coeffs = calculate_plane_from_three_points(p1, p2, p3)
        
        if plane_coeffs is not None:
            normal = np.array(plane_coeffs[:3])
            d = plane_coeffs[3]
        else:
            normal = np.array([0, 0, 1])
            d = -np.dot(normal, center)
    else:
        normal = np.array([0, 0, 1])
        d = -np.dot(normal, center)
    
    # 归一化法向量
    normal = normal / np.linalg.norm(normal)
    
    
    # 计算线段统计
    line_lengths = []
    for line in plane_lines:
        length = np.linalg.norm(np.array(line[1]) - np.array(line[0]))
        line_lengths.append(length)
    
    stats = {
        'num_lines': len(plane_lines),
        'total_length': np.sum(line_lengths),
        'avg_length': np.mean(line_lengths),
        'min_length': np.min(line_lengths),
        'max_length': np.max(line_lengths)
    }
    
    return {
        'plane_id': plane_idx + 1,
        'center': center.tolist(),
        'normal': normal.tolist(),
        'd': float(d),
        'statistics': stats,
        'line_indices': list(range(len(plane_lines)))
    }

def generate_convex_plane_mesh(plane_params, whole_points, distance_threshold=0.005, min_points_threshold=500):
    """
    生成凸多边形平面网格（更精确的边界）
    
    Args:
        plane_params: 平面参数
        whole_points: 整个物体的表面点云
        distance_threshold: 点到平面的距离阈值
        min_points_threshold: 内点数量阈值
        
    Returns:
        trimesh.Trimesh: 凸多边形平面网格
    """

    
    # 获取平面参数
    center = np.array(plane_params['center'])
    normal = np.array(plane_params['normal'])
    d = plane_params['d']
    
    # 筛选内点
    plane_coeffs = (normal[0], normal[1], normal[2], d)
    inlier_points = []
    
    for point in whole_points:
        point_array = np.array(point)
        distance = point_to_plane_distance(point_array, plane_coeffs)
        
        if distance < distance_threshold:
            inlier_points.append(point_array)
    
    inlier_points = np.array(inlier_points)
    
    # 检查内点数量
    if len(inlier_points) < min_points_threshold:
        print(f"  内点数量不足: {len(inlier_points)} < {min_points_threshold}，舍弃平面")
        return None
    
    print(f"  找到 {len(inlier_points)} 个内点")
    
    # 将内点投影到平面
    projected_points = []
    for point in inlier_points:
        # 计算点到平面中心的向量
        vec_to_center = point - center
        
        # 找到两个与法向量垂直的基向量
        if abs(normal[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        u = np.cross(v, normal)
        u = u / np.linalg.norm(u)
        
        # 投影到u和v方向
        u_coord = np.dot(vec_to_center, u)
        v_coord = np.dot(vec_to_center, v)
        
        projected_points.append([u_coord, v_coord])
    
    projected_points = np.array(projected_points)
    
    # 计算凸包
    if len(projected_points) >= 3:
        try:
            hull = ConvexHull(projected_points)
            hull_vertices_2d = projected_points[hull.vertices]
        except:
            # 如果凸包计算失败，使用边界框
            u_min, v_min = np.min(projected_points, axis=0)
            u_max, v_max = np.max(projected_points, axis=0)
            
            margin_u = (u_max - u_min) * 0.1
            margin_v = (v_max - v_min) * 0.1
            
            hull_vertices_2d = np.array([
                [u_min - margin_u, v_min - margin_v],
                [u_max + margin_u, v_min - margin_v],
                [u_max + margin_u, v_max + margin_v],
                [u_min - margin_u, v_max + margin_v]
            ])
    else:
        # 点数不足，使用边界框
        u_min, v_min = np.min(projected_points, axis=0)
        u_max, v_max = np.max(projected_points, axis=0)
        
        margin_u = (u_max - u_min) * 0.1
        margin_v = (v_max - v_min) * 0.1
        
        hull_vertices_2d = np.array([
            [u_min - margin_u, v_min - margin_v],
            [u_max + margin_u, v_min - margin_v],
            [u_max + margin_u, v_max + margin_v],
            [u_min - margin_u, v_max + margin_v]
        ])
    
    # 将2D凸包顶点转换回3D空间
    vertices_3d = []
    for vertex_2d in hull_vertices_2d:
        u_coord, v_coord = vertex_2d
        
        # 重新计算基向量（确保一致性）
        if abs(normal[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        u = np.cross(v, normal)
        u = u / np.linalg.norm(u)
        
        point_3d = center + u_coord * u + v_coord * v
        vertices_3d.append(point_3d)
    
    # 创建面片（三角化凸多边形）
    if len(vertices_3d) == 3:
        faces = [[0, 1, 2]]
    elif len(vertices_3d) == 4:
        faces = [[0, 1, 2], [0, 2, 3]]
    else:
        # 对于更多顶点，使用扇形三角化
        faces = []
        for i in range(1, len(vertices_3d) - 1):
            faces.append([0, i, i + 1])
    
    # 创建网格
    mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
    
    # 更新平面参数
    plane_params['inlier_points_count'] = len(inlier_points)
    plane_params['convex_hull_vertices'] = len(vertices_3d)
    plane_params['mesh_type'] = 'convex_polygon'
    
    print(f"  生成凸多边形网格: {len(vertices_3d)} 个顶点, {len(faces)} 个面片")
    
    return mesh



if __name__ == "__main__":
    # 设置参数
    threshold = 0.005  # 共面性阈值
    # 额外的需要调整的超参数：threshold1,threshold2

    min_group_size = 3  # 最小组大小
    
    # 平面去重参数
    center_threshold = 0.05  # 中心点距离阈值
    normal_threshold = 0.1   # 法向量夹角阈值（弧度，约5.7度）
    overlap_threshold = 0.3  # 线段重叠阈值
    use_advanced_merge = True  # 是否使用高级去重
    
    # 遍历数据集
    base_path = "/data1/yjx/research_data/mini_dataset_param"
    
    for object_name in os.listdir(base_path):
        if object_name != "00000168":
            continue
        object_path = os.path.join(base_path, object_name)
        
        # 检查是否是目录
        if not os.path.isdir(object_path):
            continue
            
        # 构建曲线文件路径
        curve_file = os.path.join(object_path, f"record_{object_name}_stage1_line.json")
        
        if not os.path.exists(curve_file):
            print(f"跳过 {object_name}: 曲线文件不存在")
            continue
        
        print(f"\n处理对象: {object_name}")
        print(f"曲线文件: {curve_file}")
        
        try:
            # 读取线段数据
            line_points = read_line_points(curve_file)
            print(f"读取到 {len(line_points)} 条线段")
            
            # 执行表面拟合
            plane_groups = surface_fitting(line_points, threshold, min_group_size)
            param_and_save(plane_groups, object_path, line_points)

            
        except Exception as e:
            print(f"处理 {object_name} 时出错: {e}")
            continue
