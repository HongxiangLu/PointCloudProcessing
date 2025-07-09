import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os


def cylindrical_projection(ply_path, output_dir="cylinder_output",
                           height=1024, width=2048):
    """
    使用柱面投影将点云转换为全景图像（铁路场景优化）

    参数：
        ply_path: PLY文件路径
        output_dir: 输出目录
        height: 图像高度
        width: 图像宽度

    返回：
        多通道全景图字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载PLY文件
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    n_points = len(vertex)

    # 提取数据
    coords = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T.astype(np.uint8)
    intensity = vertex['scalar_Intensity'].astype(np.float32)
    classification = vertex['scalar_Classification'].astype(np.uint8)

    print(f"加载 {n_points} 个点，开始投影...")

    # === 柱面投影核心算法 ===

    # 1. 自动计算铁路方向（主成分分析）
    print("自动计算铁路方向...")
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # 高效计算主方向向量
    cov = centered_coords[:10000].T @ centered_coords[:10000]  # 使用样本子集提高效率
    eigvals, eigvecs = np.linalg.eig(cov)
    rail_direction = eigvecs[:, np.argmax(eigvals)]

    print(f"检测到的铁路方向: ({rail_direction[0]:.2f}, {rail_direction[1]:.2f}, {rail_direction[2]:.2f})")

    # 2. 旋转点云使铁路方向与Z轴对齐
    target_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(rail_direction, target_axis)

    if np.linalg.norm(rotation_axis) < 1e-8:
        rotation = R.identity()
        rotated_coords = centered_coords
        print("点云已与Z轴对齐")
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(rail_direction, target_axis))
        rotation = R.from_rotvec(angle * rotation_axis)
        rotated_coords = rotation.apply(centered_coords)
        print(f"点云旋转角度: {np.rad2deg(angle):.2f}度")

    # 3. 计算柱面坐标
    x, y, z = rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]

    # 柱面投影公式
    r = np.sqrt(x ** 2 + y ** 2)  # 点到铁路轴线的距离
    theta = np.arctan2(y, x)  # 水平角度 [-π, π]

    # 转换为像素坐标
    u = np.floor((theta + np.pi) / (2 * np.pi) * width).astype(int)
    # 高度映射 - 使用Z坐标
    z_min, z_max = np.min(z), np.max(z)
    v = np.floor((z - z_min) / (z_max - z_min) * (height - 1)).astype(int)

    # 边界处理
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    v = height - 1 - v  # 翻转Y轴，使图像顶部是最高点

    # === 创建图像缓冲区 ===
    # 初始化缓冲区
    depth_buffer = np.full((height, width), np.inf)
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    intensity_buffer = np.zeros((height, width), dtype=np.float32)
    class_buffer = np.zeros((height, width), dtype=np.uint8)

    # 线性索引加速
    print("将点映射到图像...")
    lin_idx = v * width + u
    unique_lin_idx = np.unique(lin_idx)

    print(f"处理 {len(unique_lin_idx)} 个像素分区...")
    for idx in tqdm(unique_lin_idx):
        # 找到当前像素的所有点
        mask = lin_idx == idx
        if not np.any(mask):
            continue

        # 找到最近点（最小柱面半径）
        min_idx = np.argmin(r[mask])
        global_idx = np.where(mask)[0][min_idx]

        # 提取像素位置
        v_idx = idx // width
        u_idx = idx % width

        # 更新缓冲区
        depth_buffer[v_idx, u_idx] = r[global_idx]
        rgb_buffer[v_idx, u_idx] = colors[global_idx]
        intensity_buffer[v_idx, u_idx] = intensity[global_idx]
        class_buffer[v_idx, u_idx] = classification[global_idx]

    # 生成多通道图像
    panorama = {
        'RGB': rgb_buffer,
        'Depth': depth_buffer,
        'Intensity': intensity_buffer,
        'Classification': class_buffer,
    }

    # === 保存结果 ===
    # 可视化深度图
    valid_depth = np.isfinite(depth_buffer)
    depth_norm = np.zeros_like(depth_buffer)
    if np.any(valid_depth):
        depth_norm[valid_depth] = (depth_buffer[valid_depth] - np.min(depth_buffer[valid_depth])) / \
                                  (np.max(depth_buffer[valid_depth]) - np.min(depth_buffer[valid_depth]) + 1e-8)

    # 保存图像
    plt.imsave(os.path.join(output_dir, 'cylinder_rgb.png'), rgb_buffer)
    plt.imsave(os.path.join(output_dir, 'cylinder_depth.png'), depth_norm, cmap='viridis')
    plt.imsave(os.path.join(output_dir, 'cylinder_intensity.png'), intensity_buffer, cmap='plasma')

    # 分类图生成
    generate_classification_image(class_buffer, output_dir, height, width)

    print(f"结果保存在: {os.path.abspath(output_dir)}")
    return panorama


def generate_classification_image(class_buffer, output_dir, height, width):
    """生成分类图像并保存图例"""
    # 定义类别颜色映射 (铁路场景专用)
    CLASS_COLORS = {
        1: ('#f4af80', 'Ground'),  # 浅橙色，地面
        2: ('#24ac44', 'Vegetation'),  # 绿色，植被
        3: ('#b3dfc7', 'Rail'),  # 浅绿色，轨道
        4: ('#48b8c6', 'Poles'),  # 蓝绿色，电线杆
        5: ('#dd9bda', 'Wires'),  # 浅紫色，电线
        6: ('#ffff00', 'Signaling'),  # 黄色，信号设备
        7: ('#d6822e', 'Fences'),  # 橙色，围栏
        8: ('#554ae2', 'Installation'),  # 蓝紫色，装置设备
        9: ('#ff0000', 'Building')  # 红色，建筑物
    }

    # 创建预定义颜色映射数组
    color_map = np.zeros((10, 3), dtype=np.uint8)
    for class_id, (hex_color, _) in CLASS_COLORS.items():
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        color_map[class_id] = [r, g, b]

    # 创建分类图像
    class_img = np.zeros((height, width, 3), dtype=np.uint8)
    valid_mask = (class_buffer >= 1) & (class_buffer <= 9)
    class_ids = class_buffer[valid_mask].astype(np.uint8)
    class_img[valid_mask] = color_map[class_ids]

    # 保存分类图
    plt.imsave(os.path.join(output_dir, 'cylinder_classification.png'), class_img)

    # 创建并保存图例
    plt.figure(figsize=(8, 2))
    for class_id, (_, label) in CLASS_COLORS.items():
        plt.scatter([], [], color=np.array(color_map[class_id]) / 255, label=f"{class_id}: {label}")
    plt.legend(loc='center', ncol=3, frameon=False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_legend.png'), dpi=150, bbox_inches='tight')
    plt.close()


# 测试执行
if __name__ == "__main__":
    # 测试点云路径（替换为实际路径）
    test_ply = "../data/sncf_01.ply"

    if os.path.exists(test_ply):
        print("执行柱面投影转换...")
        result = cylindrical_projection(test_ply, output_dir="output")
    else:
        print(f"警告：测试文件 {test_ply} 不存在")
        print("请将测试点云文件放在脚本同目录下运行")
