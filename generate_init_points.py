import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import ndimage
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from utils.image import binarize_mask, dilate_mask
from utils.viz import compute_target_center, plot_camera_coordinates

def generate_init_points(data_dir: str) -> Optional[List[Dict[str, Any]]]:
    """
    Generate Visual Hull point clouds for 3DGS initialization based on cropped images.
    """
    base_dir = Path(data_dir)
    json_path = base_dir / "transforms.json"
    
    # Create debug directory for mask verification
    debug_dir = base_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    # 1. Load calibration and frame metadata
    if not json_path.exists():
        print(f"[Error] Metadata not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        transforms_data = json.load(f)

    frames = transforms_data.get("frames", [])
    if not frames:
        print("[Error] No frame data found in transforms.json")
        return None
    
    print(f"[Info] Loaded {len(frames)} frames from {json_path}")

    ############### 2. Mask extraction and dilation ###############
    cameras_info = []

    for idx, frame in enumerate(frames):
        img_path = base_dir / frame["file_path"]
        
        if not img_path.exists():
            print(f"[Warning] Skipping Cam {idx+1}: Image not found at {img_path}")
            continue

        # Load image and process masks
        im = cv2.imread(str(img_path))
        if im is None:
            continue

        # Generate binary and dilated masks
        binary_mask = binarize_mask(im, threshold=1)
        dilated_mask = dilate_mask(binary_mask, kernel_size=3, iterations=2)

        # Save debug images for verification
        debug_mask_path = debug_dir / f"mask_cam_{idx+1:02d}.png"
        cv2.imwrite(str(debug_mask_path), dilated_mask)

        # Store camera parameters and processed mask
        cam_data = {
            "cam_idx": idx + 1,
            "fl_x": float(frame["fl_x"]),
            "fl_y": float(frame["fl_y"]),
            "cx": float(frame["cx"]),
            "cy": float(frame["cy"]),
            "transform_matrix": np.array(frame["transform_matrix"]),
            "mask": dilated_mask
        }

        v_center, u_center = ndimage.center_of_mass(binary_mask)
        
        if np.isnan(u_center) or np.isnan(v_center):
            print(f"[警告] Cam {idx+1} 的 Mask 为空！使用默认光心作为 fallback。")
            cam_data["target_uv"] = [cam_data["cx"], cam_data["cy"]]
        else:
            cam_data["target_uv"] = [u_center, v_center]
        
        cameras_info.append(cam_data)
        # Log progress with stats
        active_pixels = np.count_nonzero(dilated_mask)
        print(f"[Log] Cam {idx+1:02d}: Mask saved, active pixels: {active_pixels}")

    print(f"--- Preparation complete: {len(cameras_info)} cameras ready for 3D sampling ---")

    # DEBUG: camera config
    plot_camera_coordinates(cameras_info)
    exit()

    
    # ---------------------------------------------------------
    # 第三步：定义 3D 微小搜索空间与“撒点”
    # ---------------------------------------------------------
    box_center = compute_target_center(cameras_info, crop_size=160)
    box_half_size = 0.006 # 10 mm
    
    num_random_points = 1_000_000
    print(f"\n--- Generating Initial 3D Points ---")
    print(f"Auto-calculated Fly 3D Center: {box_center}")
    print(f"Sampling {num_random_points} points in a 3cm cube around {box_center}...")

    # 在边界内均匀生成随机点 (N, 3)
    points_3d = np.random.uniform(
        low=box_center - box_half_size, 
        high=box_center + box_half_size, 
        size=(num_random_points, 3)
    )

    # ---------------------------------------------------------
    # 第四步：空间投影与多重视角剔除 (基于 Threshold 的软投票机制)
    # ---------------------------------------------------------
    # DEBUG Viser 3D: 可视化初始点云与相机阵列
    plot_camera_coordinates(cameras_info, points_3d)
    exit(0)
    # ==========================================
    print(f"\n--- Starting Frustum Intersection ---")
    
    # 1. 转换为齐次坐标 (N, 4) 以便进行矩阵运算
    points_h = np.hstack((points_3d, np.ones((num_random_points, 1))))
    
    # 2. 初始化一个计票器数组，记录每个点获得的合法 Mask 票数
    mask_votes = np.zeros(num_random_points, dtype=int)

    for cam in cameras_info:
        cam_idx = cam["cam_idx"]
        c2w_gl = cam["transform_matrix"]
        
        # 将 OpenGL C2W 转为 OpenCV W2C 以便进行投影计算
        flip_mat = np.diag([1, -1, -1, 1])
        c2w_cv = c2w_gl @ flip_mat
        w2c_cv = np.linalg.inv(c2w_cv)
        
        # 批量将所有 50 万个 3D 点变换到当前相机的局部坐标系
        points_c = (points_h @ w2c_cv.T) 
        x_c, y_c, z_c = points_c[:, 0], points_c[:, 1], points_c[:, 2]
        
        # 检查是否在相机正前方
        valid_depth = z_c > 0.001
        
        # 根据针孔模型投影到 2D 像素平面
        u = (cam["fl_x"] * x_c) / (z_c + 1e-8) + cam["cx"]
        v = (cam["fl_y"] * y_c) / (z_c + 1e-8) + cam["cy"]
        
        # 检查是否落在 160x160 的画面范围内
        img_w, img_h = cam["mask"].shape[1], cam["mask"].shape[0]
        valid_uv = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        
        # 提取既在正前方，又在画面内的有效点索引 (避免查表时索引越界)
        safe_idx = np.where(valid_depth & valid_uv)[0]
        
        hits_count = 0
        if len(safe_idx) > 0:
            u_int = np.floor(u[safe_idx]).astype(int)
            v_int = np.floor(v[safe_idx]).astype(int)
            
            # 查表看这些点是否落在白色的果蝇 Mask 内
            hits_mask = cam["mask"][v_int, u_int] > 0
            
            # 找出真正命中 Mask 的点的全局索引
            valid_mask_idx = safe_idx[hits_mask]
            
            # 给这些幸运的点投上神圣的一票！
            mask_votes[valid_mask_idx] += 1
            hits_count = len(valid_mask_idx)

        # [DEBUG] 打印单相机的初步命中情况
        print(f"  Cam {cam_idx} processed. Points hitting this mask: {hits_count}")

    # ==========================================
    # 最终裁决：根据 Threshold 筛选点云
    # ==========================================
    # 这里的 threshold 是全局可调参数
    VOTE_THRESHOLD = 2
    
    # 筛选出得票数大于等于 threshold 的点
    surviving_mask = mask_votes >= VOTE_THRESHOLD
    final_points = points_3d[surviving_mask]
    
    # [DEBUG] 极其详细的计票分布统计
    print(f"\n--- Intersection Complete ---")
    print(f"Points with 4 votes (Perfect overlap) : {np.sum(mask_votes == 4)}")
    print(f"Points with 3 votes (High confidence) : {np.sum(mask_votes == 3)}")
    print(f"Points with 2 votes (Good confidence) : {np.sum(mask_votes == 2)}")
    print(f"Points with 1 votes (Low confidence)  : {np.sum(mask_votes == 1)}")
    print(f"Points with 0 votes (Empty space)     : {np.sum(mask_votes == 0)}")
    print(f"-> Final Visual Hull size (>= {VOTE_THRESHOLD} votes): {final_points.shape[0]} points.")
    
    # ==========================================
    # [DEBUG 2] 启动 Viser：终极 4 色光束可视化
    # ==========================================
    import viser
    import time

    print("\n--- [DEBUG] 启动 Viser 光束交汇可视化 ---")
    server = viser.ViserServer(port=8080)
    print("Viser 服务器已启动！请在浏览器打开: http://localhost:8080")
    
    # 颜色字典 (RGB 格式 0~1)
    colors_map = {
        1: [1.0, 0.0, 0.0], # Cam 1: 纯红
        2: [0.0, 1.0, 0.0], # Cam 2: 纯绿
        3: [0.0, 0.5, 1.0], # Cam 3: 亮蓝
        4: [1.0, 1.0, 0.0]  # Cam 4: 纯黄
    }
    
    # 画一个世界坐标系，方便你定位中心
    server.scene.add_frame("/World", axes_length=0.005, axes_radius=0.0001)

    has_beams = False
    for cam in cameras_info:
        cam_idx = cam["cam_idx"]
        c2w_gl = cam["transform_matrix"]
        
        # 重新投影当前相机的点
        flip_mat = np.diag([1, -1, -1, 1])
        w2c_cv = np.linalg.inv(c2w_gl @ flip_mat)
        points_c = (points_h @ w2c_cv.T)
        z_c = points_c[:, 2]
        
        u = (cam["fl_x"] * points_c[:, 0]) / (z_c + 1e-8) + cam["cx"]
        v = (cam["fl_y"] * points_c[:, 1]) / (z_c + 1e-8) + cam["cy"]
        
        img_w, img_h = cam["mask"].shape[1], cam["mask"].shape[0]
        valid_uv = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z_c > 0.001)
        
        safe_idx = np.where(valid_uv)[0]
        if len(safe_idx) > 0:
            u_int = np.floor(u[safe_idx]).astype(int)
            v_int = np.floor(v[safe_idx]).astype(int)
            
            # 注意这里加上了 > 0 防止 NumPy 陷阱
            hits = cam["mask"][v_int, u_int] > 0 
            beam_points = points_3d[safe_idx[hits]]
            
            if len(beam_points) > 0:
                has_beams = True
                color_array = np.tile(colors_map[cam_idx], (len(beam_points), 1))
                
                # 在 Viser 中分层添加点云，这样你可以在网页左侧面板单独开关某个相机的光束！
                server.scene.add_point_cloud(
                    name=f"/Beams/Cam_{cam_idx}",
                    points=beam_points,
                    colors=color_array,
                    point_size=0.0001  # 如果点太小看不清，可以改为 0.0002
                )
                print(f"  -> 已加载 Cam {cam_idx} 光束: {len(beam_points)} 个点")

    if has_beams:
        # 阻塞进程，等待你在网页上查看完毕
        paused = True
        btn = server.gui.add_button("继续执行代码", color="green")
        
        @btn.on_click
        def _(_):
            nonlocal paused
            paused = False
            server.gui.add_markdown("**已放行！** 请返回终端。")

        print("程序已暂停，等待您在网页端点击 '继续' 按钮...")
        try:
            while paused:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[中断] 用户按下了 Ctrl+C")
    else:
        print("没有找到任何有效的光束点！")
        server.stop()

    # ---------------------------------------------------------
    # 第五步：统一着色与保存为 PLY
    # ---------------------------------------------------------
    print(f"\nSaving PLY file...")
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    
    # 统一赋予中性灰颜色 [0.6, 0.6, 0.6] (Open3D 颜色范围是 0~1)
    colors = np.tile(np.array([0.6, 0.6, 0.6]), (final_points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存到目录
    out_ply_path = base_dir / "init_points.ply"
    o3d.io.write_point_cloud(str(out_ply_path), pcd)
    print(f"Successfully saved initial point cloud to: {out_ply_path}")


if __name__ == "__main__":
    test_data_dir = r"./data"

    print(f"--- Starting Visual Hull Initialization Test ---")
    
    # 2. Run the processing pipeline
    generate_init_points(test_data_dir)
