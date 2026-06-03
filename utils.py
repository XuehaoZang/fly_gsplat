from pathlib import Path
from typing import Dict, Any, Tuple
import time
import numpy as np
import cv2
import open3d as o3d
import viser
from scipy.spatial.transform import Rotation

cam_colors = {
        1: [255, 30, 30],
        2: [30, 255, 30],
        3: [50, 150, 255],
        4: [255, 255, 30]
    }

def generate_frame_dict(img_name: str, w: int, h: int, K: np.ndarray, R: np.ndarray, X0: np.ndarray) -> Dict[str, Any]:
    """
    Format the camera parameters into a Nerfstudio-compatible frame dictionary.
    """
    fl_x = abs(K[0,0])
    fl_y = abs(K[1,1])
    cx = K[0,2]
    cy = K[1,2]

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = X0.flatten()

    flip = np.array([
            [1,  0,  0, 0],
            [0,  -1,  0, 0],
            [0,  0,  -1, 0],
            [0,  0,  0, 1]
        ])
    # transform_matrix = flip_gravity @ M @ flip_gravity
    transform_matrix =  M 

    return {
        "file_path": f"images/{img_name}",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "transform_matrix": transform_matrix.tolist()
    }

def crop_image(im: np.ndarray, cx: float, cy: float, crop_size: int = 160) -> Tuple[np.ndarray, float, float, int, int]:
    """
    Crop image around non-zero pixels and update camera principal points.
    """
    h_orig, w_orig = im.shape[:2]

    # 1. Locate target centroid from non-zero pixels
    y_coords, x_coords = np.nonzero(im)
    
    if len(y_coords) == 0:
        center_x, center_y = w_orig // 2, h_orig // 2
    else:
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

    # 2. Calculate top-left corner of the crop window
    half_size = crop_size // 2
    x_min = center_x - half_size
    y_min = center_y - half_size

    # 3. Constraint window within image boundaries
    x_min = max(0, min(x_min, w_orig - crop_size))
    y_min = max(0, min(y_min, h_orig - crop_size))

    # 4. Execute crop
    cropped_im = im[y_min:y_min + crop_size, x_min:x_min + crop_size]

    # 5. Offset principal points to maintain 3D ray consistency
    cx_new = cx - x_min
    cy_new = cy - y_min

    return cropped_im, cx_new, cy_new, crop_size, crop_size


def gray_to_rgba(gray_path: Path, rgba_path: Path) -> bool:
    """grayscale -> RGBA PNG"""
    img = cv2.imread(str(gray_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # RGB = 255, Alpha = 255
    rgba[mask > 0, 0:3] = 255
    rgba[mask > 0, 3] = 255
    
    cv2.imwrite(str(rgba_path), rgba)
    return True

def binarize_mask(im: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Convert input image to a binary mask, isolating target areas.
    """
    # 1. Convert to grayscale if image is colored (H, W, 3)
    if len(im.shape) == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im
        
    # 2. Thresholding: pixels > threshold become 255 (white)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary

def dilate_mask(im: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:
    """
    Apply morphological dilation to expand the binary mask boundaries.
    """
    # 1. Generate elliptical kernel for smoother, natural edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 2. Execute dilation
    dilated = cv2.dilate(im, kernel, iterations=iterations)
    
    return dilated

def compute_target_center(cameras_info):
    """
    通过多视角相机的质心射线相交，计算出目标在 3D 世界中的绝对坐标。
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for cam in cameras_info:
        # 1. 拿回 OpenGL 原生外参
        c2w_gl = cam["transform_matrix"]
        R_c2w = c2w_gl[:3, :3]
        T_c2w = c2w_gl[:3, 3] # 相机光心
        
        # 2. 直接从字典中提取专属的目标质心
        u_target, v_target = cam["target_uv"]
        
        # 3. 构建原生 OpenGL 局部坐标系下的射线方向
        # OpenGL 看向 -Z，且 Y 轴朝上 (图像 V 轴朝下，所以加负号)
        dir_local = np.array([
            (u_target - cam["cx"]) / cam["fl_x"],
            -(v_target - cam["cy"]) / cam["fl_y"], 
            -1.0                                   
        ])
        
        # 4. 转换到世界坐标系并归一化
        dir_w = R_c2w @ dir_local
        dir_w = dir_w / np.linalg.norm(dir_w)
        
        # 5. 最小二乘矩阵构建
        I_minus_ddT = np.eye(3) - np.outer(dir_w, dir_w)
        A += I_minus_ddT
        b += I_minus_ddT @ T_c2w
        
    # 解 Ax = b
    target_center, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return target_center

import numpy as np

def extract_camera_params_from_P(coefs_11):
    """
    从 11 个 DLT 系数中严格按照射影几何原理解析出真实的 K 和 R, t。
    绝对不使用黑盒 RQ 分解，确保物理符号完全正确。
    """
    # 构造 3x4 投影矩阵
    P = np.append(coefs_11, 1.0).reshape(3, 4)
    H = P[:, :3]
    h = P[:, 3]

    # 1. 提取光心 X0
    X0 = -np.linalg.inv(H) @ h

    # 2. 提取并归一化 Z 轴 (R 矩阵的第三行)
    r3 = H[2, :]
    norm_r3 = np.linalg.norm(r3)
    r3 = r3 / norm_r3

    # 3. 提取极其关键的真实光心 (cx, cy)
    cx = np.dot(H[0, :], r3)
    cy = np.dot(H[1, :], r3)

    # 4. 提取并归一化焦距 (fx, fy) 和 X, Y 轴
    r1_unscaled = H[0, :] - cx * r3
    fx = np.linalg.norm(r1_unscaled)
    r1 = r1_unscaled / fx

    r2_unscaled = H[1, :] - cy * r3
    fy = np.linalg.norm(r2_unscaled)
    r2 = r2_unscaled / fy

    # 5. 组装最纯正的旋转矩阵 R
    R = np.vstack((r1, r2, r3))
    
    # 6. 安全护城河：如果物理空间被镜像了，翻转它以满足右手系
    if np.linalg.det(R) < 0:
        R = -R
        # 翻转 R 意味着我们需要重新调整 fx, fy 的符号逻辑，
        # 但通常真实的 DLT 标定矩阵的 det(R) 都会是 1。
        print("[警告] 遇到左手系旋转矩阵，已强制翻转！")

    K = np.array([
            [float(fx), 0.0,  float(cx)],
            [0.0,  float(fy), float(cy)],
            [0.0,  0.0,  1.0]
        ])
    
    return K, R, X0

def plot_camera_coordinates(cameras_info, point_clouds=None):

    """
    启动 Viser Web 服务器进行交互式 3D 查看，并暂停主进程。
    在网页点击 "Continue" 按钮后，程序才会继续向下运行。
    """
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print("🌐 Viser 3D 可视化服务器已启动！")
    print("👉 请在 Windows 浏览器中打开: http://localhost:8080")
    print("⏳ 程序已暂停，等待您在网页端点击 'Continue' 按钮...")
    print("="*50 + "\n")

    # 1. 绘制世界原点 (0,0,0)
    # axes_length 控制轴的长度，红=X, 绿=Y, 蓝=Z
    server.scene.add_frame("/World", axes_length=0.002, axes_radius=0.0002)

    if point_clouds is None:
        point_clouds = np.array([[0.0, 0.0, 0.0]])
    elif point_clouds is not None:
         for idx, (pts, cols) in enumerate(point_clouds):
            if len(pts) > 0:
                server.scene.add_point_cloud(
                    name=f"/Points_{idx+1}",
                    points=pts,
                    colors=cols,
                    point_size=0.0001 # 如果嫌太淡，可以调成 0.0002
                )       
    
    # 2. 绘制每个相机的局部坐标系
    for cam in cameras_info:
        idx = cam["cam_idx"]
        c2w_gl = np.array(cam["transform_matrix"])
        
        pos = c2w_gl[:3, 3]
        rot_mat = c2w_gl[:3, :3]
        
        # 将旋转矩阵转为四元数 (scipy 默认输出 xyzw，viser 需要 wxyz)
        quat_xyzw = Rotation.from_matrix(rot_mat).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # 添加相机坐标系节点
        server.scene.add_frame(
            f"/World/Cam_{idx}",
            position=pos,
            wxyz=quat_wxyz,
            axes_length=0.05,
            axes_radius=0.001
        )

        server.scene.add_label(
            f"/World/Cam_{idx}_label",
            text=f"Cam {idx}",
            position=pos,         # 锚定在相机的光心位置
        )
                
        
    # server.scene.add_label(
    #     f"/World/legend",
    #     text=f"🔴 X \n🟢 Y \n🔵 Z",
    # )

    # 3. 在 Web UI 增加一个控制按钮来实现“暂停/继续”
    paused = True
    continue_btn = server.gui.add_button("🚀 继续执行后续代码 (Continue)", color="green")
    
    @continue_btn.on_click
    def _(_):
        nonlocal paused
        paused = False
        server.gui.add_markdown("**继续运行！** 请返回终端查看输出。")

    # 4. 阻塞主进程，直到网页上的按钮被点击 (或在终端按 Ctrl+C)
    try:
        while paused:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[中断] 用户按下了 Ctrl+C")
    
    print("关闭 Viser 可视化，程序继续运行...")
    # 可选：如果希望继续执行时不占用端口，可以停掉 server
    # server.stop()

import viser
import numpy as np
import time

def plot_point_cloud_viser(points_3d, colors=None):
    """
    使用 Viser 在浏览器中直接可视化点云，并暂停 Python 进程
    """
    server = viser.ViserServer(port=8080)
    print("🌐 Viser 服务器已启动！请在浏览器打开: http://localhost:8080")
    
    # 如果没有提供颜色，默认给一个亮蓝色
    if colors is None:
        colors = np.tile(np.array([0.2, 0.6, 1.0]), (points_3d.shape[0], 1))

    # 在场景中添加点云
    server.scene.add_point_cloud(
        name="/fly_points",
        points=points_3d,
        colors=colors,
        point_size=0.0001, # 点的物理大小，根据你的 3mm 盒子微调
    )
    
    # 添加坐标系原点方便参考
    server.scene.add_frame("/World", axes_length=0.005, axes_radius=0.0001)
    server.scene.camera.position = (0.02, 0.02, 0.02) # 把初始视角放在 2 厘米处
    server.scene.camera.look_at = (0.0, 0.0, 0.0)

    # 阻塞进程，直到在网页端点击继续
    paused = True
    btn = server.gui.add_button("🚀 阅毕，继续跑代码", color="green")
    
    @btn.on_click
    def _(_):
        nonlocal paused
        paused = False
        server.gui.add_markdown("**已放行！** 请返回终端。")

    try:
        while paused:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    print("可视化结束，进程继续...")

def generate_mask_frustum(cam, target_center, color, depth_steps=50, pixel_step=2):
    """
    生成从光心到目标质心的实心异形光束点云。
    pixel_step: 降低像素采样率防止内存爆炸 (2 表示取 1/4 的点)
    depth_steps: 在射线上采样的分层数
    """
    c2w = cam["transform_matrix"]
    cam_pos = c2w[:3, 3]
    R = c2w[:3, :3]

    # 1. 获取 Mask 像素 (进行降采样)
    v, u = np.where(cam["mask"][::pixel_step, ::pixel_step] > 0)
    if len(u) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # 还原真实像素坐标
    u = u * pixel_step
    v = v * pixel_step

    # 2.1 OpenGL 射线方向 (Z 朝后，Y 朝上)
    dirs_local = np.column_stack([
        (u - cam["cx"]) / cam["fl_x"],
        -(v - cam["cy"]) / cam["fl_y"],
        np.full_like(u, -1.0)
    ])
    
    # 2.2 OpenCV 射线方向 (Z 朝前，Y 朝下)
    # dirs_local = np.column_stack([
    #     (u - cam["cx"]) / cam["fl_x"],
    #     (v - cam["cy"]) / cam["fl_y"],     # 取消负号，Y 朝下
    #     np.full_like(u, 1.0)               # 改为正 1.0，Z 朝前
    # ])
    
    # 归一化方向向量，确保 depth 是真实的物理距离
    dirs_local = dirs_local / np.linalg.norm(dirs_local, axis=1, keepdims=True)
    dirs_world = (R @ dirs_local.T).T  # (N, 3)

    # 3. 计算相机到果蝇中心的物理总长
    max_depth = np.linalg.norm(target_center - cam_pos)

    # 4. 沿射线矩阵化批量采样点 (避免 Python for 循环的龟速)
    # t shape: (depth_steps,) -> 拓展为 (depth_steps, N, 3)
    t = np.linspace(0.0001, max_depth*1.1, depth_steps)
    points = cam_pos[None, None, :] + t[:, None, None] * dirs_world[None, :, :]
    points = points.reshape(-1, 3) # 压平为 (N*steps, 3)

    colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    
    return points, colors

def generate_view_frustum(cam, target_center, color, depth_steps=50, pixel_step=2):
    """
    生成从光心到目标质心的实心异形光束点云。
    pixel_step: 降低像素采样率防止内存爆炸 (2 表示取 1/4 的点)
    depth_steps: 在射线上采样的分层数
    """
    c2w = cam["transform_matrix"]
    cam_pos = c2w[:3, 3]
    R = c2w[:3, :3]

    # --- 1. 强制生成右上角 (Top-Right) 的像素网格 ---
    u_range = np.arange(0, 1280, pixel_step)
    v_range = np.arange(0, 800, pixel_step)
    
    # 生成网格并压平
    U, V = np.meshgrid(u_range, v_range)
    u = U.flatten()
    v = V.flatten()

    if len(u) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # 2.2 OpenCV 射线方向 (Z 朝前，Y 朝下)
    dirs_local = np.column_stack([
        (u - cam["cx"]) / cam["fl_x"],
        (v - cam["cy"]) / cam["fl_y"],     # 取消负号，Y 朝下
        np.full_like(u, 1.0)               # 改为正 1.0，Z 朝前
    ])
    dirs_local = dirs_local / np.linalg.norm(dirs_local, axis=1, keepdims=True)
    dirs_world = (R @ dirs_local.T).T  # (N, 3)

    # 3. 计算相机到果蝇中心的物理总长
    max_depth = np.linalg.norm(target_center - cam_pos)

    # 4. 沿射线矩阵化批量采样点 (避免 Python for 循环的龟速)
    # t shape: (depth_steps,) -> 拓展为 (depth_steps, N, 3)
    t = np.linspace(0.0001, max_depth*1.1, depth_steps)
    points = cam_pos[None, None, :] + t[:, None, None] * dirs_world[None, :, :]
    points = points.reshape(-1, 3) # 压平为 (N*steps, 3)

    colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    
    return points, colors