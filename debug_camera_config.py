import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import ndimage
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from utils import cam_colors, binarize_mask, dilate_mask, compute_target_center, plot_camera_coordinates, generate_mask_frustum


def main():
    data_dir = r"./data/removed_002_009"
    base_dir = Path(data_dir)
    json_path = base_dir / "transforms.json"
    
    debug_dir = base_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    with open(json_path, 'r') as f:
        transforms_data = json.load(f)
    frames = transforms_data.get("frames", [])

    import scipy.io as sio
    _mat = sio.loadmat(str(base_dir / "calibration_easyWandData.mat"),
                    struct_as_record=False, squeeze_me=True)
    _coefs = _mat['easyWandData'].coefs        # shape (11, nCams)
    
    print(f"[Info] Loaded {len(frames)} frames from {json_path}")

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

        h, w = im.shape[:2]

        v_center, u_center = ndimage.center_of_mass(binary_mask)

        if np.isnan(u_center) or np.isnan(v_center):
            print(f"[警告] Cam {idx+1} 的 Mask 为空！使用默认光心作为 fallback。")
            cam_data["target_uv"] = [cam_data["cx"], cam_data["cy"]]
        else:

            cam_data["target_uv"] = [u_center, v_center]
            
            # ---------------------------------------------------------
            debug_vis = im.copy()
            center_pt = (int(u_center), int(v_center))
            
            # 使用 OpenCV 画一个红色的十字准星 (BGR 格式，红色是 0,0,255)
            cv2.drawMarker(debug_vis, center_pt, color=(0, 0, 255), 
                           markerType=cv2.MARKER_CROSS, markerSize=40, thickness=2)
            
            # 再画一个蓝色的圆圈代表光心 cx, cy
            optical_center = (int(cam_data["cx"]), int(cam_data["cy"]))
            cv2.circle(debug_vis, optical_center, radius=10, color=(255, 0, 0), thickness=-1)
            print(f"cam:{idx + 1}, center:{optical_center}")
            
            # 保存这张带有 Debug 信息的图
            debug_vis_path = debug_dir / f"centroid_vis_cam_{idx+1:02d}.png"
            cv2.imwrite(str(debug_vis_path), debug_vis)
            # ---------------------------------------------------------

        cameras_info.append(cam_data)
        # Log progress with stats
        active_pixels = np.count_nonzero(dilated_mask)
        print(f"[Log] Cam {idx+1:02d}: Mask saved, active pixels: {active_pixels}\n")

    print(f"--- Preparation complete: {len(cameras_info)} cameras ready ---")

    # DEBUG: camera config
    # plot_camera_coordinates(cameras_info)

    # 1. view frustum

    target_center = compute_target_center(cameras_info)
    all_clouds = []
    
    sphere_r = 0.0010   # 球半径(米)，先按 ~半个果蝇调
    for cam in cameras_info:
        c2w = cam["transform_matrix"]
        Rwc = c2w[:3, :3].T
        C   = c2w[:3, 3]
        Xc  = Rwc @ (target_center - C)              # OpenGL 相机系
        u   = cam["cx"] + cam["fl_x"] * (Xc[0] / -Xc[2])
        # u   = (im.shape[1] - 1) - (cam["cx"] + cam["fl_x"] * (Xc[0] / -Xc[2]))
        v   = cam["cy"] - cam["fl_y"] * (Xc[1] / -Xc[2])
        r_px = cam["fl_x"] * sphere_r / (-Xc[2])
        vis = cv2.cvtColor(cam["mask"], cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(u), int(v)), max(int(r_px), 2), (0, 0, 255), 2)
        L = np.append(_coefs[:, cam["cam_idx"]-1], 1.0)
        X, Y, Z = target_center
        den = L[8]*X + L[9]*Y + L[10]*Z + 1.0
        u_dlt = (L[0]*X + L[1]*Y + L[2]*Z + L[3]) / den - 1   # -1: MATLAB 1-index
        v_dlt = (L[4]*X + L[5]*Y + L[6]*Z + L[7]) / den - 1
        cv2.circle(vis, (int(u_dlt), int(v_dlt)), 8, (0, 255, 0), 2)
        print(f"  Cam {cam['cam_idx']} DLT=({u_dlt:.0f},{v_dlt:.0f}) vs mine=({u:.0f},{v:.0f})")
        cv2.imwrite(str(debug_dir / f"sphere_overlay_cam_{cam['cam_idx']:02d}.png"), vis)
        print(f"  Cam {cam['cam_idx']} depth={-Xc[2]:.3f} r_px={r_px:.0f}")

    # for cam in cameras_info:
    #     pts, cols = generate_mask_frustum(
    #         cam, 
    #         target_center=target_center, 
    #         color=cam_colors.get(cam["cam_idx"], [255, 255, 255]),
    #         depth_steps=800,   # 光束切片数量，越多越密
    #         pixel_step=2     # 像素降采样，越小越密
    #     )
    #     all_clouds.append((pts, cols))
        # print(f"  Cam {cam['cam_idx']} beam generated: {len(pts)} points")

        # # ----------------------DEBUG-----------------------
        # # [Debug Section - 需要在获取修正后的 R_c2w 和 X0_2 之后执行]
        # print(f"\n--- Debug: Camera {cam['cam_idx']} Geometry Analysis ---")
        # # 假设 R_c2w 和 X0_2 是你修正后得到的 OpenGL 格式的相机位姿
        # R_c2w = cam["transform_matrix"][0:3,0:3]
        # X0 = cam["transform_matrix"][0:3,3]
        # # 1. 验证旋转矩阵的旋转对称性：R * R.T 是否接近单位阵 I (误差小于 ~1e-6)
        # det_R = np.linalg.det(R_c2w)
        # orthogonality = np.linalg.norm(R_c2w @ R_c2w.T - np.eye(3))
        # print(f"Determinant of R_c2w (Should be +1.0): {det_R:.6f}")
        # print(f"Orthogonality norm (Should be < ~1e-6): {orthogonality:.6f}")

        # # 2. 验证相机位置 (Translation) 是否在物理上合理
        # # 检查光心是否过于接近或原离目标，数值是否在传感器物理范围内
        # print(f"Camera Center (X0, World frame): {X0.flatten()}")

        # # 3. 计算相机观察方向与理论目标的夹角 (核心指标!)
        # # 根据你使用的坐标系，提取相机主轴向量。
        # # 如果是 OpenGL (Nerfstudio默认)，主轴是相机的负Z轴：[-c2w[0,2], -c2w[1,2], -c2w[2,2]]
        # # 如果是 OpenCV (标定常用)，主轴是相机的正Z轴：[c2w[0,2], c2w[1,2], c2w[2,2]]
        # # 假设此时你已经转换到了 OpenGL
        # forward_vec = -R_c2w[:3, 2]  # OpenGL Forward is -Z (第三列)
        # forward_vec = forward_vec / np.linalg.norm(forward_vec)

        # # 理论目标到相机的向量 (目标 - 相机位置)
        # target_center = compute_target_center(cameras_info)
        # vector_to_target = target_center.flatten() - X0.flatten()
        # dist_to_target = np.linalg.norm(vector_to_target)
        # target_dir = vector_to_target / dist_to_target

        # # 计算理论方向与实际视轴的夹角 (余弦定理)
        # cos_angle = np.dot(forward_vec, target_dir)
        # angle_degrees = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # print(f"Distance to target: {dist_to_target:.3f}")
        # print(f"Camera View Axis Vector (OpenGL Forward): {forward_vec}")
        # print(f"Target Direction Vector: {target_dir}")
        # print(f"Angle offset from Target (Should be very small, e.g., < 3°): {angle_degrees:.3f}°")

        # # [检查逻辑总结]
        # # 如果 orthogonality norm > ~1e-6：RQ分解或符号修正数学上出错了。
        # # 如果 Angle offset < 1° 且 cy 合理：则单个相机修正没问题，不交叠可能是多相机间全局尺度问题。
        # # 如果 Angle offset > 3°：相机旋转R或平移T有问题。
        # #   - 如果此时cy修正正确（不漂移），通常是旋转R坐标轴对齐错了，比如误用了正Z轴而非负Z轴。
        # #   - 如果所有相机的 Angle offset 普遍偏大：标定本身的质量问题。

        # -------------------------------------------------------------------
        # exit()
    # 启动 Viser 进行终极检验
    plot_camera_coordinates(cameras_info, point_clouds=all_clouds)
if __name__ == "__main__":
    main()