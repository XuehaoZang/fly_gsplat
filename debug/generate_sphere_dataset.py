import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys

# 确保能找到 utils 模块
sys.path.insert(0, "/home/computer0/fly_project/fly_gsplat")
from utils.camera import CameraConfig
from utils.calib import proj

def generate_synthetic_dataset(
    src_dir: str,
    dst_dir: str,
    radius: float = 0.001,
    n_points: int = 1000,
    color_mode: str = "GRAY",        # "GRAY", "RGB", "RGBA"
    bg_color: tuple = (0, 0, 0),     # 背景颜色 (B, G, R) 或单通道灰度
    fg_color: tuple = (255, 255, 255), # 球体颜色 (B, G, R) 或单通道灰度
    transparent_bg: bool = False,    # 是否透明背景
    generate_mask: bool = False      # 是否生成 mask 文件
) -> None:
    
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    img_out_dir = dst_path / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    
    debug_dir = dst_path / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    if generate_mask:
        mask_out_dir = dst_path / "masks"
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    with open(src_path / "transforms.json", 'r') as f:
        src_transforms = json.load(f)

    # --- 1. 提取质心并生成球体 3D 点云 ---
    hull_ply_path = src_path / "init_points.ply"
    pcd_hull = o3d.io.read_point_cloud(str(hull_ply_path))
    centroid = np.mean(np.asarray(pcd_hull.points), axis=0)

    i = np.arange(n_points)
    phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * i
    unit_sphere = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=1)
    sphere_points = centroid + radius * unit_sphere

    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)
    sphere_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (n_points, 1)))
    ply_out_path = dst_path / "init_sphere.ply"
    o3d.io.write_point_cloud(str(ply_out_path), sphere_pcd)

    # --- 2. 初始化目标 JSON 结构 ---
    dst_transforms = {
        "ply_file_path": "init_sphere.ply",
        "camera_model": "OPENCV",
        "frames": []
    }

    # --- 3. 投影、渲染与文件生成 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, frame in enumerate(src_transforms["frames"]):
        cam = CameraConfig.from_opengl(frame)
        
        us, vs = [], []
        for X in sphere_points:
            u, v, d = proj(cam.K, cam.R_w2c, cam.X0, X)
            if d > 0:
                us.append(u)
                vs.append(v)
        
        # 确定图像通道数和基础颜色
        if color_mode == "GRAY":
            img = np.full((cam.h, cam.w), bg_color[0] if isinstance(bg_color, tuple) else bg_color, dtype=np.uint8)
            draw_color = fg_color[0] if isinstance(fg_color, tuple) else fg_color
        elif color_mode == "RGB":
            img = np.full((cam.h, cam.w, 3), bg_color[:3], dtype=np.uint8)
            draw_color = fg_color[:3]
        elif color_mode == "RGBA":
            img = np.full((cam.h, cam.w, 4), bg_color[:3] + (0 if transparent_bg else 255,), dtype=np.uint8)
            draw_color = fg_color[:3] + (255,)

        mask_img = np.zeros((cam.h, cam.w), dtype=np.uint8)

        # 绘制投影凸包
        if len(us) >= 3:
            pts2d = np.stack([us, vs], axis=1).astype(np.float32).reshape(-1, 1, 2)
            hull_2d = cv2.convexHull(pts2d)
            
            # 填充合成图像
            cv2.fillConvexPoly(img, hull_2d.astype(np.int32), draw_color)
            # 填充二值掩码 (供后续可能使用)
            cv2.fillConvexPoly(mask_img, hull_2d.astype(np.int32), 255)
        
        # 保存主图
        img_name = Path(frame["file_path"]).name
        cv2.imwrite(str(img_out_dir / img_name), img)
        
        # 组装 JSON 节点
        new_frame = {
            "file_path": f"images/{img_name}",
            "fl_x": cam.fx,
            "fl_y": cam.fy,
            "cx": cam.cx,
            "cy": cam.cy,
            "w": cam.w,
            "h": cam.h,
            "transform_matrix": frame["transform_matrix"]
        }

        # 保存掩码并在 JSON 中添加记录
        if generate_mask:
            cv2.imwrite(str(mask_out_dir / img_name), mask_img)
            new_frame["mask_path"] = f"masks/{img_name}"

        dst_transforms["frames"].append(new_frame)

        # --- 绘制 Debug Overlay 图 ---
        if idx < 4:
            ax = axes[idx]
            orig_img_path = src_path / frame["file_path"]
            orig_img = cv2.imread(str(orig_img_path), cv2.IMREAD_GRAYSCALE)
            
            if orig_img is not None:
                ax.imshow(orig_img, cmap='gray')
            else:
                # 降级方案：如果没找到原图，画个空的占位
                ax.imshow(np.zeros((cam.h, cam.w)), cmap='gray')
                
            ax.scatter(us, vs, s=1.5, c='magenta', alpha=0.6)
            ax.set_title(f"CAM {idx+1}: {img_name}")
            ax.axis('off')

    with open(dst_path / "transforms.json", 'w') as f:
        json.dump(dst_transforms, f, indent=4)
        
    plt.tight_layout()
    plt.savefig(str(debug_dir / "overlay.png"), dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"[Success] Dataset '{dst_path.name}' created with mode={color_mode}, masks={generate_mask}")


if __name__ == "__main__":
    REPO = Path("/home/computer0/fly_project/fly_gsplat")
    SRC_DIR = REPO / "data" / "ctrl_009_002"
    
    # 测试集 1: 灰度、黑底白球、无 Mask (最基础的 Baseline)
    generate_synthetic_dataset(
        src_dir=str(SRC_DIR),
        dst_dir=str(REPO / "data" / "test_01_gray_nomask"),
        color_mode="GRAY",
        bg_color=0, fg_color=255, generate_mask=False
    )

    # 测试集 2: RGBA 透明背景、白色球、无 Mask
    generate_synthetic_dataset(
        src_dir=str(SRC_DIR),
        dst_dir=str(REPO / "data" / "test_02_rgba_nomask"),
        color_mode="RGBA", transparent_bg=True,
        bg_color=(0,0,0), fg_color=(255,255,255), generate_mask=False
    )

    # 测试集 3: RGB 白底灰球、无 Mask (测试高亮背景的反向梯度影响)
    generate_synthetic_dataset(
        src_dir=str(SRC_DIR),
        dst_dir=str(REPO / "data" / "test_03_whitebg_grayfg_nomask"),
        color_mode="RGB",
        bg_color=(255,255,255), fg_color=(128,128,128), generate_mask=False
    )