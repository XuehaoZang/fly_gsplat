import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/computer0/fly_project/fly_gsplat")

from utils.camera import CameraConfig
from utils.calib import proj

def verify_reprojection(data_dir: str, ply_path: str, save_path: Path) -> None:
    """
    验证 3D 点云 (Hull) 到各个相机视角的 2D 重投影是否完美对齐。
    """
    base_dir = Path(data_dir)
    json_path = base_dir / "transforms.json"
    
    # 1. 加载 3D 点云 (Visual Hull)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points_3d = np.asarray(pcd.points)
    print(f"Loaded {len(points_3d)} points from {ply_path}")

    # 2. 读取相机内外参
    with open(json_path, 'r') as f:
        transforms = json.load(f)
    frames = transforms["frames"]
    
    # 3. 准备绘图画布 (2x2 排版适配 4 台相机)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, frame in enumerate(frames):
        if idx >= 4:  # 防御性限制，仅展示前 4 个视角
            break
            
        ax = axes[idx]
        cam = CameraConfig.from_opengl(frame)
        
        # 读取原始图像 (或直接读 Mask 图像，取决于你想用什么作为背景)
        img_path = base_dir / frame["file_path"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"[Warning] Image not found: {img_path}")
            continue

        # 4. 遍历点云并应用相机投影模型
        us, vs = [], []
        for X in points_3d:
            u, v, d = proj(cam.K, cam.R_w2c, cam.X0, X)
            if d > 0:  # 仅保留在相机前方的有效深度点
                us.append(u)
                vs.append(v)
                
        # 5. 可视化：Overlay 叠加
        # 底图：使用灰度图或 Mask
        ax.imshow(img, cmap='gray')
        
        # 叠加层：高透明度的青色散点，方便观察是否超出真实果蝇的边界
        ax.scatter(us, vs, s=0.5, c='cyan', alpha=0.15, label='Reprojected Hull')
        
        # UI 装饰
        ax.set_title(f"CAM {idx+1}: {img_path.name}")
        ax.axis('off')
        
        # 如果需要放大到 Mask 附近观察，可以取消下面两行的注释 (自动动态裁剪视窗)
        # ax.set_xlim(min(us) - 20, max(us) + 20)
        # ax.set_ylim(max(vs) + 20, min(vs) - 20) # 注意图像坐标系 y 轴向下

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Reprojection verification plot saved to -> {save_path}")
    
    plt.show()

if __name__ == "__main__":
    REPO = Path("/home/computer0/fly_project/fly_gsplat")
    DATA_DIR = REPO / "data" / "ctrl_009_002"
    PLY_PATH = REPO / "data" / "ctrl_009_002" / "init_points.ply"
    
    verify_reprojection(
        data_dir=str(DATA_DIR),
        ply_path=str(PLY_PATH),
        save_path= DATA_DIR / "debug" / "reprojection_check.png"
    )