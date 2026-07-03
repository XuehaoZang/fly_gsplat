"""
generate_hull.py
Visual Hull reconstruction for 3DGS initialisation.

Pipeline:
  transforms.json + images/
      -> CameraConfig list + binary masks
      -> triangulate mask centroids -> seed point
      -> sample points in 2mm sphere around seed
      -> vote: keep points visible in ALL cameras (threshold = n_cams)
      -> statistical outlier removal
      -> save hull.ply
      -> Viser: hull point cloud + camera axes
"""

import json
import numpy as np
from pathlib import Path
import cv2
import open3d as o3d

from utils.camera import CameraConfig
from utils.calib  import backproj, triangulate, mask_centroid
from utils.image  import binarize_mask, dilate_mask
from utils.viz    import (cam_colors, start_viser, add_camera_axes,
                           add_point_cloud, stop_viser)


# ----------------------------------------------------------------- sampling --
def sample_sphere(centre: np.ndarray, radius: float, n: int) -> np.ndarray:
    """Uniform random points inside a sphere (rejection-free)."""
    dirs = np.random.randn(n, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = np.random.uniform(0, 1, n) ** (1 / 3) * radius
    return centre + dirs * r[:, None]


# ---------------------------------------------------------------- hull vote --
def visual_hull_vote(points: np.ndarray,
                     cameras: list,
                     masks: list) -> np.ndarray:
    """Return boolean mask: True where point projects inside mask in every camera.

    points  : (N, 3) world-frame points
    cameras : List[CameraConfig]
    masks   : List[np.ndarray]  binary masks, one per camera
    """
    N      = len(points)
    inside = np.ones(N, dtype=bool)   # start with all passing

    for cam, mask in zip(cameras, masks):
        # OpenCV projection: pts_cam = R_w2c @ (X - X0)
        pts_cam = (cam.R_w2c @ (points - cam.X0).T)   # (3, N)
        z       = pts_cam[2]

        valid = z > 0.001
        u = np.where(valid, cam.fx * pts_cam[0] / np.where(valid, z, 1) + cam.cx, -1)
        v = np.where(valid, cam.fy * pts_cam[1] / np.where(valid, z, 1) + cam.cy, -1)

        h, w   = mask.shape
        in_fov = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)

        # mask lookup for in-fov points
        hits = np.zeros(N, dtype=bool)
        idx  = np.where(in_fov)[0]
        if len(idx):
            hits[idx] = mask[v[idx].astype(int), u[idx].astype(int)] > 0

        inside &= hits

    return inside


# -------------------------------------------------------------------- main --
def generate_hull(data_dir: str, if_viser: bool = True) -> None:
    base_dir  = Path(data_dir)
    json_path = base_dir / "transforms.json"
    ply_path  = base_dir / "init_points.ply"

    with open(json_path) as f:
        frames = json.load(f)["frames"]

    cameras, masks, centroids = [], [], []

    for idx, frame in enumerate(frames):
        img_path = base_dir / frame["file_path"]
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"[Warning] Cam {idx+1}: image not found, skipping")
            continue

        cam = CameraConfig.from_opengl(frame)
        cam.cam_idx = idx + 1

        binary  = binarize_mask(im, threshold=1, dark_bg=False)
        dilated = dilate_mask(binary, kernel_size=3, iterations=2)

        u, v = mask_centroid(binary)
        if np.isnan(u) or np.isnan(v):
            u, v = cam.cx, cam.cy
            print(f"[Warning] Cam {idx+1}: empty mask, using principal point")

        cameras.append(cam)
        masks.append(dilated)
        centroids.append((u, v))

    print(f"{len(cameras)} cameras loaded")

    # seed point from mask centroid triangulation
    rays = [(cam.X0, backproj(cam.K, cam.R_w2c, u, v))
            for cam, (u, v) in zip(cameras, centroids)]
    seed, res = triangulate(rays)
    print(f"Seed: {seed}  triangulation residual={res*1000:.3f} mm")

    # sample points in 2mm sphere
    N_SAMPLES = 10_000      # 10k points
    RADIUS    = 0.002       # metres
    points    = sample_sphere(seed, RADIUS, N_SAMPLES)
    print(f"Sampled {N_SAMPLES} points in {RADIUS*1000:.0f}mm sphere")

    # visual hull vote (threshold = all cameras)
    inside = visual_hull_vote(points, cameras, masks)
    final  = points[inside]
    print(f"Surviving points (all {len(cameras)} cameras): {len(final)}")

    if len(final) == 0:
        print("[Error] No points survived — check masks or increase sphere radius")
        return

    # statistical outlier removal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    final  = np.asarray(pcd.points)
    print(f"After outlier removal: {len(final)} points")

    # save PLY (neutral grey)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.6, 0.6, 0.6], (len(final), 1)))
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Saved -> {ply_path}")

    # Viser: hull + camera axes
    if if_viser:
        server = start_viser()
        add_camera_axes(server, cameras)
        add_point_cloud(server, final,
                        np.tile(np.uint8([50, 150, 255]), (len(final), 1)),
                        name="/hull")
        stop_viser(server)


if __name__ == "__main__":
    generate_hull("./data/ctrl_009_002",if_viser=True)