"""第 1 步 dataset (c)：图像 + transforms.json -> 可视外壳点云 init_points.ply（训练的初始化）。

    python -m dataset.make_hull --data data/ctrl_009/003 [--frames 425-905]

每帧：
1. 前景掩码：白底上灰度 < 255 的像素（果蝇是暗的），再膨胀两次（3x3 椭圆核）。
2. 中心：四台相机掩码质心的反投影射线做最小二乘三角化。
3. 在中心周围半径 2 mm 的球内均匀撒 10000 个点（固定随机种子，可复现），只保留在每台相机里都投到掩码内的点。
4. 统计去离群（20 近邻，2σ），写 init_points.ply（灰色）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage

from utils.camera import CameraConfig

SPHERE_RADIUS_M = 0.002
N_SAMPLES = 10_000
SEED = 0


def foreground_mask(im: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    return binary


def triangulate_center(cameras: list[CameraConfig], masks: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """掩码质心射线的最小二乘交点和平均残差（米）。掩码为空时用主点。"""
    A, b, rays = np.zeros((3, 3)), np.zeros(3), []
    for cam, mask in zip(cameras, masks):
        v, u = ndimage.center_of_mass(mask > 0)
        if np.isnan(u) or np.isnan(v):
            u, v = cam.cx, cam.cy
            print(f"[make_hull] cam {cam.cam_idx}: empty mask, using principal point")
        d = cam.R_w2c.T @ (np.linalg.inv(cam.K) @ np.array([u, v, 1.0]))
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ cam.X0
        rays.append((cam.X0, d))
    X = np.linalg.solve(A, b)
    res = float(np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C)) for C, d in rays]))
    return X, res


def inside_all_masks(points: np.ndarray, cameras: list[CameraConfig], masks: list[np.ndarray]) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    for cam, mask in zip(cameras, masks):
        pc = cam.R_w2c @ (points - cam.X0).T
        z = pc[2]
        valid = z > 0.001
        zs = np.where(valid, z, 1)
        u = np.where(valid, cam.fx * pc[0] / zs + cam.cx, -1)
        v = np.where(valid, cam.fy * pc[1] / zs + cam.cy, -1)
        h, w = mask.shape
        in_fov = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        hits = np.zeros(len(points), dtype=bool)
        idx = np.where(in_fov)[0]
        hits[idx] = mask[v[idx].astype(int), u[idx].astype(int)] > 0
        inside &= hits
    return inside


def make_hull(frame_dir: Path) -> dict:
    with open(frame_dir / "transforms.json") as f:
        frames = json.load(f)["frames"]
    cameras = [CameraConfig.from_opengl(fr, cam_idx=i + 1) for i, fr in enumerate(frames)]
    raw = [foreground_mask(cv2.imread(str(frame_dir / fr["file_path"]))) for fr in frames]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = [cv2.dilate(m, kernel, iterations=2) for m in raw]

    center, res = triangulate_center(cameras, raw)
    rng = np.random.default_rng(SEED)
    dirs = rng.standard_normal((N_SAMPLES, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = center + dirs * (rng.uniform(0, 1, N_SAMPLES) ** (1 / 3) * SPHERE_RADIUS_M)[:, None]
    kept = points[inside_all_masks(points, cameras, dilated)]
    if len(kept) == 0:
        raise RuntimeError(f"{frame_dir}: no sample inside all masks")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kept)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.6, 0.6, 0.6], (len(pcd.points), 1)))
    o3d.io.write_point_cloud(str(frame_dir / "init_points.ply"), pcd)
    return {"center": center, "residual_mm": 1e3 * res, "n_inside": len(kept), "n_points": len(pcd.points)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--frames", help="a-b，含 a 不含 b；默认 <data> 下所有含 transforms.json 的帧")
    args = ap.parse_args()

    dirs = sorted(d for d in args.data.glob("f*") if re.fullmatch(r"f\d+", d.name) and (d / "transforms.json").exists())
    if args.frames:
        a, b = (int(x) for x in args.frames.split("-"))
        dirs = [d for d in dirs if a <= int(d.name[1:]) < b]
    if not dirs:
        raise SystemExit(f"no f####/transforms.json under {args.data}")
    n = []
    for d in dirs:
        r = make_hull(d)
        n.append(r["n_points"])
    print(f"[make_hull] {len(dirs)} frames, init points per frame median {int(np.median(n))} "
          f"(min {min(n)}, max {max(n)})")


if __name__ == "__main__":
    main()
