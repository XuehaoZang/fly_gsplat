"""splat.ply + dataparser_transforms.json -> 逐高斯特征表（实验室坐标系，单位米）。

nerfstudio 训练时把实验室坐标变换成 x_ns = s · (R x + t)（R、t、s 来自 dataparser_transforms.json）。
这里做逆变换：
  位置      x = Rᵀ (x_ns / s − t)
  方向向量  v = Rᵀ v_ns            （只受线性部分影响，与平移无关，归一化后与 s 无关）
  尺度      scale = exp(log_scale) / s
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814
"""0 阶球谐基常数：RGB = 0.5 + SH_C0 · f_dc。"""
K_DENSITY = 10


def load_dataparser_transform(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with open(path) as f:
        dp = json.load(f)
    T = np.array(dp["transform"])
    return T[:3, :3], T[:3, 3], float(dp["scale"])


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(N,4) 单位四元数 (w,x,y,z) -> (N,3,3)；第 i 列是高斯第 i 根局部轴（对应 scale_i）的方向。"""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    R = np.empty((quat_wxyz.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def compute_features(splat_path: Path, transform_path: Path) -> pd.DataFrame:
    """每个高斯一行，不删点。

    列：x,y,z（米）；dist_to_centroid、dist_to_principal_axis（相对本帧全部高斯）；R,G,B、color_oob（超出 [0,1]）；
    opacity（sigmoid 后）；scale_phys_0..2（米）；scale_ratio = 最大/最小尺度；linearity、planarity、sphericity
    （由排序后的尺度 λ1≥λ2≥λ3 算出）；orientation_x/y/z（最短轴方向，实验室坐标系）；
    local_density = 1 / 平均 10 近邻距离。
    """
    v = PlyData.read(str(splat_path))["vertex"]
    n = len(v.data)
    R_ns, t_ns, s_ns = load_dataparser_transform(transform_path)

    xyz_ns = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    xyz = (R_ns.T @ (xyz_ns.T / s_ns - t_ns[:, None])).T

    centroid = xyz.mean(axis=0)
    rel = xyz - centroid
    dist_to_centroid = np.linalg.norm(rel, axis=1)
    eigvals, eigvecs = np.linalg.eigh(np.cov(rel.T))
    principal_axis = eigvecs[:, np.argsort(eigvals)[::-1][0]]
    dist_to_principal_axis = np.linalg.norm(rel - np.outer(rel @ principal_axis, principal_axis), axis=1)

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float64)
    rgb = 0.5 + SH_C0 * f_dc
    color_oob = np.any((rgb < 0) | (rgb > 1), axis=1)

    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))

    scale_phys = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float64)) / s_ns
    lam = np.sort(scale_phys, axis=1)[:, ::-1]
    lam1, lam2, lam3 = lam[:, 0], lam[:, 1], lam[:, 2]
    lam1_safe = np.clip(lam1, 1e-12, None)

    quat = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float64)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    axis_ns = quat_to_rotmat(quat)[np.arange(n), :, np.argmin(scale_phys, axis=1)]
    orientation = axis_ns @ R_ns  # 行向量形式的 Rᵀ v_ns：转到实验室坐标系

    knn_dists = cKDTree(xyz).query(xyz, k=K_DENSITY + 1)[0][:, 1:]
    local_density = 1.0 / knn_dists.mean(axis=1)

    return pd.DataFrame({
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "dist_to_centroid": dist_to_centroid,
        "dist_to_principal_axis": dist_to_principal_axis,
        "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
        "color_oob": color_oob,
        "opacity": opacity,
        "scale_phys_0": scale_phys[:, 0], "scale_phys_1": scale_phys[:, 1], "scale_phys_2": scale_phys[:, 2],
        "scale_ratio": lam1_safe / np.clip(lam3, 1e-12, None),
        "linearity": (lam1 - lam2) / lam1_safe,
        "planarity": (lam2 - lam3) / lam1_safe,
        "sphericity": lam3 / lam1_safe,
        "orientation_x": orientation[:, 0], "orientation_y": orientation[:, 1], "orientation_z": orientation[:, 2],
        "local_density": local_density,
    })
