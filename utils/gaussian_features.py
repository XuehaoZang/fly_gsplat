import json
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import cKDTree

from utils.ply import unrescale

SH_C0 = 0.28209479177387814  # SH degree-0 basis constant


def _load_dataparser_transform(transform_path: Path) -> tuple:
    with open(transform_path) as f:
        dp = json.load(f)
    R_ns = np.array(dp["transform"])[:3, :3]
    t_ns = np.array(dp["transform"])[:3, 3]
    scale_ns = float(dp["scale"])
    return R_ns, t_ns, scale_ns


def _quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(N,4) 归一化四元数(w,x,y,z顺序，nerfstudio/gsplat约定) -> (N,3,3)旋转矩阵。
    每个矩阵的第i列 = 该高斯局部第i根轴(对应scale_i)在世界系下的方向。"""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    n = quat_wxyz.shape[0]
    R = np.empty((n, 3, 3))
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


def compute_gaussian_features(splat_path: Path, transform_path: Path, k: int = 10) -> pd.DataFrame:
    """
    读取 splat.ply，还原全部激活函数，构造逐点特征表。
    不做任何过滤/删点，输出行数必须等于 ply 里的高斯点数。
    """
    ply = PlyData.read(str(splat_path))
    v = ply["vertex"]
    n = len(v.data)

    R_ns, t_ns, scale_ns = _load_dataparser_transform(transform_path)

    # ---- 空间 ----
    xyz_rescaled = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    xyz = unrescale(xyz_rescaled, R_ns, t_ns, scale_ns)  # 物理坐标 (m)

    centroid = xyz.mean(axis=0)
    dist_to_centroid = np.linalg.norm(xyz - centroid, axis=1)

    cov = np.cov((xyz - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal_axis = eigvecs[:, order[0]]  # 第一主轴方向(单位向量)
    rel = xyz - centroid
    proj_len = rel @ principal_axis
    proj_vec = np.outer(proj_len, principal_axis)
    dist_to_principal_axis = np.linalg.norm(rel - proj_vec, axis=1)

    # ---- 外观 ----
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float64)
    rgb = 0.5 + SH_C0 * f_dc
    color_oob = np.any((rgb < 0) | (rgb > 1), axis=1)

    # ---- 形状 ----
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))

    scale_raw = np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float64)
    scale_phys = np.exp(scale_raw) / scale_ns  # rescaled -> 物理尺度 (m)

    scale_sorted = np.sort(scale_phys, axis=1)[:, ::-1]  # lam1 >= lam2 >= lam3
    lam1, lam2, lam3 = scale_sorted[:, 0], scale_sorted[:, 1], scale_sorted[:, 2]
    lam1_safe = np.clip(lam1, 1e-12, None)
    linearity = (lam1 - lam2) / lam1_safe
    planarity = (lam2 - lam3) / lam1_safe
    sphericity = lam3 / lam1_safe

    quat_raw = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float64)
    quat_norm = np.linalg.norm(quat_raw, axis=1, keepdims=True)
    quat = quat_raw / quat_norm
    rotmat = _quat_to_rotmat(quat)
    min_axis_idx = np.argmin(scale_phys, axis=1)  # 最小scale对应的局部轴编号
    orientation = rotmat[np.arange(n), :, min_axis_idx]  # 取该局部轴在世界系下的方向(单位向量)

    # ---- 局部结构 ----
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)  # 含自身(距离0)，去掉第0列
    knn_dists = dists[:, 1:]
    local_density = 1.0 / knn_dists.mean(axis=1)

    df = pd.DataFrame({
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "dist_to_centroid": dist_to_centroid,
        "dist_to_principal_axis": dist_to_principal_axis,
        "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
        "color_oob": color_oob,
        "opacity": opacity,
        "scale_phys_0": scale_phys[:, 0], "scale_phys_1": scale_phys[:, 1], "scale_phys_2": scale_phys[:, 2],
        "scale_ratio": lam1_safe / np.clip(lam3, 1e-12, None),
        "linearity": linearity, "planarity": planarity, "sphericity": sphericity,
        "orientation_x": orientation[:, 0], "orientation_y": orientation[:, 1], "orientation_z": orientation[:, 2],
        "local_density": local_density,
    })
    return df
