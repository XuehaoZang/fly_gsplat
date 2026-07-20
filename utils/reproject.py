import json
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.calib import proj
from utils.camera import CameraConfig


def load_dataparser_transform(json_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """读取nerfstudio dataparser_transforms.json，返回 (R_ns, t_ns, scale)。"""
    with open(json_path) as f:
        d = json.load(f)
    transform = np.array(d["transform"])
    R_ns = transform[:, :3]
    t_ns = transform[:, 3]
    scale = float(d["scale"])
    return R_ns, t_ns, scale


def project_points(xyz: np.ndarray, cam: CameraConfig) -> Tuple[np.ndarray, np.ndarray]:
    """对每个点算 proj(K, R_w2c, X0, X)，返回 (uv: (N,2), depth: (N,))。"""
    n = len(xyz)
    uv = np.zeros((n, 2))
    depth = np.zeros(n)
    for i, X in enumerate(xyz):
        u, v, d = proj(cam.K, cam.R_w2c, cam.X0, X)
        uv[i] = (u, v)
        depth[i] = d
    return uv, depth


def lookup_labels(uv: np.ndarray, depth: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """depth<=0 或投影出边界的点标 -1，否则查 label_map[round(v), round(u)]。

    label_map 是 (H,W) 的0-based数组，和 load_sparse_frame/segment_body_wing
    用的是同一套坐标系，这里不需要再做1-based转换。
    """
    H, W = label_map.shape
    labels = np.full(len(uv), -1, dtype=np.int8)
    rows = np.round(uv[:, 1]).astype(int)
    cols = np.round(uv[:, 0]).astype(int)
    valid = (depth > 0) & (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    labels[valid] = label_map[rows[valid], cols[valid]]
    return labels
