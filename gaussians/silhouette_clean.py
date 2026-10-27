"""轮廓一致性去脏点：把每个高斯中心投影到各相机的前景掩码（与 dataset.make_hull 同一掩码），不删点，只标 if_keep。

去掉的高斯（满足任一条）：
  - 白色（RGB 均值 > WHITE_RGB_MEAN），且至少一台相机里落在掩码外；
  - 至少 MIN_FAR_CAMS 台相机里离掩码超过 MARGIN_PX 像素。
只要求"多数相机"一致而不是全部：真实数据里常有一台相机看翅膀很淡（ctrl_009_003 左翅约 24% 的高斯只落在 3 台
相机的掩码里），全部相机都要求会删掉正常的翅膀。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from dataset.make_hull import foreground_mask
from utils.camera import CameraConfig

MARGIN_PX = 2.0
MIN_FAR_CAMS = 2
WHITE_RGB_MEAN = 0.8


def mask_distances(xyz: np.ndarray, frame_dir: Path) -> np.ndarray:
    """(N, n_cam)：各中心投影点到该相机前景掩码的像素距离（掩码内为 0，出视场为 inf）。"""
    with open(frame_dir / "transforms.json") as f:
        frames = json.load(f)["frames"]
    out = np.empty((xyz.shape[0], len(frames)))
    for k, fr in enumerate(frames):
        cam = CameraConfig.from_opengl(fr, cam_idx=k + 1)
        mask = foreground_mask(cv2.imread(str(frame_dir / fr["file_path"]))) > 0
        dist = ndimage.distance_transform_edt(~mask)
        pc = cam.R_w2c @ (xyz - cam.X0).T
        z = np.where(pc[2] > 1e-6, pc[2], np.nan)
        u, v = cam.fx * pc[0] / z + cam.cx, cam.fy * pc[1] / z + cam.cy
        h, w = mask.shape
        ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.full(xyz.shape[0], np.inf)
        d[ok] = dist[v[ok].astype(int), u[ok].astype(int)]
        out[:, k] = d
    return out


def mark_silhouette(df: pd.DataFrame, frame_dir: Path) -> pd.DataFrame:
    """在特征表后面加（或与已有的 if_keep 取与）if_keep 列。"""
    d = mask_distances(df[["x", "y", "z"]].to_numpy(dtype=float), frame_dir)
    white = df[["R", "G", "B"]].to_numpy(dtype=float).mean(axis=1) > WHITE_RGB_MEAN
    drop = (white & (d > 0).any(axis=1)) | ((d > MARGIN_PX).sum(axis=1) >= MIN_FAR_CAMS)
    out = df.copy()
    prev = out["if_keep"].astype(bool).to_numpy() if "if_keep" in out.columns else np.ones(len(out), bool)
    out["if_keep"] = prev & ~drop
    return out
