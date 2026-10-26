"""基于运动累积的 body / wing_L / wing_R 分割（单帧，需要前后若干帧作为窗口）。

1. body：在 ±half_window 帧窗口内，把各帧 if_keep 点体素化（2.5e-4 m）。身体几乎不动，
   所以被足够多帧命中的体素是 body 候选（阈值见 body_hit_thresh）；只保留其最大连通分量。
   本帧落在这些体素里的点即为 body，其余为翅候选。
2. 翅候选做 kNN 连通分量：最大两个分量为 wing_A / wing_B，其余碎块按最近邻归入 body / A / B。
   如果只有一个显著分量（两翅贴在一起），按 x_body × up 方向投影的中位数强行切成两半，confidence 记为 low。
3. 左右：两翅质心在 right_axis = x_body × up 上投影大的为 wing_R。
4. if_keep=0 的点按最近的已标点传播标签。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from utils.geometry import orient_to_reference, weighted_pca
from utils.pointcloud import knn_component_labels

UP = np.array([0.0, 0.0, 1.0])

HALF_WINDOW_AT_16K = 36
"""16000 fps 下累积窗口的半宽（帧）。其他帧率按比例换算，保持窗口覆盖的物理时长不变。"""
VOXEL_SIZE_M = 2.5e-4
BODY_HIT_FRACTION = 19 / 73
"""体素被窗口内多于该比例的帧命中时算 body 候选。原值是 16000 fps、73 帧窗口下的 19 帧；
其他帧率按窗口帧数等比换算（5000 fps：23 帧窗口 -> 6 帧）。"""
BODY_CC_K = 10
BODY_CC_PERCENTILE = 75.0
WING_CC_K = 10
WING_CC_PERCENTILE = 75.0
WING_MERGE_MIN_FRAC = 0.05
WING_MERGE_MIN_ABS = 5
"""显著翅分量：点数 >= 5 且 >= 翅候选点数的 5%。"""
MIN_BODY_POINTS = 20
BODY_FRAC_MAX = 0.9
"""body 点数 < 20 或 > 0.9 × if_keep 点数时 confidence 记为 low。"""


def half_window_for_fps(fps: float) -> int:
    return int(round(HALF_WINDOW_AT_16K * fps / 16000.0))


def body_hit_thresh(half_window: int) -> int:
    """16000 fps（half_window=36）时为 19。"""
    return int(round(BODY_HIT_FRACTION * (2 * half_window + 1)))


def voxel_keys(xyz: np.ndarray) -> np.ndarray:
    return np.floor(xyz / VOXEL_SIZE_M).astype(np.int64)


def frame_voxels(xyz_kept: np.ndarray) -> np.ndarray:
    """本帧命中的体素（去重，(M,3) int64）。"""
    return np.unique(voxel_keys(xyz_kept), axis=0)


def body_voxels(window_voxels: list[np.ndarray], hit_thresh: int) -> set[tuple[int, int, int]]:
    """窗口内各帧的体素集合 -> body 体素集合（命中帧数 > hit_thresh，取最大连通分量）。"""
    keys, counts = np.unique(np.concatenate(window_voxels, axis=0), axis=0, return_counts=True)
    cand = keys[counts > hit_thresh]
    if len(cand) == 0:
        return set()
    if len(cand) == 1:
        return {tuple(cand[0])}
    centers = (cand.astype(float) + 0.5) * VOXEL_SIZE_M
    labels = knn_component_labels(centers, k=min(BODY_CC_K, len(centers) - 1), dist_percentile=BODY_CC_PERCENTILE)
    is_main = labels == int(np.argmax(np.bincount(labels)))
    return {tuple(vk) for vk in cand[is_main]}


def _split_wings(xyz_kept: np.ndarray, is_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (每点 body/wing_A/wing_B, 翅候选各连通分量大小)。"""
    semantic = np.where(is_body, "body", "wing_candidate").astype(object)
    wing_idx = np.where(~is_body)[0]
    if len(wing_idx) < 2:
        semantic[wing_idx] = "wing_A"
        return semantic, np.array([len(wing_idx)])

    labels = knn_component_labels(xyz_kept[wing_idx], k=min(WING_CC_K, len(wing_idx) - 1),
                                  dist_percentile=WING_CC_PERCENTILE)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    is_a = labels == order[0]
    is_b = (labels == (order[1] if len(order) > 1 else order[0])) & ~is_a
    semantic[wing_idx[is_a]] = "wing_A"
    semantic[wing_idx[is_b]] = "wing_B"

    frag_idx = wing_idx[~is_a & ~is_b]
    if len(frag_idx) > 0:
        trees = {name: cKDTree(xyz_kept[semantic == name]) for name in ("body", "wing_A", "wing_B")
                 if (semantic == name).any()}
        for i in frag_idx:
            dists = {name: tree.query(xyz_kept[i])[0] for name, tree in trees.items()}
            semantic[i] = min(dists, key=dists.get)
    return semantic, sizes


def _body_axes(xyz_kept: np.ndarray, semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(right_axis = x_body × up 单位化, body 质心)；x_body 为 body 点 PCA 主轴（朝上）。"""
    body_xyz = xyz_kept[semantic == "body"]
    _, eigvecs, _ = weighted_pca(body_xyz)
    x_body = orient_to_reference(eigvecs[:, -1], UP)
    right_axis = np.cross(x_body, UP)
    return right_axis / np.linalg.norm(right_axis), body_xyz.mean(axis=0)


def label_frame(df_full: pd.DataFrame, body_vox: set[tuple[int, int, int]]) -> tuple[np.ndarray, str]:
    """一帧 `_marked` 表 + body 体素 -> (全部点的 part_label, confidence)。"""
    kept_mask = df_full["if_keep"].astype(bool).to_numpy()
    xyz_kept = df_full.loc[kept_mask, ["x", "y", "z"]].to_numpy()
    n_kept = len(xyz_kept)

    is_body = np.array([tuple(vk) in body_vox for vk in voxel_keys(xyz_kept)])
    semantic, sizes = _split_wings(xyz_kept, is_body)

    n_wing = int((~is_body).sum())
    merged = n_wing > 0 and int(np.sum((sizes >= WING_MERGE_MIN_ABS) & (sizes >= WING_MERGE_MIN_FRAC * n_wing))) <= 1
    if merged:
        right_axis, body_cm = _body_axes(xyz_kept, semantic)
        idx = np.where((semantic == "wing_A") | (semantic == "wing_B"))[0]
        proj = (xyz_kept[idx] - body_cm) @ right_axis
        semantic[idx] = np.where(proj > np.median(proj), "wing_A", "wing_B")

    n_body = int(is_body.sum())
    low = merged or n_body < MIN_BODY_POINTS or n_body > BODY_FRAC_MAX * n_kept
    confidence = "low" if low else "high"

    right_axis, body_cm = _body_axes(xyz_kept, semantic)
    proj_a = float(np.dot(xyz_kept[semantic == "wing_A"].mean(axis=0) - body_cm, right_axis))
    proj_b = float(np.dot(xyz_kept[semantic == "wing_B"].mean(axis=0) - body_cm, right_axis))
    lr = {"body": "body"}
    lr["wing_A"], lr["wing_B"] = ("wing_R", "wing_L") if proj_a > proj_b else ("wing_L", "wing_R")
    final_kept = np.array([lr[s] for s in semantic], dtype=object)

    part_label = np.empty(len(df_full), dtype=object)
    part_label[kept_mask] = final_kept
    dropped = np.where(~kept_mask)[0]
    if len(dropped) > 0:
        xyz_full = df_full[["x", "y", "z"]].to_numpy()
        _, nn = cKDTree(xyz_kept).query(xyz_full[dropped], k=1)
        part_label[dropped] = final_kept[nn]
    return part_label, confidence
