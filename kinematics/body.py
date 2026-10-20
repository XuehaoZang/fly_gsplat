"""单帧体坐标系与身体角度 yaw / pitch / roll。

x_body：体长轴（头向），通常由 body_axis.sequence_x_body 给出；缺失时退回本帧 PCA 主轴并令其朝上。
y_body：右翅根 -> 左翅根，投影到垂直 x_body 的平面。
z_body = x_body × y_body。
n_sp：拍动平面法向，x_body 绕 y_body 转 -45°。
翅根位置用"远端点 + 最大连通分量"求质心（far+CC）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import geometry as geo

UP = np.array([0.0, 0.0, 1.0])
"""实验室坐标系的向上方向。"""
STROKE_PLANE_PITCH_DEG = 45.0

FAR_FRACTION = 0.15
"""离质心最远的这部分点构成候选池。"""
MIN_CANDIDATE_POOL = 6
DELTA_FRACTION = 0.15
"""候选池沿引导轴两端各取这部分跨度作为端点区。"""
CONNECT_RADIUS_FACTOR = 2.5
"""连通半径 = 该倍数 × 最近邻距离中位数。"""
MIN_CC_POINTS = 3


@dataclass
class BodyFrame:
    x_body: np.ndarray
    y_body: np.ndarray
    z_body: np.ndarray
    n_sp: np.ndarray
    yaw: float
    pitch: float
    roll: float
    body_cm: np.ndarray


def _weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (points * weights[:, None]).sum(axis=0) / weights.sum()


def _largest_connected_component(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最大连通分量；不足 MIN_CC_POINTS 时原样返回。"""
    n = points.shape[0]
    if n < 2:
        return points, weights
    tree = cKDTree(points)
    nn_dist = tree.query(points, k=2)[0][:, 1]
    radius = CONNECT_RADIUS_FACTOR * max(float(np.median(nn_dist)), geo._EPS)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        adjacency = coo_matrix((n, n))
    else:
        adjacency = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)
    mask = labels == int(np.argmax(np.bincount(labels)))
    if int(mask.sum()) < MIN_CC_POINTS:
        return points, weights
    return points[mask], weights[mask]


def end_centroids(
    xyz: np.ndarray, center: np.ndarray, guide_axis: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """点云沿 guide_axis 正端、负端的端点质心（far+CC）。"""
    n = xyz.shape[0]
    dist = np.linalg.norm(xyz - center, axis=1)
    n_pool = min(max(MIN_CANDIDATE_POOL, int(np.ceil(n * FAR_FRACTION))), n)
    pool_idx = np.argsort(dist)[::-1][:n_pool]
    pool_xyz = xyz[pool_idx]
    pool_w = weights[pool_idx]

    proj = (pool_xyz - center) @ guide_axis
    delta = float(proj.max() - proj.min()) * DELTA_FRACTION
    pos_mask = proj >= proj.max() - delta
    neg_mask = proj <= proj.min() + delta

    pos_xyz, pos_w = _largest_connected_component(pool_xyz[pos_mask], pool_w[pos_mask])
    neg_xyz, neg_w = _largest_connected_component(pool_xyz[neg_mask], pool_w[neg_mask])
    return _weighted_centroid(pos_xyz, pos_w), _weighted_centroid(neg_xyz, neg_w)


def wing_hinge(wing_xyz: np.ndarray, body_cm: np.ndarray) -> np.ndarray:
    """翅根：翅点云朝向身体质心一端的端点质心。"""
    w = np.ones(wing_xyz.shape[0])
    wing_cm = _weighted_centroid(wing_xyz, w)
    guide_axis = geo.unit(np.asarray(body_cm, dtype=float) - wing_cm)
    hinge_cm, _tip_cm = end_centroids(wing_xyz, wing_cm, guide_axis, w)
    return hinge_cm


def _calculate_roll(yaw_rad: float, pitch_rad: float, y_body: np.ndarray) -> float:
    """y_body 相对零滚转系 (e_y, e_z) 的有符号角（假定 up = +z）。"""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    e_y = np.array([-sy, cy, 0.0])
    e_z = np.array([-sp * cy, -sp * sy, cp])
    return math.atan2(float(np.dot(y_body, e_z)), float(np.dot(y_body, e_y)))


def estimate_body_frame(
    body_xyz: np.ndarray,
    wingL_xyz: np.ndarray,
    wingR_xyz: np.ndarray,
    x_body: np.ndarray | None = None,
) -> BodyFrame:
    body_cm = body_xyz.mean(axis=0)

    if x_body is not None:
        x_body = geo.unit(np.asarray(x_body, dtype=float))
    else:
        _, eigvecs, _ = geo.weighted_pca(body_xyz)
        x_body = geo.orient_to_reference(eigvecs[:, -1], UP)

    hinge_L = wing_hinge(wingL_xyz, body_cm)
    hinge_R = wing_hinge(wingR_xyz, body_cm)
    y_body = geo.project_onto_plane(hinge_L - hinge_R, x_body)

    z_body = np.cross(x_body, y_body)
    z_body = z_body / np.linalg.norm(z_body)

    n_sp = geo.rodrigues_rotate(x_body, y_body, math.radians(-STROKE_PLANE_PITCH_DEG))

    yaw = math.degrees(math.atan2(x_body[1], x_body[0]))
    pitch = 90.0 - math.degrees(math.acos(np.clip(np.dot(x_body, UP), -1.0, 1.0)))
    roll = math.degrees(_calculate_roll(math.radians(yaw), math.radians(pitch), y_body))

    return BodyFrame(
        x_body=x_body, y_body=y_body, z_body=z_body, n_sp=n_sp,
        yaw=yaw, pitch=pitch, roll=roll, body_cm=body_cm,
    )
