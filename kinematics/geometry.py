"""几何原语：归一化、加权 PCA、RANSAC 平面、投影、旋转。所有长度单位为米。"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < _EPS:
        raise ValueError(f"unit(): cannot normalize a zero-length vector (norm={n:.3e})")
    return v / n


def normalize_rows(V: np.ndarray) -> np.ndarray:
    """逐行归一化；零向量行置 NaN。"""
    V = np.asarray(V, dtype=float)
    norms = np.linalg.norm(V, axis=-1, keepdims=True)
    zero = (norms < _EPS).squeeze(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = V / norms
    out[zero] = np.nan
    return out


def weighted_pca(
    points: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (特征值升序, 特征向量按列同序, 加权质心)。主轴是 eigvecs[:, -1]。"""
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError(f"weighted_pca: sum of weights must be positive, got {w_sum}")

    centroid = (w[:, None] * points).sum(axis=0) / w_sum
    rel = points - centroid
    cov = (rel * w[:, None]).T @ rel / w_sum

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order], centroid


def ransac_plane_inliers(
    points: np.ndarray, threshold: float, iters: int = 300, rng: int = 0
) -> np.ndarray:
    """三点采样 RANSAC 平面，返回最佳内点 mask。内点数不足一半时报错。"""
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    min_inliers = max(3, n // 2)
    rng = np.random.default_rng(rng)

    best_mask, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < _EPS:
            continue  # 近共线采样
        normal = normal / norm
        dist = np.abs((points - p0) @ normal)
        mask = dist < threshold
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count

    if best_mask is None or best_count < min_inliers:
        raise ValueError(
            f"fit_plane RANSAC: no sample reached min_inliers={min_inliers} "
            f"in {iters} iters (best={best_count})"
        )
    return best_mask


def project_onto_plane(vectors: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """投影到法向为 normal 的平面并归一化（单个向量或 (N,3)）。"""
    v = np.asarray(vectors, dtype=float)
    single = v.ndim == 1
    V = v.reshape(1, 3) if single else v

    n_hat = unit(np.asarray(normal, dtype=float))
    comp = V @ n_hat
    in_plane = V - comp[:, None] * n_hat
    out = normalize_rows(in_plane)
    return out[0] if single else out


def rodrigues_rotate(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    single = v.ndim == 1
    V = v.reshape(1, 3) if single else v

    k = unit(np.asarray(axis, dtype=float))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cross_kv = np.cross(k, V)
    dot_kv = V @ k
    out = V * cos_a + cross_kv * sin_a + np.outer(dot_kv, k) * (1 - cos_a)
    return out[0] if single else out


def orient_to_reference(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """翻转 v 的符号使 dot(v, ref) >= 0。"""
    v = np.asarray(v, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return -v if np.dot(v, ref) < 0 else v.copy()
