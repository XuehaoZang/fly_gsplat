"""Small numpy helpers: quaternions (wxyz), rotations, slerp, image filters."""

from __future__ import annotations

import numpy as np


def quat_to_mat(q) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix (body -> world)."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quats_to_mats(q) -> np.ndarray:
    """(N, 4) -> (N, 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(R) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def slerp(q0, q1, u: float) -> np.ndarray:
    """Spherical interpolation between two wxyz quaternions, u in [0, 1]."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * u
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def continuous_quaternions(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions stay in the same hemisphere."""
    q = np.array(q, dtype=np.float64, copy=True)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0.0:
            q[i] *= -1.0
    return q


def interp_series(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation of (T, D) values onto new times."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty((len(t_dst), values.shape[1]))
    for d in range(values.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, values[:, d])
    return out


def interp_quaternions(t_src: np.ndarray, quats: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Slerp a (T, 4) wxyz series onto new times."""
    quats = continuous_quaternions(quats)
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 2)
    out = np.empty((len(t_dst), 4))
    for k, (i, t) in enumerate(zip(idx, t_dst)):
        span = t_src[i + 1] - t_src[i]
        u = 0.0 if span <= 0 else float(np.clip((t - t_src[i]) / span, 0.0, 1.0))
        out[k] = slerp(quats[i], quats[i + 1], u)
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a 2-D float image (edge-replicated)."""
    if sigma <= 0:
        return img
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(img.astype(np.float64), radius, mode="edge")
    rows = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, padded)
    cols = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, rows)
    return cols


def apply_photometry(gray: np.ndarray, blur_sigma_px: float, noise_sigma: float, rng) -> np.ndarray:
    """Optional blur and Gaussian sensor noise on a uint8 image; returns uint8."""
    if blur_sigma_px <= 0 and noise_sigma <= 0:
        return gray
    img = gray.astype(np.float64)
    if blur_sigma_px > 0:
        img = gaussian_blur(img, blur_sigma_px)
    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, size=img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)
