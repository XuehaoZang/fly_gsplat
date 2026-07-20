"""S1 geometry primitives for T4 kinematics.

Pure geometry only — no fly/wing/body semantics and no angle *definitions*
(those live in S2-S4). Every function is stateless, works on `(N,3)` numpy
arrays (or a single `(3,)` vector where noted), and documents its own
sign/orientation convention since several of the later angle formulas in
`reference/calc_kinematics.md` are sign-sensitive (§3 roll, §4 phi/theta, §5
eta all build on `rodrigues_rotate` / `signed_angle` / `project_onto_plane`
defined here).

Units follow calc_kinematics.md §0 (meters); nothing here assumes a specific
unit, but callers should keep RANSAC `threshold` in the same units as the
input points.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize a single 3-vector.

    Raises `ValueError` on (near-)zero norm — a single vector with no defined
    direction is treated as a caller error, not something to silently paper
    over. For batches, use `normalize_rows`, which returns NaN rows instead
    (so one degenerate row doesn't abort a whole array).
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < _EPS:
        raise ValueError(f"unit(): cannot normalize a zero-length vector (norm={n:.3e})")
    return v / n


def normalize_rows(V: np.ndarray) -> np.ndarray:
    """Normalize each row of `(N,3)` (or any `(..., D)` array) to unit length.

    Rows with (near-)zero norm are returned as NaN (not raised) — this is a
    batch operation and a single degenerate point should not crash the whole
    call; callers that care must check `np.isnan(out).any(axis=-1)`.
    """
    V = np.asarray(V, dtype=float)
    norms = np.linalg.norm(V, axis=-1, keepdims=True)
    zero = (norms < _EPS).squeeze(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = V / norms
    out[zero] = np.nan
    return out


# ---------------------------------------------------------------------------
# PCA / plane / line fitting
# ---------------------------------------------------------------------------


def weighted_pca(
    points: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted PCA of an `(N,3)` point cloud.

    Returns `(eigvals_ascending, eigvecs_columns, centroid)`:
    - `eigvals_ascending`: shape `(3,)`, sorted smallest -> largest.
    - `eigvecs_columns`: shape `(3,3)`, `eigvecs[:, i]` is the eigenvector for
      `eigvals_ascending[i]`. So `eigvecs[:, 0]` = smallest-variance axis
      (plane normal for a flat cloud), `eigvecs[:, -1]` = major axis
      (span/long-axis direction). Column signs are arbitrary (PCA gives an
      axis, not a direction) — disambiguate with `orient_to_reference`.
    - `centroid`: shape `(3,)`, the (weighted) mean.

    `weights` defaults to uniform; must sum to a positive value. Degenerate
    input (fewer than 3 distinct points, or a perfectly collinear/coincident
    cloud) still returns a valid orthonormal eigenbasis, just with near-zero
    eigenvalues along the undetermined direction(s) — callers relying on a
    specific axis in that regime should check `eigvals_ascending` for
    near-degeneracy first.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError(f"weighted_pca: sum of weights must be positive, got {w_sum}")

    centroid = (w[:, None] * points).sum(axis=0) / w_sum
    rel = points - centroid
    cov = (rel * w[:, None]).T @ rel / w_sum

    eigvals, eigvecs = np.linalg.eigh(cov)  # eigh already returns ascending order
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order], centroid


def fit_plane(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "pca",
    *,
    threshold: float | None = None,
    iters: int = 300,
    min_inliers: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a plane to `(N,3)` points.

    Returns `(normal, point_on_plane, inlier_mask)`. `normal` is unit length;
    its **sign is unspecified** (a plane fit alone can't tell you which side
    is "outward") — callers must orient it, e.g. with `orient_to_reference`.

    `method="pca"`: weighted PCA plane through the centroid; `inlier_mask` is
    all-True (every point was used).

    `method="ransac"`: minimal-sample (3 points) RANSAC. Requires
    `threshold` (max point-to-plane distance to count as inlier, same units
    as `points`). Runs `iters` random samples, keeps the sample with the most
    inliers, then refits the final plane via weighted PCA on just those
    inliers (weights subset accordingly, if given). Raises `ValueError` if no
    sample reaches `min_inliers` (default `max(3, N // 2)`).
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]

    if method == "pca":
        _, eigvecs, centroid = weighted_pca(points, weights)
        normal = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
        return normal, centroid, np.ones(n, dtype=bool)

    if method == "ransac":
        if threshold is None:
            raise ValueError("fit_plane(method='ransac') requires `threshold`")
        if min_inliers is None:
            min_inliers = max(3, n // 2)
        rng = np.random.default_rng(rng)

        best_mask, best_count = None, -1
        for _ in range(iters):
            idx = rng.choice(n, size=3, replace=False)
            p0, p1, p2 = points[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < _EPS:
                continue  # degenerate (near-collinear) sample
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
        w_in = None if weights is None else np.asarray(weights, dtype=float)[best_mask]
        _, eigvecs, centroid = weighted_pca(points[best_mask], w_in)
        normal = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
        return normal, centroid, best_mask

    raise ValueError(f"fit_plane: unknown method {method!r}, expected 'pca' or 'ransac'")


def fit_line(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "pca",
    *,
    threshold: float | None = None,
    iters: int = 300,
    min_inliers: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a line to `(N,3)` points (used later for leading-edge fitting, §4/§5).

    Returns `(direction, point_on_line, inlier_mask)`. `direction` is unit
    length; sign unspecified — orient with `orient_to_reference` (e.g. toward
    a known wingtip).

    `method="pca"`: major axis (largest-eigenvalue eigenvector) of the
    weighted PCA through the centroid; `inlier_mask` all-True.

    `method="ransac"`: minimal-sample (2 points) RANSAC. Requires
    `threshold` (max perpendicular point-to-line distance). Best sample (by
    inlier count over `iters` draws) is refit via weighted PCA on its
    inliers. Raises `ValueError` if no sample reaches `min_inliers` (default
    `max(2, N // 2)`).
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]

    if method == "pca":
        _, eigvecs, centroid = weighted_pca(points, weights)
        direction = eigvecs[:, -1] / np.linalg.norm(eigvecs[:, -1])
        return direction, centroid, np.ones(n, dtype=bool)

    if method == "ransac":
        if threshold is None:
            raise ValueError("fit_line(method='ransac') requires `threshold`")
        if min_inliers is None:
            min_inliers = max(2, n // 2)
        rng = np.random.default_rng(rng)

        best_mask, best_count = None, -1
        for _ in range(iters):
            idx = rng.choice(n, size=2, replace=False)
            p0, p1 = points[idx]
            d = p1 - p0
            norm = np.linalg.norm(d)
            if norm < _EPS:
                continue  # degenerate (coincident) sample
            d = d / norm
            rel = points - p0
            proj = rel @ d
            perp = rel - np.outer(proj, d)
            dist = np.linalg.norm(perp, axis=1)
            mask = dist < threshold
            count = int(mask.sum())
            if count > best_count:
                best_mask, best_count = mask, count

        if best_mask is None or best_count < min_inliers:
            raise ValueError(
                f"fit_line RANSAC: no sample reached min_inliers={min_inliers} "
                f"in {iters} iters (best={best_count})"
            )
        w_in = None if weights is None else np.asarray(weights, dtype=float)[best_mask]
        _, eigvecs, centroid = weighted_pca(points[best_mask], w_in)
        direction = eigvecs[:, -1] / np.linalg.norm(eigvecs[:, -1])
        return direction, centroid, best_mask

    raise ValueError(f"fit_line: unknown method {method!r}, expected 'pca' or 'ransac'")


# ---------------------------------------------------------------------------
# Vector algebra
# ---------------------------------------------------------------------------


def project_onto_plane(vectors: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project `vectors` onto the plane perpendicular to `normal`, normalized.

    `vectors` may be a single `(3,)` vector or `(N,3)` (vectorized over
    rows); `normal` is a single `(3,)` vector applied to every row. Removes
    the component of each vector along `normal`, then unit-normalizes the
    remainder (this is what §4's `x_sp`/`y_sp`-style projections need — a
    direction *within* the plane, not just the residual).

    A vector (near-)parallel to `normal` has (near-)zero in-plane residual;
    per `normalize_rows`, that row comes back as NaN rather than raising,
    since this is a vectorized operation.
    """
    v = np.asarray(vectors, dtype=float)
    single = v.ndim == 1
    V = v.reshape(1, 3) if single else v

    n_hat = unit(np.asarray(normal, dtype=float))
    comp = V @ n_hat
    in_plane = V - comp[:, None] * n_hat
    out = normalize_rows(in_plane)
    return out[0] if single else out


def rodrigues_rotate(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate `vector` about unit `axis` by `angle_rad` (right-hand rule).

    `vector` may be a single `(3,)` vector or `(N,3)` (vectorized over rows);
    `axis` is a single `(3,)` vector, internally normalized. Standard
    Rodrigues formula:
    `v*cos(a) + (axis x v)*sin(a) + axis*(axis.v)*(1-cos(a))`.
    This matches the sign convention `mock.py::_rotate_about_axis` and
    calc_kinematics.md §0/§2's "`n_sp` = `x_body` rotated -45 deg about
    `y_body`" (a positive angle is a right-handed turn about `axis`).
    """
    v = np.asarray(vector, dtype=float)
    single = v.ndim == 1
    V = v.reshape(1, 3) if single else v

    k = unit(np.asarray(axis, dtype=float))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cross_kv = np.cross(k, V)
    dot_kv = V @ k
    out = V * cos_a + cross_kv * sin_a + np.outer(dot_kv, k) * (1 - cos_a)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Quaternions / Gaussian local frame
# ---------------------------------------------------------------------------


def quat_to_rotmat(q: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """Convert quaternion(s) to rotation matrix/matrices.

    `q` is `(4,)` or `(N,4)`; `order` is `"wxyz"` (nerfstudio/gsplat
    convention, default) or `"xyzw"`. Returns `(3,3)` or `(N,3,3)`.
    Column `i` of the returned matrix is the world-frame direction of the
    quaternion's local axis `i` — same convention as
    `utils/gaussian_features.py::_quat_to_rotmat`, which this mirrors so T4
    stays consistent with how T1 derived `orientation_*` from `rot_i`/`scale_i`.
    Quaternions are re-normalized internally (not assumed unit).
    """
    q = np.asarray(q, dtype=float)
    single = q.ndim == 1
    Q = q.reshape(1, 4) if single else q

    if order == "wxyz":
        w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    elif order == "xyzw":
        x, y, z, w = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    else:
        raise ValueError(f"quat_to_rotmat: unknown order {order!r}, expected 'wxyz' or 'xyzw'")

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    n = Q.shape[0]
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
    return R[0] if single else R


def local_normal_from_gaussian(
    scale_phys: np.ndarray, quat: np.ndarray, order: str = "wxyz"
) -> np.ndarray:
    """World-frame direction of one Gaussian's smallest-scale local axis.

    `scale_phys` is `(3,)` (this Gaussian's `scale_phys_0/1/2`), `quat` is
    `(4,)`. This is the sheet-normal proxy §5 step 4 uses to reject points
    whose local normal disagrees with the fitted wing-plane normal
    (`orientation_*` in the input CSV, per §1, is exactly this quantity
    precomputed at the per-point-cloud-extraction stage — see
    `utils/gaussian_features.py`). Returns a unit `(3,)` vector; like any
    principal-axis direction its **sign is arbitrary** (rotating the
    quaternion by 180 deg about that axis leaves the Gaussian unchanged but
    flips this vector) — orient with `orient_to_reference` if a consistent
    sign is needed.
    """
    scale_phys = np.asarray(scale_phys, dtype=float)
    R = quat_to_rotmat(quat, order=order)
    min_idx = int(np.argmin(scale_phys))
    normal = R[:, min_idx]
    return normal / np.linalg.norm(normal)


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------


def signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from `a` to `b`, right-handed about `axis`, in `(-pi, pi]`.

    `a` and `b` are first projected onto the plane perpendicular to `axis`
    (components along `axis` are dropped), so this is well-defined even if
    `a`/`b` are not already in-plane. This is the workhorse the later
    `atan2`-based formulas build on (§3 roll, §4 phi, §5 eta all reduce to a
    `signed_angle` of two in-(stroke-)plane vectors about the plane normal).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = unit(np.asarray(axis, dtype=float))

    a_perp = a - np.dot(a, k) * k
    b_perp = b - np.dot(b, k) * k
    sin_part = np.dot(k, np.cross(a_perp, b_perp))
    cos_part = np.dot(a_perp, b_perp)
    return float(np.arctan2(sin_part, cos_part))


def orient_to_reference(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip `v` so it points toward `ref` (`dot(v, ref) >= 0`); else return as-is.

    Sign-disambiguation helper for the axis outputs of `weighted_pca` /
    `fit_plane` / `fit_line` / `local_normal_from_gaussian`, whose sign is
    otherwise arbitrary. `v` may be `(3,)` or `(N,3)` (vectorized over rows);
    `ref` is `(3,)` (broadcast) or matches `v`'s shape for a per-row reference.
    """
    v = np.asarray(v, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if v.ndim == 1:
        return -v if np.dot(v, ref) < 0 else v.copy()
    dots = np.sum(v * ref, axis=-1)
    sign = np.where(dots < 0, -1.0, 1.0)
    return v * sign[:, None]
