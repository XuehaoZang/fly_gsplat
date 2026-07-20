"""S1 verification: geometry primitives (`geometry.py`).

References calc_kinematics.md §0 (up = +z, stroke-plane via -45 deg
Rodrigues) and §5 (`orientation_*` = per-point local normal). Pure geometry
only: no fly/wing/body semantics are exercised here, and no angle
*definition* from §2-§5 is tested directly -- only the primitives those
definitions are built from (rodrigues_rotate, signed_angle,
project_onto_plane, quat/PCA/RANSAC fits).

Runnable both under pytest and standalone: `python test_s1.py`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from postprocessing.kinematics import geometry as geo

# ---------------------------------------------------------------------------
# unit / normalize_rows
# ---------------------------------------------------------------------------


def test_unit_normalizes_and_raises_on_zero():
    v = geo.unit(np.array([3.0, 0.0, 4.0]))
    assert np.allclose(v, [0.6, 0.0, 0.8])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12

    raised = False
    try:
        geo.unit(np.zeros(3))
    except ValueError:
        raised = True
    assert raised


def test_normalize_rows_returns_nan_for_zero_rows_without_raising():
    V = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    out = geo.normalize_rows(V)
    assert np.allclose(out[0], [0.6, 0.0, 0.8])
    assert np.all(np.isnan(out[1]))
    assert np.allclose(out[2], [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# weighted_pca
# ---------------------------------------------------------------------------


def test_weighted_pca_recovers_flat_sheet_normal():
    rng = np.random.default_rng(0)
    true_normal = geo.unit(np.array([0.2, -0.3, 1.0]))
    # build an orthonormal in-plane basis
    tmp = np.array([1.0, 0.0, 0.0])
    e1 = geo.unit(tmp - np.dot(tmp, true_normal) * true_normal)
    e2 = np.cross(true_normal, e1)

    n = 500
    a = rng.normal(scale=1.0, size=n)
    b = rng.normal(scale=1.0, size=n)
    thickness = rng.normal(scale=1e-4, size=n)  # tiny out-of-plane noise
    pts = a[:, None] * e1 + b[:, None] * e2 + thickness[:, None] * true_normal

    eigvals, eigvecs, centroid = geo.weighted_pca(pts)
    assert eigvals[0] < eigvals[1] < eigvals[2]
    smallest_axis = eigvecs[:, 0]
    cos_sim = abs(np.dot(smallest_axis, true_normal))
    assert cos_sim > 0.99, cos_sim
    assert np.linalg.norm(centroid) < 0.1


def test_weighted_pca_recovers_elongated_major_axis():
    rng = np.random.default_rng(1)
    true_axis = geo.unit(np.array([1.0, 1.0, 0.5]))
    tmp = np.array([0.0, 0.0, 1.0])
    e1 = geo.unit(tmp - np.dot(tmp, true_axis) * true_axis)
    e2 = np.cross(true_axis, e1)

    n = 500
    along = rng.normal(scale=5.0, size=n)
    off1 = rng.normal(scale=0.05, size=n)
    off2 = rng.normal(scale=0.05, size=n)
    pts = along[:, None] * true_axis + off1[:, None] * e1 + off2[:, None] * e2

    eigvals, eigvecs, _ = geo.weighted_pca(pts)
    major_axis = eigvecs[:, -1]
    cos_sim = abs(np.dot(major_axis, true_axis))
    assert cos_sim > 0.99, cos_sim


def test_weighted_pca_weights_bias_the_fit():
    # Two tight clusters of points along two different directions from the
    # origin. Unweighted PCA splits the difference; heavily weighting one
    # cluster should pull the major axis toward that cluster's direction.
    dir_a = geo.unit(np.array([1.0, 0.0, 0.0]))
    dir_b = geo.unit(np.array([0.0, 1.0, 0.0]))
    n = 200
    rng = np.random.default_rng(2)
    cluster_a = np.outer(rng.normal(loc=1.0, scale=0.05, size=n), dir_a)
    cluster_b = np.outer(rng.normal(loc=1.0, scale=0.05, size=n), dir_b)
    pts = np.vstack([cluster_a, cluster_b])

    _, eigvecs_uniform, _ = geo.weighted_pca(pts)
    major_uniform = eigvecs_uniform[:, -1]
    # roughly equidistant between dir_a and dir_b when unweighted
    assert abs(np.dot(major_uniform, dir_a)) < 0.9
    assert abs(np.dot(major_uniform, dir_b)) < 0.9

    weights = np.concatenate([np.full(n, 1000.0), np.full(n, 1.0)])
    _, eigvecs_biased, _ = geo.weighted_pca(pts, weights)
    major_biased = eigvecs_biased[:, -1]
    assert abs(np.dot(major_biased, dir_a)) > abs(np.dot(major_biased, dir_b))
    assert abs(np.dot(major_biased, dir_a)) > 0.9


# ---------------------------------------------------------------------------
# fit_plane / fit_line RANSAC
# ---------------------------------------------------------------------------


def test_fit_plane_ransac_recovers_plane_under_outliers():
    rng = np.random.default_rng(3)
    true_normal = geo.unit(np.array([0.0, 0.0, 1.0]))
    n_in, n_out = 300, 60
    inliers = np.column_stack(
        [rng.uniform(-1, 1, n_in), rng.uniform(-1, 1, n_in), rng.normal(scale=1e-4, size=n_in)]
    )
    outliers = np.column_stack(
        [rng.uniform(-1, 1, n_out), rng.uniform(-1, 1, n_out), rng.uniform(-1, 1, n_out)]
    )
    # drop outliers that accidentally land near the plane
    outliers = outliers[np.abs(outliers[:, 2]) > 0.05]
    pts = np.vstack([inliers, outliers])
    true_inlier_mask = np.concatenate(
        [np.ones(n_in, dtype=bool), np.zeros(outliers.shape[0], dtype=bool)]
    )

    normal, point, mask = geo.fit_plane(
        pts, method="ransac", threshold=1e-2, iters=200, min_inliers=n_in // 2, rng=0
    )
    cos_sim = abs(np.dot(normal, true_normal))
    assert cos_sim > 0.99, cos_sim
    assert abs(point[2]) < 1e-2

    # recall/precision against the true inlier set
    recall = np.sum(mask & true_inlier_mask) / true_inlier_mask.sum()
    precision = np.sum(mask & true_inlier_mask) / max(mask.sum(), 1)
    assert recall > 0.95, recall
    assert precision > 0.95, precision


def test_fit_line_ransac_recovers_line_under_outliers():
    rng = np.random.default_rng(4)
    true_dir = geo.unit(np.array([1.0, 2.0, -1.0]))
    n_in, n_out = 300, 60
    t = rng.uniform(-2, 2, n_in)
    noise = rng.normal(scale=1e-4, size=(n_in, 3))
    inliers = t[:, None] * true_dir + noise
    outliers = rng.uniform(-2, 2, size=(n_out, 3))
    # ensure outliers are actually far from the line
    proj = outliers @ true_dir
    perp = outliers - np.outer(proj, true_dir)
    outliers = outliers[np.linalg.norm(perp, axis=1) > 0.1]
    pts = np.vstack([inliers, outliers])
    true_inlier_mask = np.concatenate(
        [np.ones(n_in, dtype=bool), np.zeros(outliers.shape[0], dtype=bool)]
    )

    direction, point, mask = geo.fit_line(
        pts, method="ransac", threshold=1e-2, iters=200, min_inliers=n_in // 2, rng=0
    )
    cos_sim = abs(np.dot(direction, true_dir))
    assert cos_sim > 0.99, cos_sim

    recall = np.sum(mask & true_inlier_mask) / true_inlier_mask.sum()
    precision = np.sum(mask & true_inlier_mask) / max(mask.sum(), 1)
    assert recall > 0.95, recall
    assert precision > 0.95, precision


def test_fit_plane_ransac_raises_when_min_inliers_unreachable():
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1, 1, size=(50, 3))  # pure noise, no real plane
    raised = False
    try:
        geo.fit_plane(pts, method="ransac", threshold=1e-6, iters=50, min_inliers=40, rng=0)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# rodrigues_rotate
# ---------------------------------------------------------------------------


def test_rodrigues_rotate_minus45_about_y():
    x_body = np.array([1.0, 0.0, 0.0])
    y_body = np.array([0.0, 1.0, 0.0])
    n_sp = geo.rodrigues_rotate(x_body, y_body, math.radians(-45.0))
    assert abs(np.linalg.norm(n_sp) - 1.0) < 1e-12
    angle_to_x = math.degrees(math.acos(np.clip(np.dot(n_sp, x_body), -1, 1)))
    angle_to_y = math.degrees(math.acos(np.clip(np.dot(n_sp, y_body), -1, 1)))
    assert abs(angle_to_x - 45.0) < 1e-9
    assert abs(angle_to_y - 90.0) < 1e-9  # rotation about y_body keeps y_body-orthogonality


def test_rodrigues_rotate_360_round_trip_is_identity():
    v = geo.unit(np.array([0.3, -0.7, 1.1]))
    axis = geo.unit(np.array([0.1, 0.2, 0.9]))
    out = geo.rodrigues_rotate(v, axis, math.radians(360.0))
    assert np.allclose(out, v, atol=1e-9)


def test_rodrigues_rotate_composition_is_additive_about_same_axis():
    v = geo.unit(np.array([1.0, 0.2, -0.3]))
    axis = geo.unit(np.array([0.0, 0.0, 1.0]))
    a, b = math.radians(30.0), math.radians(50.0)
    step = geo.rodrigues_rotate(geo.rodrigues_rotate(v, axis, a), axis, b)
    combined = geo.rodrigues_rotate(v, axis, a + b)
    assert np.allclose(step, combined, atol=1e-9)


def test_rodrigues_rotate_vectorized_over_rows():
    V = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    axis = np.array([0.0, 0.0, 1.0])
    out = geo.rodrigues_rotate(V, axis, math.radians(90.0))
    expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(out, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# quat_to_rotmat / local_normal_from_gaussian
# ---------------------------------------------------------------------------


def test_quat_to_rotmat_matches_scipy_both_orders():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(6)
    n = 50
    raw = rng.normal(size=(n, 4))
    xyzw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    wxyz = xyzw[:, [3, 0, 1, 2]]

    expected = Rotation.from_quat(xyzw).as_matrix()

    got_xyzw = geo.quat_to_rotmat(xyzw, order="xyzw")
    got_wxyz = geo.quat_to_rotmat(wxyz, order="wxyz")
    assert np.allclose(got_xyzw, expected, atol=1e-8)
    assert np.allclose(got_wxyz, expected, atol=1e-8)


def test_quat_to_rotmat_single_quat_no_batch_dim():
    from scipy.spatial.transform import Rotation

    xyzw = np.array([0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)])  # 45deg about z
    wxyz = xyzw[[3, 0, 1, 2]]
    expected = Rotation.from_quat(xyzw).as_matrix()
    got = geo.quat_to_rotmat(wxyz, order="wxyz")
    assert got.shape == (3, 3)
    assert np.allclose(got, expected, atol=1e-8)


def test_local_normal_from_gaussian_is_min_scale_axis():
    # identity quaternion (wxyz) -> world axes == local axes
    identity_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    scale_phys = np.array([5.0, 1.0, 3.0])  # min is axis index 1 -> +y
    normal = geo.local_normal_from_gaussian(scale_phys, identity_wxyz, order="wxyz")
    assert np.allclose(np.abs(normal), [0.0, 1.0, 0.0], atol=1e-9)

    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)
    raw = rng.normal(size=4)
    xyzw = raw / np.linalg.norm(raw)
    wxyz = xyzw[[3, 0, 1, 2]]
    scale_phys2 = np.array([2.0, 0.1, 4.0])  # min is axis index 1
    normal2 = geo.local_normal_from_gaussian(scale_phys2, wxyz, order="wxyz")
    expected_col = Rotation.from_quat(xyzw).as_matrix()[:, 1]
    assert np.allclose(normal2, expected_col, atol=1e-8)
    assert abs(np.linalg.norm(normal2) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# signed_angle / project_onto_plane / orient_to_reference
# ---------------------------------------------------------------------------


def test_signed_angle_axis_aligned_hand_computed():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    assert abs(geo.signed_angle(x, y, z) - math.radians(90.0)) < 1e-9
    assert abs(geo.signed_angle(y, x, z) - math.radians(-90.0)) < 1e-9
    assert abs(geo.signed_angle(x, x, z)) < 1e-9
    assert abs(abs(geo.signed_angle(x, -x, z)) - math.pi) < 1e-9


def test_signed_angle_antisymmetric():
    rng = np.random.default_rng(8)
    axis = geo.unit(np.array([0.2, 0.4, 1.0]))
    for _ in range(20):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        fwd = geo.signed_angle(a, b, axis)
        bwd = geo.signed_angle(b, a, axis)
        assert abs(fwd + bwd) < 1e-9


def test_project_onto_plane_unit_length_and_orthogonal():
    rng = np.random.default_rng(9)
    normal = geo.unit(np.array([0.1, 0.2, 1.0]))
    V = rng.normal(size=(30, 3))
    out = geo.project_onto_plane(V, normal)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    dots = out @ normal
    assert np.allclose(dots, 0.0, atol=1e-9)


def test_project_onto_plane_parallel_vector_is_nan():
    normal = np.array([0.0, 0.0, 1.0])
    v = np.array([0.0, 0.0, 5.0])  # purely along normal
    out = geo.project_onto_plane(v, normal)
    assert np.all(np.isnan(out))


def test_orient_to_reference_flips_only_when_needed():
    ref = np.array([1.0, 0.0, 0.0])
    same = np.array([0.5, 0.1, 0.1])
    opposite = np.array([-0.5, 0.1, 0.1])
    assert np.allclose(geo.orient_to_reference(same, ref), same)
    assert np.allclose(geo.orient_to_reference(opposite, ref), -opposite)

    V = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = geo.orient_to_reference(V, ref)
    assert np.allclose(out, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _run_all():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
