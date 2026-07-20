"""S3 verification: wing stroke & deviation angles (`wing_angles.py`).

References calc_kinematics.md §4 (wing stroke & deviation, phi/theta). Uses
`mock.py`'s forward-constructed ground truth to check the *estimator* in
`wing_angles.py` recovers what the mock was built from.

Most tests here drive `stroke_deviation`/`estimate_leading_edge` with a
**true** `BodyFrame` built directly from `GroundTruth` (`_true_body_frame`),
not `body_frame.estimate_body_frame`'s output. `body_frame`'s own recovery
noise floor (~1-3 deg, see test_s2.py) is that module's concern; isolating it
here keeps this file's tolerances about what S3 (leading-edge fit + phi/theta
formulas) is actually responsible for.

No chord/eta here -- that's S4.

Runnable both under pytest and standalone: `python test_s3.py`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from postprocessing.kinematics import body_frame as bf
from postprocessing.kinematics import geometry as geo
from postprocessing.kinematics import io_schema, mock, wing_angles as wa


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference `a - b` wrapped into `(-180, 180]`."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _true_body_frame(gt: mock.GroundTruth) -> bf.BodyFrame:
    """Wrap `GroundTruth`'s own forward geometry into a `BodyFrame`, with no
    fitting/estimation involved -- the exact axes `mock.py` built the point
    cloud from.
    """
    x_body, y_body, z_body = mock.body_axes(gt)
    n_sp = mock.stroke_plane_normal(gt)
    return bf.BodyFrame(
        x_body=x_body, y_body=y_body, z_body=z_body, n_sp=n_sp,
        yaw=gt.yaw_deg, pitch=gt.pitch_deg, roll=gt.roll_deg,
        hinge_L=gt.wing_L.root, hinge_R=gt.wing_R.root, body_cm=gt.body_center,
    )


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos_sim = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
    return math.degrees(math.acos(cos_sim))


# ---------------------------------------------------------------------------
# scenario_clean: phi/theta recovery against ground truth
# ---------------------------------------------------------------------------


def test_clean_scenario_recovers_phi_theta_both_wings():
    df, gt = mock.scenario_clean(seed=0)
    frame = _true_body_frame(gt)

    for side, wing_gt in (("wing_L", gt.wing_L), ("wing_R", gt.wing_R)):
        wing_xyz = io_schema.get_part(df, side)
        sweep = wa.stroke_deviation(wing_xyz, frame, side)

        # le_dir recovers the true span direction directly.
        angle_err = _angle_between_deg(sweep.leading_edge.le_dir, wing_gt.span_dir)
        assert angle_err < 2.0, (side, angle_err)

        # theta matches the true deviation exactly as stored (both are the
        # same §4 formula applied to true vs. estimated `le_dir`).
        assert abs(_angular_diff_deg(sweep.theta, wing_gt.deviation_deg)) < 2.0, (side, sweep.theta)

        # phi matches the value §4's formula gives for the TRUE span_dir
        # (independently recomputed here, not read off any stored constant --
        # WingGroundTruth deliberately does not store phi, see mock.py).
        true_phi, true_theta = wa._phi_theta(
            wing_gt.span_dir, frame.x_body, frame.y_body, frame.n_sp, wa._SIGN_LEFT[side]
        )
        assert abs(_angular_diff_deg(true_theta, wing_gt.deviation_deg)) < 1e-9  # self-consistency
        assert abs(_angular_diff_deg(sweep.phi, true_phi)) < 2.0, (side, sweep.phi, true_phi)


def test_estimate_leading_edge_oriented_outward_both_sides():
    df, gt = mock.scenario_clean(seed=1)
    frame = _true_body_frame(gt)
    for side, wing_gt in (("wing_L", gt.wing_L), ("wing_R", gt.wing_R)):
        wing_xyz = io_schema.get_part(df, side)
        le = wa.estimate_leading_edge(wing_xyz, frame, side)
        wing_centroid = wing_xyz.mean(axis=0)
        out_ref = wing_centroid - frame.body_cm
        assert np.dot(le.le_dir, out_ref) >= 0.0, (side, le.le_dir, out_ref)
        assert abs(np.linalg.norm(le.le_dir) - 1.0) < 1e-9
        # root closer to the body than tip, along le_dir
        assert np.dot(le.tip - le.root, le.le_dir) > 0.0


# ---------------------------------------------------------------------------
# Numerical-identity check: _phi_theta vs. inline cell-3 reimplementation
# ---------------------------------------------------------------------------


def _reference_phi_theta(le, n_sp, x_body, y_body, sign_left):
    """Inline re-derivation of `reference/python_snippets.py` cell 3
    `project_on_plane` + `calculate_phi` geometry, single-vector, for direct
    numerical comparison against `wing_angles._phi_theta` (not just via the
    mock).
    """
    def project_on_plane(normal, vector):
        projected = vector - np.dot(normal, vector) * normal
        return projected / np.linalg.norm(projected)

    xbody_on_sp = project_on_plane(n_sp, x_body)
    ybody_on_sp = project_on_plane(n_sp, y_body)
    le_on_sp = project_on_plane(n_sp, le)

    xle = np.dot(le_on_sp, xbody_on_sp)
    yle = np.dot(le_on_sp, ybody_on_sp)
    phi = math.atan2(sign_left * yle, xle) * 180.0 / math.pi
    theta = 90.0 - math.acos(np.dot(n_sp, le)) * 180.0 / math.pi
    return phi, theta


def test_phi_theta_matches_reference_formula_numerically():
    rng = np.random.default_rng(11)
    for _ in range(50):
        x_body = geo.unit(rng.normal(size=3))
        tmp = geo.unit(rng.normal(size=3))
        y_body = geo.unit(tmp - np.dot(tmp, x_body) * x_body)
        n_sp = geo.rodrigues_rotate(x_body, y_body, math.radians(-45.0))
        le = geo.unit(rng.normal(size=3))
        sign_left = rng.choice([-1.0, 1.0])

        got_phi, got_theta = wa._phi_theta(le, x_body, y_body, n_sp, sign_left)
        exp_phi, exp_theta = _reference_phi_theta(le, n_sp, x_body, y_body, sign_left)
        assert abs(got_phi - exp_phi) < 1e-9, (got_phi, exp_phi)
        assert abs(got_theta - exp_theta) < 1e-9, (got_theta, exp_theta)


# ---------------------------------------------------------------------------
# Symmetry: geometric mirror flips phi sign, preserves theta
# ---------------------------------------------------------------------------


def test_mirrored_wing_yields_mirrored_phi_equal_theta():
    """Mirrors `wing_L`'s point cloud across the plane spanned by
    `(x_body, n_sp)` through `body_cm` (i.e. negates each point's `y_body`
    component -- `y_body` is exactly in-plane and orthogonal to `n_sp`, per
    `body_frame.py`, so this is a reflection of the stroke-plane geometry
    about `x_sp`). Feeding the mirrored cloud back in under the SAME side
    label isolates the phi/theta formulas' response to a pure geometric
    mirror from the `sign_left` L/R convention (which is deliberately built
    so that a physically mirror-symmetric wing *pair*, one per side, yields
    matching-sign phi -- see `mock._SIGN_LEFT` / §4). A geometric mirror on
    one fixed side must flip `phi`'s sign and leave `theta` unchanged.
    """
    df, gt = mock.scenario_clean(seed=2)
    frame = _true_body_frame(gt)
    wing_xyz = io_schema.get_part(df, "wing_L")

    sweep = wa.stroke_deviation(wing_xyz, frame, "wing_L")

    rel = wing_xyz - frame.body_cm
    y_comp = rel @ frame.y_body
    mirrored_xyz = wing_xyz - 2.0 * np.outer(y_comp, frame.y_body)

    sweep_mirrored = wa.stroke_deviation(mirrored_xyz, frame, "wing_L")

    assert abs(_angular_diff_deg(sweep_mirrored.phi, -sweep.phi)) < 2.0, (sweep.phi, sweep_mirrored.phi)
    assert abs(sweep_mirrored.theta - sweep.theta) < 2.0, (sweep.theta, sweep_mirrored.theta)


# ---------------------------------------------------------------------------
# estimate_leading_edge robustness to outliers
# ---------------------------------------------------------------------------


def test_estimate_leading_edge_robust_to_outliers():
    df, gt = mock.scenario_clean(seed=3)
    frame = _true_body_frame(gt)
    rng = np.random.default_rng(4)

    for side, wing_gt in (("wing_L", gt.wing_L), ("wing_R", gt.wing_R)):
        wing_xyz = io_schema.get_part(df, side)
        n_clean = wing_xyz.shape[0]

        extent = wing_xyz.max(axis=0) - wing_xyz.min(axis=0)
        n_outliers = int(0.15 * n_clean)
        lo, hi = wing_xyz.min(axis=0) - extent, wing_xyz.max(axis=0) + extent
        outliers = rng.uniform(lo, hi, size=(n_outliers, 3))
        contaminated = np.vstack([wing_xyz, outliers])

        le = wa.estimate_leading_edge(contaminated, frame, side, rng=0)

        angle_err = _angle_between_deg(le.le_dir, wing_gt.span_dir)
        assert angle_err < 3.0, (side, angle_err)

        wing_centroid = contaminated.mean(axis=0)
        out_ref = wing_centroid - frame.body_cm
        assert np.dot(le.le_dir, out_ref) >= 0.0

        # every flagged inlier must come from the original (clean) points
        assert not np.any(le.inlier_mask[n_clean:]), "RANSAC line fit picked up an injected outlier"
        assert np.any(le.inlier_mask[:n_clean])


# ---------------------------------------------------------------------------
# Sweep: several (span, deviation) configs, including deviation~0 and large phi
# ---------------------------------------------------------------------------


def test_sweep_recovers_phi_theta_across_configs():
    """Sweeps phi/theta configs including near-zero deviation and large
    |phi| (near the atan2 wrap boundary at +/-180). Single-frame `phi`/`theta`
    are raw (no `unwrap`, per §4 and `stroke_deviation`'s docstring) -- a
    true value near +180 can be recovered as a value near -180 (or vice
    versa) since both represent the same physical angle; `_angular_diff_deg`
    (wrap-aware) is used for the comparison rather than raw subtraction, and
    that wrap is the only "wrap behavior" this single-frame function has --
    stitching a trajectory's -180/+180 crossings into a continuous curve is a
    multi-frame concern, not implemented here.
    """
    cases = [
        dict(phi_L_deg=140.0, phi_R_deg=40.0, theta_L_deg=10.0, theta_R_deg=10.0),
        dict(phi_L_deg=90.0, phi_R_deg=90.0, theta_L_deg=0.0, theta_R_deg=0.0),  # deviation ~ 0
        dict(phi_L_deg=170.0, phi_R_deg=10.0, theta_L_deg=25.0, theta_R_deg=-15.0),  # large |phi|
        dict(phi_L_deg=-170.0, phi_R_deg=-10.0, theta_L_deg=5.0, theta_R_deg=5.0),  # near -180 wrap
        dict(phi_L_deg=5.0, phi_R_deg=175.0, theta_L_deg=-20.0, theta_R_deg=20.0),
    ]
    for i, case in enumerate(cases):
        gt = mock.default_ground_truth(**case)
        df, gt = mock.make_frame(gt, seed=100 + i)
        frame = _true_body_frame(gt)

        for side, wing_gt, true_phi in (
            ("wing_L", gt.wing_L, case["phi_L_deg"]),
            ("wing_R", gt.wing_R, case["phi_R_deg"]),
        ):
            wing_xyz = io_schema.get_part(df, side)
            sweep = wa.stroke_deviation(wing_xyz, frame, side)
            assert abs(_angular_diff_deg(sweep.phi, true_phi)) < 3.0, (case, side, sweep.phi, true_phi)
            assert abs(_angular_diff_deg(sweep.theta, wing_gt.deviation_deg)) < 3.0, (
                case, side, sweep.theta, wing_gt.deviation_deg
            )


def test_unknown_side_raises():
    df, gt = mock.scenario_clean(seed=0)
    frame = _true_body_frame(gt)
    wing_xyz = io_schema.get_part(df, "wing_L")
    for fn in (wa.estimate_leading_edge, wa.stroke_deviation):
        raised = False
        try:
            fn(wing_xyz, frame, "wing_bogus")
        except ValueError:
            raised = True
        assert raised, fn


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
