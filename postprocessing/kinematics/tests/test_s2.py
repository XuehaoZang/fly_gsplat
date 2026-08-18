"""S2 verification: body frame + body angles (`body_frame.py`).

References calc_kinematics.md §2 (body frame construction) and §3 (body
angles). Uses `mock.py`'s forward-constructed ground truth to check the
*estimator* in `body_frame.py` recovers what the mock was built from — the
mock itself is checked separately in test_s0.py.

Note: S3/S4 (wing phi/theta/eta, chord) depend only on this module's
`x_body, y_body, z_body, n_sp` outputs — nothing else from `BodyFrame` is
part of their contract.

Runnable both under pytest and standalone: `python test_s2.py`.
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
from postprocessing.kinematics import io_schema, mock


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference `a - b` wrapped into `(-180, 180]`."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _estimate_from_gt(gt: mock.GroundTruth, seed: int = 0, root_mode: str = "root", **kwargs) -> bf.BodyFrame:
    df, _ = mock.make_frame(gt, seed=seed)
    body_xyz = io_schema.body_xyz(df)
    wingL_xyz = io_schema.wingL_xyz(df)
    wingR_xyz = io_schema.wingR_xyz(df)
    return bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz, root_mode=root_mode, **kwargs)


# ---------------------------------------------------------------------------
# scenario_clean: tight recovery + frame algebra
# ---------------------------------------------------------------------------


def test_clean_scenario_recovers_yaw_pitch_roll():
    """Tolerance note: `scenario_clean()`'s default point counts (n_body=300,
    n_wing=400) leave a real, measured few-degree noise floor on this
    estimator — `x_body` is a PCA axis of a finite ellipsoid-surface sample
    (body aspect ratio ~2.8:1, not infinitely elongated), measured (not `<1`
    deg as the spec's illustrative bound suggested) at ~1-2.5 deg for this
    seed, so `yaw`/`pitch`'s tolerance reflects that real floor rather than a
    looser stand-in.

    `roll` (via `hinge_*`, §2 step 2) has a separate, wider floor since
    `_wing_hinge`'s `root_mode="root"` moved from a single most-extreme
    sampled point along the wing's own PCA span axis to
    `robust_body_axis.compute_wing_hinge_far_cc` (far-from-wing-centroid +
    connected-component root cluster, oriented by `guide_axis = unit(body_cm
    - wing_cm)` -- see that function's docstring, and `_wing_hinge`'s, for
    why this is the wired-in default despite being *noisier* on this
    particular synthetic scenario: it fixes a large real-data roll-jump
    problem, measured on `correct_body_axis/diag/i_roll_source_isolation.py`,
    that this idealized clean synthetic mock never exercises). On this exact
    scenario `guide_axis` is a genuinely different, slightly-off-span
    direction from the wing's own PCA axis (the mock's wing sweep/root
    offset means `body_cm - wing_cm` isn't perfectly spanwise), so the
    far/CC method's root-cluster centroid picks up a small systematic bias
    the old single-extreme-point method didn't have here -- measured at
    mean=2.76 deg, max=4.97 deg over 30 seeds (`np.random.default_rng`
    seeds 0-29, same `scenario_clean` call), vs the old method's ~1-2.5 deg.
    6.0 deg is a bound with headroom above that measured max, not a guess.
    """
    df, gt = mock.scenario_clean(seed=0)
    frame = bf.estimate_body_frame(
        io_schema.body_xyz(df), io_schema.wingL_xyz(df), io_schema.wingR_xyz(df)
    )
    assert abs(_angular_diff_deg(frame.yaw, gt.yaw_deg)) < 3.0, frame.yaw
    assert abs(_angular_diff_deg(frame.pitch, gt.pitch_deg)) < 3.0, frame.pitch
    assert abs(_angular_diff_deg(frame.roll, gt.roll_deg)) < 6.0, frame.roll


def test_clean_scenario_frame_is_orthonormal_right_handed():
    df, gt = mock.scenario_clean(seed=0)
    frame = bf.estimate_body_frame(
        io_schema.body_xyz(df), io_schema.wingL_xyz(df), io_schema.wingR_xyz(df)
    )
    for v in (frame.x_body, frame.y_body, frame.z_body):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-9

    assert abs(np.dot(frame.x_body, frame.y_body)) < 1e-9
    assert abs(np.dot(frame.y_body, frame.z_body)) < 1e-9
    assert abs(np.dot(frame.x_body, frame.z_body)) < 1e-9

    # right-handed: z_body == x_body x y_body
    assert np.allclose(np.cross(frame.x_body, frame.y_body), frame.z_body, atol=1e-9)


def test_clean_scenario_stroke_plane_normal_geometry():
    df, gt = mock.scenario_clean(seed=0)
    frame = bf.estimate_body_frame(
        io_schema.body_xyz(df), io_schema.wingL_xyz(df), io_schema.wingR_xyz(df),
        stroke_plane_pitch_deg=45.0,
    )
    assert abs(np.linalg.norm(frame.n_sp) - 1.0) < 1e-9

    angle_to_x = math.degrees(math.acos(np.clip(np.dot(frame.n_sp, frame.x_body), -1, 1)))
    assert abs(angle_to_x - 45.0) < 1e-6, angle_to_x
    assert abs(np.dot(frame.n_sp, frame.y_body)) < 1e-9


def test_external_stroke_plane_normal_overrides_and_is_normalized():
    df, gt = mock.scenario_clean(seed=0)
    raw = np.array([1.0, 2.0, 3.0])
    frame = bf.estimate_body_frame(
        io_schema.body_xyz(df), io_schema.wingL_xyz(df), io_schema.wingR_xyz(df),
        stroke_plane_normal=raw,
    )
    assert abs(np.linalg.norm(frame.n_sp) - 1.0) < 1e-12
    assert np.allclose(frame.n_sp, raw / np.linalg.norm(raw))


# ---------------------------------------------------------------------------
# roll: numerical identity against the reference cell-2 formula
# ---------------------------------------------------------------------------


def _reference_calculate_roll(yaw_rad: float, pitch_rad: float, y_body: np.ndarray) -> float:
    """Inline re-derivation of `reference/python_snippets.py` cell 2
    `calculate_roll`, non-vectorized, for direct numerical comparison
    against `body_frame._calculate_roll` (not just via the mock).
    """
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    ey = np.array([-sy, cy, 0.0])
    ez = np.array([-sp * cy, -sp * sy, cp])
    Yy = np.dot(y_body, ey)
    Yz = np.dot(y_body, ez)
    return math.atan2(Yz, Yy)


def test_calculate_roll_matches_reference_formula_numerically():
    rng = np.random.default_rng(42)
    for _ in range(50):
        yaw_rad = rng.uniform(-math.pi, math.pi)
        pitch_rad = rng.uniform(-math.pi / 2 + 0.01, math.pi / 2 - 0.01)
        y_body = geo.unit(rng.normal(size=3))
        got = bf._calculate_roll(yaw_rad, pitch_rad, y_body)
        expected = _reference_calculate_roll(yaw_rad, pitch_rad, y_body)
        assert abs(got - expected) < 1e-12, (got, expected)


# ---------------------------------------------------------------------------
# sweep: several ground-truth (yaw, pitch, roll), including edge configs
# ---------------------------------------------------------------------------


def test_sweep_recovers_yaw_pitch_roll_across_configs():
    """Tolerance note: same noise floor as `test_clean_scenario_recovers_*`
    (finite-sample body PCA + single-extreme-point wing hinge, §2 steps 1-2),
    measured up to ~6 deg worst-case on some of these configs at default
    point counts. The point of this sweep is confirming the yaw/pitch/roll
    *formulas* (§3) hold up across wrap/near-gimbal configs, not sub-degree
    precision — see the dedicated `scenario_clean` test for the tighter,
    single-config check.
    """
    cases = [
        dict(yaw_deg=0.0, pitch_deg=10.0, roll_deg=0.0),
        dict(yaw_deg=45.0, pitch_deg=20.0, roll_deg=30.0),
        dict(yaw_deg=179.0, pitch_deg=5.0, roll_deg=-20.0),   # yaw wrap near +180
        dict(yaw_deg=-179.0, pitch_deg=5.0, roll_deg=20.0),   # yaw wrap near -180
        dict(yaw_deg=10.0, pitch_deg=15.0, roll_deg=88.0),    # roll near +90
        dict(yaw_deg=10.0, pitch_deg=15.0, roll_deg=-88.0),   # roll near -90
        dict(yaw_deg=-60.0, pitch_deg=30.0, roll_deg=-45.0),
    ]
    for i, case in enumerate(cases):
        gt = mock.default_ground_truth(**case)
        frame = _estimate_from_gt(gt, seed=i)
        assert abs(_angular_diff_deg(frame.yaw, case["yaw_deg"])) < 7.0, (case, frame.yaw)
        assert abs(_angular_diff_deg(frame.pitch, case["pitch_deg"])) < 7.0, (case, frame.pitch)
        assert abs(_angular_diff_deg(frame.roll, case["roll_deg"])) < 7.0, (case, frame.roll)


def test_negative_pitch_head_sign_heuristic_documented_failure():
    """Documents (does not fix) the §2-step-1 head-sign heuristic failure:
    the body PCA axis has no intrinsic head/tail sign, so it's oriented via
    `dot(x_body, up) > 0`. For a genuinely nose-down (negative-pitch) body,
    that heuristic picks the tail as "head", flipping `x_body` end-to-end.
    `y_body` (built from wing hinges, independent of `x_body`'s sign via
    `project_onto_plane`'s symmetric projection) is unaffected, but
    `yaw`/`pitch` derived from the flipped `x_body` come out as
    `yaw + 180` / `-pitch` instead of the true values.
    """
    case = dict(yaw_deg=30.0, pitch_deg=-15.0, roll_deg=10.0)
    gt = mock.default_ground_truth(**case)
    frame = _estimate_from_gt(gt, seed=0)

    # frame stays finite / orthonormal despite the sign flip
    assert np.all(np.isfinite(frame.x_body))
    assert abs(np.linalg.norm(frame.x_body) - 1.0) < 1e-9

    # documented flip: pitch negates, yaw shifts by 180 deg
    assert abs(_angular_diff_deg(frame.pitch, -case["pitch_deg"])) < 2.0, frame.pitch
    assert abs(_angular_diff_deg(frame.yaw, case["yaw_deg"] + 180.0)) < 2.0, frame.yaw


# ---------------------------------------------------------------------------
# root_mode="centroid" fallback
# ---------------------------------------------------------------------------


def test_centroid_root_mode_runs_and_is_a_valid_frame():
    """`root_mode="centroid"` runs end-to-end and yields a valid unit
    `y_body`. It is NOT numerically close to `root_mode="root"` on
    `scenario_clean()`'s default pose (measured cos similarity ~-0.02, i.e.
    near-orthogonal, not "close"): the whole-wing centroid sits roughly
    half a span-length away from the true hinge, offset along that wing's
    own (obliquely-angled, `phi_L=140/phi_R=40` by default) span direction,
    and that offset dominates the small (~1.75mm) true inter-hinge distance.
    This is the "acknowledged simplification" calc_kinematics.md §2 step 2
    flags for the centroid fallback, measured rather than assumed — not
    fixed here, since a real root-region-truncated wing cloud (the case
    this fallback exists for) would look different again. See
    `test_sweep_...` / `_wing_hinge` for the `"root"` estimate this is
    contrasted with.
    """
    df, gt = mock.scenario_clean(seed=0)
    body_xyz = io_schema.body_xyz(df)
    wingL_xyz = io_schema.wingL_xyz(df)
    wingR_xyz = io_schema.wingR_xyz(df)

    frame_centroid = bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz, root_mode="centroid")

    assert np.all(np.isfinite(frame_centroid.y_body))
    assert abs(np.linalg.norm(frame_centroid.y_body) - 1.0) < 1e-9
    assert abs(np.dot(frame_centroid.y_body, frame_centroid.x_body)) < 1e-9


def test_unknown_root_mode_raises():
    df, gt = mock.scenario_clean(seed=0)
    raised = False
    try:
        bf.estimate_body_frame(
            io_schema.body_xyz(df), io_schema.wingL_xyz(df), io_schema.wingR_xyz(df),
            root_mode="bogus",
        )
    except ValueError:
        raised = True
    assert raised


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
