"""S4a verification: single-frame, xyz-only chord/eta baseline (`chord.py`).

References calc_kinematics.md §5 (wing chord & pitch eta). Uses `mock.py`'s
forward-constructed ground truth to check the *estimator* in `chord.py`
recovers what the mock was built from, plus formula-level and robustness
checks matching test_s2.py/test_s3.py's style.

Most tests here drive `estimate_chord` with a **true** `BodyFrame` built
directly from `GroundTruth` (`_true_body_frame`), isolating S4a's own error
(LE fit + binning + eta formula) from `body_frame.py`'s recovery noise floor
(that module's own concern, see test_s2.py).

`robust=True`/`use_gaussian_normals=True` and the stroke-reversal-contaminated
scenario are S4b's targets, not S4a's -- see the dedicated tests below for
what S4a is (and isn't) expected to do with them.

Runnable both under pytest and standalone: `python test_s4a.py`.
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
from postprocessing.kinematics import chord as ch
from postprocessing.kinematics import io_schema, mock, wing_angles as wa


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference `a - b` wrapped into `(-180, 180]`."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _true_body_frame(gt: mock.GroundTruth) -> bf.BodyFrame:
    """Wrap `GroundTruth`'s own forward geometry into a `BodyFrame`, with no
    fitting/estimation involved -- the exact axes `mock.py` built the point
    cloud from. Copied from test_s3.py (kept per-file per that file's own
    convention, see test_s2.py/test_s3.py).
    """
    x_body, y_body, z_body = mock.body_axes(gt)
    n_sp = mock.stroke_plane_normal(gt)
    return bf.BodyFrame(
        x_body=x_body, y_body=y_body, z_body=z_body, n_sp=n_sp,
        yaw=gt.yaw_deg, pitch=gt.pitch_deg, roll=gt.roll_deg,
        hinge_L=gt.wing_L.root, hinge_R=gt.wing_R.root, body_cm=gt.body_center,
    )


# ---------------------------------------------------------------------------
# scenario_clean: eta recovery + chord vector sanity, both wings
# ---------------------------------------------------------------------------


def test_clean_scenario_recovers_eta_both_wings():
    df, gt = mock.scenario_clean(seed=0)
    frame = _true_body_frame(gt)

    for side, wing_gt in (("wing_L", gt.wing_L), ("wing_R", gt.wing_R)):
        wing_xyz = io_schema.get_part(df, side)
        sign_left = wa._SIGN_LEFT[side]
        le = wa.estimate_leading_edge(wing_xyz, frame, side)
        result = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le)

        # eta matches ground truth (target < 2-3 deg; measured ~0.3-0.4 deg
        # for this seed).
        assert abs(_angular_diff_deg(result.eta, wing_gt.eta_deg)) < 3.0, (side, result.eta)

        # chord is unit length.
        assert abs(np.linalg.norm(result.chord) - 1.0) < 1e-9, (side, result.chord)

        # chord is in-plane (perp to n_w): loose tolerance -- S4a only
        # removes the *span* component per bin (see chord.py's
        # `_bin_chord_vectors` docstring), not the plane-normal component, so
        # near-orthogonality here is an empirical property of the clean
        # flat-sheet mock (thickness << chord, averaged over many span bins),
        # not an algebraic guarantee.
        cos_to_normal = abs(np.dot(result.chord, le.plane_normal))
        angle_from_perp = 90.0 - math.degrees(math.acos(np.clip(cos_to_normal, -1.0, 1.0)))
        assert angle_from_perp < 2.0, (side, angle_from_perp)

        # chord is oriented LE->TE: aligned with the true chord direction
        # (built from the same estimated le_dir, per mock.py's forward
        # geometry), not anti-parallel.
        true_chord = mock._chord_dir(le.le_dir, frame.n_sp, wing_gt.eta_deg, sign_left)
        assert np.dot(result.chord, true_chord) > 0.9, (side, result.chord, true_chord)

        assert result.per_bin_chords.shape == (result.n_bins_used, 3)
        assert result.n_bins_used >= 3
        assert 0.0 <= result.chord_conf <= 1.0


# ---------------------------------------------------------------------------
# Numerical-identity check: _eta vs. inline cell-3 reimplementation
# ---------------------------------------------------------------------------


def _reference_eta(chord: np.ndarray, le: np.ndarray, n_sp: np.ndarray, sign_left: float) -> float:
    """Inline re-derivation of calc_kinematics.md §5 / `reference/python_snippets.py`
    cell 3's `sp_chord`/`le_sp_normal` construction, for direct numerical
    comparison against `chord._eta` (not just via the mock).
    """
    if sign_left > 0:
        le_sp_normal = np.cross(n_sp, le)
    else:
        le_sp_normal = np.cross(le, n_sp)
    le_sp_normal = le_sp_normal / np.linalg.norm(le_sp_normal)

    sp_chord = np.cross(le, le_sp_normal)
    sp_chord = sp_chord / np.linalg.norm(sp_chord)

    x = np.dot(chord, le_sp_normal)
    y = np.dot(chord, sp_chord)
    return math.degrees(math.atan2(sign_left * y, x))


def test_eta_matches_reference_formula_numerically():
    rng = np.random.default_rng(21)
    for _ in range(50):
        le = _unit(rng.normal(size=3))
        tmp = _unit(rng.normal(size=3))
        n_sp = _unit(tmp - np.dot(tmp, le) * le)  # roughly perpendicular, like a real le/n_sp pair
        chord = _unit(rng.normal(size=3))
        sign_left = float(rng.choice([-1.0, 1.0]))

        got = ch._eta(chord, le, n_sp, sign_left)
        expected = _reference_eta(chord, le, n_sp, sign_left)
        assert abs(got - expected) < 1e-9, (got, expected)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Mirror symmetry: an L/R pair with mirrored ground-truth eta recovers
# mirrored-sign eta
# ---------------------------------------------------------------------------


def test_mirrored_lr_pair_recovers_mirrored_eta_sign():
    """Builds a wing_L/wing_R pair with `eta_R_deg = -eta_L_deg` (a
    physically mirrored chord pitch between the two sides) and checks the
    recovered etas mirror accordingly: each matches its own ground truth,
    and `eta_L + eta_R ~= 0`. This isolates the `sign_left` L/R convention
    (§4/§5) baked into `_le_sp_normal`/`_eta`, independent of any spatial
    reflection of a single wing's point cloud (which does NOT reduce to a
    clean eta sign flip -- the (le_sp_normal, sp_chord) reference frame is
    itself rebuilt from the reflected `le_dir`, so a same-side spatial mirror
    algebraically gives `eta -> 180 - eta`, not `-eta`; verified directly
    against `mock._chord_dir` while developing this test).
    """
    for eta_l in (25.0, -15.0, 40.0):
        gt = mock.default_ground_truth(eta_L_deg=eta_l, eta_R_deg=-eta_l)
        df, gt = mock.make_frame(gt, seed=7)
        frame = _true_body_frame(gt)

        wl = io_schema.get_part(df, "wing_L")
        wr = io_schema.get_part(df, "wing_R")
        res_l = ch.estimate_chord(wl, frame, "wing_L")
        res_r = ch.estimate_chord(wr, frame, "wing_R")

        assert abs(_angular_diff_deg(res_l.eta, eta_l)) < 3.0, (eta_l, res_l.eta)
        assert abs(_angular_diff_deg(res_r.eta, -eta_l)) < 3.0, (eta_l, res_r.eta)
        assert abs(res_l.eta + res_r.eta) < 1.0, (eta_l, res_l.eta, res_r.eta)


# ---------------------------------------------------------------------------
# scenario_reversal_contaminated: S4a must not crash; accuracy is S4b's job
# ---------------------------------------------------------------------------


def test_reversal_contaminated_runs_without_crashing():
    """S4a has no normal-consistency filtering or outlier rejection (that's
    S4b, §5 step 4), so a mislabeled/overlapping wing pair near stroke
    reversal is expected to degrade accuracy -- this test only asserts
    `estimate_chord` runs to completion and returns a finite number here, NOT
    that it's accurate. Measured baseline error at these default params
    (overlap=0.9, contam_frac=0.15, seed=0, eta_L=eta_R=25 deg per
    `default_ground_truth`'s own default): wing_L ~0.14 deg, wing_R ~0.97 deg
    -- recorded here as the number S4b's normal-consistency filtering should
    beat, not a target this test enforces.
    """
    df, gt = mock.scenario_reversal_contaminated(overlap=0.9, contam_frac=0.15, seed=0)
    frame = _true_body_frame(gt)

    for side, wing_gt in (("wing_L", gt.wing_L), ("wing_R", gt.wing_R)):
        wing_xyz = io_schema.get_part(df, side)
        result = ch.estimate_chord(wing_xyz, frame, side)

        assert math.isfinite(result.eta)
        assert abs(np.linalg.norm(result.chord) - 1.0) < 1e-6
        err = abs(_angular_diff_deg(result.eta, wing_gt.eta_deg))
        assert math.isfinite(err)  # S4b TODO: assert this shrinks vs. the baseline above


# ---------------------------------------------------------------------------
# robust / use_gaussian_normals: accepted but S4a always falls back to baseline
# ---------------------------------------------------------------------------


def test_robust_and_gaussian_normals_flags_accepted_without_orientation_data():
    """S4b implements `robust`/`use_gaussian_normals` (see `chord.py`), so
    they no longer exact-fallback to the baseline -- that comparison is
    `tests/test_s4b.py`'s job. This only checks the flags are safe to pass
    without `orientation`/`planarity` (S4b's real inputs): no crash, a
    finite/valid result, and no contaminant rejection (nothing to reject
    without per-point orientation data).
    """
    df, gt = mock.scenario_clean(seed=0)
    frame = _true_body_frame(gt)
    wing_xyz = io_schema.get_part(df, "wing_L")
    le = wa.estimate_leading_edge(wing_xyz, frame, "wing_L", rng=0)

    flagged = ch.estimate_chord(
        wing_xyz, frame, "wing_L", leading_edge=le, robust=True, use_gaussian_normals=True
    )

    assert math.isfinite(flagged.eta)
    assert abs(np.linalg.norm(flagged.chord) - 1.0) < 1e-9
    assert 0.0 <= flagged.chord_conf <= 1.0
    assert flagged.per_bin_chords.shape == (flagged.n_bins_used, 3)
    assert not flagged.rejected_mask.any()


# ---------------------------------------------------------------------------
# Sign-logic / bin-aggregation helpers are independently testable
# ---------------------------------------------------------------------------


def test_oriented_chord_axis_and_aggregate_chords_are_independently_testable():
    """§5's sign logic (`_oriented_chord_axis`) and bin aggregation
    (`_bin_chord_vectors` / `_aggregate_chords`) are separate helpers so S4b
    can swap the aggregator without touching sign/eta code -- exercise each
    directly, not just through `estimate_chord`.
    """
    df, gt = mock.scenario_clean(seed=0)
    frame = _true_body_frame(gt)
    wing_xyz = io_schema.get_part(df, "wing_L")
    le = wa.estimate_leading_edge(wing_xyz, frame, "wing_L")

    axis = ch._oriented_chord_axis(wing_xyz, le.plane_normal, le.le_dir, le.inlier_mask)
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-9
    assert abs(np.dot(axis, le.le_dir)) < 1e-9  # perpendicular to span

    per_bin = ch._bin_chord_vectors(
        wing_xyz, le.le_dir, axis, n_bins=ch._N_SPAN_BINS, min_bin_points=ch._MIN_BIN_POINTS
    )
    assert per_bin.shape[0] >= 3
    assert np.allclose(np.linalg.norm(per_bin, axis=1), 1.0, atol=1e-9)

    aggregated = ch._aggregate_chords(per_bin)
    assert abs(np.linalg.norm(aggregated) - 1.0) < 1e-9


def test_unknown_side_raises():
    df, gt = mock.scenario_clean(seed=0)
    frame = _true_body_frame(gt)
    wing_xyz = io_schema.get_part(df, "wing_L")
    raised = False
    try:
        ch.estimate_chord(wing_xyz, frame, "wing_bogus")
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
