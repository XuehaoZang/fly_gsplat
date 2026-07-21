"""S4b verification: Gaussian-normal contaminant rejection + robust
aggregation/confidence (`chord.py`'s `use_gaussian_normals=True`/`robust=True`
paths), measured against the S4a baseline (both flags `False`).

References calc_kinematics.md §5 step 4 (normal-consistency filtering) and
step 3 (robust/weighted aggregation). Uses `mock.py`'s forward-constructed
ground truth plus its `mock_contaminant`/`mock_bad_orientation` bookkeeping
columns (S4b additions, see `mock.py`) to score rejection precision/recall
directly, not just downstream eta error.

Runnable both under pytest and standalone: `python test_s4b.py`.
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

_ORIENT_COLS = ["orientation_x", "orientation_y", "orientation_z"]


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference `a - b` wrapped into `(-180, 180]`."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _true_body_frame(gt: mock.GroundTruth) -> bf.BodyFrame:
    """Exact ground-truth `BodyFrame`, no fitting -- copied from
    test_s3.py/test_s4a.py's own convention (kept per-file, see those files).
    """
    x_body, y_body, z_body = mock.body_axes(gt)
    n_sp = mock.stroke_plane_normal(gt)
    return bf.BodyFrame(
        x_body=x_body, y_body=y_body, z_body=z_body, n_sp=n_sp,
        yaw=gt.yaw_deg, pitch=gt.pitch_deg, roll=gt.roll_deg,
        hinge_L=gt.wing_L.root, hinge_R=gt.wing_R.root, body_cm=gt.body_center,
    )


def _wing_inputs(df, side: str):
    """`(wing_xyz, orientation, planarity)`, row-aligned (S4b's new
    `estimate_chord` kwargs) -- the same row selection `io_schema.get_part`
    uses, via `get_part_columns`."""
    wing_xyz = io_schema.get_part(df, side)
    orientation = io_schema.get_part_columns(df, side, _ORIENT_COLS)
    planarity = io_schema.get_part_columns(df, side, ["planarity"])[:, 0]
    return wing_xyz, orientation, planarity


def _both(wing_xyz, frame, side, orientation, planarity):
    """`(baseline, enhanced)` `ChordResult`s from the same inputs, same call
    site modulo the flags -- the comparison this whole test file is about.
    """
    baseline = ch.estimate_chord(wing_xyz, frame, side)
    enhanced = ch.estimate_chord(
        wing_xyz, frame, side, robust=True, use_gaussian_normals=True,
        orientation=orientation, planarity=planarity,
    )
    return baseline, enhanced


# ---------------------------------------------------------------------------
# Headline: eta error vs. contamination fraction, baseline vs. enhanced
# ---------------------------------------------------------------------------


def test_headline_contamination_sweep_enhanced_beats_baseline():
    """`scenario_reversal_contaminated` with mirrored (`+25`/`-25` deg) eta on
    the two wings -- modeling asynchronous supination/pronation at a real
    stroke-reversal moment -- so the injected wing_L points landing in
    wing_R's cloud carry a genuinely different local normal (~54 deg apart,
    not just a different xyz region). `wing_R` is the contaminated side
    (`scenario_reversal_contaminated` only ever relabels wing_L points into
    wing_R, never the reverse).
    """
    overlap = 0.9
    contam_fracs = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
    side = "wing_R"

    rows = []
    for contam_frac in contam_fracs:
        df, gt = mock.scenario_reversal_contaminated(
            overlap=overlap, contam_frac=contam_frac, seed=0, eta_L_deg=25.0, eta_R_deg=-25.0
        )
        frame = _true_body_frame(gt)
        wing_xyz, orientation, planarity = _wing_inputs(df, side)
        baseline, enhanced = _both(wing_xyz, frame, side, orientation, planarity)

        base_err = abs(_angular_diff_deg(baseline.eta, gt.wing_R.eta_deg))
        enh_err = abs(_angular_diff_deg(enhanced.eta, gt.wing_R.eta_deg))
        rows.append((contam_frac, base_err, enh_err))

    print(f"\n{'contam_frac':>12} {'baseline_err_deg':>18} {'enhanced_err_deg':>18}")
    for contam_frac, base_err, enh_err in rows:
        print(f"{contam_frac:12.2f} {base_err:18.3f} {enh_err:18.3f}")

    # Clean (contam_frac=0.0): no regression, both near the true-eta noise floor.
    _, base_err0, enh_err0 = rows[0]
    assert base_err0 < 3.0, rows[0]
    assert enh_err0 < 3.0, rows[0]

    # Nonzero contamination: baseline actually degrades (sanity the scenario
    # stresses it), and the enhanced path is meaningfully -- not just
    # marginally -- better at every contaminated level.
    for contam_frac, base_err, enh_err in rows[1:]:
        assert base_err > 3.0, (contam_frac, base_err)  # baseline genuinely hurt
        assert enh_err < base_err * 0.5, (contam_frac, base_err, enh_err)
        assert enh_err < 2.0, (contam_frac, enh_err)  # enhanced stays near the noise floor


# ---------------------------------------------------------------------------
# scenario_clean: no regression, enhanced ~= baseline
# ---------------------------------------------------------------------------


def test_clean_scenario_enhanced_matches_baseline_both_wings():
    df, gt = mock.scenario_clean(seed=0)
    frame = _true_body_frame(gt)

    for side, wing_gt in (("wing_L", gt.wing_L), ("wing_R", gt.wing_R)):
        wing_xyz, orientation, planarity = _wing_inputs(df, side)
        baseline, enhanced = _both(wing_xyz, frame, side, orientation, planarity)

        assert abs(_angular_diff_deg(baseline.eta, wing_gt.eta_deg)) < 3.0, (side, baseline.eta)
        assert abs(_angular_diff_deg(enhanced.eta, wing_gt.eta_deg)) < 3.0, (side, enhanced.eta)
        assert abs(baseline.eta - enhanced.eta) < 1.0, (side, baseline.eta, enhanced.eta)
        assert not enhanced.rejected_mask.any(), (side, enhanced.rejected_mask.sum())


# ---------------------------------------------------------------------------
# Contaminant rejection: precision/recall against mock's own bookkeeping
# ---------------------------------------------------------------------------


def test_contaminant_rejection_precision_recall():
    df, gt = mock.scenario_reversal_contaminated(
        overlap=0.9, contam_frac=0.2, seed=0, eta_L_deg=25.0, eta_R_deg=-25.0
    )
    frame = _true_body_frame(gt)
    side = "wing_R"
    wing_xyz, orientation, planarity = _wing_inputs(df, side)
    true_contaminant = io_schema.get_part_columns(df, side, ["mock_contaminant"])[:, 0].astype(bool)

    _, enhanced = _both(wing_xyz, frame, side, orientation, planarity)
    rejected = enhanced.rejected_mask

    assert true_contaminant.sum() > 0  # sanity: this scenario actually injected some
    tp = int((rejected & true_contaminant).sum())
    fp = int((rejected & ~true_contaminant).sum())
    fn = int((~rejected & true_contaminant).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    assert precision >= 0.95, (precision, tp, fp, fn)
    assert recall >= 0.95, (recall, tp, fp, fn)


# ---------------------------------------------------------------------------
# Low-planarity robustness: soft-weighting vs. naive unweighted normals
# ---------------------------------------------------------------------------


def test_low_planarity_soft_weighting_beats_unweighted_average():
    """`scenario_noisy_orientation` replaces most of one wing's `orientation`
    with random directions (and low `planarity`), xyz/ground-truth
    untouched. The planarity-trust weighting in
    `_robust_wing_normal_and_survivors` should recover the true wing normal
    almost exactly (it excludes the untrusted points outright), while a
    naive unweighted mean over *all* points' local normals -- the thing S4b
    replaces -- is measurably worse, and increasingly so as the corrupted
    fraction grows.
    """
    side = "wing_L"

    for bad_frac, min_naive_err_deg in ((0.3, 0.15), (0.85, 1.0)):
        df, gt = mock.scenario_noisy_orientation(bad_frac=bad_frac, seed=0)
        frame = _true_body_frame(gt)
        wing_xyz, orientation, planarity = _wing_inputs(df, side)

        le = wa.estimate_leading_edge(wing_xyz, frame, side, rng=0)
        true_normal = np.cross(
            gt.wing_L.span_dir,
            mock._chord_dir(gt.wing_L.span_dir, mock.stroke_plane_normal(gt), gt.wing_L.eta_deg, -1.0),
        )
        true_normal /= np.linalg.norm(true_normal)

        n_w, _survivors = ch._robust_wing_normal_and_survivors(orientation, planarity, le.plane_normal)
        weighted_err = math.degrees(math.acos(np.clip(abs(np.dot(n_w, true_normal)), -1.0, 1.0)))

        oriented_all = ch.geo.orient_to_reference(orientation, le.plane_normal)
        naive_mean = oriented_all.mean(axis=0)
        naive_mean /= np.linalg.norm(naive_mean)
        naive_err = math.degrees(math.acos(np.clip(abs(np.dot(naive_mean, true_normal)), -1.0, 1.0)))

        assert weighted_err < 0.1, (bad_frac, weighted_err)
        assert naive_err > min_naive_err_deg, (bad_frac, naive_err)
        assert weighted_err < naive_err, (bad_frac, weighted_err, naive_err)


# ---------------------------------------------------------------------------
# chord_conf: lower on contaminated/degraded frames than on clean ones
# ---------------------------------------------------------------------------


def test_chord_conf_drops_on_contaminated_frame():
    df_clean, gt_clean = mock.scenario_clean(seed=0)
    frame_clean = _true_body_frame(gt_clean)
    wing_xyz_c, orient_c, plan_c = _wing_inputs(df_clean, "wing_R")
    _, enhanced_clean = _both(wing_xyz_c, frame_clean, "wing_R", orient_c, plan_c)

    df_contam, gt_contam = mock.scenario_reversal_contaminated(
        overlap=0.9, contam_frac=0.3, seed=0, eta_L_deg=25.0, eta_R_deg=-25.0
    )
    frame_contam = _true_body_frame(gt_contam)
    wing_xyz_x, orient_x, plan_x = _wing_inputs(df_contam, "wing_R")
    _, enhanced_contam = _both(wing_xyz_x, frame_contam, "wing_R", orient_x, plan_x)

    assert 0.0 <= enhanced_contam.chord_conf <= 1.0
    assert 0.0 <= enhanced_clean.chord_conf <= 1.0
    assert enhanced_contam.chord_conf < enhanced_clean.chord_conf, (
        enhanced_contam.chord_conf, enhanced_clean.chord_conf
    )


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
