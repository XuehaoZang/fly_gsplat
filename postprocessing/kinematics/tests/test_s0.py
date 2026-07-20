"""S0 verification: I/O contract (`io_schema.py`) + mock generator (`mock.py`).

References calc_kinematics.md §0 (units, up-vector), §1 (input/output
contract), §4/§5 (phi/theta/eta, chord/eta conventions used by the mock's
forward construction). No angle estimation is exercised here — only that the
mock's own ground truth is self-consistent with the formulas it was built
from, checked directly (never via a fitted estimator).

Runnable both under pytest and standalone: `python test_s0.py`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from postprocessing.kinematics import io_schema, mock

SCENARIOS = {
    "clean": lambda: mock.scenario_clean(seed=0),
    "reversal_contaminated": lambda: mock.scenario_reversal_contaminated(
        overlap=0.9, contam_frac=0.2, seed=1
    ),
    "noisy": lambda: mock.scenario_noisy(pos_noise_std=5e-5, density_imbalance=0.6, seed=2),
}


def test_scenarios_pass_schema_validation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, build in SCENARIOS.items():
            df, gt = build()
            csv_path = tmp_path / f"s0_{name}.csv"
            df.to_csv(csv_path, index=False)
            loaded = io_schema.load_frame(csv_path)
            assert list(loaded.columns) == list(df.columns), name
            assert len(loaded) == len(df), name


def test_missing_mandatory_column_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        df, _ = mock.scenario_clean()
        bad = df.drop(columns=["part_label"])
        csv_path = tmp_path / "s0_bad.csv"
        bad.to_csv(csv_path, index=False)
        try:
            io_schema.load_frame(csv_path)
            raised = False
        except ValueError as e:
            raised = True
            assert "part_label" in str(e)
        assert raised


def test_part_label_values_and_units():
    for name, build in SCENARIOS.items():
        df, gt = build()
        labels = set(df["part_label"].unique())
        assert labels <= io_schema.PART_LABELS, f"{name}: unexpected labels {labels}"
        assert "body" in labels and "wing_L" in labels

        body_extent = df.loc[df["part_label"] == "body", ["x", "y", "z"]].to_numpy()
        span = body_extent.max(axis=0) - body_extent.min(axis=0)
        # body length ~2.5mm: overall extent should be O(1e-3) m, not cm/mm-as-if-mm
        assert 1e-4 < np.linalg.norm(span) < 1e-2, f"{name}: implausible body scale {span}"

        wing_l = df.loc[df["part_label"] == "wing_L", ["x", "y", "z"]].to_numpy()
        root = gt.wing_L.root
        reach = np.linalg.norm(wing_l - root, axis=1).max()
        assert 1e-4 < reach < 1e-2, f"{name}: implausible wing_L reach {reach}"


def test_get_part_filters_if_keep():
    df, gt = mock.scenario_clean()
    df = df.copy()
    wl_idx = df.index[df["part_label"] == "wing_L"].to_numpy()
    dropped = wl_idx[: len(wl_idx) // 2]
    df.loc[dropped, "if_keep"] = False

    all_pts = io_schema.get_part(df, "wing_L", apply_if_keep=False)
    kept_pts = io_schema.get_part(df, "wing_L", apply_if_keep=True)
    assert all_pts.shape[0] == len(wl_idx)
    assert kept_pts.shape[0] == len(wl_idx) - len(dropped)

    assert io_schema.wingL_xyz(df).shape[0] == kept_pts.shape[0]
    assert io_schema.body_xyz(df).shape[1] == 3
    assert io_schema.wingR_xyz(df).shape[1] == 3


def test_ground_truth_span_direction_is_self_consistent():
    """Direct-construction check (no estimator): the wing's true span
    direction reproduces its own ground-truth deviation (theta) via §4's
    formula, and the farthest wing point from the root lies (near) along
    `span_dir` at ~ the true wing length.
    """
    for name, build in SCENARIOS.items():
        df, gt = build()
        n_sp = mock.stroke_plane_normal(gt)
        for side, wing_gt in (("wing_L", gt.wing_L), ("wing_R", gt.wing_R)):
            assert abs(np.linalg.norm(wing_gt.span_dir) - 1.0) < 1e-9, side

            theta_direct = mock.deviation_of(wing_gt.span_dir, n_sp)
            assert abs(theta_direct - wing_gt.deviation_deg) < 1e-6, (name, side)

            pts = df.loc[df["part_label"] == side, ["x", "y", "z"]].to_numpy()
            if pts.shape[0] == 0:
                continue
            rel = pts - wing_gt.root
            proj = rel @ wing_gt.span_dir
            tip_idx = np.argmax(proj)
            tip_rel = rel[tip_idx]
            cos_angle = np.dot(tip_rel, wing_gt.span_dir) / np.linalg.norm(tip_rel)
            assert cos_angle > 0.9, (name, side, cos_angle)
            assert 0.5 * wing_gt.length_m < proj[tip_idx] < 1.2 * wing_gt.length_m, (name, side)


def test_reversal_contamination_relabels_without_moving_points():
    n_wing = 400  # mock.make_frame's default
    contam_frac = 0.2
    df, gt = mock.scenario_reversal_contaminated(overlap=0.9, contam_frac=contam_frac, seed=1)
    labels = df["part_label"].value_counts()
    expected_contam = round(contam_frac * n_wing)
    assert labels["wing_R"] == n_wing + expected_contam
    assert labels["wing_L"] == n_wing - expected_contam
    assert df["part_label"].isin(io_schema.PART_LABELS).all()


def test_csv_roundtrip_preserves_columns_and_dtypes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        df, gt = mock.scenario_clean()
        csv_path = tmp_path / "s0_roundtrip.csv"
        df.to_csv(csv_path, index=False)
        loaded = io_schema.load_frame(csv_path)

    assert list(loaded.columns) == io_schema.INPUT_COLUMNS
    for col in ("x", "y", "z", "opacity", "scale_phys_0", "local_density"):
        assert pd.api.types.is_float_dtype(loaded[col]), col
    assert loaded["part_label"].dtype == object
    assert set(loaded["part_label"].unique()) <= io_schema.PART_LABELS
    assert pd.api.types.is_bool_dtype(loaded["if_keep"])


def test_empty_output_row_and_columns():
    row = io_schema.empty_output_row(frame_id=42)
    d = row.to_dict()
    assert list(d.keys()) == io_schema.OUTPUT_COLUMNS
    assert d["frame_id"] == 42
    assert np.isnan(d["yaw"]) and np.isnan(d["chord_conf_L"])


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
