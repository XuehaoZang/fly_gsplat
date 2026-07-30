"""S5 verification: labeled per-point CSVs -> per-frame kinematics table (`pipeline.py`).

References calc_kinematics.md §0/§1 (conventions, input/output contract) and
§6 (T4-adopted definitions), exercised end to end: S2 (`body_frame.py`) -> S3
(`wing_angles.py`) -> S4a (`chord.py`), chained by `pipeline.py`. No
multi-frame logic is tested here -- S5 is stateless (§ "no unwrap/smoothing";
see pipeline.py module docstring).

Full-pipeline tolerances (below) are looser than S2/S3/S4a's own isolated
tests (which drive S3/S4a with the *true* `BodyFrame`, per test_s3.py /
test_s4a.py's own docstrings): here `estimate_body_frame`'s ~1-3 deg
recovery noise floor feeds forward into the S3/S4a fits, so errors compound.
Measured empirically (10 seeds of `scenario_clean`, see below): worst case
~6 deg (roll), ~5 deg (phi); an 8 deg tolerance is used throughout with
margin.

Runnable both under pytest and standalone: `python test_s5.py`. The real
dataset smoke test at the bottom is non-strict by design (see its own
docstring) and never used as the pass/fail source of truth -- the tests
above it are.
"""
from __future__ import annotations

import math
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from postprocessing.kinematics import io_schema, mock, pipeline


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference `a - b` wrapped into `(-180, 180]`."""
    return ((a - b + 180.0) % 360.0) - 180.0


_ANGLE_TOL_DEG = 8.0
"""Chained S2->S3->S4a tolerance, see module docstring."""


# ---------------------------------------------------------------------------
# Mock-dataset helpers: real nested f<NNNN>/splatfacto-checkpoint/<ts>/ layout
# ---------------------------------------------------------------------------


def _write_mock_dataset(tmp_root: Path, frames: dict[int, pd.DataFrame], ts: str = "2026-07-20_120000") -> None:
    """Write `{frame_id: df}` out as `*_marked.csv` in the real layout S5's
    globbing expects (`f<NNNN>/splatfacto-checkpoint/<ts>/gaussian_features_f<NNNN>_marked.csv`).
    """
    for frame_id, df in frames.items():
        frame_dir = tmp_root / f"f{frame_id:04d}" / "splatfacto-checkpoint" / ts
        frame_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(frame_dir / f"gaussian_features_f{frame_id:04d}_marked.csv", index=False)


def _drop_part(df: pd.DataFrame, label: str) -> pd.DataFrame:
    return df[df["part_label"] != label].reset_index(drop=True)


def _shrink_part(df: pd.DataFrame, label: str, n_keep: int, rng: np.random.Generator) -> pd.DataFrame:
    idx_label = df.index[df["part_label"] == label].to_numpy()
    idx_other = df.index[df["part_label"] != label].to_numpy()
    keep = rng.choice(idx_label, size=min(n_keep, idx_label.size), replace=False)
    keep_idx = np.sort(np.concatenate([idx_other, keep]))
    return df.loc[keep_idx].reset_index(drop=True)


def _tmp_dir(name: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=f"test_s5_{name}_"))
    return d


# ---------------------------------------------------------------------------
# estimate_frame: clean scenario recovers ground truth (chained pipeline)
# ---------------------------------------------------------------------------


def test_estimate_frame_clean_recovers_ground_truth():
    df, gt = mock.scenario_clean(seed=0)
    config = pipeline.PipelineConfig(write_debug=False)
    row = pipeline.estimate_frame(df, 42, config)

    assert row["status"] == "ok"
    assert row["frame_id"] == 42

    assert abs(_angular_diff_deg(row["yaw"], gt.yaw_deg)) < _ANGLE_TOL_DEG, row["yaw"]
    assert abs(_angular_diff_deg(row["pitch"], gt.pitch_deg)) < _ANGLE_TOL_DEG, row["pitch"]
    assert abs(_angular_diff_deg(row["roll"], gt.roll_deg)) < _ANGLE_TOL_DEG, row["roll"]

    assert abs(_angular_diff_deg(row["phi_L"], 140.0)) < _ANGLE_TOL_DEG, row["phi_L"]
    assert abs(_angular_diff_deg(row["theta_L"], gt.wing_L.deviation_deg)) < _ANGLE_TOL_DEG, row["theta_L"]
    assert abs(_angular_diff_deg(row["eta_L"], gt.wing_L.eta_deg)) < _ANGLE_TOL_DEG, row["eta_L"]

    assert abs(_angular_diff_deg(row["phi_R"], 40.0)) < _ANGLE_TOL_DEG, row["phi_R"]
    assert abs(_angular_diff_deg(row["theta_R"], gt.wing_R.deviation_deg)) < _ANGLE_TOL_DEG, row["theta_R"]
    assert abs(_angular_diff_deg(row["eta_R"], gt.wing_R.eta_deg)) < _ANGLE_TOL_DEG, row["eta_R"]

    n_sp = np.array([row["sp_normal_x"], row["sp_normal_y"], row["sp_normal_z"]])
    assert abs(np.linalg.norm(n_sp) - 1.0) < 1e-6
    assert 0.0 <= row["chord_conf_L"] <= 1.0
    assert 0.0 <= row["chord_conf_R"] <= 1.0

    assert set(row.keys()) == set(io_schema.OUTPUT_COLUMNS) | {"status"}


# ---------------------------------------------------------------------------
# apply_if_keep actually filters (rejected rows change nothing)
# ---------------------------------------------------------------------------


def test_apply_if_keep_matches_physically_removed_rows():
    df, gt = mock.scenario_clean(seed=0)
    rng = np.random.default_rng(3)
    body_idx = df.index[df["part_label"] == "body"].to_numpy()
    drop_idx = rng.choice(body_idx, size=20, replace=False)

    df_flagged = df.copy()
    df_flagged.loc[drop_idx, "if_keep"] = False
    df_removed = df.drop(index=drop_idx).reset_index(drop=True)

    config = pipeline.PipelineConfig(write_debug=False)
    row_flagged = pipeline.estimate_frame(df_flagged, 0, config)
    row_removed = pipeline.estimate_frame(df_removed, 0, config)

    assert row_flagged == row_removed, (row_flagged, row_removed)


# ---------------------------------------------------------------------------
# Failure handling: never drop a frame; body angles survive a wing-only failure
# ---------------------------------------------------------------------------


def test_dropped_wing_yields_nan_row_with_status_and_does_not_raise():
    df, gt = mock.scenario_clean(seed=0)
    df_dropped = _drop_part(df, "wing_L")
    config = pipeline.PipelineConfig(min_points=10, write_debug=False)

    row = pipeline.estimate_frame(df_dropped, 7, config)

    assert row["frame_id"] == 7
    assert row["status"] == "wing_L:too_few_points"
    for col in io_schema.OUTPUT_COLUMNS:
        if col == "frame_id":
            continue
        assert math.isnan(row[col]), (col, row[col])


def test_empty_body_yields_nan_row_with_status():
    df, gt = mock.scenario_clean(seed=0)
    df_dropped = _drop_part(df, "body")
    config = pipeline.PipelineConfig(min_points=10, write_debug=False)

    row = pipeline.estimate_frame(df_dropped, 3, config)
    assert row["status"] == "body:too_few_points"
    assert math.isnan(row["yaw"])


def test_tiny_wing_point_count_leaves_body_angles_intact():
    """Shrink wing_L to fewer points than S3's leading-edge RANSAC needs
    (empirically: `estimate_leading_edge` needs ~3 populated span bins on
    each candidate edge, i.e. more than the handful left here) but still
    above `min_points`, so the frame-level pre-check passes and the failure
    surfaces inside the per-wing try/except instead -- body angles and
    wing_R must survive.
    """
    df, gt = mock.scenario_clean(seed=0)
    df_tiny = _shrink_part(df, "wing_L", n_keep=15, rng=np.random.default_rng(1))
    config = pipeline.PipelineConfig(min_points=10, write_debug=False)

    row = pipeline.estimate_frame(df_tiny, 11, config)

    assert row["status"].startswith("wing_L:"), row["status"]
    assert row["status"] != "wing_L:too_few_points"  # must be the RANSAC/bin failure, not the pre-check

    # Body angles survive.
    assert not math.isnan(row["yaw"])
    assert not math.isnan(row["pitch"])
    assert not math.isnan(row["roll"])
    assert not math.isnan(row["sp_normal_x"])

    # wing_R (untouched) survives.
    assert not math.isnan(row["phi_R"])
    assert not math.isnan(row["theta_R"])
    assert not math.isnan(row["eta_R"])
    assert not math.isnan(row["chord_conf_R"])

    # wing_L fields are NaN.
    assert math.isnan(row["phi_L"])
    assert math.isnan(row["theta_L"])
    assert math.isnan(row["eta_L"])
    assert math.isnan(row["chord_conf_L"])


# ---------------------------------------------------------------------------
# run_dataset: discovery, per-frame dispatch, sorted output, batch doesn't abort
# ---------------------------------------------------------------------------


def test_run_dataset_discovers_frames_sorted_with_correct_frame_ids():
    tmp_root = _tmp_dir("discover")
    try:
        seeds = {90: 0, 3: 1, 41: 2}  # deliberately out-of-order frame ids
        frames = {fid: mock.scenario_clean(seed=seed)[0] for fid, seed in seeds.items()}
        _write_mock_dataset(tmp_root, frames)

        config = pipeline.PipelineConfig(min_points=10, output_dir=tmp_root, write_debug=True)
        out_df = pipeline.run_dataset(tmp_root, config)

        assert list(out_df["frame_id"]) == sorted(seeds.keys())
        assert len(out_df) == 3
        assert (out_df["status"] == "ok").all(), out_df["status"].tolist()
        assert list(out_df.columns) == io_schema.OUTPUT_COLUMNS + ["status"]

        csv_path = tmp_root / f"kinematics_{tmp_root.name}.csv"
        assert csv_path.exists()
        reloaded = pd.read_csv(csv_path)
        assert list(reloaded["frame_id"]) == sorted(seeds.keys())
        assert list(reloaded.columns) == io_schema.OUTPUT_COLUMNS + ["status"]

        debug_path = tmp_root / f"kinematics_{tmp_root.name}_debug.pkl"
        assert debug_path.exists()
        with open(debug_path, "rb") as f:
            debug = pickle.load(f)
        assert set(debug.keys()) == set(seeds.keys())
        for fid in seeds:
            assert set(debug[fid].keys()) == set(pipeline.DEBUG_KEYS)
            assert debug[fid]["x_body"] is not None
            assert np.asarray(debug[fid]["per_bin_chords_L"]).shape[1] == 3
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_run_dataset_broken_frame_does_not_abort_batch():
    tmp_root = _tmp_dir("broken")
    try:
        df_clean, _ = mock.scenario_clean(seed=0)
        df_broken = _drop_part(mock.scenario_clean(seed=1)[0], "wing_R")
        _write_mock_dataset(tmp_root, {0: df_clean, 1: df_broken})

        config = pipeline.PipelineConfig(min_points=10, output_dir=tmp_root, write_debug=False)
        out_df = pipeline.run_dataset(tmp_root, config)

        assert len(out_df) == 2
        row0 = out_df[out_df["frame_id"] == 0].iloc[0]
        row1 = out_df[out_df["frame_id"] == 1].iloc[0]
        assert row0["status"] == "ok"
        assert row1["status"] == "wing_R:too_few_points"
        assert not math.isnan(row0["yaw"])
        assert math.isnan(row1["yaw"])
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_run_dataset_unparseable_csv_is_a_load_failure_not_a_crash():
    tmp_root = _tmp_dir("badcsv")
    try:
        frame_dir = tmp_root / "f0005" / "splatfacto-checkpoint" / "2026-07-20_120000"
        frame_dir.mkdir(parents=True)
        # Missing the mandatory `part_label` column -> io_schema.load_frame raises.
        pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}).to_csv(
            frame_dir / "gaussian_features_f0005_marked.csv", index=False
        )

        config = pipeline.PipelineConfig(min_points=10, output_dir=tmp_root, write_debug=False)
        out_df = pipeline.run_dataset(tmp_root, config)

        assert len(out_df) == 1
        assert out_df.iloc[0]["frame_id"] == 5
        assert out_df.iloc[0]["status"].startswith("load:")
        assert math.isnan(out_df.iloc[0]["yaw"])
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_run_dataset_empty_root_returns_empty_frame_no_crash():
    tmp_root = _tmp_dir("empty")
    try:
        config = pipeline.PipelineConfig(min_points=10, output_dir=tmp_root, write_debug=False)
        out_df = pipeline.run_dataset(tmp_root, config)
        assert len(out_df) == 0
        assert list(out_df.columns) == io_schema.OUTPUT_COLUMNS + ["status"]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Real-dataset smoke test -- report only, never the pass/fail source of truth
# ---------------------------------------------------------------------------


def test_smoke_real_dataset():
    """Runs `run_dataset` against the real
    `outputs/ctrl_009_002_8groups_100frames/G2b_G9` root as a smoke test.
    Writes outputs to a scratch directory (never into the tracked `outputs/`
    tree). Skips cleanly (prints, does not fail) if the root or any frames
    are absent in this environment -- the synthetic tests above are the
    actual source of truth for correctness, per the task spec.
    """
    real_root = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
    if not real_root.exists():
        print(f"SKIP  real dataset root not found: {real_root}")
        return

    scratch_out = _tmp_dir("real_smoke")
    try:
        config = pipeline.PipelineConfig(
            min_points=10,
            output_dir=scratch_out,
            write_debug=False,
            frame_glob="f*/splatfacto-checkpoint/*/*_labeled.csv",
        )
        out_df = pipeline.run_dataset(real_root, config)

        print(f"\nreal dataset smoke test: root={real_root}")
        print(f"  frames found: {len(out_df)}")
        if len(out_df) == 0:
            print("  SKIP  no *_labeled.csv frames discovered")
            return
        print("  status breakdown:")
        for status, count in out_df["status"].value_counts().items():
            print(f"    {status}: {count}")
        print("  first rows:")
        print(out_df.head().to_string())

        assert isinstance(out_df, pd.DataFrame)
        assert list(out_df.columns) == io_schema.OUTPUT_COLUMNS + ["status"]
    finally:
        shutil.rmtree(scratch_out, ignore_errors=True)


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
