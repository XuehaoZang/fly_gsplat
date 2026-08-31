"""Score `segment.segment_frame_kmeans_v2` + `pipeline`'s T4 estimator against
`scene.py` ground truth for one frame -- the "does the whole T3+T4 chain
actually work" harness the rest of this package builds data/labeling for.

Every frame is run through **three** conditions, so a plotted discrepancy
can be attributed to segmentation (T3) vs angle estimation (T4) instead of
lumping both into one number:

- `row_t3` / `debug_t3`: `segment.segment_frame_kmeans_v2`'s *predicted* labels fed
  into T4 -- the real end-to-end pipeline a real dataset would go through.
- `row_t4_only` / `debug_t4_only`: the scene's *exact* ground-truth labels
  (never estimated) fed into the same T4 code -- isolates T4's own
  estimator error from segmentation error.
- `row_gt`: the analytic ground-truth angles themselves (`scene.py`'s
  `FrameGroundTruth`, never touched by any estimator) -- the reference line
  both of the above are trying to reproduce.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics import io_schema  # noqa: E402
from postprocessing.kinematics import pipeline as pl  # noqa: E402
from postprocessing.kinematics.simulate_gt import segment  # noqa: E402
from postprocessing.kinematics.simulate_gt.scene import FrameGroundTruth  # noqa: E402


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _wrap180(delta_deg: float) -> float:
    """Shortest signed distance for a difference of two `(-180, 180]`-valued
    angles (e.g. `yaw_pred - yaw_gt`), same convention as
    `diagnostics.circular_delta_deg` but for a single scalar difference
    rather than a consecutive-frame series."""
    return ((delta_deg + 180.0) % 360.0) - 180.0


def _row_errors(row: dict, debug: dict, frame_gt: FrameGroundTruth) -> dict:
    """Per-metric absolute error of one pipeline output `row`/`debug`
    (`pipeline._estimate_frame_impl`'s return) against `frame_gt`. Shared by
    both the T3-predicted and the exact-segmentation-T4-only conditions."""
    errors: dict[str, float] = {}

    if np.isfinite(row["yaw"]):
        errors["yaw_deg"] = abs(_wrap180(row["yaw"] - frame_gt.yaw_deg))
    if np.isfinite(row["pitch"]):
        errors["pitch_deg"] = abs(row["pitch"] - frame_gt.pitch_deg)
    if np.isfinite(row["roll"]):
        errors["roll_deg"] = abs(_wrap180(row["roll"] - frame_gt.roll_deg))

    if debug["x_body"] is not None:
        errors["x_body_deg"] = _angle_between_deg(debug["x_body"], frame_gt.x_body)
        errors["y_body_deg"] = _angle_between_deg(debug["y_body"], frame_gt.y_body)
        errors["z_body_deg"] = _angle_between_deg(debug["z_body"], frame_gt.z_body)
        errors["n_sp_deg"] = _angle_between_deg(debug["n_sp"], frame_gt.n_sp)

    for suffix in ("L", "R"):
        phi_gt = frame_gt.phi_L_deg if suffix == "L" else frame_gt.phi_R_deg
        theta_gt = frame_gt.theta_L_deg if suffix == "L" else frame_gt.theta_R_deg
        eta_gt = frame_gt.eta_L_deg if suffix == "L" else frame_gt.eta_R_deg
        span_gt = frame_gt.span_L if suffix == "L" else frame_gt.span_R
        chord_gt = frame_gt.chord_L if suffix == "L" else frame_gt.chord_R

        phi_pred = row[f"phi_{suffix}"]
        theta_pred = row[f"theta_{suffix}"]
        eta_pred = row[f"eta_{suffix}"]
        if np.isfinite(phi_pred):
            errors[f"phi_{suffix}_deg"] = abs(_wrap180(phi_pred - phi_gt))
        if np.isfinite(theta_pred):
            errors[f"theta_{suffix}_deg"] = abs(theta_pred - theta_gt)
        if np.isfinite(eta_pred):
            errors[f"eta_{suffix}_deg"] = abs(_wrap180(eta_pred - eta_gt))

        span_dbg = debug.get(f"span_{suffix}")
        if span_dbg is not None:
            errors[f"span_{suffix}_deg"] = _angle_between_deg(span_dbg, span_gt)
        chord_dbg = debug.get(f"chord_{suffix}")
        if chord_dbg is not None:
            errors[f"chord_{suffix}_deg"] = _angle_between_deg(chord_dbg, chord_gt)

    return errors


def _gt_row(frame_gt: FrameGroundTruth) -> dict:
    """`frame_gt`'s angles, keyed the same as `io_schema.OUTPUT_COLUMNS` so
    plotting code can treat `row_t3`/`row_t4_only`/`row_gt` uniformly."""
    return dict(
        yaw=frame_gt.yaw_deg, pitch=frame_gt.pitch_deg, roll=frame_gt.roll_deg,
        phi_L=frame_gt.phi_L_deg, theta_L=frame_gt.theta_L_deg, eta_L=frame_gt.eta_L_deg,
        phi_R=frame_gt.phi_R_deg, theta_R=frame_gt.theta_R_deg, eta_R=frame_gt.eta_R_deg,
    )


@dataclass
class FrameEvalResult:
    frame_id: int
    status_t3: str
    status_t4_only: str
    seg_accuracy: float
    seg_confusion: pd.DataFrame | None
    row_t3: dict
    row_t4_only: dict
    row_gt: dict
    errors_t3: dict = field(default_factory=dict)
    errors_t4_only: dict = field(default_factory=dict)


def evaluate_frame(
    df_unlabeled: pd.DataFrame,
    frame_gt: FrameGroundTruth,
    config: pl.PipelineConfig | None = None,
    segment_fn=None,
) -> FrameEvalResult:
    """Run all three conditions (see module docstring) on one frame and
    compare each against `frame_gt`. Never raises: a segmentation failure
    only degrades the T3 condition (`status_t3="segment:<reason>"`,
    `seg_accuracy=nan`, `row_t3` all-NaN) -- the exact-segmentation
    T4-only condition does not depend on `segment.segment_frame_kmeans_v2` at all and
    always runs, matching `pipeline.py`'s own never-drop-a-frame convention.

    `segment_fn` defaults to `segment.segment_frame_kmeans_v2` (unchanged
    behavior for every existing caller). Pass a different single-argument
    callable (`df_unlabeled -> pd.Series` of `part_label`) to score an
    alternative segmenter through the same T3/T4-only/GT error-report path
    -- e.g. a `functools.partial` binding `segment.
    segment_frame_kmeans_motion_fusion`'s extra `window_xyz_by_frame`/
    `center_frame_idx` arguments down to this one-argument shape.
    """
    config = config if config is not None else pl.PipelineConfig()
    segment_fn = segment_fn if segment_fn is not None else segment.segment_frame_kmeans_v2
    gt_label = pd.Series(frame_gt.part_label, index=df_unlabeled.index)

    try:
        pred_label = segment_fn(df_unlabeled)
        seg_accuracy = float((pred_label.to_numpy() == gt_label.to_numpy()).mean())
        seg_confusion = pd.crosstab(gt_label, pred_label, rownames=["gt"], colnames=["pred"])

        df_t3 = df_unlabeled.copy()
        df_t3["part_label"] = pred_label
        row_t3, debug_t3 = pl._estimate_frame_impl(df_t3, frame_gt.frame_id, config)
        status_t3 = row_t3["status"]
        errors_t3 = _row_errors(row_t3, debug_t3, frame_gt)
    except Exception as e:  # noqa: BLE001
        seg_accuracy = float("nan")
        seg_confusion = None
        status_t3 = f"segment:{e}"
        row_t3 = io_schema.empty_output_row(frame_gt.frame_id).to_dict()
        errors_t3 = {}

    df_t4_only = df_unlabeled.copy()
    df_t4_only["part_label"] = gt_label
    row_t4_only, debug_t4_only = pl._estimate_frame_impl(df_t4_only, frame_gt.frame_id, config)
    status_t4_only = row_t4_only["status"]
    errors_t4_only = _row_errors(row_t4_only, debug_t4_only, frame_gt)

    row_gt = _gt_row(frame_gt)

    return FrameEvalResult(
        frame_id=frame_gt.frame_id,
        status_t3=status_t3,
        status_t4_only=status_t4_only,
        seg_accuracy=seg_accuracy,
        seg_confusion=seg_confusion,
        row_t3=row_t3,
        row_t4_only=row_t4_only,
        row_gt=row_gt,
        errors_t3=errors_t3,
        errors_t4_only=errors_t4_only,
    )
