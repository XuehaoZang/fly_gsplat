"""Run `continuity.compute_continuous_x_body` over the 640-frame dataset and
dump one intermediate CSV that all `diag/` scripts read (so the PCA/
continuity math only runs once, not once per diagnostic).

Only reads existing `*_labeled.csv` (T3 output) — does not re-run motion/T3
and does not modify anything under `postprocessing/labeling/`. Frame
discovery (`discover_labeled_frames`) and the lab `UP` vector are imported
read-only from that package, same as the earlier diagnostic scripts
(`flip_root_cause_check.py` etc.) already do.

For each frame we compute *two* parallel `x_body` series from the same PCA:
- "before": `orient_to_reference(major_axis, UP)` independently every frame
  — the existing `body_frame.py::estimate_body_frame` behavior, no
  continuity.
- "after": `continuity.compute_continuous_x_body`, seeded from the previous
  *successful* frame's "after" value.

Reset policy (hard, not a CLI flag): whenever the gap between the current
frame_id and the last successfully-processed frame_id is >1, or there is no
previous successful frame yet, the "after" chain restarts from `None`
(re-anchored to UP). A frame is "failed" (and skipped, not counted in the
chain) when it has fewer than `MIN_BODY_POINTS` body points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.calc_kinematics import DEFAULT_DATASET_ROOT  # noqa: E402
from postprocessing.kinematics import geometry as geo  # noqa: E402
from postprocessing.kinematics.correct_body_axis.continuity import (  # noqa: E402
    compute_continuous_x_body,
)
from postprocessing.labeling.labeling import UP  # noqa: E402
from postprocessing.labeling.motion.diag.identity_flip_stats import (  # noqa: E402
    discover_labeled_frames,
    load_labeled,
)

OUT_DIR = Path(__file__).resolve().parent / "diag"
SEQUENCE_CSV = OUT_DIR / "sequence_body_axis.csv"

MIN_BODY_POINTS = 10
"""Below this many body points (part_label=="body", if_keep=True), a frame's
PCA is considered too underdetermined to trust at all (not just sign-
ambiguous) and the frame is marked failed / skipped, per task spec §2. 640
frames' body clouds are normally in the hundreds of points (see
flip_point_cloud_diag.py's n_body_points stats), so 10 is a conservative
"basically nothing there" floor, not a tuned threshold."""


def frame_body_xyz(df: pd.DataFrame) -> np.ndarray:
    df_kept = df[df["if_keep"].astype(bool)] if "if_keep" in df.columns else df
    return df_kept.loc[df_kept["part_label"] == "body", ["x", "y", "z"]].to_numpy(dtype=float)


def build_sequence(root: Path = DEFAULT_DATASET_ROOT) -> pd.DataFrame:
    frame_paths = discover_labeled_frames(root)
    frame_idxs = sorted(frame_paths)
    print(f"[build_sequence] {len(frame_idxs)} labeled frames under {root}: "
          f"f{frame_idxs[0]:04d}..f{frame_idxs[-1]:04d}")

    rows: list[dict] = []
    x_body_after_prev: np.ndarray | None = None
    x_body_before_prev: np.ndarray | None = None
    last_success_frame_id: int | None = None
    n_failed = 0

    for frame_id in frame_idxs:
        df = load_labeled(frame_paths[frame_id])
        body_xyz = frame_body_xyz(df)

        if len(body_xyz) < MIN_BODY_POINTS:
            n_failed += 1
            rows.append({
                "frame_id": frame_id, "n_body_points": len(body_xyz), "failed": True,
                "frame_id_gap": np.nan, "eigval_ratio": np.nan, "method_after": "failed",
                "x_before_x": np.nan, "x_before_y": np.nan, "x_before_z": np.nan,
                "x_after_x": np.nan, "x_after_y": np.nan, "x_after_z": np.nan,
                "angle_to_prev_deg_before": np.nan, "angle_to_prev_deg_after": np.nan,
            })
            # 失败帧不更新 prev/last_success -> 下一帧的 gap 用最后一次成功帧计算，
            # continuity 链也从失败帧之前的最后一个成功帧继续算 gap。
            continue

        gap = np.nan if last_success_frame_id is None else frame_id - last_success_frame_id
        if last_success_frame_id is None or gap > 1:
            x_body_after_prev = None  # 硬性重置，不做成可调参数

        eigvals, eigvecs, _centroid = geo.weighted_pca(body_xyz)
        eigval_ratio = float(eigvals[-2] / eigvals[-1]) if eigvals[-1] > 0 else float("nan")
        x_body_before = geo.orient_to_reference(eigvecs[:, -1], UP)

        angle_before = float("nan")
        if x_body_before_prev is not None:
            cos_a = float(np.clip(np.dot(x_body_before, x_body_before_prev), -1.0, 1.0))
            angle_before = float(np.degrees(np.arccos(cos_a)))

        x_body_after, diag = compute_continuous_x_body(body_xyz, x_body_after_prev, up=UP)

        rows.append({
            "frame_id": frame_id, "n_body_points": len(body_xyz), "failed": False,
            "frame_id_gap": gap, "eigval_ratio": eigval_ratio, "method_after": diag["method"],
            "x_before_x": x_body_before[0], "x_before_y": x_body_before[1], "x_before_z": x_body_before[2],
            "x_after_x": x_body_after[0], "x_after_y": x_body_after[1], "x_after_z": x_body_after[2],
            "angle_to_prev_deg_before": angle_before,
            "angle_to_prev_deg_after": diag["angle_to_prev_deg"],
        })

        x_body_after_prev = x_body_after
        x_body_before_prev = x_body_before
        last_success_frame_id = frame_id

    print(f"[build_sequence] {n_failed}/{len(frame_idxs)} frames failed "
          f"(n_body_points < {MIN_BODY_POINTS})")
    return pd.DataFrame(rows)


def run(root: Path = DEFAULT_DATASET_ROOT) -> pd.DataFrame:
    seq_df = build_sequence(root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(SEQUENCE_CSV, index=False)
    print(f"[build_sequence] csv -> {SEQUENCE_CSV} ({len(seq_df)} rows)")
    method_counts = seq_df.loc[~seq_df["failed"], "method_after"].value_counts()
    print(f"[build_sequence] method_after counts:\n{method_counts.to_string()}")
    return seq_df


if __name__ == "__main__":
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_ROOT
    run(dataset_root)
