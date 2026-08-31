"""Segmentation-only check for `segment.segment_frame_motion` (the
cross-frame voxel-density T3 method, `HALF_WINDOW=36` frames each side) on
the step-2 100-frame flapping sequence.

Only frames with a *full* window available -- `[HALF_WINDOW, n_frames-1-
HALF_WINDOW]` = `[36, 63]` for a 100-frame sequence -- have a well-defined
prediction; frame 0 (or any frame within 36 of either edge) is not a valid
test case for this method (see chat history: testing frame 0 for the
window-based method was flagged as invalid for exactly this reason). This
script only scores/visualizes frames in that valid range, and only checks
segmentation accuracy -- it does not run T4 on top (see `run_step1.py`/
`run_step2.py` for the full segment+T4 comparison, using the single-frame
`segment_frame_kmeans_v2` method instead).

Run: python -m postprocessing.kinematics.simulate_gt.run_step2_motion
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics.simulate_gt import plots, segment  # noqa: E402
from postprocessing.kinematics.simulate_gt.scene import scenario_step2_flapping  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"
N_FRAMES = 100
HALF_WINDOW = segment.HALF_WINDOW
VIZ_FRAME_IDS = None
"""Which valid center frames get a multi-view point-cloud render -- filled
in `main()` with (worst, median, best) by accuracy once results are in, so
the plots are chosen by what's actually interesting rather than fixed
up-front."""


def main() -> None:
    frames = scenario_step2_flapping(n_frames=N_FRAMES)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    xyz_by_frame: dict[int, np.ndarray] = {}
    gt_by_frame: dict[int, np.ndarray] = {}
    for df_unlabeled, frame_gt in frames:
        xyz_by_frame[frame_gt.frame_id] = df_unlabeled[["x", "y", "z"]].to_numpy()
        gt_by_frame[frame_gt.frame_id] = frame_gt.part_label

    valid_centers = range(HALF_WINDOW, N_FRAMES - HALF_WINDOW)
    print(f"n_frames={N_FRAMES}, HALF_WINDOW={HALF_WINDOW} -> valid center frames: "
          f"{valid_centers.start}..{valid_centers.stop - 1} ({len(valid_centers)} frames)")

    rows = []
    pred_by_frame: dict[int, np.ndarray] = {}
    for center in valid_centers:
        pred = segment.segment_frame_motion(xyz_by_frame, center).to_numpy()
        pred_by_frame[center] = pred
        gt = gt_by_frame[center]
        acc = float((pred == gt).mean())
        rows.append({"frame_id": center, "seg_accuracy": acc})
        print(f"[frame {center}] seg_acc={acc:.4f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(DIAG_DIR / "step2_motion_seg_results.csv", index=False)
    print("\n=== summary ===")
    print(out_df["seg_accuracy"].describe().to_string())

    worst = int(out_df.loc[out_df["seg_accuracy"].idxmin(), "frame_id"])
    best = int(out_df.loc[out_df["seg_accuracy"].idxmax(), "frame_id"])
    median_acc = out_df["seg_accuracy"].median()
    median_frame = int(out_df.iloc[(out_df["seg_accuracy"] - median_acc).abs().idxmin()]["frame_id"])
    viz_frames = sorted(set([worst, median_frame, best]))
    print(f"\nrendering multi-view plots for frames {viz_frames} (worst/median/best by seg_accuracy)")

    for center in viz_frames:
        gt = gt_by_frame[center]
        pred = pred_by_frame[center]
        confusion = pd.crosstab(pd.Series(gt, name="gt"), pd.Series(pred, name="pred"))
        print(f"\n[frame {center}] confusion:\n{confusion.to_string()}")
        out_path = DIAG_DIR / f"step2_motion_seg_multiview_f{center:04d}.png"
        plots.plot_point_cloud_segmentation_multiview(xyz_by_frame[center], gt, pred, center, out_path)
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
