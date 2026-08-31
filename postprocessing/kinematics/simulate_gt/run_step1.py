"""Step 1 (task spec: simplest end-to-end smoke test): static ellipsoid
body at pitch=45deg + two flat elliptical wings, 10 frames, no positional
noise, no density imbalance -- see `scene.scenario_step1_static`. Purpose is
to prove the full segment -> T4 -> compare-to-ground-truth chain actually
runs end to end, scored against three conditions (T3-predicted seg -> T4,
exact seg -> T4-only, ground truth -- see `evaluate.py`'s module docstring)
and rendered as time-series comparison plots plus a one-frame point-cloud
segmentation view (see `plots.py`). Later steps (task spec) add positional
noise, density imbalance, and time-varying yaw/pitch/roll/phi/theta/eta.

Run: python -m postprocessing.kinematics.simulate_gt.run_step1
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics.simulate_gt import segment  # noqa: E402
from postprocessing.kinematics.simulate_gt import plots  # noqa: E402
from postprocessing.kinematics.simulate_gt.evaluate import evaluate_frame  # noqa: E402
from postprocessing.kinematics.simulate_gt.scene import scenario_step1_static  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"
VIZ_FRAME_INDEX = 0
"""Which frame (index into the returned `frames` list, not `frame_id`) the
point-cloud segmentation viz is rendered for."""


def _flat_row(result) -> dict:
    row = {
        "frame_id": result.frame_id,
        "status_t3": result.status_t3,
        "status_t4_only": result.status_t4_only,
        "seg_accuracy": result.seg_accuracy,
    }
    row.update({f"t3_{k}": v for k, v in result.errors_t3.items()})
    row.update({f"t4only_{k}": v for k, v in result.errors_t4_only.items()})
    return row


def main() -> None:
    frames = scenario_step1_static(n_frames=10)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for df_unlabeled, frame_gt in frames:
        result = evaluate_frame(df_unlabeled, frame_gt)
        results.append(result)
        print(f"[frame {result.frame_id}] status_t3={result.status_t3} "
              f"status_t4_only={result.status_t4_only} seg_acc={result.seg_accuracy}")
        if result.seg_confusion is not None:
            print(result.seg_confusion.to_string())

    out_df = pd.DataFrame([_flat_row(r) for r in results])
    out_df.to_csv(DIAG_DIR / "step1_static_results.csv", index=False)

    print("\n=== per-frame table ===")
    print(out_df.to_string(index=False))

    print("\n=== summary (mean over frames) ===")
    numeric_cols = [c for c in out_df.columns if not c.startswith("status") and c != "frame_id"]
    print(out_df[numeric_cols].mean(numeric_only=True).to_string())

    body_plot = DIAG_DIR / "step1_body_angles_compare.png"
    wing_plot = DIAG_DIR / "step1_wing_angles_compare.png"
    plots.plot_body_angles_compare(results, body_plot)
    plots.plot_wing_angles_compare(results, wing_plot)

    df_viz, frame_gt_viz = frames[VIZ_FRAME_INDEX]
    pred_label_viz = segment.segment_frame_kmeans_v2(df_viz).to_numpy()
    xyz_viz = df_viz[["x", "y", "z"]].to_numpy()
    seg_plot = DIAG_DIR / "step1_point_cloud_segmentation.png"
    plots.plot_point_cloud_segmentation(
        xyz_viz, frame_gt_viz.part_label, pred_label_viz, frame_gt_viz.frame_id, seg_plot,
    )

    print(f"\nwritten: {DIAG_DIR / 'step1_static_results.csv'}")
    print(f"written: {body_plot}")
    print(f"written: {wing_plot}")
    print(f"written: {seg_plot}")


if __name__ == "__main__":
    main()
