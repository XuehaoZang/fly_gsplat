"""Step 2 (task spec: "flap effect" + 100-frame validation): slowly-drifting
body yaw/pitch/roll + sinusoidally flapping wings (see
`scene.scenario_step2_flapping` for the literature/real-data-calibrated
amplitude/period constants), 100 frames -- unlike step 1, the wings actually
move frame to frame. Still no positional noise / density imbalance.

Run: python -m postprocessing.kinematics.simulate_gt.run_step2
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
from postprocessing.kinematics.simulate_gt.scene import scenario_step2_flapping  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"
VIZ_FRAME_INDEX = 10
"""Which frame (index into the returned `frames` list) the point-cloud
segmentation viz is rendered for -- 10 is mid-way through the first quarter
of the ~80-frame wingbeat cycle, a representative mid-flap pose."""


def _flat_row(result) -> dict:
    row = {
        "frame_id": result.frame_id,
        "status_t3": result.status_t3,
        "status_t4_only": result.status_t4_only,
        "seg_accuracy": result.seg_accuracy,
    }
    row.update({f"gt_{k}": v for k, v in result.row_gt.items()})
    row.update({f"t3_{k}": v for k, v in result.errors_t3.items()})
    row.update({f"t4only_{k}": v for k, v in result.errors_t4_only.items()})
    return row


def main() -> None:
    frames = scenario_step2_flapping(n_frames=100)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (df_unlabeled, frame_gt) in enumerate(frames):
        result = evaluate_frame(df_unlabeled, frame_gt)
        results.append(result)
        if i % 10 == 0 or result.status_t3 != "ok" or result.status_t4_only != "ok":
            print(f"[frame {result.frame_id}] status_t3={result.status_t3} "
                  f"status_t4_only={result.status_t4_only} seg_acc={result.seg_accuracy}")

    out_df = pd.DataFrame([_flat_row(r) for r in results])
    out_df.to_csv(DIAG_DIR / "step2_flapping_results.csv", index=False)

    n_ok_t3 = int((out_df["status_t3"] == "ok").sum())
    n_ok_t4only = int((out_df["status_t4_only"] == "ok").sum())
    print(f"\nstatus_t3 ok: {n_ok_t3}/{len(out_df)}, status_t4_only ok: {n_ok_t4only}/{len(out_df)}")

    print("\n=== summary (mean over frames) ===")
    numeric_cols = [c for c in out_df.columns if c.startswith("t3_") or c.startswith("t4only_") or c == "seg_accuracy"]
    print(out_df[numeric_cols].mean(numeric_only=True).to_string())

    body_plot = DIAG_DIR / "step2_body_angles_compare.png"
    wing_plot = DIAG_DIR / "step2_wing_angles_compare.png"
    plots.plot_body_angles_compare(results, body_plot)
    plots.plot_wing_angles_compare(results, wing_plot)

    df_viz, frame_gt_viz = frames[VIZ_FRAME_INDEX]
    pred_label_viz = segment.segment_frame_kmeans_v2(df_viz).to_numpy()
    xyz_viz = df_viz[["x", "y", "z"]].to_numpy()
    seg_plot = DIAG_DIR / "step2_point_cloud_segmentation.png"
    plots.plot_point_cloud_segmentation(
        xyz_viz, frame_gt_viz.part_label, pred_label_viz, frame_gt_viz.frame_id, seg_plot,
    )

    print(f"\nwritten: {DIAG_DIR / 'step2_flapping_results.csv'}")
    print(f"written: {body_plot}")
    print(f"written: {wing_plot}")
    print(f"written: {seg_plot}")


if __name__ == "__main__":
    main()
