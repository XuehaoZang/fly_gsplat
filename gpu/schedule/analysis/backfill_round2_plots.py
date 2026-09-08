"""
backfill_round2_plots.py

One-off backfill: the first real round2 run wrote body_angles.png/wing_angles.png/
reprojection/*.png to a path shared by every param_set under the same sweep_name
(outputs/<sweep_name>/kinematics/), so concurrent groups in the multiprocessing pool
clobbered each other down to 1 survivor per sweep_name (6/56 groups kept their angle
plots). run_round2_kinematics.py's run_group() is now fixed to namespace these by
group; this script regenerates the missing ones from data that's already on disk
(kinematics_<group>.csv + the labeled per-frame csvs + raw images) -- no GPU/T1-T4
rerun needed, this is pure replotting.

用法:
    python -m gpu.schedule.analysis.backfill_round2_plots
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from postprocessing import calc_kinematics as ck  # noqa: E402
from postprocessing.kinematics.diagnostics import plot_body_angles  # noqa: E402
from postprocessing.viz import reprojection_viewer  # noqa: E402
from gpu.schedule.analysis.run_round2_kinematics import enumerate_groups  # noqa: E402


def backfill_one(sweep_name: str, group: str, raw_data_dir_rel: str) -> str:
    dataset_root = REPO_ROOT / "outputs" / sweep_name / group
    raw_data_dir = REPO_ROOT / raw_data_dir_rel
    out_dir = dataset_root.parent / "kinematics"
    csv_path = out_dir / f"kinematics_{group}.csv"

    body_path = out_dir / f"{group}_body_angles.png"
    wing_path = out_dir / f"{group}_wing_angles.png"
    reproj_dir = out_dir / "reprojection" / group

    if not csv_path.exists():
        return "skip:no_csv"
    if body_path.exists() and wing_path.exists() and reproj_dir.exists():
        return "skip:already_backfilled"

    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].reset_index(drop=True)
    if ok.empty:
        return "skip:no_ok_frames"

    plot_body_angles(ok, body_path)
    ck.plot_wing_angles(ok, wing_path)

    frames = ck.pick_reprojection_frames(ok["frame_id"].tolist(), ck.N_FRAMES)
    reprojection_viewer.run_batch(frames, dataset_root, raw_data_dir, reproj_dir)
    return "backfilled"


def main() -> None:
    tasks = enumerate_groups()
    counts: dict[str, int] = {}
    failed = []
    for t in tasks:
        tag = f"{t['sweep_name']}/{t['group']}"
        try:
            status = backfill_one(t["sweep_name"], t["group"], t["raw_data_dir"])
        except Exception as e:  # noqa: BLE001
            status = "failed"
            failed.append((tag, f"{type(e).__name__}: {e}"))
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        counts[status] = counts.get(status, 0) + 1
        print(f"[{tag}] {status}")

    print(f"\n{counts}")
    if failed:
        print(f"failed: {failed}")


if __name__ == "__main__":
    main()
