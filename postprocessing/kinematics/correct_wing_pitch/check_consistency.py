"""Step 1 required gate: verify `le_repro.estimate_leading_edge_diag`'s
`le_dir` matches `wing_angles.estimate_leading_edge`'s own `le_dir` exactly,
on both synthetic (`mock.py`) and real frames. If this fails, the
reproduction has diverged from the real algorithm and nothing downstream
(synthetic/real-data validation) can be trusted.

Run: python -m postprocessing.kinematics.correct_wing_pitch.check_consistency
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import io_schema, mock  # noqa: E402
from postprocessing.kinematics import wing_angles as wa  # noqa: E402
from postprocessing.kinematics.correct_wing_pitch.le_repro import estimate_leading_edge_diag  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"


def _true_body_frame(gt: mock.GroundTruth) -> bf.BodyFrame:
    x_body, y_body, z_body = mock.body_axes(gt)
    n_sp = mock.stroke_plane_normal(gt)
    return bf.BodyFrame(
        x_body=x_body, y_body=y_body, z_body=z_body, n_sp=n_sp,
        yaw=gt.yaw_deg, pitch=gt.pitch_deg, roll=gt.roll_deg,
        hinge_L=gt.wing_L.root, hinge_R=gt.wing_R.root, body_cm=gt.body_center,
    )


def _compare_one(label: str, wing_xyz: np.ndarray, frame: bf.BodyFrame, side: str) -> tuple[bool, str]:
    real = wa.estimate_leading_edge(wing_xyz, frame, side, rng=0)
    diag = estimate_leading_edge_diag(wing_xyz, frame, side, rng=0)

    checks = {
        "le_dir": np.allclose(real.le_dir, diag.le_dir, atol=0.0, rtol=0.0),
        "tip": np.allclose(real.tip, diag.tip, atol=0.0, rtol=0.0),
        "root": np.allclose(real.root, diag.root, atol=0.0, rtol=0.0),
        "inlier_mask": np.array_equal(real.inlier_mask, diag.inlier_mask),
        "plane_normal": np.allclose(real.plane_normal, diag.plane_normal, atol=0.0, rtol=0.0),
    }
    ok = all(checks.values())
    detail = f"{label}/{side}: " + ", ".join(f"{k}={'OK' if v else 'MISMATCH'}" for k, v in checks.items())
    if not ok:
        detail += f"\n  real.le_dir={real.le_dir} diag.le_dir={diag.le_dir}"
    return ok, detail


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# LE reproduction consistency check\n"]
    all_ok = True

    # --- synthetic frames (mock.py scenarios) ---
    scenarios = [
        ("scenario_clean_seed0", mock.scenario_clean(seed=0)),
        ("scenario_clean_seed1", mock.scenario_clean(seed=1)),
        ("scenario_reversal_contaminated", mock.scenario_reversal_contaminated(overlap=0.9, contam_frac=0.15, seed=0)),
        ("scenario_noisy", mock.scenario_noisy(seed=0)),
        ("scenario_noisy_orientation", mock.scenario_noisy_orientation(seed=0)),
    ]
    for label, (df, gt) in scenarios:
        frame = _true_body_frame(gt)
        for side in ("wing_L", "wing_R"):
            wing_xyz = io_schema.get_part(df, side)
            ok, detail = _compare_one(label, wing_xyz, frame, side)
            all_ok &= ok
            lines.append(f"- {'PASS' if ok else 'FAIL'} {detail}")
            print(("PASS " if ok else "FAIL ") + detail)

    # --- real frames, if the dataset is present ---
    real_root = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
    real_frames = sorted(real_root.glob("f*/splatfacto-checkpoint/*/*_labeled.csv"))[:5] if real_root.exists() else []
    if not real_frames:
        lines.append("\n(real dataset not found or no labeled frames matched -- synthetic-only check)")
        print("real dataset not found; skipping real-frame consistency check")
    for csv_path in real_frames:
        df = io_schema.load_frame(csv_path)
        try:
            body_xyz = io_schema.get_part(df, "body")
            wingL_xyz = io_schema.get_part(df, "wing_L")
            wingR_xyz = io_schema.get_part(df, "wing_R")
            frame = bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz)
        except Exception as e:  # noqa: BLE001
            lines.append(f"- SKIP {csv_path.name}: body_frame failed ({e})")
            continue
        for side, wing_xyz in (("wing_L", wingL_xyz), ("wing_R", wingR_xyz)):
            try:
                ok, detail = _compare_one(csv_path.parent.parent.parent.name, wing_xyz, frame, side)
            except Exception as e:  # noqa: BLE001
                lines.append(f"- SKIP {csv_path.name}/{side}: {e}")
                continue
            all_ok &= ok
            lines.append(f"- {'PASS' if ok else 'FAIL'} {detail}")
            print(("PASS " if ok else "FAIL ") + detail)

    lines.append(f"\n**Overall: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}**")
    (DIAG_DIR / "00_consistency_check.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'ALL PASS' if all_ok else 'FAILURES PRESENT'} -- written to {DIAG_DIR / '00_consistency_check.md'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
