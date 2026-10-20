"""第 5 步 kinematics：一段连续帧的 `_labeled.csv` -> angles.csv。

    python -m kinematics.run --run <recon_dir> [--param-set ratio3_sh0_dense]

输入：<recon_dir>/<param_set>/f####/<method>/<时间戳>/gaussian_features_f####_labeled.csv
输出：<recon_dir>/kinematics/angles.csv，每帧一行（列见 io.OUTPUT_COLUMNS）。
     有任一部分失败的帧，对应列为 NaN，原因写在 status 列；成功的帧 status 为 "ok"。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .body import estimate_body_frame
from .body_axis import sequence_x_body
from .io import OUTPUT_COLUMNS, FrameParts, discover_frames, load_parts
from .wing import estimate_chord, estimate_span, phi_theta

MIN_POINTS = 10
"""body / wing_L / wing_R 各自至少需要的点数。"""


def _all_finite(*values) -> bool:
    return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)


def compute_angles(frames: dict[int, FrameParts]) -> pd.DataFrame:
    """frame_id -> FrameParts（必须是一段连续帧）-> 每帧一行的角度表。"""
    x_body_table = sequence_x_body(frames)
    prev_chord: dict[str, np.ndarray | None] = {"wing_L": None, "wing_R": None}

    rows = []
    for fid in sorted(frames):
        parts = frames[fid]
        row = {c: np.nan for c in OUTPUT_COLUMNS}
        row["frame_id"] = fid
        rows.append(row)

        too_few = [label for label in ("body", "wing_L", "wing_R") if len(getattr(parts, label)) < MIN_POINTS]
        if too_few:
            row["status"] = f"{too_few[0]}:too_few_points"
            continue

        try:
            frame = estimate_body_frame(parts.body, parts.wing_L, parts.wing_R, x_body=x_body_table.get(fid))
            if not _all_finite(frame.x_body, frame.y_body, frame.z_body, frame.n_sp,
                               frame.yaw, frame.pitch, frame.roll):
                raise ValueError("non-finite result (degenerate hinge/PCA geometry)")
        except Exception as e:  # noqa: BLE001
            row["status"] = f"body:{e}"
            continue
        row["yaw"], row["pitch"], row["roll"] = frame.yaw, frame.pitch, frame.roll
        row["sp_normal_x"], row["sp_normal_y"], row["sp_normal_z"] = (float(v) for v in frame.n_sp)

        wing_errors = []
        for side, s in (("wing_L", "L"), ("wing_R", "R")):
            wing_xyz = getattr(parts, side)
            try:
                span = estimate_span(wing_xyz, frame)
                phi, theta = phi_theta(span, frame, side)
                chord = estimate_chord(wing_xyz, frame, side, span, prev_chord[side])
                if not _all_finite(phi, theta, chord.eta, chord.chord_conf, span):
                    raise ValueError("non-finite result")
            except Exception as e:  # noqa: BLE001
                wing_errors.append(f"{side}:{e}")
                continue
            row[f"phi_{s}"], row[f"theta_{s}"], row[f"eta_{s}"] = phi, theta, chord.eta
            row[f"chord_conf_{s}"] = chord.chord_conf
            row[f"span_{s}_x"], row[f"span_{s}_y"], row[f"span_{s}_z"] = (float(v) for v in span)
            prev_chord[side] = chord.chord.copy()

        row["status"] = ";".join(wing_errors) if wing_errors else "ok"

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="重建结果目录，例如 outputs/ctrl_009/<run>")
    ap.add_argument("--param-set", default="ratio3_sh0_dense")
    args = ap.parse_args()

    paths = discover_frames(args.run / args.param_set)
    if not paths:
        raise SystemExit(f"no *_labeled.csv under {args.run / args.param_set}")
    print(f"[kinematics] {len(paths)} frames f{min(paths):04d}..f{max(paths):04d}")
    frames = {fid: load_parts(p) for fid, p in paths.items()}

    df = compute_angles(frames)
    out = args.run / "kinematics" / "angles.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[kinematics] status=ok {int((df['status'] == 'ok').sum())}/{len(df)} -> {out}")


if __name__ == "__main__":
    main()
