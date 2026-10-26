"""第 4 步 segment：一段连续帧的 `_marked.csv` -> 逐帧 `_labeled.csv`。

    python -m segment.run --run <recon_dir> --fps 16000 [--param-set ratio3_sh0_dense]

输入：<recon_dir>/<param_set>/f####/<method>/<时间戳>/gaussian_features_f####_marked.csv
输出：同目录下的 gaussian_features_f####_labeled.csv，即在输入表后面加 part_label ∈ {body, wing_L, wing_R}
     和 confidence ∈ {high, low} 两列。
累积窗口为 ±round(36 × fps / 16000) 帧，body 命中阈值按窗口帧数等比换算（16000 fps 时 >19 帧）；
靠近段首、段尾的帧窗口会变短。某帧失败时删除该帧已有的 _labeled.csv。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .motion import body_hit_thresh, body_voxels, frame_voxels, half_window_for_fps, label_frame

MARKED_GLOB = "f*/*/*/gaussian_features_f*_marked.csv"
_FRAME_RE = re.compile(r"gaussian_features_f(\d+)_marked\.csv$")


def discover_marked(param_dir: Path) -> dict[int, Path]:
    """frame_id -> `_marked.csv`。同一帧有多个时间戳目录时取最新的。"""
    by_frame: dict[int, list[Path]] = {}
    for p in Path(param_dir).glob(MARKED_GLOB):
        m = _FRAME_RE.search(p.name)
        if m:
            by_frame.setdefault(int(m.group(1)), []).append(p)
    return {fid: sorted(paths)[-1] for fid, paths in sorted(by_frame.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="重建结果目录，例如 outputs/ctrl_009/<run>")
    ap.add_argument("--fps", type=float, required=True, help="拍摄帧率：真实 4 相机 16000，合成数据 5000")
    ap.add_argument("--param-set", default="ratio3_sh0_dense")
    args = ap.parse_args()

    paths = discover_marked(args.run / args.param_set)
    if not paths:
        raise SystemExit(f"no *_marked.csv under {args.run / args.param_set}")
    half_window = half_window_for_fps(args.fps)
    hit_thresh = body_hit_thresh(half_window)
    print(f"[segment] {len(paths)} frames f{min(paths):04d}..f{max(paths):04d}, "
          f"half_window={half_window}, body hit thresh={hit_thresh}")

    tables = {fid: pd.read_csv(p) for fid, p in paths.items()}
    voxels = {fid: frame_voxels(df.loc[df["if_keep"].astype(bool), ["x", "y", "z"]].to_numpy(dtype=float))
              for fid, df in tables.items()}

    n_low, failed = 0, []
    for fid, df in tables.items():
        window = [voxels[i] for i in range(fid - half_window, fid + half_window + 1) if i in voxels]
        labeled_path = paths[fid].with_name(paths[fid].name.replace("_marked.csv", "_labeled.csv"))
        try:
            part_label, confidence = label_frame(df, body_voxels(window, hit_thresh))
        except Exception as e:  # noqa: BLE001
            failed.append(fid)
            labeled_path.unlink(missing_ok=True)  # 不留旧标签，避免下游混用
            print(f"[segment] f{fid:04d} FAILED: {type(e).__name__}: {e}")
            continue
        out = df.copy()
        out["part_label"] = part_label
        out["confidence"] = confidence
        out.to_csv(labeled_path, index=False)
        n_low += confidence == "low"

    print(f"[segment] ok {len(tables) - len(failed)}/{len(tables)} (low confidence {n_low}), failed {failed}")


if __name__ == "__main__":
    main()
