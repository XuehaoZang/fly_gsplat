"""第 3 步 gaussians：逐帧 splat.ply -> 物理单位特征表（可选去 floater）。

    python -m gaussians.run --run <recon_dir> [--param-set ratio3_sh0_dense] [--remove-floaters]

输入：<recon_dir>/<param_set>/f####/<method>/<时间戳>/{splat.ply, dataparser_transforms.json}
输出：同目录下的 gaussian_features_f####.csv；加 --remove-floaters 时另写 gaussian_features_f####_marked.csv
     （在特征表后面加 if_keep 列）。同一帧有多个时间戳目录时只处理最新的那个。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .extract_features import compute_features
from .remove_floaters import mark_floaters

_FRAME_RE = re.compile(r"^f(\d+)$")


def discover_splats(param_dir: Path) -> dict[int, Path]:
    """frame_id -> 含 splat.ply 的最新时间戳目录。"""
    by_frame: dict[int, list[Path]] = {}
    for p in Path(param_dir).glob("f*/*/*/splat.ply"):
        m = _FRAME_RE.match(p.relative_to(param_dir).parts[0])
        if m:
            by_frame.setdefault(int(m.group(1)), []).append(p.parent)
    return {fid: sorted(dirs, key=lambda d: d.name)[-1] for fid, dirs in sorted(by_frame.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="重建结果目录，例如 outputs/ctrl_009/<run>")
    ap.add_argument("--param-set", default="ratio3_sh0_dense")
    ap.add_argument("--remove-floaters", action="store_true")
    args = ap.parse_args()

    dirs = discover_splats(args.run / args.param_set)
    if not dirs:
        raise SystemExit(f"no splat.ply under {args.run / args.param_set}")
    print(f"[gaussians] {len(dirs)} frames f{min(dirs):04d}..f{max(dirs):04d}")

    n_floater = n_total = 0
    for fid, d in dirs.items():
        df = compute_features(d / "splat.ply", d / "dataparser_transforms.json")
        df.to_csv(d / f"gaussian_features_f{fid:04d}.csv", index=False)
        if args.remove_floaters:
            marked = mark_floaters(df)
            marked.to_csv(d / f"gaussian_features_f{fid:04d}_marked.csv", index=False)
            n_floater += int((~marked["if_keep"]).sum())
        n_total += len(df)

    msg = f"[gaussians] {n_total} gaussians"
    if args.remove_floaters:
        msg += f", floaters {n_floater} ({100 * n_floater / n_total:.1f}%)"
    print(msg)


if __name__ == "__main__":
    main()
