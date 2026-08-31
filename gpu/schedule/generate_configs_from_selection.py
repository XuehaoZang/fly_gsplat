"""
generate_configs_from_selection.py

把 select_frame_window.py 的输出(frame_selection.csv，每个视频一行，含是否通过筛选
+ 最终训练帧范围)转成 schedule.py 能吃的per-视频config，是"有效帧选择"接入实际部署
pipeline的胶水脚本：

    select_frame_window.py --out-csv frame_selection.csv
        -> generate_configs_from_selection.py --selection-csv frame_selection.csv
        -> 每个通过筛选的视频一个 configs/<out-subdir>/<name>.json
        -> schedule.py --config <name>.json (逐个顺序跑)

只处理selected==True的行；筛掉的视频不生成config，不会被训练。

用法:
    python gpu/schedule/generate_configs_from_selection.py \\
        --selection-csv gpu/schedule/configs/ctrl_009_valid480/frame_selection.csv \\
        --sparse-root "X:\\antenna\\control\\009_25052026\\Sparse" \\
        --data-root data/ctrl_009 \\
        --out-dir gpu/schedule/configs/ctrl_009_valid480 \\
        --name-prefix ctrl_009 \\
        --param-set-name ratio3_sh0_dense
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# ratio3_sh0_dense: 当前生产用参数组(见README.md "Config schema"一节)，跟
# ctrl_009_002_ratio3_sh0_dense.json / ctrl_009_mid200/*.json一致。
RATIO3_SH0_DENSE_ARGS = [
    "--pipeline.model.use-scale-regularization", "True",
    "--pipeline.model.max-gauss-ratio", "3.0",
    "--pipeline.model.sh-degree", "0",
    "--pipeline.model.warmup-length", "50",
    "--pipeline.model.stop-split-at", "1800",
    "--pipeline.model.densify-grad-thresh", "0.0004",
    "--pipeline.model.refine-every", "50",
]
MAX_ITERS = 2000


def _find_video_dir(sparse_root: Path, mov: str) -> Path:
    matches = list(sparse_root.glob(f"Expr_*_mov_{mov}"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 dir for mov={mov!r} under {sparse_root}, found {matches}")
    return matches[0]


def generate(selection_csv: Path, sparse_root: Path, data_root: str, out_dir: Path,
             name_prefix: str, param_set_name: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with open(selection_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["selected"] != "True":
                continue
            mov = row["mov"]
            video_dir = _find_video_dir(sparse_root, mov)
            name = f"{name_prefix}_{mov}_{param_set_name}_valid480"
            cfg = {
                "name": name,
                "sparse_dir": str(video_dir).replace("/mnt/x", "X:").replace("/", "\\"),
                "base_name": f"{data_root}/{mov}",
                "max_iters": MAX_ITERS,
                "param_sets": {param_set_name: RATIO3_SH0_DENSE_ARGS},
                "frames": {"start": int(row["train_start"]), "end": int(row["train_end"])},
            }
            out_path = out_dir / f"{name}.json"
            out_path.write_text(json.dumps(cfg, indent=2))
            written.append(name)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-csv", type=str, required=True)
    ap.add_argument("--sparse-root", type=str, required=True,
                     help=r'session的Sparse根目录，如 "X:\antenna\control\009_25052026\Sparse"')
    ap.add_argument("--data-root", type=str, required=True,
                     help='base_name前缀(相对仓库根目录)，如 "data/ctrl_009" -> base_name="data/ctrl_009/<mov>"')
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--name-prefix", type=str, required=True)
    ap.add_argument("--param-set-name", type=str, default="ratio3_sh0_dense")
    args = ap.parse_args()

    sparse_root = Path(args.sparse_root.replace("X:", "/mnt/x").replace("\\", "/"))
    written = generate(Path(args.selection_csv), sparse_root, args.data_root,
                        Path(args.out_dir), args.name_prefix, args.param_set_name)
    print(f"[configs] wrote {len(written)} config(s) -> {args.out_dir}")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
