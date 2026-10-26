"""第 2 步 train：逐帧 3D 高斯训练（多 GPU 并行，可断点续跑）。

    python -m train.run --config train/configs/<run>.json [--gpus 0,1] [--workers-per-gpu 6] [--cpu-threads 28]

配置（JSON）：
    name        运行名，也是输出目录名
    data_dir    第 1 步输出的帧目录的上一级（含 f####/{images, transforms.json, init_points.ply}），相对仓库根目录
    out_root    输出根目录，相对仓库根目录
    frames      {"start": 起始帧（含）, "end": 结束帧（不含）}
    param_set   train/param_sets.json 里的参数组名
    max_iters   训练迭代数
输出：<out_root>/<name>/<param_set>/f####/splatfacto/<时间戳>/{splat.ply, dataparser_transforms.json, config.yml}
     <out_root>/<name>/_progress/*.jsonl（每帧一行：耗时、高斯数、scale_ratio 中位数）、train_summary.json
已有 splat.ply 的帧会跳过。同一 <out_root>/<name> 下的参数组和迭代数必须和第一次运行一致（记在 train_meta.json）。
GPU 长任务请用 nohup setsid 放到后台。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import task


def check_inputs(cfg: dict, frames: list[int]) -> None:
    missing = [f for f in frames
               if not ((task.frame_dir(cfg, f) / "transforms.json").exists()
                       and (task.frame_dir(cfg, f) / "init_points.ply").exists())]
    if missing:
        raise SystemExit(f"{len(missing)} frame(s) lack transforms.json/init_points.ply (run step 1 first): "
                         f"f{missing[0]:04d}..f{missing[-1]:04d}")


def check_meta(run_dir: Path, cfg: dict) -> None:
    meta = {"param_set": cfg["param_set"], "param_args": task.PARAM_SETS[cfg["param_set"]],
            "max_iters": cfg["max_iters"], "data_dir": cfg["data_dir"]}
    path = run_dir / "train_meta.json"
    if path.exists() and json.loads(path.read_text()) != meta:
        raise SystemExit(f"{path} differs from this config; use a new name for different training settings")
    path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers-per-gpu", type=int, default=6)
    ap.add_argument("--cpu-threads", type=int, default=28)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    frames = list(range(cfg["frames"]["start"], cfg["frames"]["end"]))
    check_inputs(cfg, frames)
    run_dir = (task.REPO / cfg["out_root"]).resolve() / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    check_meta(run_dir, cfg)

    queue, progress = run_dir / "_queue", run_dir / "_progress"
    for d in (queue / "pending", queue / "claimed", progress):
        d.mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames if not task.is_done(cfg, f)]
    for f in todo:
        (queue / "pending" / f"{cfg['param_set']}__f{f:04d}.json").write_text(json.dumps({"frame": f}))
    print(f"[train] {cfg['name']}: {len(todo)} to train, {len(frames) - len(todo)} already done")
    if not todo:
        return

    gpus = [g for g in args.gpus.split(",") if g]
    n_workers = len(gpus) * args.workers_per_gpu
    threads = str(max(1, args.cpu_threads // n_workers))
    procs = []
    t0 = time.perf_counter()
    for g in gpus:
        for w in range(args.workers_per_gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads)
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "train.worker", "--config", str(args.config.resolve()),
                 "--queue-dir", str(queue), "--progress-dir", str(progress), "--tag", f"gpu{g}_w{w}"],
                cwd=str(task.REPO), env=env))
    for p in procs:
        p.wait()

    records = [json.loads(line) for p in progress.glob("*.jsonl") for line in p.read_text().splitlines() if line]
    failed = sorted({r["frame"] for r in records if r["status"] != "ok"} - {r["frame"] for r in records if r["status"] == "ok"})
    summary = {"wall_s": time.perf_counter() - t0, "n_done": sum(task.is_done(cfg, f) for f in frames),
               "n_frames": len(frames), "failed_frames": failed}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done in {summary['wall_s']:.0f}s: {summary['n_done']}/{len(frames)} frames have splat.ply; "
          f"failed {failed}")


if __name__ == "__main__":
    main()
