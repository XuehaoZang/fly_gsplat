"""
time_pipeline.py
G2: 单帧训练全流程耗时分解。固定(数据,参数)组合，重复N次，把墙钟时间拆成5段：
  1. generate_dataset  (读X:盘sparse + 写images/transforms.json)
  2. generate_hull      (三角化+采样+可见性投票，CPU/GPU占用旁证)
  3. ns-train 冷启动     (子进程起 -> import torch/nerfstudio/gsplat/CUDA context -> 第一个iteration真正开始跑之前)
  4. ns-train 训练循环   (第一个iteration开始 -> 子进程退出)，全程采样GPU利用率曲线
  5. 训练后处理         (ns-export gaussian-splat + load_ply/stats读取 + DBSCAN clean_ply)

不改动 generate_dataset.py / generate_hull.py / models/ 任何代码，只在外部包一层计时。
用独立 scratch 数据目录(data/timing_g2_scratch)，不触碰 data/ctrl_009_002/f0000~f0099
正式实验帧；calib_dir 复用 data/ctrl_009_002 下已有的 calibration_easyWandData.mat（只读）。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from generate_dataset import generate_dataset
from generate_hull import generate_hull
from utils.ply import export_splat, load_ply, load_ply_with_attrs, unrescale, clean_ply

from samplers import GPUSampler, ProcResourceSampler, read_proc_io, gpu_busy_start_time

# ------------------------------------------------------------------ config --
SPARSE_DIR = r"X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002"
CALIB_DIR = REPO / "data" / "ctrl_009_002"          # 只读，复用已有的 calibration mat
SCRATCH_DATA_DIR = REPO / "data" / "timing_g2_scratch" / "f0000"
FRAME_IDX = 0
N_REPEATS = 5
MAX_ITERS = 2000
EXP_NAME = "timing_g2_scratch/f0000"
GPU_INDEX = 0
GPU_SAMPLE_MS = 200
CPU_SAMPLE_S = 0.2
GPU_BUSY_THRESHOLD = 15.0   # util% 阈值，判定"训练循环真正开始"
GPU_BUSY_SUSTAIN = 2        # 连续N个采样点超阈值才算数，避免瞬时抖动误判

TRAIN_EXTRA_ARGS = [
    "--pipeline.model.warmup-length", "50",
    "--pipeline.model.stop-split-at", "1800",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def now():
    return time.perf_counter()


def stage_generate_dataset() -> dict:
    t0 = now()
    p = psutil.Process(os.getpid())
    io0 = p.io_counters()
    cpu0 = p.cpu_times()
    generate_dataset(str(SCRATCH_DATA_DIR), SPARSE_DIR, target_frame=FRAME_IDX,
                      if_crop=False, white_bg=True, if_mask=False, calib_dir=str(CALIB_DIR))
    t1 = now()
    io1 = p.io_counters()
    cpu1 = p.cpu_times()
    wall = t1 - t0
    cpu_busy = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    read_mb = (io1.read_chars - io0.read_chars) / 1e6
    return {
        "wall_s": wall,
        "cpu_busy_s": cpu_busy,
        "cpu_busy_frac": cpu_busy / wall if wall > 0 else float("nan"),
        "read_mb": read_mb,
        "read_mb_per_s": read_mb / wall if wall > 0 else float("nan"),
    }


def stage_generate_hull() -> dict:
    t0 = now()
    p = psutil.Process(os.getpid())
    cpu0 = p.cpu_times()
    generate_hull(str(SCRATCH_DATA_DIR), if_viser=False)
    t1 = now()
    cpu1 = p.cpu_times()
    wall = t1 - t0
    cpu_busy = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    return {
        "wall_s": wall,
        "cpu_busy_s": cpu_busy,
        "cpu_busy_frac": cpu_busy / wall if wall > 0 else float("nan"),
    }


def stage_train() -> dict:
    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(SCRATCH_DATA_DIR),
        "--vis", "tensorboard",
        "--max-num-iterations", str(MAX_ITERS),
        "--pipeline.model.background-color", "white",
        "--experiment-name", EXP_NAME,
    ] + TRAIN_EXTRA_ARGS + [
        "nerfstudio-data", "--eval-mode", "all",
    ]

    t0 = now()
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    gpu_sampler = GPUSampler(gpu_index=GPU_INDEX, interval_ms=GPU_SAMPLE_MS)
    cpu_sampler = ProcResourceSampler(pid=proc.pid, interval_s=CPU_SAMPLE_S)
    gpu_sampler.start()
    cpu_sampler.start()

    proc.wait()
    t1 = now()

    gpu_samples = gpu_sampler.stop()
    cpu_samples = cpu_sampler.stop()

    if proc.returncode != 0:
        raise RuntimeError(f"ns-train exited with code {proc.returncode}")

    total_wall = t1 - t0
    t_split = gpu_busy_start_time(gpu_samples, GPU_BUSY_THRESHOLD, GPU_BUSY_SUSTAIN)
    if t_split is None:
        coldstart_s = float("nan")
        train_loop_s = float("nan")
    else:
        coldstart_s = t_split
        train_loop_s = total_wall - t_split

    util_vals = [s[1] for s in gpu_samples]
    return {
        "total_wall_s": total_wall,
        "coldstart_s": coldstart_s,
        "train_loop_s": train_loop_s,
        "gpu_util_mean": float(np.mean(util_vals)) if util_vals else float("nan"),
        "gpu_util_max": float(np.max(util_vals)) if util_vals else float("nan"),
        "gpu_util_p50": float(np.percentile(util_vals, 50)) if util_vals else float("nan"),
        "gpu_samples": gpu_samples,     # (t_rel, util_pct, mem_mib)
        "cpu_samples": cpu_samples,     # (t_rel, cpu_pct_sum)
    }


def find_splat_dir() -> Path:
    return sorted((REPO / "outputs" / EXP_NAME / "splatfacto-checkpoint").iterdir())[-1]


def stage_postprocess(splat_dir: Path, hull_extent: np.ndarray, eps: float) -> dict:
    p = psutil.Process(os.getpid())

    t0 = now()
    export_splat(splat_dir)
    t1 = now()

    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R, t, scale = np.array(dp["transform"])[:3, :3], np.array(dp["transform"])[:3, 3], float(dp["scale"])

    attrs = load_ply_with_attrs(splat_dir / "splat.ply")
    t2 = now()

    splat_pts_physical = unrescale(attrs["xyz"], R, t, scale)
    _, removed = clean_ply(splat_pts_physical, eps=eps, min_samples=5, min_cluster_frac=0.02)
    t3 = now()

    return {
        "export_splat_s": t1 - t0,
        "load_ply_s": t2 - t1,
        "dbscan_clean_s": t3 - t2,
        "total_s": t3 - t0,
        "n_splat_points": len(attrs["xyz"]),
    }


def hull_eps() -> tuple:
    hull_pts = load_ply(SCRATCH_DATA_DIR / "init_points.ply")
    hull_extent = hull_pts.max(0) - hull_pts.min(0)
    nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
    dists, _ = nn.kneighbors(hull_pts)
    median_nn = float(np.median(dists[:, 1]))
    return hull_extent, 2.5 * median_nn


def run_once(rep_idx: int) -> dict:
    print(f"\n{'='*20} repeat {rep_idx} {'='*20}")
    SCRATCH_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] generate_dataset ...")
    r1 = stage_generate_dataset()
    print(f"  wall={r1['wall_s']:.3f}s cpu_busy={r1['cpu_busy_s']:.3f}s "
          f"read={r1['read_mb']:.2f}MB ({r1['read_mb_per_s']:.2f} MB/s)")

    print("[2/5] generate_hull ...")
    r2 = stage_generate_hull()
    print(f"  wall={r2['wall_s']:.3f}s cpu_busy={r2['cpu_busy_s']:.3f}s "
          f"(cpu_busy_frac={r2['cpu_busy_frac']:.2f})")

    hull_extent, eps = hull_eps()

    print("[3-4/5] ns-train (cold start + train loop) ...")
    r34 = stage_train()
    print(f"  total={r34['total_wall_s']:.3f}s coldstart={r34['coldstart_s']:.3f}s "
          f"train_loop={r34['train_loop_s']:.3f}s gpu_util_mean={r34['gpu_util_mean']:.1f}% "
          f"gpu_util_max={r34['gpu_util_max']:.1f}%")

    splat_dir = find_splat_dir()
    print("[5/5] postprocess (export + load + dbscan clean) ...")
    r5 = stage_postprocess(splat_dir, hull_extent, eps)
    print(f"  export={r5['export_splat_s']:.3f}s load_ply={r5['load_ply_s']:.3f}s "
          f"dbscan={r5['dbscan_clean_s']:.3f}s total={r5['total_s']:.3f}s "
          f"n_splat={r5['n_splat_points']}")

    return {
        "rep": rep_idx,
        "generate_dataset": r1,
        "generate_hull": r2,
        "train": r34,
        "postprocess": r5,
        "splat_dir": str(splat_dir),
    }


def main():
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else N_REPEATS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for i in range(n_repeats):
        all_results.append(run_once(i))
        # 每轮结束立刻落盘，防止中途中断丢数据
        with open(RESULTS_DIR / "timing_raw.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone: {len(all_results)}/{n_repeats} repeats. Saved -> {RESULTS_DIR / 'timing_raw.json'}")


if __name__ == "__main__":
    main()
