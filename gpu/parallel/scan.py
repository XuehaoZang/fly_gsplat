"""
scan.py
G3: 单卡并发训练吞吐扫描。固定(数据,参数)组合与G2的time_pipeline.py完全一致
(frame=0 of ctrl_009_002, MAX_ITERS=2000, warmup-length=50, stop-split-at=1800,
background=white, --vis tensorboard)，只加"并发进程数"这一维度。

设计依据G2结论：数据准备(generate_dataset+generate_hull)只占端到端0.6%，不单独设并发档位，
和训练/导出合并在同一个worker循环里一起扫。

对 CONC_LEVELS 中每个并发数N，重复 N_REPEATS 次：
  1. 用 CUDA_VISIBLE_DEVICES=GPU_INDEX 固定同一张卡，启动N个worker.py子进程，
     每个子进程分配 OMP_NUM_THREADS=28//N 条BLAS/OMP线程(下取整,至少1)，避免冷启动阶段
     线程超订放大争抢。
  2. 每个worker循环跑 TOTAL_FRAMES_PER_REPEAT // N 帧完整流程
     (generate_dataset+generate_hull+ns-train+ns-export+postprocess)。
  3. 全程用整卡GPU利用率采样(GPUSampler, 200ms间隔) + 多进程根pid聚合CPU%采样
     (MultiProcResourceSampler, 200ms间隔)。
  4. 记录本次repeat的总墙钟时间、总完成帧数 -> 折算 frames/hour 聚合吞吐，
     以及每个worker每帧的 e2e_s (供后续和level=1基线比较，算"被并发拖慢的比例")。

不改动 generate_dataset.py / generate_hull.py / models/ 任何代码，只是外部调度wrapper。
用法: python3 scan.py [--levels 1,2,3,4,6] [--frames-per-repeat 12] [--repeats 2]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "gpu" / "timing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from samplers import GPUSampler          # gpu/timing/samplers.py，整卡聚合，无需改动
from samplers_multi import MultiProcResourceSampler

GPU_INDEX = 0          # G1结论: GPU0基线显存占用最低(17MiB)、无Windows桌面常驻占用，选它做单卡扫描
                        # G3b: 可用 --gpu-index 覆盖，用于补测GPU1(见run_gpu1_note)
TOTAL_CPU_THREADS = 28  # G1实测 nproc=28

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_DIR = RESULTS_DIR / "raw"


def worker_omp_threads(n_conc: int) -> int:
    return max(1, TOTAL_CPU_THREADS // n_conc)


def run_level(n_conc: int, rep_idx: int, frames_per_repeat: int) -> dict:
    run_tag = f"L{n_conc}_r{rep_idx}"
    print(f"\n{'='*20} {run_tag} (concurrency={n_conc}) {'='*20}")

    frames_per_worker = max(1, frames_per_repeat // n_conc)
    omp_threads = worker_omp_threads(n_conc)

    out_paths = [RAW_DIR / f"{run_tag}_w{w}.json" for w in range(n_conc)]
    for p in out_paths:
        if p.exists():
            p.unlink()

    env = os.environ.copy()
    procs = []
    t_start = time.perf_counter()
    for w in range(n_conc):
        cmd = [
            sys.executable, str(Path(__file__).resolve().parent / "worker.py"),
            "--worker-id", str(w),
            "--n-frames", str(frames_per_worker),
            "--gpu-index", str(GPU_INDEX),
            "--omp-threads", str(omp_threads),
            "--run-tag", run_tag,
            "--out", str(out_paths[w]),
        ]
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(proc)

    gpu_sampler = GPUSampler(gpu_index=GPU_INDEX, interval_ms=200)
    cpu_sampler = MultiProcResourceSampler(root_pids=[p.pid for p in procs], interval_s=0.2)
    gpu_sampler.start()
    cpu_sampler.start()

    returncodes = [p.wait() for p in procs]
    t_end = time.perf_counter()

    gpu_samples = gpu_sampler.stop()
    cpu_samples = cpu_sampler.stop()

    worker_records = []
    for p in out_paths:
        if p.exists():
            with open(p) as f:
                worker_records.append(json.load(f))
        else:
            worker_records.append([])

    n_frames_ok = sum(1 for wr in worker_records for r in wr if "error" not in r)
    n_frames_failed = sum(1 for wr in worker_records for r in wr if "error" in r)
    wall_s = t_end - t_start

    util_vals = [s[1] for s in gpu_samples]
    mem_vals = [s[2] for s in gpu_samples]
    cpu_vals = [s[1] for s in cpu_samples]

    result = {
        "run_tag": run_tag,
        "n_conc": n_conc,
        "rep": rep_idx,
        "frames_per_worker": frames_per_worker,
        "omp_threads": omp_threads,
        "returncodes": returncodes,
        "wall_s": wall_s,
        "n_frames_ok": n_frames_ok,
        "n_frames_failed": n_frames_failed,
        "throughput_frames_per_hour": (n_frames_ok / wall_s * 3600) if wall_s > 0 else float("nan"),
        "gpu_util_mean": (sum(util_vals) / len(util_vals)) if util_vals else float("nan"),
        "gpu_util_max": max(util_vals) if util_vals else float("nan"),
        "gpu_mem_max_mib": max(mem_vals) if mem_vals else float("nan"),
        "cpu_pct_mean": (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else float("nan"),
        "cpu_pct_max": max(cpu_vals) if cpu_vals else float("nan"),
        "gpu_samples": gpu_samples,
        "cpu_samples": cpu_samples,
        "worker_records": worker_records,
    }

    print(f"  wall={wall_s:.1f}s ok={n_frames_ok} failed={n_frames_failed} "
          f"throughput={result['throughput_frames_per_hour']:.1f} f/h "
          f"gpu_mean={result['gpu_util_mean']:.1f}% gpu_max={result['gpu_util_max']:.1f}% "
          f"cpu_mean={result['cpu_pct_mean']:.0f}% cpu_max={result['cpu_pct_max']:.0f}% "
          f"mem_max={result['gpu_mem_max_mib']:.0f}MiB")

    if any(rc != 0 for rc in returncodes):
        print(f"  [WARN] non-zero worker returncodes: {returncodes}")
    if n_frames_failed:
        print(f"  [WARN] {n_frames_failed} frame(s) failed inside workers, see worker_records/errors")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=str, default="1,2,3,4,6")
    ap.add_argument("--frames-per-repeat", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--gpu-index", type=int, default=0,
                     help="G3b: 补测GPU1时用 --gpu-index 1")
    ap.add_argument("--results-dir", type=str, default="results",
                     help="G3b: 补测GPU1时用独立目录(如results_gpu1)，"
                          "避免run_tag(如L6_r0)和GPU0已有结果冲突")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    global GPU_INDEX, RESULTS_DIR, RAW_DIR
    GPU_INDEX = args.gpu_index
    RESULTS_DIR = Path(__file__).resolve().parent / args.results_dir
    RAW_DIR = RESULTS_DIR / "raw"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_file = RESULTS_DIR / "scan_raw.json"
    all_results = []
    if out_file.exists():
        with open(out_file) as f:
            all_results = json.load(f)
    done_tags = {r["run_tag"] for r in all_results}

    for n_conc in levels:
        for rep in range(args.repeats):
            run_tag = f"L{n_conc}_r{rep}"
            if run_tag in done_tags:
                print(f"[skip] {run_tag} already in {out_file}, not re-running")
                continue
            r = run_level(n_conc, rep, args.frames_per_repeat)
            all_results.append(r)
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nDone. Saved -> {out_file}")


if __name__ == "__main__":
    main()
