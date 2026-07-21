"""
schedule.py
G4: 通用队列式并行调度器。任务=(param_set, frame)，全局单队列(文件系统目录+os.rename
原子认领)，GPU0/GPU1各起WORKERS_PER_GPU(=6，G3/G3b定稿)个worker子进程，worker启动时
用CUDA_VISIBLE_DEVICES固定绑卡；两卡的worker池共享同一个pending/队列，不做"跨卡抢任务"
以外的静态预切分——哪个池先腾出手就先从队列取下一个任务，避免几千任务规模下任务耗时
不均导致某张卡的worker提前空闲(而另一张卡还有积压)。

Phase A(数据准备)在起worker之前、主进程里串行做一遍：按frame去重(同一帧被多个
param_set共用同一份输入数据)，避免多个worker并发对同一个data_dir做generate_dataset/
generate_hull写入产生竞态。G2已确认这部分只占端到端0.6%，串行做不影响整体吞吐。

幂等/断点续跑：入队前用common.is_task_done()过滤掉已完成的任务，只把未完成/半成品的
任务写进pending/；重跑本脚本时已完成的任务自动不会再被派发。

不做自动重试：任务级失败由worker.py捕获异常后如实记录到它自己的progress JSONL里，
不悄悄重跑、不掩盖。想重跑失败子集，直接重新执行本脚本即可。

不改动 generate_dataset.py / generate_hull.py / models/ 任何代码，只是外部调度wrapper。

用法:
  /home/computer0/anaconda3/envs/fly_gsplat/bin/python gpu/schedule/schedule.py --sweep-name ctrl_009_002_ratio3_sh0_full
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import common

WORKERS_PER_GPU = 6            # G3/G3b定稿：两卡对称6/6并发，不做动态调整
GPU_INDICES = [0, 1]
TOTAL_CPU_THREADS = 28          # G1实测nproc=28


def enumerate_tasks(param_sets: dict, frames: list) -> list:
    return [{"param_set": ps, "frame": f, "extra_args": extra_args}
            for ps, extra_args in param_sets.items() for f in frames]


def prepare_all_frames(frames: list) -> None:
    """Phase A: 数据准备按frame去重、在主进程里串行执行，避免worker并发写同一个
    data_dir产生竞态。已存在的帧(transforms.json+init_points.ply都在)直接跳过。"""
    from generate_dataset import generate_dataset
    from generate_hull import generate_hull

    for frame_idx in frames:
        data_dir = common.data_dir_for(frame_idx)
        if (data_dir / "transforms.json").exists() and (data_dir / "init_points.ply").exists():
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        generate_dataset(str(data_dir), common.SPARSE_DIR, target_frame=frame_idx,
                          if_crop=False, white_bg=True, if_mask=False,
                          calib_dir=str(common.REPO / "data" / common.BASE_NAME))
        generate_hull(str(data_dir), if_viser=False)
        print(f"[prepare] frame {frame_idx} data ready")


def build_queue(sweep_name: str, tasks: list, queue_dir: Path) -> int:
    """幂等过滤 + 写pending/任务文件。只有过滤后仍未完成的任务才会被派发给worker。"""
    pending_dir = queue_dir / "pending"
    claimed_dir = queue_dir / "claimed"
    pending_dir.mkdir(parents=True, exist_ok=True)
    claimed_dir.mkdir(parents=True, exist_ok=True)

    n_queued = n_skipped = 0
    for task in tasks:
        if common.is_task_done(sweep_name, task["param_set"], task["frame"]):
            n_skipped += 1
            continue
        tid = common.task_id(task["param_set"], task["frame"])
        with open(pending_dir / f"{tid}.json", "w") as f:
            json.dump(task, f)
        n_queued += 1
    print(f"[queue] {n_queued} tasks queued, {n_skipped} already done (skipped)")
    return n_queued


def spawn_workers(sweep_name: str, queue_dir: Path, progress_dir: Path) -> list:
    n_total_workers = len(GPU_INDICES) * WORKERS_PER_GPU
    omp_threads = max(1, TOTAL_CPU_THREADS // n_total_workers)
    worker_script = Path(__file__).resolve().parent / "worker.py"

    procs = []
    for gpu_index in GPU_INDICES:
        for w in range(WORKERS_PER_GPU):
            worker_tag = f"gpu{gpu_index}_w{w}"
            cmd = [
                sys.executable, str(worker_script),
                "--gpu-index", str(gpu_index),
                "--worker-tag", worker_tag,
                "--queue-dir", str(queue_dir),
                "--progress-dir", str(progress_dir),
                "--sweep-name", sweep_name,
                "--omp-threads", str(omp_threads),
            ]
            procs.append(subprocess.Popen(cmd, cwd=str(common.REPO)))
    return procs


def aggregate_progress(progress_dir: Path) -> dict:
    n_ok = n_failed = 0
    failed_tasks = []
    for jsonl_path in sorted(progress_dir.glob("*.jsonl")):
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["status"] == "ok":
                n_ok += 1
            else:
                n_failed += 1
                failed_tasks.append(rec["task_id"])
    return {"n_ok": n_ok, "n_failed": n_failed, "failed_tasks": failed_tasks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-name", type=str, required=True)
    args = ap.parse_args()

    param_sets = common.PARAM_SETS
    frames = common.FRAMES

    out_base_dir = common.REPO / "outputs" / args.sweep_name
    queue_dir = out_base_dir / "_queue"
    progress_dir = out_base_dir / "_progress"
    out_base_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase A: data prep ===")
    prepare_all_frames(frames)

    print("=== Phase B: build queue ===")
    tasks = enumerate_tasks(param_sets, frames)
    n_queued = build_queue(args.sweep_name, tasks, queue_dir)
    if n_queued == 0:
        print("Nothing to do, all tasks already done.")
        return

    print(f"=== Phase C: dispatch {len(GPU_INDICES)}x{WORKERS_PER_GPU} workers ===")
    t0 = time.perf_counter()
    procs = spawn_workers(args.sweep_name, queue_dir, progress_dir)
    returncodes = [p.wait() for p in procs]
    wall_s = time.perf_counter() - t0

    if any(rc != 0 for rc in returncodes):
        print(f"[WARN] non-zero worker returncodes: {returncodes}")

    summary = aggregate_progress(progress_dir)
    summary["wall_s"] = wall_s
    summary_path = out_base_dir / "schedule_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done in {wall_s:.1f}s: {summary['n_ok']} ok, {summary['n_failed']} failed "
          f"-> {summary_path}")
    if summary["failed_tasks"]:
        print(f"[failed tasks] {summary['failed_tasks']}")


if __name__ == "__main__":
    main()
