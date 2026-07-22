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
  /home/computer0/anaconda3/envs/fly_gsplat/bin/python gpu/schedule/schedule.py --config a.json
  加 --debug-checkpoint 用splatfacto-checkpoint(带debug_checkpoints/stats等调试dump)跑，
  默认(不加)用原版splatfacto，训练/export完立刻删tensorboard事件文件+nerfstudio_models。

  一次只跑一组config(一个config = 一组完整的sweep参数)。多组config顺序多次调用本脚本
  (--config a.json、--config b.json、...)，不做多config一键并行——不同config的
  name字段天然对应不同的outputs/<name>输出路径，互不干扰。

config schema:
  {
    "name": "ctrl_009_002_ratio3_sh0_full",
    "sparse_dir": "X:\\antenna\\control\\009_25052026\\Sparse\\Expr_009_mov_002",
    "base_name": "ctrl_009_002",
    "max_iters": 2000,
    "param_sets": {"ratio3_sh0": ["--pipeline.model.use-scale-regularization", "True", ...]},
    "frames": {"start": 0, "end": 640}
  }
  frames按Python range语义处理(list(range(start, end))，end不包含在内)。
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

_CONFIG_REQUIRED_KEYS = ["name", "sparse_dir", "base_name", "max_iters", "param_sets", "frames"]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    missing = [k for k in _CONFIG_REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config {config_path} missing required keys: {missing}")
    return cfg


def check_sweep_meta(out_base_dir: Path, meta: dict) -> None:
    """一个run的元信息(sparse_dir/base_name/max_iters/param_sets/frames)落盘到
    outputs/<name>/sweep_meta.json。同一个name第一次跑时写入；之后每次跑(包括断点
    续跑)都拿这次--config解析出的meta和落盘的比对，任何字段不一致就报错退出并打印
    出哪些字段不一致——绝不静默覆盖或掩盖用config手误/改动导致的不一致。"""
    meta_path = out_base_dir / "sweep_meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2))
        return
    old_meta = json.loads(meta_path.read_text())
    mismatches = {k: (old_meta.get(k), v) for k, v in meta.items() if old_meta.get(k) != v}
    if mismatches:
        lines = [f"  {k}: existing={old!r} vs new={new!r}" for k, (old, new) in mismatches.items()]
        print(f"[ERROR] sweep_meta.json mismatch for name={meta['name']!r} ({meta_path}):", file=sys.stderr)
        print("\n".join(lines), file=sys.stderr)
        sys.exit(1)


def enumerate_tasks(param_sets: dict, frames: list, base_name: str, max_iters: int,
                     use_checkpoint_model: bool) -> list:
    return [{"param_set": ps, "frame": f, "extra_args": extra_args,
              "base_name": base_name, "max_iters": max_iters,
              "use_checkpoint_model": use_checkpoint_model}
            for ps, extra_args in param_sets.items() for f in frames]


def prepare_all_frames(frames: list, sparse_dir: str, base_name: str) -> None:
    """Phase A: 数据准备按frame去重、在主进程里串行执行，避免worker并发写同一个
    data_dir产生竞态。已存在的帧(transforms.json+init_points.ply都在)直接跳过。"""
    from generate_dataset import generate_dataset
    from generate_hull import generate_hull

    for frame_idx in frames:
        data_dir = common.data_dir_for(base_name, frame_idx)
        if (data_dir / "transforms.json").exists() and (data_dir / "init_points.ply").exists():
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        generate_dataset(str(data_dir), sparse_dir, target_frame=frame_idx,
                          if_crop=False, white_bg=True, if_mask=False,
                          calib_dir=str(common.REPO / "data" / base_name))
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
        if common.is_task_done(sweep_name, task["param_set"], task["frame"], task["use_checkpoint_model"]):
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
    ap.add_argument("--config", type=str, required=True,
                     help="sweep config json路径(schema见文件头注释)，一次只跑一组")
    ap.add_argument("--debug-checkpoint", action="store_true",
                     help="用splatfacto-checkpoint(带debug_checkpoints/stats等调试dump)代替默认的原版splatfacto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sweep_name = cfg["name"]
    sparse_dir = cfg["sparse_dir"]
    base_name = cfg["base_name"]
    max_iters = cfg["max_iters"]
    param_sets = cfg["param_sets"]
    frames = list(range(cfg["frames"]["start"], cfg["frames"]["end"]))

    out_base_dir = common.REPO / "outputs" / sweep_name
    queue_dir = out_base_dir / "_queue"
    progress_dir = out_base_dir / "_progress"
    out_base_dir.mkdir(parents=True, exist_ok=True)

    meta = {k: cfg[k] for k in _CONFIG_REQUIRED_KEYS}
    check_sweep_meta(out_base_dir, meta)

    print("=== Phase A: data prep ===")
    prepare_all_frames(frames, sparse_dir, base_name)

    print("=== Phase B: build queue ===")
    tasks = enumerate_tasks(param_sets, frames, base_name, max_iters, args.debug_checkpoint)
    n_queued = build_queue(sweep_name, tasks, queue_dir)
    if n_queued == 0:
        print("Nothing to do, all tasks already done.")
        return

    print(f"=== Phase C: dispatch {len(GPU_INDICES)}x{WORKERS_PER_GPU} workers ===")
    t0 = time.perf_counter()
    procs = spawn_workers(sweep_name, queue_dir, progress_dir)
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
