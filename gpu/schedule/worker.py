"""
worker.py
G4: 队列worker子进程。启动时用CUDA_VISIBLE_DEVICES固定绑一张GPU，之后循环从共享的
pending/目录里认领任务、跑完整流程、把结果append到自己独享的progress JSONL，
直到pending/见底才退出。

认领机制：os.rename在同一文件系统内是原子操作，谁先把某个任务文件从pending/
rename到claimed/谁就拥有这个任务；另一个worker(不管是不是同一张卡的池)撞上
同一个文件名会拿到OSError，直接跳过试下一个候选——不需要任何跨进程锁，也不需要
静态预切分队列，两张卡的worker池天然从同一个源里动态抢活，谁先腾出手谁先拿下一个。

不做自动重试：单个任务失败(ns-train非零退出/后处理异常/收尾幂等检查不过)只记一行
failed记录(带异常文本)，continue处理下一个任务，绝不吞掉/掩盖异常也不悄悄重跑。
想重跑失败子集，重新执行schedule.py即可——is_task_done()会自动跳过已成功的任务。

用法(由schedule.py以子进程形式调用，也可单独调试):
  python3 gpu/schedule/worker.py --gpu-index 0 --worker-tag gpu0_w3 \
      --queue-dir outputs/xxx/_queue --progress-dir outputs/xxx/_progress \
      --sweep-name xxx --omp-threads 2
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--worker-tag", type=str, required=True)
    p.add_argument("--queue-dir", type=str, required=True)
    p.add_argument("--progress-dir", type=str, required=True)
    p.add_argument("--sweep-name", type=str, required=True)
    p.add_argument("--omp-threads", type=int, required=True)
    return p.parse_args()


def _set_env(gpu_index: int, omp_threads: int):
    # 必须在import torch/numpy/sklearn等重量级模块之前设置：
    # CUDA_VISIBLE_DEVICES要在CUDA context创建前生效；OMP/MKL线程数同理(G3已验证，
    # 延迟设置会被BLAS库缓存的默认线程池大小覆盖，12进程共享28核会超订)。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(omp_threads)


def claim_next(pending_dir: Path, claimed_dir: Path):
    try:
        names = sorted(os.listdir(pending_dir))
    except FileNotFoundError:
        return None
    for name in names:
        src = pending_dir / name
        dst = claimed_dir / name
        try:
            os.rename(src, dst)
            return dst
        except OSError:
            continue  # 被别的worker先抢走了(文件已不在原位)，试下一个候选
    return None


def main():
    args = parse_args()
    _set_env(args.gpu_index, args.omp_threads)

    repo = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(repo))
    import common  # 延迟import：确保上面设置的env在torch/numpy/sklearn真正加载前生效

    pending_dir = Path(args.queue_dir) / "pending"
    claimed_dir = Path(args.queue_dir) / "claimed"
    progress_path = Path(args.progress_dir) / f"{args.worker_tag}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = 0
    while True:
        claimed_path = claim_next(pending_dir, claimed_dir)
        if claimed_path is None:
            break

        task = json.loads(claimed_path.read_text())
        param_set, frame_idx = task["param_set"], task["frame"]
        started_at = datetime.now().isoformat(timespec="seconds")
        t0 = time.perf_counter()
        try:
            result = common.run_task(args.sweep_name, param_set, frame_idx, task["extra_args"])
            status = "ok"
            n_ok += 1
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            status = "failed"
            n_failed += 1
        finished_at = datetime.now().isoformat(timespec="seconds")

        record = {
            "task_id": claimed_path.stem,
            "param_set": param_set,
            "frame": frame_idx,
            "status": status,
            "worker_tag": args.worker_tag,
            "gpu_index": args.gpu_index,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_s": time.perf_counter() - t0,
            **result,
        }
        with open(progress_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{args.worker_tag}] {claimed_path.stem} -> {status} ({record['wall_s']:.1f}s)", flush=True)

    print(f"[{args.worker_tag}] queue drained: {n_ok} ok, {n_failed} failed", flush=True)


if __name__ == "__main__":
    main()
