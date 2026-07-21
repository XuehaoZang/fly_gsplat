"""
worker.py
G3单卡并发扫描的单个并发worker：在固定GPU上，重复跑N次完整流程
(generate_dataset + generate_hull + ns-train + ns-export + postprocess)，
每帧记录墙钟时间，落盘到独立JSON文件(按worker_id/run_tag区分路径，避免多进程互相覆盖)。

用法（由 scan.py 以子进程形式调用，也可单独调试）:
  python3 worker.py --worker-id 0 --n-frames 4 --gpu-index 0 --omp-threads 7 \
      --run-tag L4_r0 --out results/raw/L4_r0_w0.json

不改动 generate_dataset.py / generate_hull.py / models/ 任何代码，只是外部调度脚本。
scratch数据/输出目录用 data/parallel_g3_scratch/{run_tag}/w{worker_id}/f{i}，
和正式实验帧(data/ctrl_009_002)完全隔离；每帧计时完成后立刻删除，避免5个并发档位
x2重复x多帧堆积磁盘占用。
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _set_thread_env(omp_threads: int):
    # 必须在import torch/numpy/generate_dataset等重量级模块之前设置，
    # 否则BLAS/OMP库可能在import时已经探测CPU核数并缓存了默认线程池大小，
    # 导致并发多进程时线程超订（G2发现冷启动阶段CPU瞬时冲到11-12核，这里就是要压制的对象）。
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(omp_threads)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--n-frames", type=int, required=True)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--omp-threads", type=int, required=True)
    p.add_argument("--run-tag", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    _set_thread_env(args.omp_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # 延迟import：确保上面设置的env在torch/numpy/cv2/open3d真正被加载前生效
    import subprocess
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    REPO = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(REPO))
    from generate_dataset import generate_dataset
    from generate_hull import generate_hull
    from utils.ply import export_splat, load_ply, load_ply_with_attrs, unrescale, clean_ply

    SPARSE_DIR = r"X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002"
    CALIB_DIR = REPO / "data" / "ctrl_009_002"
    FRAME_IDX = 0
    MAX_ITERS = 2000
    TRAIN_EXTRA_ARGS = [
        "--pipeline.model.warmup-length", "50",
        "--pipeline.model.stop-split-at", "1800",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def now():
        return time.perf_counter()

    def flush():
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

    def find_splat_dir(exp_name):
        return sorted((REPO / "outputs" / exp_name / "splatfacto-checkpoint").iterdir())[-1]

    def hull_eps(scratch_dir):
        hull_pts = load_ply(scratch_dir / "init_points.ply")
        nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
        dists, _ = nn.kneighbors(hull_pts)
        median_nn = float(np.median(dists[:, 1]))
        return 2.5 * median_nn

    for i in range(args.n_frames):
        scratch_dir = REPO / "data" / "parallel_g3_scratch" / args.run_tag / f"w{args.worker_id}" / f"f{i}"
        exp_name = f"parallel_g3_scratch/{args.run_tag}/w{args.worker_id}/f{i}"

        t0 = now()
        try:
            scratch_dir.mkdir(parents=True, exist_ok=True)
            generate_dataset(str(scratch_dir), SPARSE_DIR, target_frame=FRAME_IDX,
                              if_crop=False, white_bg=True, if_mask=False, calib_dir=str(CALIB_DIR))
            generate_hull(str(scratch_dir), if_viser=False)
            t1 = now()

            eps = hull_eps(scratch_dir)

            cmd = [
                "ns-train", "splatfacto-checkpoint",
                "--data", str(scratch_dir),
                "--vis", "tensorboard",
                "--max-num-iterations", str(MAX_ITERS),
                "--pipeline.model.background-color", "white",
                "--experiment-name", exp_name,
            ] + TRAIN_EXTRA_ARGS + [
                "nerfstudio-data", "--eval-mode", "all",
            ]
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
            t2 = now()
            if proc.returncode != 0:
                records.append({"frame": i, "error": f"ns-train exit={proc.returncode}",
                                 "data_prep_s": t1 - t0})
                flush()
                continue

            splat_dir = find_splat_dir(exp_name)
            export_splat(splat_dir)
            t3 = now()

            with open(splat_dir / "dataparser_transforms.json") as f:
                dp = json.load(f)
            R = np.array(dp["transform"])[:3, :3]
            t_vec = np.array(dp["transform"])[:3, 3]
            scale = float(dp["scale"])
            attrs = load_ply_with_attrs(splat_dir / "splat.ply")
            splat_pts_physical = unrescale(attrs["xyz"], R, t_vec, scale)
            _, removed = clean_ply(splat_pts_physical, eps=eps, min_samples=5, min_cluster_frac=0.02)
            t4 = now()

            records.append({
                "frame": i,
                "data_prep_s": t1 - t0,
                "train_total_s": t2 - t1,
                "postprocess_s": t4 - t2,
                "e2e_s": t4 - t0,
                "n_splat_points": len(attrs["xyz"]),
            })
            flush()
        except Exception as e:
            records.append({"frame": i, "error": f"{type(e).__name__}: {e}"})
            flush()
        finally:
            # 一次性benchmark数据，记完即删，避免5档x2重复x多帧堆积磁盘
            shutil.rmtree(scratch_dir, ignore_errors=True)
            shutil.rmtree(REPO / "outputs" / exp_name, ignore_errors=True)

    print(f"[worker {args.worker_id}] done: "
          f"{sum(1 for r in records if 'error' not in r)}/{args.n_frames} frames ok -> {out_path}")


if __name__ == "__main__":
    main()
