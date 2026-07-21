"""
common.py
G4: schedule.py(主进程，产任务+起worker) 和 worker.py(worker子进程，消费任务) 共用的
路径规则/幂等判定/单任务执行逻辑，避免两边各写一份导致路径拼接不一致。

任务粒度=(param_set, frame)。数据(data_dir)按frame去重(schedule.py的Phase A串行准备，
worker不再重新生成)；训练产出(output/exp_name)按(param_set, frame)隔离，天然对应
"输出路径按任务身份隔离"——不用worker_id，因为一个worker进程生命周期内会连续处理很多个
不同任务，用worker_id做输出路径反而会导致同一worker前后两个任务互相覆盖。

不改动 generate_dataset.py / generate_hull.py / models/ 任何代码，只是外部调度wrapper，
和 debug/batch_8groups_100frames.py::run_group_frame 用同一套训练CLI和指标口径。
"""
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------- 任务空间配置 --
SPARSE_DIR = r"X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002"
BASE_NAME = "ctrl_009_002"
MAX_ITERS = 2000
_COMMON_DENSIFY = ["--pipeline.model.warmup-length", "50", "--pipeline.model.stop-split-at", "1800"]

PARAM_SETS = {
    "ratio3_sh0": ["--pipeline.model.use-scale-regularization", "True",
               "--pipeline.model.max-gauss-ratio", "3.0",
               "--pipeline.model.sh-degree", "0"] + _COMMON_DENSIFY,
}
FRAMES = list(range(640))  # ctrl_009_002 f0000-f0639，和run_batch.py的FRAME_RANGE对齐


# -------------------------------------------------------------------- 路径规则 --

def data_dir_for(frame_idx: int) -> Path:
    return REPO / "data" / BASE_NAME / f"f{frame_idx:04d}"


def task_id(param_set: str, frame_idx: int) -> str:
    return f"{param_set}__f{frame_idx:04d}"


def exp_name_for(sweep_name: str, param_set: str, frame_idx: int) -> str:
    return f"{sweep_name}/{param_set}/f{frame_idx:04d}"


def find_splat_dir(exp_name: str):
    d = REPO / "outputs" / exp_name / "splatfacto-checkpoint"
    if not d.exists():
        return None
    subdirs = sorted(d.iterdir())
    return subdirs[-1] if subdirs else None


# -------------------------------------------------------------------- 幂等判定 --

def is_task_done(sweep_name: str, param_set: str, frame_idx: int) -> bool:
    """splat.ply + stats.json 都存在、stats.json能被json.load解析、scale_ratio.median非空
    才算done；只要有一项不满足(目录整个不存在、或被中断的半成品)，都当作待办重新入队。"""
    splat_dir = find_splat_dir(exp_name_for(sweep_name, param_set, frame_idx))
    if splat_dir is None:
        return False
    if not (splat_dir / "splat.ply").exists():
        return False
    stats_files = sorted((splat_dir / "debug_checkpoints" / "stats").glob("step_*_stats.json"))
    if not stats_files:
        return False
    try:
        stats = json.loads(stats_files[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return stats.get("scale_ratio", {}).get("median") is not None


# ------------------------------------------------------------------ 单任务执行 --

def run_task(sweep_name: str, param_set: str, frame_idx: int, extra_args: list) -> dict:
    """单个(param_set, frame)任务的完整训练+后处理。假设Phase A已经把这一帧的数据
    (transforms.json+init_points.ply)准备好了，这里不再调用generate_dataset/generate_hull。
    指标口径和debug/batch_8groups_100frames.py::run_group_frame保持一致。
    失败(ns-train非零退出/找不到产出/stats.json缺scale_ratio)直接抛异常，
    不在这里重试或兜底——由调用方(worker.py)捕获后如实记录failed，不悄悄重跑。
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from utils.ply import export_splat, load_ply, load_ply_with_attrs, unrescale, clean_ply

    data_dir = data_dir_for(frame_idx)
    exp_name = exp_name_for(sweep_name, param_set, frame_idx)

    hull_pts = load_ply(data_dir / "init_points.ply")
    hull_extent = hull_pts.max(0) - hull_pts.min(0)
    nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
    dists, _ = nn.kneighbors(hull_pts)
    eps = 2.5 * float(np.median(dists[:, 1]))

    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(data_dir),
        "--vis", "tensorboard",
        "--max-num-iterations", str(MAX_ITERS),
        "--pipeline.model.background-color", "white",
        "--experiment-name", exp_name,
    ] + list(extra_args) + [
        "nerfstudio-data", "--eval-mode", "all",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), env=os.environ.copy(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"ns-train exit={proc.returncode} for {exp_name}")

    splat_dir = find_splat_dir(exp_name)
    if splat_dir is None:
        raise RuntimeError(f"no splatfacto-checkpoint output dir for {exp_name}")

    stats_files = sorted((splat_dir / "debug_checkpoints" / "stats").glob("step_*_stats.json"))
    if not stats_files:
        raise RuntimeError(f"no stats.json found under {splat_dir}")
    final = json.loads(stats_files[-1].read_text())
    if final.get("scale_ratio", {}).get("median") is None:
        raise RuntimeError(f"scale_ratio.median missing/null at final step for {exp_name}")

    export_splat(splat_dir)
    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R = np.array(dp["transform"])[:3, :3]
    t = np.array(dp["transform"])[:3, 3]
    scale = float(dp["scale"])

    attrs = load_ply_with_attrs(splat_dir / "splat.ply")
    splat_pts_physical = unrescale(attrs["xyz"], R, t, scale)
    splat_extent = splat_pts_physical.max(0) - splat_pts_physical.min(0)
    extent_overshoot = float(splat_extent.max() / hull_extent.max())
    low_opacity_frac = float((attrs["opacity"] < 0.05).mean()) if attrs["opacity"] is not None else float("nan")
    _, removed = clean_ply(splat_pts_physical, eps=eps, min_samples=5, min_cluster_frac=0.02)
    dbscan_floater_frac = float(len(removed) / len(splat_pts_physical)) if len(splat_pts_physical) else float("nan")

    return {
        "n_gaussians": final["n_gaussians"],
        "scale_ratio_median": final["scale_ratio"]["median"],
        "scale_ratio_p95": final["scale_ratio"]["p95"],
        "scale_ratio_frac_over_10": final["scale_ratio"]["frac_over_10"],
        "opacity_median": final["opacity"]["median"],
        "low_opacity_frac": low_opacity_frac,
        "bbox_extent_max": max(final["bbox_extent"]),
        "extent_overshoot": extent_overshoot,
        "dbscan_floater_frac": dbscan_floater_frac,
        "splat_dir": str(splat_dir),
    }
