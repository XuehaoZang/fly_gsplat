"""
common.py
G4: schedule.py(主进程，产任务+起worker) 和 worker.py(worker子进程，消费任务) 共用的
路径规则/幂等判定/单任务执行逻辑，避免两边各写一份导致路径拼接不一致。

任务粒度=(param_set, frame)。数据(data_dir)按frame去重(schedule.py的Phase A串行准备，
worker不再重新生成)；训练产出(output/exp_name)按(param_set, frame)隔离，天然对应
"输出路径按任务身份隔离"——不用worker_id，因为一个worker进程生命周期内会连续处理很多个
不同任务，用worker_id做输出路径反而会导致同一worker前后两个任务互相覆盖。

不改动 generate_dataset.py / generate_hull.py / models/ 任何代码，只是外部调度wrapper。

默认method用原版splatfacto(不dump debug_checkpoints，见method_name_for)；schedule.py
的--debug-checkpoint可以切回splatfacto-checkpoint。两种method下run_task()的指标都从
export后的splat.ply现算，不依赖训练期dump的stats.json。tensorboard事件文件和
nerfstudio_models在每个任务export_splat()完成后立刻删——splat.ply才是下游唯一需要的
持久产出。
"""
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


# -------------------------------------------------------------------- 路径规则 --

def data_dir_for(base_name: str, frame_idx: int) -> Path:
    return REPO / "data" / base_name / f"f{frame_idx:04d}"


def task_id(param_set: str, frame_idx: int) -> str:
    return f"{param_set}__f{frame_idx:04d}"


def exp_name_for(sweep_name: str, param_set: str, frame_idx: int) -> str:
    return f"{sweep_name}/{param_set}/f{frame_idx:04d}"


def method_name_for(use_checkpoint_model: bool) -> str:
    """默认用原版splatfacto(不dump debug_checkpoints)；use_checkpoint_model=True时
    切回带调试dump的splatfacto-checkpoint(见models/splatfacto_checkpoint.py)。"""
    return "splatfacto-checkpoint" if use_checkpoint_model else "splatfacto"


def find_splat_dir(exp_name: str, method_name: str):
    d = REPO / "outputs" / exp_name / method_name
    if not d.exists():
        return None
    subdirs = sorted(d.iterdir())
    return subdirs[-1] if subdirs else None


# -------------------------------------------------------------------- 幂等判定 --

def is_task_done(sweep_name: str, param_set: str, frame_idx: int, use_checkpoint_model: bool = False) -> bool:
    """splat.ply存在、且能被analyze_scale_ratio解析出非空scale_ratio.median才算done；
    只要有一项不满足(目录整个不存在、损坏的半成品)，都当作待办重新入队。
    不再依赖debug_checkpoints/stats.json——run_task每个任务结束就清掉了nerfstudio_models，
    splat.ply才是唯一持久留下的产出，幂等判定也只能认它。"""
    from utils.ply import analyze_scale_ratio

    method_name = method_name_for(use_checkpoint_model)
    splat_dir = find_splat_dir(exp_name_for(sweep_name, param_set, frame_idx), method_name)
    if splat_dir is None:
        return False
    ply_path = splat_dir / "splat.ply"
    if not ply_path.exists():
        return False
    try:
        stats = analyze_scale_ratio(ply_path)
    except Exception:
        return False
    return stats.get("median") is not None


# ------------------------------------------------------------------ 单任务执行 --

def run_task(sweep_name: str, param_set: str, frame_idx: int, extra_args: list,
             base_name: str, max_iters: int, use_checkpoint_model: bool = False) -> dict:
    """单个(param_set, frame)任务的完整训练+后处理。假设Phase A已经把这一帧的数据
    (transforms.json+init_points.ply)准备好了，这里不再调用generate_dataset/generate_hull。
    失败(ns-train非零退出/找不到产出/splat.ply缺scale_ratio)直接抛异常，
    不在这里重试或兜底——由调用方(worker.py)捕获后如实记录failed，不悄悄重跑。

    下游只消费splat.ply，不需要clean过的点云(DBSCAN floater清理只是历史上的一个
    QC指标，从没写回过splat.ply本身，这里直接不算了)。指标全部从export后的
    splat.ply现算，不依赖训练期dump的stats.json，vanilla splatfacto和
    splatfacto-checkpoint两种method都能用同一套口径。

    export_splat()跑完、指标算完之后立刻删掉tensorboard事件文件和nerfstudio_models
    checkpoint权重——这两样东西训练/debug时才有用，splat.ply导出后就是唯一需要的
    产出，逐任务删而不是等整批sweep跑完再统一清理，避免几百帧的磁盘占用滚雪球。
    """
    import shutil
    import numpy as np
    from utils.ply import export_splat, load_ply, load_ply_with_attrs, unrescale, analyze_scale_ratio

    data_dir = data_dir_for(base_name, frame_idx)
    exp_name = exp_name_for(sweep_name, param_set, frame_idx)
    method_name = method_name_for(use_checkpoint_model)

    hull_pts = load_ply(data_dir / "init_points.ply")
    hull_extent = hull_pts.max(0) - hull_pts.min(0)

    cmd = [
        "ns-train", method_name,
        "--data", str(data_dir),
        "--vis", "tensorboard",
        "--max-num-iterations", str(max_iters),
        "--pipeline.model.background-color", "white",
        "--experiment-name", exp_name,
    ] + list(extra_args) + [
        "nerfstudio-data", "--eval-mode", "all",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), env=os.environ.copy(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"ns-train exit={proc.returncode} for {exp_name}")

    splat_dir = find_splat_dir(exp_name, method_name)
    if splat_dir is None:
        raise RuntimeError(f"no {method_name} output dir for {exp_name}")

    export_splat(splat_dir)
    ply_path = splat_dir / "splat.ply"
    if not ply_path.exists():
        raise RuntimeError(f"export_splat did not produce splat.ply for {exp_name}")

    scale_stats = analyze_scale_ratio(ply_path)
    if scale_stats.get("median") is None:
        raise RuntimeError(f"scale_ratio.median missing/null for {exp_name}")

    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R = np.array(dp["transform"])[:3, :3]
    t = np.array(dp["transform"])[:3, 3]
    scale = float(dp["scale"])

    attrs = load_ply_with_attrs(ply_path)
    splat_pts_physical = unrescale(attrs["xyz"], R, t, scale)
    splat_extent = splat_pts_physical.max(0) - splat_pts_physical.min(0)
    extent_overshoot = float(splat_extent.max() / hull_extent.max())
    opacity_median = float(np.median(attrs["opacity"])) if attrs["opacity"] is not None else float("nan")
    low_opacity_frac = float((attrs["opacity"] < 0.05).mean()) if attrs["opacity"] is not None else float("nan")

    for tf_event in splat_dir.glob("events.out.tfevents.*"):
        tf_event.unlink()
    shutil.rmtree(splat_dir / "nerfstudio_models", ignore_errors=True)

    return {
        "n_gaussians": scale_stats["n"],
        "scale_ratio_median": scale_stats["median"],
        "scale_ratio_p95": scale_stats["p95"],
        "scale_ratio_frac_over_10": scale_stats["frac_over_10"],
        "opacity_median": opacity_median,
        "low_opacity_frac": low_opacity_frac,
        "bbox_extent_max": float(splat_extent.max()),
        "extent_overshoot": extent_overshoot,
        "splat_dir": str(splat_dir),
    }
