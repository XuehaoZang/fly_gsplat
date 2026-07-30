"""
6组(H1-H6) x 100帧(f0000-f0099)，共600次训练。全部在G2b_G9(use-scale-regularization True,
max-gauss-ratio 3.0, sh-degree 0, warmup-length 50, stop-split-at 1800)基础上叠加，不改
densify窗口、不加iteration预算，测试3个提升点数的杠杆(hull加密/降低densify-grad-thresh/
提高refine频率)及其两两组合。baseline(G2b_G9)不重跑，直接复用
outputs/ctrl_009_002_8groups_100frames/summary.json里的数字做对比。

hull加密(H1/H4/H5)：generate_hull的N_SAMPLES从10000改到30000时不覆盖原init_points.ply
(其他组和历史实验都依赖它)。改用一份镜像数据目录 data/ctrl_009_002_dense/f{idx}/，
images/ 软链接到 data/ctrl_009_002/f{idx}/images/(不物理复制)，transforms.json 是原文件的
拷贝+ply_file_path改成init_points_dense.ply，nerfstudio据此加载稠密点云。

统一加 --pipeline.model.stats-every 100，用来诊断"hull加密后训练刚init完/第一次
refine附近n_gaussians断崖式下跌"的猜测：这次不只存final step的stats，把每个run的
全部stats_*.json(step=0,100,...,1900,final)都存进raw_records，重点看0-500这段。

通宵执行：外层按组循环，每组100帧跑完后往batch_progress.log追加一行状态再继续下一组，
单帧失败不中断整体批次(try/except记录，继续下一帧)。
"""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from generate_hull import generate_hull
from utils.ply import export_splat, load_ply, load_ply_with_attrs, unrescale, clean_ply

REPO = Path(__file__).resolve().parent.parent.parent
BASE_NAME = "ctrl_009_002"
SWEEP_NAME = "ctrl_009_002_densify_6groups_100frames"
BASELINE_SWEEP_NAME = "ctrl_009_002_8groups_100frames"
BASELINE_GROUP = "G2b_G9"
FRAMES = list(range(100))
MAX_ITERS = 2000

DATA_BASE_DIR = REPO / "data" / BASE_NAME              # 原有100帧数据，H2/H3/H6复用，不重新生成
DATA_DENSE_DIR = REPO / "data" / (BASE_NAME + "_dense")  # H1/H4/H5用的稠密hull镜像目录
OUT_BASE_DIR = REPO / "outputs" / SWEEP_NAME
PROGRESS_LOG = OUT_BASE_DIR / "batch_progress.log"

DENSE_N_SAMPLES = 30_000
DENSE_PLY_NAME = "init_points_dense.ply"

# 6组共同的base flags：等价于G2b_G9 + 本轮统一加的stats-every诊断
_COMMON_BASE = [
    "--pipeline.model.use-scale-regularization", "True",
    "--pipeline.model.max-gauss-ratio", "3.0",
    "--pipeline.model.sh-degree", "0",
    "--pipeline.model.warmup-length", "50",
    "--pipeline.model.stop-split-at", "1800",
    "--pipeline.model.stats-every", "100",
]

GRAD_THRESH_LOW = ["--pipeline.model.densify-grad-thresh", "0.0004"]
REFINE_FAST = ["--pipeline.model.refine-every", "50"]

GROUPS = {
    "H1_hull_dense": list(_COMMON_BASE),
    "H2_grad_thresh_low": _COMMON_BASE + GRAD_THRESH_LOW,
    "H3_refine_fast": _COMMON_BASE + REFINE_FAST,
    "H4_hull_dense_grad_thresh_low": _COMMON_BASE + GRAD_THRESH_LOW,
    "H5_hull_dense_refine_fast": _COMMON_BASE + REFINE_FAST,
    "H6_grad_thresh_low_refine_fast": _COMMON_BASE + GRAD_THRESH_LOW + REFINE_FAST,
}

USES_DENSE_HULL = {"H1_hull_dense", "H4_hull_dense_grad_thresh_low", "H5_hull_dense_refine_fast"}


def log_progress(line: str) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")
    print(line)


# ---------------------------------------------------------------- data prep --

def prepare_dense_frame(frame_idx: int) -> None:
    """images/ 软链到原frame目录，生成一份N_SAMPLES=30000的init_points_dense.ply，
    transforms.json 是原文件的拷贝+ply_file_path改指向dense ply。不动原frame目录。"""
    src_dir = DATA_BASE_DIR / f"f{frame_idx:04d}"
    dst_dir = DATA_DENSE_DIR / f"f{frame_idx:04d}"

    if (dst_dir / "transforms.json").exists() and (dst_dir / DENSE_PLY_NAME).exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)

    images_link = dst_dir / "images"
    if not images_link.exists():
        target = os.path.relpath(src_dir / "images", dst_dir)
        os.symlink(target, images_link, target_is_directory=True)

    with open(src_dir / "transforms.json") as f:
        transforms = json.load(f)
    transforms["ply_file_path"] = DENSE_PLY_NAME
    with open(dst_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)

    generate_hull(str(dst_dir), if_viser=False, n_samples=DENSE_N_SAMPLES, out_name=DENSE_PLY_NAME)


def prepare_all_dense_frames() -> None:
    for frame_idx in FRAMES:
        prepare_dense_frame(frame_idx)
    print(f"[Generated] dense hull data for {len(FRAMES)} frames -> {DATA_DENSE_DIR}")


def hull_eps_cache(data_base_dir: Path, ply_name: str) -> dict:
    """(hull_extent, eps) 每帧算一次。eps = 2.5x hull点间中位最近邻距离。"""
    cache = {}
    for frame_idx in FRAMES:
        data_dir = data_base_dir / f"f{frame_idx:04d}"
        hull_pts = load_ply(data_dir / ply_name)
        hull_extent = hull_pts.max(0) - hull_pts.min(0)
        nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
        dists, _ = nn.kneighbors(hull_pts)
        median_nn = float(np.median(dists[:, 1]))
        cache[frame_idx] = (hull_extent, 2.5 * median_nn)
    return cache


# ------------------------------------------------------------------ training --

def run_group_frame(group_name: str, extra_args: list, frame_idx: int,
                     hull_extent: np.ndarray, eps: float, use_dense: bool) -> dict:
    data_dir = (DATA_DENSE_DIR if use_dense else DATA_BASE_DIR) / f"f{frame_idx:04d}"
    exp_name = f"{SWEEP_NAME}/{group_name}/f{frame_idx:04d}"
    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(data_dir),
        "--vis", "tensorboard",
        "--max-num-iterations", str(MAX_ITERS),
        "--pipeline.model.background-color", "white",
        "--experiment-name", exp_name,
    ] + extra_args + [
        "nerfstudio-data", "--eval-mode", "all",
    ]

    subprocess.run(cmd, check=True)

    splat_dir = sorted((REPO / "outputs" / exp_name / "splatfacto-checkpoint").iterdir())[-1]

    stats_files = sorted((splat_dir / "debug_checkpoints" / "stats").glob("step_*_stats.json"))
    if not stats_files:
        raise RuntimeError(f"no stats.json found under {splat_dir}")
    all_stats = [json.loads(p.read_text()) for p in stats_files]
    trajectory = [{"step": s["step"], "n_gaussians": s["n_gaussians"]} for s in all_stats]
    final = all_stats[-1]
    if "scale_ratio" not in final:
        raise RuntimeError(f"n_gaussians=0 at final step for {exp_name}")

    export_splat(splat_dir)
    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R, t, scale = np.array(dp["transform"])[:3, :3], np.array(dp["transform"])[:3, 3], float(dp["scale"])

    attrs = load_ply_with_attrs(splat_dir / "splat.ply")
    splat_pts_physical = unrescale(attrs["xyz"], R, t, scale)
    splat_extent = splat_pts_physical.max(0) - splat_pts_physical.min(0)
    extent_overshoot = float(splat_extent.max() / hull_extent.max())

    low_opacity_frac = float((attrs["opacity"] < 0.05).mean()) if attrs["opacity"] is not None else float("nan")

    _, removed = clean_ply(splat_pts_physical, eps=eps, min_samples=5, min_cluster_frac=0.02)
    dbscan_floater_frac = float(len(removed) / len(splat_pts_physical)) if len(splat_pts_physical) else float("nan")

    return {
        "group": group_name, "frame": frame_idx, "status": "ok",
        "n_gaussians": final["n_gaussians"],
        "n_gaussians_trajectory": trajectory,
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


def run_group(group_name: str, extra_args: list, hull_cache_sparse: dict, hull_cache_dense: dict) -> list:
    use_dense = group_name in USES_DENSE_HULL
    hull_cache = hull_cache_dense if use_dense else hull_cache_sparse
    t0 = time.time()
    records = []
    for frame_idx in FRAMES:
        hull_extent, eps = hull_cache[frame_idx]
        try:
            records.append(run_group_frame(group_name, extra_args, frame_idx, hull_extent, eps, use_dense))
        except Exception as e:
            print(f"[{group_name}/f{frame_idx:04d}] failed: {e}")
            records.append({"group": group_name, "frame": frame_idx, "status": "failed", "error": str(e)})

    n_ok = sum(r["status"] == "ok" for r in records)
    elapsed = time.time() - t0
    log_progress(f"group={group_name} success={n_ok}/{len(FRAMES)} failed={len(FRAMES) - n_ok} "
                 f"wall_time={elapsed:.1f}s")
    return records


def smoke_test(frames=(0, 1, 2)) -> None:
    """冒烟：6组 x 少量帧，验证命令能跑通、hull加密3组真的读到了dense版本init points。
    不写入summary/raw_records。"""
    for frame_idx in frames:
        prepare_dense_frame(frame_idx)

    hull_cache_sparse, hull_cache_dense = {}, {}
    for frame_idx in frames:
        sparse_dir = DATA_BASE_DIR / f"f{frame_idx:04d}"
        dense_dir = DATA_DENSE_DIR / f"f{frame_idx:04d}"

        sparse_pts = load_ply(sparse_dir / "init_points.ply")
        dense_pts = load_ply(dense_dir / DENSE_PLY_NAME)
        print(f"[smoke f{frame_idx:04d}] sparse hull points={len(sparse_pts)} "
              f"dense hull points={len(dense_pts)}")
        assert len(dense_pts) > len(sparse_pts), \
            f"dense hull ({len(dense_pts)}) should have more points than sparse hull ({len(sparse_pts)})"

        for name, base_dir, ply_name, cache in [
            ("sparse", DATA_BASE_DIR, "init_points.ply", hull_cache_sparse),
            ("dense", DATA_DENSE_DIR, DENSE_PLY_NAME, hull_cache_dense),
        ]:
            hull_pts = load_ply(base_dir / f"f{frame_idx:04d}" / ply_name)
            hull_extent = hull_pts.max(0) - hull_pts.min(0)
            nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
            dists, _ = nn.kneighbors(hull_pts)
            eps = 2.5 * float(np.median(dists[:, 1]))
            cache[frame_idx] = (hull_extent, eps)

    for group_name, extra_args in GROUPS.items():
        use_dense = group_name in USES_DENSE_HULL
        hull_cache = hull_cache_dense if use_dense else hull_cache_sparse
        for frame_idx in frames:
            hull_extent, eps = hull_cache[frame_idx]
            result = run_group_frame(group_name, extra_args, frame_idx, hull_extent, eps, use_dense)
            step0_n = result["n_gaussians_trajectory"][0]["n_gaussians"]
            print(f"[smoke] {group_name}/f{frame_idx:04d}: use_dense={use_dense} "
                  f"step0_n_gaussians={step0_n} final_n_gaussians={result['n_gaussians']} "
                  f"n_stats_points={len(result['n_gaussians_trajectory'])}")


# ------------------------------------------------------------- aggregation --

def jitter_score(records: list) -> dict:
    """相邻(按frame_idx排序、跳过失败帧后)一阶差分的标准差，bbox_extent_max和n_gaussians各一个。"""
    ok = sorted([r for r in records if r["status"] == "ok"], key=lambda r: r["frame"])
    if len(ok) < 2:
        return {"bbox_extent_jitter": float("nan"), "n_gaussians_jitter": float("nan")}
    extent_seq = np.array([r["bbox_extent_max"] for r in ok])
    ngauss_seq = np.array([r["n_gaussians"] for r in ok])
    return {
        "bbox_extent_jitter": float(np.std(np.diff(extent_seq))),
        "n_gaussians_jitter": float(np.std(np.diff(ngauss_seq))),
    }


def early_trajectory_summary(records: list) -> list:
    """按step对齐，跨帧求n_gaussians的mean/std，只用成功帧。"""
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        return []
    by_step = {}
    for r in ok:
        for pt in r["n_gaussians_trajectory"]:
            by_step.setdefault(pt["step"], []).append(pt["n_gaussians"])
    return [
        {"step": step, "n_gaussians_mean": float(np.mean(vals)), "n_gaussians_std": float(np.std(vals)), "n": len(vals)}
        for step, vals in sorted(by_step.items())
    ]


def summarize(all_records: dict) -> dict:
    """all_records: group_name -> list of per-frame records。n_gaussians放最前面(核心指标)。"""
    summary = {}
    for group_name, records in all_records.items():
        ok = [r for r in records if r["status"] == "ok"]
        n_ok = len(ok)
        row = {"success_rate": n_ok / len(FRAMES)}
        if ok:
            for key in ["n_gaussians", "scale_ratio_median", "scale_ratio_p95", "scale_ratio_frac_over_10",
                        "opacity_median", "low_opacity_frac",
                        "extent_overshoot", "dbscan_floater_frac"]:
                vals = [r[key] for r in ok]
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_median"] = float(np.median(vals))
        row.update(jitter_score(records))
        row["n_gaussians_trajectory_mean"] = early_trajectory_summary(records)
        summary[group_name] = row
    return summary


# -------------------------------------------------------------------- plots --

def plot_bar_scale_ratio(summary: dict, out_dir: Path) -> None:
    names = list(GROUPS.keys())
    p95_vals = [summary[g].get("scale_ratio_p95_mean", float("nan")) for g in names]
    frac_vals = [summary[g].get("scale_ratio_frac_over_10_mean", float("nan")) for g in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].bar(names, p95_vals, color="tab:blue")
    axes[0].set_title("scale_ratio.p95 (100帧均值)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(names, frac_vals, color="tab:blue")
    axes[1].set_title("scale_ratio.frac_over_10 (100帧均值)")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Final-step tail severity by group (n=100 per group)")
    plt.tight_layout()
    plt.savefig(out_dir / "01_bar_scale_ratio.png", dpi=150)
    print(f"[Saved] {out_dir / '01_bar_scale_ratio.png'}")
    plt.close(fig)


def plot_box_distributions(all_records: dict, out_dir: Path) -> None:
    names = list(GROUPS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key, title in zip(
        axes,
        ["n_gaussians", "extent_overshoot", "dbscan_floater_frac"],
        ["n_gaussians", "extent_overshoot", "dbscan_floater_frac"],
    ):
        data = [[r[key] for r in all_records[g] if r["status"] == "ok"] for g in names]
        ax.boxplot(data, labels=names)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Distribution across 100 frames per group")
    plt.tight_layout()
    plt.savefig(out_dir / "02_box_distributions.png", dpi=150)
    print(f"[Saved] {out_dir / '02_box_distributions.png'}")
    plt.close(fig)


def plot_bbox_extent_timeseries(all_records: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for group_name, records in all_records.items():
        ok = sorted([r for r in records if r["status"] == "ok"], key=lambda r: r["frame"])
        xs = [r["frame"] for r in ok]
        ys = [r["bbox_extent_max"] for r in ok]
        ax.plot(xs, ys, marker="o", markersize=2, linewidth=1, label=group_name)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("bbox_extent (max component)")
    ax.set_title("bbox_extent over 100 consecutive frames, by group")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "03_bbox_extent_timeseries.png", dpi=150)
    print(f"[Saved] {out_dir / '03_bbox_extent_timeseries.png'}")
    plt.close(fig)


def plot_early_trajectory(summary: dict, out_dir: Path) -> None:
    """n_gaussians vs step 早期轨迹(0/100/.../1999)，6组一起画，重点看0-500这段。
    hull加密3组用实线，非加密3组用虚线，便于对比"断崖下跌"猜测。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for group_name in GROUPS:
        traj = summary[group_name].get("n_gaussians_trajectory_mean", [])
        if not traj:
            continue
        xs = [p["step"] for p in traj]
        ys = [p["n_gaussians_mean"] for p in traj]
        style = "-" if group_name in USES_DENSE_HULL else "--"
        for ax in axes:
            ax.plot(xs, ys, style, marker="o", markersize=3, linewidth=1.3, label=group_name)

    for ax, xlim, title in [
        (axes[0], None, "n_gaussians vs step (full 0-1999)"),
        (axes[1], (0, 500), "n_gaussians vs step (zoom 0-500)"),
    ]:
        ax.axvline(50, color="grey", linestyle=":", linewidth=1, label="warmup_end=50" if ax is axes[1] else None)
        ax.set_xlabel("step")
        ax.set_ylabel("n_gaussians (mean across 100 frames)")
        ax.set_title(title)
        if xlim:
            ax.set_xlim(*xlim)
        ax.legend(fontsize=7)
    fig.suptitle("Early n_gaussians trajectory — solid=hull_dense (H1/H4/H5), dashed=sparse (H2/H3/H6)")
    plt.tight_layout()
    plt.savefig(out_dir / "04_early_trajectory.png", dpi=150)
    print(f"[Saved] {out_dir / '04_early_trajectory.png'}")
    plt.close(fig)


# --------------------------------------------------------------------- main --

def main():
    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    log_progress("=== batch start ===")

    prepare_all_dense_frames()
    hull_cache_sparse = hull_eps_cache(DATA_BASE_DIR, "init_points.ply")
    hull_cache_dense = hull_eps_cache(DATA_DENSE_DIR, DENSE_PLY_NAME)

    all_records = {}
    for group_name, extra_args in GROUPS.items():
        all_records[group_name] = run_group(group_name, extra_args, hull_cache_sparse, hull_cache_dense)
        # 每组跑完立刻落盘一次原始记录，防止中途中断丢数据
        with open(OUT_BASE_DIR / "raw_records.json", "w") as f:
            json.dump(all_records, f, indent=2)

    summary = summarize(all_records)
    with open(OUT_BASE_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {OUT_BASE_DIR / 'summary.json'}")

    plot_bar_scale_ratio(summary, OUT_BASE_DIR)
    plot_box_distributions(all_records, OUT_BASE_DIR)
    plot_bbox_extent_timeseries(all_records, OUT_BASE_DIR)
    plot_early_trajectory(summary, OUT_BASE_DIR)

    log_progress("=== batch done ===")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke_test()
    else:
        main()
