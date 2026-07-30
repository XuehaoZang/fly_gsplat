"""
8组(G2/G2b/G3/G6/G7/G8/G9/G10) x 100帧(f0000-f0099)正式批次，共800次训练。
除G3/G6外，其余6组统一固定用G3的densify窗口(warmup-length 50, stop-split-at 1800)，
保证各自只相对baseline改了自己那一个变量。数据生成(generate_dataset+generate_hull)
按帧只做一次、8组共用，已存在的帧(检测transforms.json+init_points.ply)自动跳过。

这轮全部走final-state：不加stats-every/save-points等checkpoint对齐flag，
默认splatfacto-checkpoint(save_stats=True, stats_every=1000)配合final_step强制dump，
免费拿到每帧最后一步(step=max_iters-1=1999)的stats.json。extent_overshoot/
dbscan_floater_frac：export_splat产出splat.ply后读取+反归一化现算。

通宵执行：外层按组循环，每组100帧跑完后往batch_progress.log追加一行状态再继续下一组，
单帧失败不中断整体批次(try/except记录，继续下一帧)。
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from generate_dataset import generate_dataset
from generate_hull import generate_hull
from utils.ply import export_splat, load_ply, load_ply_with_attrs, unrescale, clean_ply

REPO = Path(__file__).resolve().parent.parent.parent
SPARSE_DIR = r"X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002"
BASE_NAME = "ctrl_009_002"
SWEEP_NAME = "ctrl_009_002_8groups_100frames"
FRAMES = list(range(100))
MAX_ITERS = 2000

DATA_BASE_DIR = REPO / "data" / BASE_NAME
OUT_BASE_DIR = REPO / "outputs" / SWEEP_NAME
PROGRESS_LOG = OUT_BASE_DIR / "batch_progress.log"

# 除 G3/G6 外统一复用的densify窗口
_COMMON_DENSIFY = ["--pipeline.model.warmup-length", "50", "--pipeline.model.stop-split-at", "1800"]

GROUPS = {
    "G2_scale_reg_ratio1": ["--pipeline.model.use-scale-regularization", "True",
                             "--pipeline.model.max-gauss-ratio", "1.0"] + _COMMON_DENSIFY,
    "G2b_scale_reg_ratio3": ["--pipeline.model.use-scale-regularization", "True",
                              "--pipeline.model.max-gauss-ratio", "3.0"] + _COMMON_DENSIFY,
    "G3_densify_50_1800": list(_COMMON_DENSIFY),
    "G6_densify_200_1200": ["--pipeline.model.warmup-length", "200",
                             "--pipeline.model.stop-split-at", "1200"],
    "G7_cull_strict": ["--pipeline.model.cull-alpha-thresh", "0.2",
                        "--pipeline.model.cull-scale-thresh", "0.3",
                        "--pipeline.model.cull-screen-size", "0.10"] + _COMMON_DENSIFY,
    "G8_cull_loose": ["--pipeline.model.cull-alpha-thresh", "0.05",
                       "--pipeline.model.cull-scale-thresh", "0.8",
                       "--pipeline.model.cull-screen-size", "0.25"] + _COMMON_DENSIFY,
    "G9_sh_degree_0": ["--pipeline.model.sh-degree", "0"] + _COMMON_DENSIFY,
    "G10_use_absgrad_false": ["--pipeline.model.use-absgrad", "False"] + _COMMON_DENSIFY,
    "G2_G9": ["--pipeline.model.use-scale-regularization", "True",
              "--pipeline.model.max-gauss-ratio", "1.0",
              "--pipeline.model.sh-degree", "0"] + _COMMON_DENSIFY,
    "G2b_G9": ["--pipeline.model.use-scale-regularization", "True",
               "--pipeline.model.max-gauss-ratio", "3.0",
               "--pipeline.model.sh-degree", "0"] + _COMMON_DENSIFY,
}

NEW_GROUPS = ["G2_G9", "G2b_G9"]  # 本次只跑这2组，其余8组复用已有结果不重跑


def log_progress(line: str) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")
    print(line)


# ---------------------------------------------------------------- data prep --

def prepare_all_frames() -> None:
    for frame_idx in FRAMES:
        data_dir = DATA_BASE_DIR / f"f{frame_idx:04d}"
        if (data_dir / "transforms.json").exists() and (data_dir / "init_points.ply").exists():
            print(f"[skip] frame {frame_idx} data already exists")
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        generate_dataset(str(data_dir), SPARSE_DIR, target_frame=frame_idx,
                          if_crop=False, white_bg=True, if_mask=False, calib_dir=str(DATA_BASE_DIR))
        generate_hull(str(data_dir), if_viser=False)
        print(f"[Generated] frame {frame_idx}")


def hull_eps_cache() -> dict:
    """(hull_extent, eps) 每帧算一次，8组共用同一份hull。eps = 2.5x hull点间中位最近邻距离。"""
    cache = {}
    for frame_idx in FRAMES:
        data_dir = DATA_BASE_DIR / f"f{frame_idx:04d}"
        hull_pts = load_ply(data_dir / "init_points.ply")
        hull_extent = hull_pts.max(0) - hull_pts.min(0)
        nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
        dists, _ = nn.kneighbors(hull_pts)
        median_nn = float(np.median(dists[:, 1]))
        cache[frame_idx] = (hull_extent, 2.5 * median_nn)
    return cache


# ------------------------------------------------------------------ training --

def run_group_frame(group_name: str, extra_args: list, frame_idx: int,
                     hull_extent: np.ndarray, eps: float) -> dict:
    data_dir = DATA_BASE_DIR / f"f{frame_idx:04d}"
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
    final = json.loads(stats_files[-1].read_text())
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


def run_group(group_name: str, extra_args: list, hull_cache: dict) -> list:
    t0 = time.time()
    records = []
    for frame_idx in FRAMES:
        hull_extent, eps = hull_cache[frame_idx]
        try:
            records.append(run_group_frame(group_name, extra_args, frame_idx, hull_extent, eps))
        except Exception as e:
            print(f"[{group_name}/f{frame_idx:04d}] failed: {e}")
            records.append({"group": group_name, "frame": frame_idx, "status": "failed", "error": str(e)})

    n_ok = sum(r["status"] == "ok" for r in records)
    elapsed = time.time() - t0
    log_progress(f"group={group_name} success={n_ok}/{len(FRAMES)} failed={len(FRAMES) - n_ok} "
                 f"wall_time={elapsed:.1f}s")
    return records


def run_new_groups() -> None:
    """只跑 NEW_GROUPS，合并进已有的 raw_records.json/summary.json，不动其余8组，图重画成10组版本。"""
    log_progress(f"=== new groups start: {NEW_GROUPS} ===")

    hull_cache = hull_eps_cache()  # 只读取已存在的 init_points.ply，不重新生成数据

    raw_path = OUT_BASE_DIR / "raw_records.json"
    with open(raw_path) as f:
        all_records = json.load(f)

    for group_name in NEW_GROUPS:
        all_records[group_name] = run_group(group_name, GROUPS[group_name], hull_cache)
        with open(raw_path, "w") as f:
            json.dump(all_records, f, indent=2)

    summary = summarize(all_records)
    with open(OUT_BASE_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {OUT_BASE_DIR / 'summary.json'}")

    plot_bar_scale_ratio(summary, OUT_BASE_DIR)
    plot_box_distributions(all_records, OUT_BASE_DIR)
    plot_bbox_extent_timeseries(all_records, OUT_BASE_DIR)

    log_progress("=== new groups done ===")


def smoke_test_new_groups(frames=(0, 1, 2)) -> None:
    """冒烟：只跑 NEW_GROUPS x 少量帧，验证命令能跑通，不写入 summary/raw_records。"""
    for frame_idx in frames:
        data_dir = DATA_BASE_DIR / f"f{frame_idx:04d}"
        hull_pts = load_ply(data_dir / "init_points.ply")
        hull_extent = hull_pts.max(0) - hull_pts.min(0)
        nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
        dists, _ = nn.kneighbors(hull_pts)
        eps = 2.5 * float(np.median(dists[:, 1]))
        for group_name in NEW_GROUPS:
            result = run_group_frame(group_name, GROUPS[group_name], frame_idx, hull_extent, eps)
            print(result)


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


def summarize(all_records: dict) -> dict:
    """all_records: group_name -> list of per-frame records."""
    summary = {}
    for group_name, records in all_records.items():
        ok = [r for r in records if r["status"] == "ok"]
        n_ok = len(ok)
        row = {"success_rate": n_ok / len(FRAMES)}
        if ok:
            for key in ["scale_ratio_median", "scale_ratio_p95", "scale_ratio_frac_over_10",
                        "opacity_median", "low_opacity_frac", "n_gaussians",
                        "extent_overshoot", "dbscan_floater_frac"]:
                vals = [r[key] for r in ok]
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_median"] = float(np.median(vals))
        row.update(jitter_score(records))
        summary[group_name] = row
    return summary


# -------------------------------------------------------------------- plots --

def plot_bar_scale_ratio(summary: dict, out_dir: Path) -> None:
    names = list(GROUPS.keys())
    p95_vals = [summary[g].get("scale_ratio_p95_mean", float("nan")) for g in names]
    frac_vals = [summary[g].get("scale_ratio_frac_over_10_mean", float("nan")) for g in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
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
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
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


# --------------------------------------------------------------------- main --

def main():
    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    log_progress("=== batch start ===")

    prepare_all_frames()
    hull_cache = hull_eps_cache()

    all_records = {}
    for group_name, extra_args in GROUPS.items():
        all_records[group_name] = run_group(group_name, extra_args, hull_cache)
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

    log_progress("=== batch done ===")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke_test_new_groups()
    elif len(sys.argv) > 1 and sys.argv[1] == "new_groups":
        run_new_groups()
    else:
        main()
