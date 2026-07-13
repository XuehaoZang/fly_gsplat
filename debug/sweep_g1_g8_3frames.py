"""
G1-G8 x 3帧(f0000/f0001/f0002) x 2000步，24次run，方向性预览（样本量太小，非最终结论）。
沿用冒烟测试验证过的checkpoint对齐flags：stats-every/save-points/points-every = 500。
6张图(4条trajectory + 2张final-state对比) + 1张汇总表，全部从stats.json读取；
extent_overshoot/dbscan_floater_frac作为报告问题的补充分析，额外读取hull ply + dataparser
transform(+dbscan还要读最终npz的means)，不影响6张图的"只读stats.json"范围。
"""
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils.ply import load_ply, unrescale, clean_ply

REPO = Path(__file__).resolve().parent.parent
BASE_NAME = "ctrl_009_002"
SWEEP_NAME = "ctrl_009_002_sweep_g1_g8"
FRAMES = [0, 1, 2]
MAX_ITERS = 2000
CKPT_STEPS = [0, 500, 1000, 1500, MAX_ITERS - 1]  # 见冒烟测试记录：最后一档实际落盘在 max_iters-1

CKPT_ALIGN_ARGS = [
    "--pipeline.model.stats-every", "500",
    "--pipeline.model.save-points", "True",
    "--pipeline.model.points-every", "500",
]

GROUPS = {
    "G1_scale_reg_ratio10": ["--pipeline.model.use-scale-regularization", "True",
                              "--pipeline.model.max-gauss-ratio", "10.0",
                              "--pipeline.model.warmup-length", "50",
                              "--pipeline.model.stop-split-at", "1800"],
    "G2_scale_reg_ratio1": ["--pipeline.model.use-scale-regularization", "True",
                             "--pipeline.model.max-gauss-ratio", "1.0",
                             "--pipeline.model.warmup-length", "50",
                             "--pipeline.model.stop-split-at", "1800"],
    "G3_densify_50_1800": ["--pipeline.model.warmup-length", "50",
                            "--pipeline.model.stop-split-at", "1800"],
    "G4_densify_50_1200": ["--pipeline.model.warmup-length", "50",
                            "--pipeline.model.stop-split-at", "1200"],
    "G5_densify_200_1800": ["--pipeline.model.warmup-length", "200",
                             "--pipeline.model.stop-split-at", "1800"],
    "G6_densify_200_1200": ["--pipeline.model.warmup-length", "200",
                             "--pipeline.model.stop-split-at", "1200"],
    "G7_cull_strict": ["--pipeline.model.cull-alpha-thresh", "0.2",
                        "--pipeline.model.cull-scale-thresh", "0.3",
                        "--pipeline.model.cull-screen-size", "0.10",
                        "--pipeline.model.warmup-length", "50",
                        "--pipeline.model.stop-split-at", "1800"],
    "G8_cull_loose": ["--pipeline.model.cull-alpha-thresh", "0.05",
                       "--pipeline.model.cull-scale-thresh", "0.8",
                       "--pipeline.model.cull-screen-size", "0.25",
                       "--pipeline.model.warmup-length", "50",
                       "--pipeline.model.stop-split-at", "1800"],
}

STOP_SPLIT_AT = {name: int(args[args.index("--pipeline.model.stop-split-at") + 1])
                  for name, args in GROUPS.items()}

SCALE_REG_CAVEAT = {"G1_scale_reg_ratio10", "G2_scale_reg_ratio1"}


def run_group_frame(group_name: str, extra_args: list, frame_idx: int) -> Path:
    data_dir = REPO / "data" / BASE_NAME / f"f{frame_idx:04d}"
    exp_name = f"{SWEEP_NAME}/{group_name}/f{frame_idx:04d}"
    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(data_dir),
        "--vis", "tensorboard",
        "--max-num-iterations", str(MAX_ITERS),
        "--pipeline.model.background-color", "white",
        "--experiment-name", exp_name,
    ] + CKPT_ALIGN_ARGS + extra_args + [
        "nerfstudio-data", "--eval-mode", "all",
    ]
    print(f"\n{'=' * 20} {group_name} / frame {frame_idx} {'=' * 20}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return sorted((REPO / "outputs" / exp_name / "splatfacto-checkpoint").iterdir())[-1]


def load_curve(splat_dir: Path) -> dict:
    """step -> stats dict"""
    stats_dir = splat_dir / "debug_checkpoints" / "stats"
    curve = {}
    for step in CKPT_STEPS:
        p = stats_dir / f"step_{step:05d}_stats.json"
        if not p.exists():
            print(f"[MISSING] {p}")
            continue
        curve[step] = json.loads(p.read_text())
    return curve


def compute_extent_overshoot(splat_dir: Path, hull_extent: np.ndarray) -> float:
    final = json.loads((splat_dir / "debug_checkpoints" / "stats" /
                         f"step_{CKPT_STEPS[-1]:05d}_stats.json").read_text())
    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R, t, scale = np.array(dp["transform"])[:3, :3], np.array(dp["transform"])[:3, 3], float(dp["scale"])
    bbox_min, bbox_max = np.array(final["bbox_min"]), np.array(final["bbox_max"])
    import itertools
    corners = np.array(list(itertools.product(*zip(bbox_min, bbox_max))))
    corners_physical = unrescale(corners, R, t, scale)
    final_extent = corners_physical.max(0) - corners_physical.min(0)
    return float(final_extent.max() / hull_extent.max())


def compute_dbscan_floater_frac(splat_dir: Path, eps_physical: float) -> float:
    points_path = (splat_dir / "debug_checkpoints" / "points" /
                    f"step_{CKPT_STEPS[-1]:05d}_gaussians.npz")
    means = np.load(points_path)["means"]
    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R, t, scale = np.array(dp["transform"])[:3, :3], np.array(dp["transform"])[:3, 3], float(dp["scale"])
    means_physical = unrescale(means, R, t, scale)
    _, removed = clean_ply(means_physical, eps=eps_physical, min_samples=5, min_cluster_frac=0.02)
    return float(len(removed) / len(means_physical)) if len(means_physical) else float("nan")


def hull_eps(frame_idx: int) -> tuple:
    """(hull_extent, eps) 每帧算一次，8组共用同一份hull。eps = 2.5x hull点间中位最近邻距离。"""
    data_dir = REPO / "data" / BASE_NAME / f"f{frame_idx:04d}"
    hull_pts = load_ply(data_dir / "init_points.ply")
    hull_extent = hull_pts.max(0) - hull_pts.min(0)
    nn = NearestNeighbors(n_neighbors=2).fit(hull_pts)
    dists, _ = nn.kneighbors(hull_pts)
    median_nn = float(np.median(dists[:, 1]))
    return hull_extent, 2.5 * median_nn


def main():
    hull_cache = {f: hull_eps(f) for f in FRAMES}

    records = []  # one row per (group, frame): {"group":..., "frame":..., "curve": {...}, "extent_overshoot":..., "dbscan_floater_frac":...}
    failed = []
    for group_name, extra_args in GROUPS.items():
        for frame_idx in FRAMES:
            try:
                splat_dir = run_group_frame(group_name, extra_args, frame_idx)
                curve = load_curve(splat_dir)
                hull_extent, eps = hull_cache[frame_idx]
                extent_overshoot = compute_extent_overshoot(splat_dir, hull_extent)
                dbscan_floater_frac = compute_dbscan_floater_frac(splat_dir, eps)
                records.append({
                    "group": group_name, "frame": frame_idx, "curve": curve,
                    "extent_overshoot": extent_overshoot,
                    "dbscan_floater_frac": dbscan_floater_frac,
                })
            except Exception as e:
                print(f"[{group_name}/f{frame_idx:04d}] failed: {e}")
                failed.append({"group": group_name, "frame": frame_idx, "error": str(e)})

    out_dir = REPO / "outputs" / SWEEP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    dump = [{k: v for k, v in r.items()} for r in records]
    with open(out_dir / "raw_records.json", "w") as f:
        json.dump({"records": dump, "failed": failed}, f, indent=2)
    print(f"[Saved] {out_dir / 'raw_records.json'}")

    plot_all(records, out_dir)
    print_summary_table(records, out_dir)


# ---------- aggregation helpers ----------

def avg_trajectory(records: list, group_name: str, path: tuple) -> dict:
    """按step取3帧平均。path例如("scale_ratio","median")或("n_gaussians",)。"""
    per_step = {s: [] for s in CKPT_STEPS}
    for r in records:
        if r["group"] != group_name:
            continue
        for step, stats in r["curve"].items():
            v = stats
            for key in path:
                v = v[key]
            per_step[step].append(v)
    return {s: float(np.mean(vs)) for s, vs in per_step.items() if vs}


def final_avg(records: list, group_name: str, path: tuple) -> float:
    vals = []
    for r in records:
        if r["group"] != group_name:
            continue
        stats = r["curve"].get(CKPT_STEPS[-1])
        if stats is None:
            continue
        v = stats
        for key in path:
            v = v[key]
        vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def plot_trajectory(records: list, path: tuple, ylabel: str, title: str, out_path: Path, logy: bool = False):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for group_name in GROUPS:
        traj = avg_trajectory(records, group_name, path)
        steps = sorted(traj.keys())
        vals = [traj[s] for s in steps]
        ax.plot(steps, vals, marker='o', markersize=3, label=group_name)
        ax.axvline(STOP_SPLIT_AT[group_name], linestyle=':', alpha=0.08, color='gray')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(dotted vlines = each group's stop_split_at, 3-frame avg)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")
    plt.close(fig)


def plot_all(records: list, out_dir: Path):
    plot_trajectory(records, ("scale_ratio", "median"), "scale_ratio median",
                     "1. scale_ratio.median trajectory", out_dir / "01_scale_ratio_median_trajectory.png")
    plot_trajectory(records, ("n_gaussians",), "n_gaussians (log)",
                     "2. n_gaussians trajectory", out_dir / "02_n_gaussians_trajectory.png", logy=True)

    # bbox_extent: 取三轴最大分量
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for group_name in GROUPS:
        per_step = {s: [] for s in CKPT_STEPS}
        for r in records:
            if r["group"] != group_name:
                continue
            for step, stats in r["curve"].items():
                per_step[step].append(max(stats["bbox_extent"]))
        steps = sorted(s for s, vs in per_step.items() if vs)
        vals = [float(np.mean(per_step[s])) for s in steps]
        ax.plot(steps, vals, marker='o', markersize=3, label=group_name)
        ax.axvline(STOP_SPLIT_AT[group_name], linestyle=':', alpha=0.08, color='gray')
    ax.set_xlabel("step")
    ax.set_ylabel("bbox_extent max component")
    ax.set_title("3. bbox_extent trajectory (max axis, 3-frame avg)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "03_bbox_extent_trajectory.png", dpi=150)
    print(f"[Saved] {out_dir / '03_bbox_extent_trajectory.png'}")
    plt.close(fig)

    # opacity median + p10, 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for group_name in GROUPS:
        traj_med = avg_trajectory(records, group_name, ("opacity", "median"))
        traj_p10 = avg_trajectory(records, group_name, ("opacity", "p10"))
        steps = sorted(traj_med.keys())
        axes[0].plot(steps, [traj_med[s] for s in steps], marker='o', markersize=3, label=group_name)
        steps10 = sorted(traj_p10.keys())
        axes[1].plot(steps10, [traj_p10[s] for s in steps10], marker='o', markersize=3, label=group_name)
    axes[0].set_title("opacity.median trajectory")
    axes[1].set_title("opacity.p10 trajectory")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("opacity")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("4. opacity trajectory (3-frame avg)")
    plt.tight_layout()
    plt.savefig(out_dir / "04_opacity_trajectory.png", dpi=150)
    print(f"[Saved] {out_dir / '04_opacity_trajectory.png'}")
    plt.close(fig)

    # final-state bar: scale_ratio.p95 + frac_over_10
    names = list(GROUPS.keys())
    p95_vals = [final_avg(records, g, ("scale_ratio", "p95")) for g in names]
    frac_vals = [final_avg(records, g, ("scale_ratio", "frac_over_10")) for g in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ['tab:orange' if g in SCALE_REG_CAVEAT else 'tab:blue' for g in names]
    axes[0].bar(names, p95_vals, color=colors)
    axes[0].set_title("scale_ratio.p95 (final step, 3-frame avg)")
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].bar(names, frac_vals, color=colors)
    axes[1].set_title("scale_ratio.frac_over_10 (final step, 3-frame avg)")
    axes[1].tick_params(axis='x', rotation=45)
    fig.suptitle("5. Final-step tail severity (orange = scale_reg groups, caveat applies)")
    plt.tight_layout()
    plt.savefig(out_dir / "05_final_scale_ratio_tail.png", dpi=150)
    print(f"[Saved] {out_dir / '05_final_scale_ratio_tail.png'}")
    plt.close(fig)

    # final-state scatter: n_gaussians vs bbox_extent(max)
    fig, ax = plt.subplots(figsize=(7, 6))
    for g in names:
        ng = final_avg(records, g, ("n_gaussians",))
        ext_vals = []
        for r in records:
            if r["group"] != g:
                continue
            stats = r["curve"].get(CKPT_STEPS[-1])
            if stats is not None:
                ext_vals.append(max(stats["bbox_extent"]))
        ext = float(np.mean(ext_vals)) if ext_vals else float("nan")
        ax.scatter(ng, ext, s=60)
        ax.annotate(g, (ng, ext), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("n_gaussians (final step, 3-frame avg)")
    ax.set_ylabel("bbox_extent max component (final step, 3-frame avg)")
    ax.set_title("6. n_gaussians vs bbox_extent (final step)")
    plt.tight_layout()
    plt.savefig(out_dir / "06_ngaussians_vs_extent_scatter.png", dpi=150)
    print(f"[Saved] {out_dir / '06_ngaussians_vs_extent_scatter.png'}")
    plt.close(fig)


def print_summary_table(records: list, out_dir: Path):
    rows = []
    for g in GROUPS:
        row = {
            "group": g,
            "n_gaussians": final_avg(records, g, ("n_gaussians",)),
            "scale_ratio_median": final_avg(records, g, ("scale_ratio", "median")),
            "scale_ratio_p90": final_avg(records, g, ("scale_ratio", "p90")),
            "scale_ratio_p95": final_avg(records, g, ("scale_ratio", "p95")),
            "scale_ratio_max": final_avg(records, g, ("scale_ratio", "max")),
            "scale_ratio_frac_over_10": final_avg(records, g, ("scale_ratio", "frac_over_10")),
            "opacity_mean": final_avg(records, g, ("opacity", "mean")),
            "opacity_median": final_avg(records, g, ("opacity", "median")),
            "opacity_p10": final_avg(records, g, ("opacity", "p10")),
            "bbox_extent_max": float(np.mean([max(r["curve"][CKPT_STEPS[-1]]["bbox_extent"])
                                               for r in records
                                               if r["group"] == g and CKPT_STEPS[-1] in r["curve"]])),
            "extent_overshoot": float(np.mean([r["extent_overshoot"] for r in records if r["group"] == g])),
            "dbscan_floater_frac": float(np.mean([r["dbscan_floater_frac"] for r in records if r["group"] == g])),
            "scale_reg_caveat": "有正则化loss拉低，跨组比较需谨慎" if g in SCALE_REG_CAVEAT else "",
        }
        rows.append(row)

    with open(out_dir / "summary_table.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[Saved] {out_dir / 'summary_table.json'}")

    header = (f"{'group':<24}{'n_gauss':>9}{'ratio_med':>11}{'ratio_p95':>11}{'frac>10':>9}"
              f"{'op_med':>8}{'op_p10':>8}{'extent':>9}{'overshoot':>11}{'floater':>9}")
    print("\n" + "=" * len(header))
    print("SUMMARY (8组 x 3帧均值, 最终态 step=%d)" % CKPT_STEPS[-1])
    print("=" * len(header))
    print(header)
    for r in sorted(rows, key=lambda x: x["scale_ratio_median"]):
        print(f"{r['group']:<24}{r['n_gaussians']:>9.0f}{r['scale_ratio_median']:>11.2f}"
              f"{r['scale_ratio_p95']:>11.2f}{r['scale_ratio_frac_over_10']:>9.2%}"
              f"{r['opacity_median']:>8.3f}{r['opacity_p10']:>8.3f}{r['bbox_extent_max']:>9.4f}"
              f"{r['extent_overshoot']:>11.3f}{r['dbscan_floater_frac']:>9.2%}")
        if r["scale_reg_caveat"]:
            print(f"    ^ {r['group']}: {r['scale_reg_caveat']}")


if __name__ == "__main__":
    main()
