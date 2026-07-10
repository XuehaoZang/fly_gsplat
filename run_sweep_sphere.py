"""
Sphere sweep v2: reuse the 4-camera calibration under data/ctrl_009_002_test,
generate uniform / striped textured sphere datasets, and test how packaged
"timing_profile" hyperparameters (warmup-length / refine-every /
stop-split-at / stop-screen-size-at / reset-alpha-every / resolution-schedule)
plus scale regularization affect spiky gaussians (scale_ratio) and
post-densify recovery, across two compute budgets (2k / 20k iterations).

26 groups total:
  2 budget x 3 timing_profile x 2 scale_reg x 2 texture = 24 tuned groups
  2 budget x 1 baseline (scale_reg=off, texture=uniform only, no 2x2 cross)  = 2 groups

`nerfstudio-data --eval-mode all` is always appended for every group (including
baseline) so the train/eval split logic is identical across the whole sweep --
it is a dataparser-level setting, not a model hyperparameter being tuned.

Caveat: hull_extent is read from data_dir/init_sphere.ply (the synthetic
ground truth sphere), not init_points.ply, because this synthetic dataset
never writes an init_points.ply (that file only exists for real fly data).
"""
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from debug.generate_sphere_dataset import generate_synthetic_dataset
from utils.ply import load_ply, unrescale
import subprocess

REPO = Path(__file__).resolve().parent
SRC_DIR = REPO / "data" / "ctrl_009_002_test"

BASE_NAME = "sphere_sweep_v2"
DATA_BASE_DIR = REPO / "data" / BASE_NAME
OUT_BASE_DIR = Path("outputs") / BASE_NAME

# White background everywhere; only the sphere texture differs between datasets
DATASETS = {
    "uniform": dict(color_mode="GRAY", bg_color=255, fg_color=128,
                     texture_mode="uniform", generate_mask=False),
    "striped": dict(color_mode="GRAY", bg_color=255, fg_color=180, fg_color2=80,
                     texture_mode="striped", n_stripes=8, point_px_radius=3,
                     generate_mask=False),
}
TEXTURES = list(DATASETS.keys())

# nerfstudio SplatfactoModelConfig defaults (from `ns-train splatfacto-checkpoint --help`).
# Used as the nominal value for metrics whenever a profile intentionally leaves a flag unset.
DEFAULTS = dict(warmup_length=500, refine_every=100, stop_split_at=15000,
                 stop_screen_size_at=4000, reset_alpha_every=30, resolution_schedule=3000)

BUDGETS = {
    "2k":  dict(max_iters=2000, stats_every=50),
    "20k": dict(max_iters=20000, stats_every=500),
}

PROFILES = ["early_finish", "mid_finish", "late_finish"]

# per (budget, profile) overrides; a key absent/None means "leave at nerfstudio default, don't pass the flag"
TIMING_TABLE = {
    ("2k", "early_finish"): dict(warmup_length=150, refine_every=75, stop_split_at=800,
                                  stop_screen_size_at=800, reset_alpha_every=4),
    ("2k", "mid_finish"):   dict(warmup_length=250, refine_every=75, stop_split_at=1300,
                                  stop_screen_size_at=1300, reset_alpha_every=6),
    ("2k", "late_finish"):  dict(warmup_length=350, refine_every=75, stop_split_at=1800,
                                  stop_screen_size_at=1800, reset_alpha_every=8),
    ("2k", "baseline"):     dict(),
    ("20k", "early_finish"): dict(warmup_length=1500, refine_every=750, stop_split_at=8000,
                                   stop_screen_size_at=8000, reset_alpha_every=4, resolution_schedule=21000),
    ("20k", "mid_finish"):   dict(warmup_length=2500, refine_every=750, stop_split_at=13000,
                                   stop_screen_size_at=13000, reset_alpha_every=6, resolution_schedule=21000),
    ("20k", "late_finish"):  dict(warmup_length=3500, refine_every=750, stop_split_at=18000,
                                   stop_screen_size_at=18000, reset_alpha_every=8, resolution_schedule=21000),
    ("20k", "baseline"):     dict(),
}

TIMING_FLAG_MAP = {
    "warmup_length": "--pipeline.model.warmup-length",
    "refine_every": "--pipeline.model.refine-every",
    "stop_split_at": "--pipeline.model.stop-split-at",
    "stop_screen_size_at": "--pipeline.model.stop-screen-size-at",
    "reset_alpha_every": "--pipeline.model.reset-alpha-every",
    "resolution_schedule": "--pipeline.model.resolution-schedule",
}

SCALE_REG = {
    "off": [],
    "ratio5": ["--pipeline.model.use-scale-regularization", "True",
               "--pipeline.model.max-gauss-ratio", "5.0"],
}


def timing_cli_args(budget: str, profile: str) -> list:
    overrides = TIMING_TABLE[(budget, profile)]
    args = []
    for key, flag in TIMING_FLAG_MAP.items():
        if overrides.get(key) is not None:
            args += [flag, str(overrides[key])]
    return args


def nominal_stop_split_at(budget: str, profile: str) -> int:
    return TIMING_TABLE[(budget, profile)].get("stop_split_at", DEFAULTS["stop_split_at"])


def build_groups():
    groups = []
    for budget in BUDGETS:
        for profile in PROFILES:
            for scale_reg_name, scale_reg_args in SCALE_REG.items():
                for texture in TEXTURES:
                    group_name = f"{budget}_{profile}_{scale_reg_name}_{texture}"
                    groups.append({
                        "budget": budget, "profile": profile, "scale_reg": scale_reg_name,
                        "texture": texture, "group_name": group_name,
                        "extra_args": timing_cli_args(budget, profile) + scale_reg_args,
                    })
        # single reference group per budget: pure nerfstudio defaults, off + uniform only
        groups.append({
            "budget": budget, "profile": "baseline", "scale_reg": "off", "texture": "uniform",
            "group_name": f"{budget}_baseline_off_uniform",
            "extra_args": timing_cli_args(budget, "baseline") + SCALE_REG["off"],
        })
    return groups


def prepare_datasets():
    """Generate uniform / striped sphere datasets once, skip if already present."""
    for name, kwargs in DATASETS.items():
        dst_dir = DATA_BASE_DIR / name
        if dst_dir.exists():
            print(f"[skip] {dst_dir} already exists, reusing.")
            continue
        generate_synthetic_dataset(src_dir=str(SRC_DIR), dst_dir=str(dst_dir), **kwargs)
        print(f"[Generated] {dst_dir}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_group(group: dict) -> dict:
    budget_cfg = BUDGETS[group["budget"]]
    max_iters = budget_cfg["max_iters"]
    stats_every = budget_cfg["stats_every"]

    dataset_name = group["texture"]
    group_name = group["group_name"]
    extra_args = group["extra_args"]

    data_dir = DATA_BASE_DIR / dataset_name
    exp_name = f"{BASE_NAME}/{group_name}"
    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(data_dir),
        "--vis", "tensorboard",
        "--max-num-iterations", str(max_iters),
        "--pipeline.model.background-color", "white",
        "--pipeline.model.stats-every", str(stats_every),
        # points are only needed at the final step for low_opacity_frac, so keep
        # save-points on but set points-every past max_iters to skip periodic dumps
        "--pipeline.model.save-points", "True",
        "--pipeline.model.points-every", str(max_iters + 1),
        "--experiment-name", exp_name,
    ] + extra_args + [
        "nerfstudio-data", "--eval-mode", "all"
    ]

    print(f"\n{'=' * 20} {exp_name} {'=' * 20}")
    print(" ".join(cmd))

    t0 = time.time()
    subprocess.run(cmd, check=True)
    wall_time_sec = time.time() - t0

    splat_dir = sorted((Path("outputs") / exp_name / "splatfacto-checkpoint").iterdir())[-1]
    ckpt_dir = splat_dir / "debug_checkpoints"

    stats_files = sorted((ckpt_dir / "stats").glob("step_*_stats.json"))
    if not stats_files:
        raise RuntimeError(f"no stats.json found under {ckpt_dir / 'stats'}")
    curve = sorted((json.loads(p.read_text()) for p in stats_files), key=lambda s: s["step"])
    final = curve[-1]

    points_files = sorted((ckpt_dir / "points").glob("step_*_gaussians.npz"))
    if not points_files:
        raise RuntimeError(f"no gaussians.npz found under {ckpt_dir / 'points'}")
    final_points = np.load(points_files[-1])
    final_opacities = sigmoid(final_points["opacities"].squeeze(-1))
    low_opacity_frac = float((final_opacities < 0.05).mean())

    hull_pts = load_ply(data_dir / "init_sphere.ply")
    hull_extent = hull_pts.max(0) - hull_pts.min(0)

    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R_ns = np.array(dp["transform"])[:3, :3]
    t_ns = np.array(dp["transform"])[:3, 3]
    scale = float(dp["scale"])

    stop_split_at = nominal_stop_split_at(group["budget"], group["profile"])
    metrics = compute_post_metrics(curve, stop_split_at, final, hull_extent, R_ns, t_ns, scale)

    return {
        "budget": group["budget"],
        "profile": group["profile"],
        "texture": group["texture"],
        "scale_reg": group["scale_reg"],
        "group": group_name,
        "stop_split_at": stop_split_at,
        "max_iters": max_iters,
        "wall_time_sec": wall_time_sec,
        "n_gaussians_final": final.get("n_gaussians"),
        "final_scale_ratio_median": final["scale_ratio"]["median"],
        "final_scale_ratio_p90": final["scale_ratio"]["p90"],
        "final_scale_ratio_p95": final["scale_ratio"]["p95"],
        "final_scale_ratio_max": final["scale_ratio"]["max"],
        "final_scale_ratio_frac_over_10": final["scale_ratio"]["frac_over_10"],
        "low_opacity_frac": low_opacity_frac,
        "exp_dir": str(splat_dir),
        **metrics,
        "_curve": curve,  # kept in-memory only for plotting, stripped before summary.json
    }


def compute_post_metrics(curve: list, stop_split_at: int, final: dict, hull_extent: np.ndarray,
                          R_ns: np.ndarray, t_ns: np.ndarray, scale: float) -> dict:
    with_ratio = [s for s in curve if "scale_ratio" in s]
    step_at_split = min(with_ratio, key=lambda s: abs(s["step"] - stop_split_at))
    median_at_split = step_at_split["scale_ratio"]["median"]
    final_median = final["scale_ratio"]["median"]

    # If the run ended before the nominal stop_split_at was ever reached (e.g. baseline
    # groups leave stop-split-at at its nerfstudio default of 15000, far past a 2k-iter
    # budget), there is no post-densify recovery window to measure -- recovery_delta and
    # time_to_stable are undefined rather than 0 / a huge negative number.
    split_finished_within_budget = step_at_split["step"] >= stop_split_at

    if not split_finished_within_budget:
        recovery_delta = float("nan")
        recovery_delta_pct = float("nan")
        time_to_stable = -1
    else:
        recovery_delta = median_at_split - final_median
        recovery_delta_pct = recovery_delta / median_at_split * 100 if median_at_split != 0 else float("nan")

        post_split = [s for s in with_ratio if s["step"] >= step_at_split["step"]]
        tol = 0.10 * abs(final_median)
        time_to_stable = -1
        for i, s in enumerate(post_split):
            tail = post_split[i:]
            if all(abs(t["scale_ratio"]["median"] - final_median) <= tol for t in tail):
                time_to_stable = s["step"] - stop_split_at
                break

    n_splits_total = step_at_split["n_gaussians"] - curve[0]["n_gaussians"]

    # Transform the axis-aligned bbox corners (training/rescaled frame) back into physical
    # space before comparing to hull_extent -- rotation can reorient axes, so we transform
    # all 8 corners and re-take the AABB rather than naively dividing by the scalar `scale`.
    bbox_min = np.array(final["bbox_min"])
    bbox_max = np.array(final["bbox_max"])
    corners = np.array(list(itertools.product(*zip(bbox_min, bbox_max))))  # (8, 3)
    corners_physical = unrescale(corners, R_ns, t_ns, scale)
    final_extent_physical = corners_physical.max(0) - corners_physical.min(0)
    extent_overshoot = float(final_extent_physical.max() / hull_extent.max())

    return {
        "recovery_delta": recovery_delta,
        "recovery_delta_pct": recovery_delta_pct,
        "time_to_stable": time_to_stable,
        "n_splits_total": n_splits_total,
        "extent_overshoot": extent_overshoot,
        "split_finished_within_budget": split_finished_within_budget,
    }


def fit_trend_line(ax, x: list, y: list):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.ptp(x[mask]) == 0:
        return
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 50)
    ax.plot(xs, slope * xs + intercept, linestyle='--', color='gray', alpha=0.7, label="trend")


MARKERS = {"uniform": "o", "striped": "^"}
COLORS = {"off": "tab:blue", "ratio5": "tab:orange"}
HATCHES = {"uniform": "", "striped": "//"}


def scatter_by_group(ax, records: list, x_key: str, y_key: str):
    for r in records:
        ax.scatter(r[x_key], r[y_key],
                   marker=MARKERS[r["texture"]], color=COLORS[r["scale_reg"]],
                   s=60, edgecolor='black', linewidth=0.5)
    fit_trend_line(ax, [r[x_key] for r in records], [r[y_key] for r in records])

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker=MARKERS[t], color='w', markerfacecolor='gray',
               markeredgecolor='black', label=f"texture={t}", markersize=8)
        for t in TEXTURES
    ] + [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS[s],
               label=f"scale_reg={s}", markersize=8)
        for s in SCALE_REG
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc='best')


def plot_hypothesis_check(records: list):
    """tail_length vs recovery_delta_pct, faceted by budget. Baseline groups excluded --
    their nominal stop_split_at (15000, unmodified default) is not part of the tuned
    early/mid/late hypothesis and would badly skew the x-axis scale."""
    tuned = [r for r in records if r["profile"] in PROFILES]
    if not tuned:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, budget in zip(axes, BUDGETS):
        sub = [r for r in tuned if r["budget"] == budget]
        if not sub:
            continue
        for r in sub:
            r["_tail_length"] = r["max_iters"] - r["stop_split_at"]
        scatter_by_group(ax, sub, "_tail_length", "recovery_delta_pct")
        ax.set_xlabel("tail_length (max_iters - stop_split_at)")
        ax.set_ylabel("recovery_delta_pct (%)")
        ax.set_title(f"budget={budget}")
    fig.suptitle("Recovery Delta % vs Tail Length (tuned profiles only)")
    plt.tight_layout()
    out_path = OUT_BASE_DIR / "hypothesis_check.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


def plot_confound_check(records: list):
    """n_splits_total vs recovery_delta_pct, faceted by budget. Same exclusion as above."""
    tuned = [r for r in records if r["profile"] in PROFILES]
    if not tuned:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, budget in zip(axes, BUDGETS):
        sub = [r for r in tuned if r["budget"] == budget]
        if not sub:
            continue
        scatter_by_group(ax, sub, "n_splits_total", "recovery_delta_pct")
        ax.set_xlabel("n_splits_total")
        ax.set_ylabel("recovery_delta_pct (%)")
        ax.set_title(f"budget={budget}")
    fig.suptitle("Recovery Delta % vs N Splits (confound check, tuned profiles only)")
    plt.tight_layout()
    out_path = OUT_BASE_DIR / "confound_check.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


def plot_convergence(records: list):
    if not records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, budget in zip(axes, BUDGETS):
        sub = [r for r in records if r["budget"] == budget]
        for r in sub:
            steps = [s["step"] for s in r["_curve"] if "scale_ratio" in s]
            medians = [s["scale_ratio"]["median"] for s in r["_curve"] if "scale_ratio" in s]
            ax.plot(steps, medians, marker='o', markersize=2, linewidth=1, label=r["group"])
            ax.axvline(r["stop_split_at"], linestyle=':', color='gray', alpha=0.15)
        ax.set_xlabel("step")
        ax.set_ylabel("scale_ratio median")
        ax.set_title(f"budget={budget}")
        ax.legend(fontsize=5, ncol=2)
    fig.suptitle("Scale Ratio Median Convergence (dotted lines = stop_split_at)")
    plt.tight_layout()
    out_path = OUT_BASE_DIR / "convergence_all_groups.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


def plot_marginal_effects(records: list):
    if not records:
        return
    x_categories = PROFILES + ["baseline"]
    combos = [(sr, tx) for sr in SCALE_REG for tx in TEXTURES]
    n_combos = len(combos)
    width = 0.8 / n_combos

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, budget in zip(axes, BUDGETS):
        sub = [r for r in records if r["budget"] == budget]
        for ci, (sr, tx) in enumerate(combos):
            positions, data = [], []
            for pi, profile in enumerate(x_categories):
                match = [r for r in sub if r["profile"] == profile and r["scale_reg"] == sr and r["texture"] == tx]
                if match:
                    positions.append(pi + (ci - (n_combos - 1) / 2) * width)
                    data.append([match[0]["recovery_delta_pct"]])
            if not data:
                continue
            bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True)
            for box in bp['boxes']:
                box.set(facecolor=COLORS[sr], hatch=HATCHES[tx], alpha=0.6)
        ax.set_xticks(range(len(x_categories)))
        ax.set_xticklabels(x_categories, rotation=20)
        ax.set_ylabel("recovery_delta_pct (%)")
        ax.set_title(f"budget={budget}")

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=COLORS[sr], hatch=HATCHES[tx], alpha=0.6, label=f"{sr}/{tx}")
                    for sr, tx in combos]
    axes[-1].legend(handles=legend_elems, fontsize=8)
    fig.suptitle("Recovery Delta % by Timing Profile (color=scale_reg, hatch=texture)")
    plt.tight_layout()
    out_path = OUT_BASE_DIR / "marginal_effects.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


def print_summary_table(records: list):
    ranked = [r for r in records if r["split_finished_within_budget"]]
    not_finished = [r for r in records if not r["split_finished_within_budget"]]
    rows = sorted(ranked, key=lambda r: r["recovery_delta_pct"], reverse=True)

    header = (f"{'group':<34}{'budget':>8}{'recovery_delta_pct':>20}{'time_to_stable':>16}"
              f"{'n_splits':>10}{'extent_overshoot':>18}{'low_opacity_frac':>18}{'frac_over_10':>14}")
    print("\n" + "=" * len(header))
    print("SUMMARY (sorted by recovery_delta_pct, descending)")
    print("=" * len(header))
    print(header)
    for r in rows:
        print(f"{r['group']:<34}{r['budget']:>8}{r['recovery_delta_pct']:>20.2f}{r['time_to_stable']:>16d}"
              f"{r['n_splits_total']:>10d}{r['extent_overshoot']:>18.3f}{r['low_opacity_frac']:>18.3f}"
              f"{r['final_scale_ratio_frac_over_10']:>14.3f}")

    if not_finished:
        print("\nSPLIT NEVER FINISHED WITHIN BUDGET (stop_split_at default exceeds max_iters, "
              "recovery_delta/time_to_stable not applicable):")
        print("  " + ", ".join(r["group"] for r in not_finished))

    unstable = [r["group"] for r in ranked if r["time_to_stable"] == -1]
    print("\nUNSTABLE GROUPS (time_to_stable == -1, never settled within final_median +/- 10%):")
    print("  " + (", ".join(unstable) if unstable else "none"))


def main():
    prepare_datasets()

    results, failed = [], []
    for group in build_groups():
        try:
            results.append(run_group(group))
        except Exception as e:
            print(f"[{group['group_name']}] failed: {e}")
            failed.append({"group": group["group_name"], "error": str(e)})

    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    flat_results = [{k: v for k, v in r.items() if k != "_curve"} for r in results]
    with open(OUT_BASE_DIR / "summary.json", "w") as f:
        json.dump({"results": flat_results, "failed": failed}, f, indent=2)
    print(f"[Saved] {OUT_BASE_DIR / 'summary.json'}")
    if failed:
        print(f"[Warning] {len(failed)} group(s) failed: {[f['group'] for f in failed]}")

    plot_hypothesis_check(results)
    plot_confound_check(results)
    plot_convergence(results)
    plot_marginal_effects(results)
    print_summary_table(results)


if __name__ == "__main__":
    main()
