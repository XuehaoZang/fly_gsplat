"""
单帧 densify 参数 sweep：复用 data/ctrl_009_002/f0010 的现有数据（images/transforms.json/
init_points.ply），只跑训练阶段，测试 6 组超参数配置，对比收敛曲线。
"""
import json
import shutil
import subprocess
from pathlib import Path

from debug.debug_checkpoints import debug_checkpoints
from utils.ply import analyze_scale_ratio
import matplotlib.pyplot as plt

SOURCE_DATA_DIR = Path("./data/ctrl_009_002/f0010")

# BASE_NAME       = "test_04_sweep_cull_alpha"
# SWEEP_GROUPS = {
#     "00_baseline":           [],
#     "01_reset3":             ["--pipeline.model.reset-alpha-every", "3"],
#     "02_stopsplit6k":        ["--pipeline.model.stop-split-at", "6000"],
#     "03_cull03":             ["--pipeline.model.cull-alpha-thresh", "0.3"],
#     "04_stopsplit6k_reset3": ["--pipeline.model.stop-split-at", "6000",
#                                "--pipeline.model.reset-alpha-every", "3"],
#     "05_cull03_reset3":      ["--pipeline.model.cull-alpha-thresh", "0.3",
#                                "--pipeline.model.reset-alpha-every", "3"],
# }

BASE_NAME = "test_05_sweep_spiky"

SWEEP_GROUPS = {
    "00_stopsplit6k":        ["--pipeline.model.stop-split-at", "6000"],
    "01_scalereg":           ["--pipeline.model.stop-split-at", "6000",
                               "--pipeline.model.use-scale-regularization", "True"],
    "02_scalereg_ratio5":    ["--pipeline.model.stop-split-at", "6000",
                               "--pipeline.model.use-scale-regularization", "True",
                               "--pipeline.model.max-gauss-ratio", "5.0"],
    "03_densizethresh02":    ["--pipeline.model.stop-split-at", "6000",
                               "--pipeline.model.densify-size-thresh", "0.02"],
    "04_antialiased":        ["--pipeline.model.stop-split-at", "6000",
                               "--pipeline.model.rasterize-mode", "antialiased"],
    "05_combo":              ["--pipeline.model.stop-split-at", "6000",
                               "--pipeline.model.use-scale-regularization", "True",
                               "--pipeline.model.densify-size-thresh", "0.02"],
}

DATA_DIR        = Path(f"./data/{BASE_NAME}")
MAX_ITERS       = 20000
CHECKPOINT_EVERY = 2000

def prepare_data():
    """把 f0010 现有数据复制到 sweep 专用目录，只做一次，六组共用同一份初始化。"""
    if DATA_DIR.exists():
        print(f"[skip] {DATA_DIR} already exists, reusing.")
        return
    shutil.copytree(SOURCE_DATA_DIR, DATA_DIR)
    print(f"[Copied] {SOURCE_DATA_DIR} -> {DATA_DIR}")


def run_group(group_name: str, extra_args: list) -> dict:
    exp_name = f"{BASE_NAME}/{group_name}"
    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(DATA_DIR),
        "--vis", "tensorboard",
        "--max-num-iterations", str(MAX_ITERS),
        "--pipeline.model.background-color", "white",
        "--pipeline.model.checkpoint-every", str(CHECKPOINT_EVERY),
        "--experiment-name", exp_name,
    ] + extra_args + [
        "nerfstudio-data", "--eval-mode", "all"
    ]

    print(f"\n{'='*20} {group_name} {'='*20}")
    subprocess.run(cmd, check=True)

    splat_dir = sorted((Path("outputs") / exp_name / "splatfacto-checkpoint").iterdir())[-1]
    result = debug_checkpoints(
        data_dir=str(DATA_DIR),
        splat_dir=str(splat_dir),
        checkpoint_dir=str(splat_dir / "debug_checkpoints"),
    )
    result["group"] = group_name

    scale_stats = analyze_scale_ratio(splat_dir / "splat.ply")
    result["scale_stats"] = scale_stats
    print(f"[{group_name}] scale ratio: median={scale_stats.get('median'):.2f} "
          f"p95={scale_stats.get('p95'):.2f} frac_over_10={scale_stats.get('frac_over_10'):.2%}")

    return result


def plot_comparison(results: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        axes[0].plot(r["steps"], r["n_raw"], marker='o', label=r["group"])

        extents = r["extents"]
        x_extent = [e[0] for e in extents]
        axes[1].plot(r["steps"], x_extent, marker='o', label=r["group"])

    hull_extent_x = results[0]["hull_extent"][0]
    axes[1].axhline(hull_extent_x, linestyle='--', color='gray', alpha=0.5, label="hull X")

    axes[0].set_title("gaussian count (raw)")
    axes[0].set_xlabel("step")
    axes[0].legend(fontsize=7)

    axes[1].set_title("extent X vs step (dashed = hull)")
    axes[1].set_xlabel("step")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    out_path = Path("outputs") / BASE_NAME / "sweep_comparison.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"\n[Saved] {out_path}")


def main():
    prepare_data()
    results = []
    # for group_name, extra_args in SWEEP_GROUPS.items():
    #     try:
    #         results.append(run_group(group_name, extra_args))
    #     except Exception as e:
    #         print(f"[{group_name}] failed: {e}")

    with open(Path("outputs") / BASE_NAME / "sweep_results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "extents"} |
                   {"extents": [list(e) for e in r["extents"]]} for r in results], f, indent=2)

    plot_comparison(results)


if __name__ == "__main__":
    main()