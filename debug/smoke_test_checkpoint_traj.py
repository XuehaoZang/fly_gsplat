"""
冒烟测试：checkpoint频率对齐到500 + 轨迹指标提取 (G3 baseline, 3帧, 2000步)。
只跑 G3 (--pipeline.model.warmup-length 50 --pipeline.model.stop-split-at 1800)，
frame 0/1/2，验证 stats.json / gaussians.npz 在 500/1000/1500/2000 四个 step 都正常生成，
并画出 scale_ratio.median 的轨迹折线图。
"""
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
BASE_NAME = "ctrl_009_002"
FRAMES = [0, 1, 2]
MAX_ITERS = 2000
# nerfstudio 是 0-indexed 训练循环，最后一次强制 dump 发生在 step == max_num_iterations - 1，
# 不是 step == max_num_iterations，所以名义上的“2000”这一档实际文件名是 step_01999。
CKPT_STEPS = [500, 1000, 1500, MAX_ITERS - 1]

G3_ARGS = [
    "--pipeline.model.warmup-length", "50",
    "--pipeline.model.stop-split-at", "1800",
]


def run_frame(frame_idx: int) -> Path:
    data_dir = REPO / "data" / BASE_NAME / f"f{frame_idx:04d}"
    exp_name = f"{BASE_NAME}_smoke_g3/f{frame_idx:04d}"
    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(data_dir),
        "--vis", "tensorboard",
        "--max-num-iterations", str(MAX_ITERS),
        "--pipeline.model.background-color", "white",
        "--pipeline.model.stats-every", "500",
        "--pipeline.model.save-points", "True",
        "--pipeline.model.points-every", "500",
        "--experiment-name", exp_name,
    ] + G3_ARGS + [
        "nerfstudio-data", "--eval-mode", "all",
    ]
    print(f"\n{'=' * 20} frame {frame_idx} {'=' * 20}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    splat_dir = sorted((REPO / "outputs" / exp_name / "splatfacto-checkpoint").iterdir())[-1]
    return splat_dir


def collect_trajectory(splat_dirs: dict) -> dict:
    """frame_idx -> {step: scale_ratio_median}"""
    traj = {}
    for frame_idx, splat_dir in splat_dirs.items():
        stats_dir = splat_dir / "debug_checkpoints" / "stats"
        points_dir = splat_dir / "debug_checkpoints" / "points"
        per_step = {}
        for step in CKPT_STEPS:
            stats_path = stats_dir / f"step_{step:05d}_stats.json"
            points_path = points_dir / f"step_{step:05d}_gaussians.npz"
            if not stats_path.exists():
                print(f"[MISSING] frame {frame_idx} step {step}: {stats_path}")
                continue
            if not points_path.exists():
                print(f"[MISSING] frame {frame_idx} step {step}: {points_path}")
            stats = json.loads(stats_path.read_text())
            per_step[step] = stats["scale_ratio"]["median"]
            size_kb = points_path.stat().st_size / 1024 if points_path.exists() else float("nan")
            print(f"frame {frame_idx} step {step}: scale_ratio.median={stats['scale_ratio']['median']:.3f} "
                  f"n_gaussians={stats.get('n_gaussians')} npz_size={size_kb:.1f}KB")
        traj[frame_idx] = per_step
    return traj


def plot_trajectory(traj: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for frame_idx, per_step in traj.items():
        steps = sorted(per_step.keys())
        vals = [per_step[s] for s in steps]
        ax.plot(steps, vals, marker='o', label=f"frame {frame_idx}")
    ax.set_xlabel("step")
    ax.set_ylabel("scale_ratio median")
    ax.set_title("G3 baseline: scale_ratio median trajectory (3 frames)")
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


def main():
    splat_dirs = {f: run_frame(f) for f in FRAMES}
    traj = collect_trajectory(splat_dirs)
    out_dir = REPO / "outputs" / f"{BASE_NAME}_smoke_g3"
    plot_trajectory(traj, out_dir / "scale_ratio_trajectory.png")
    with open(out_dir / "trajectory.json", "w") as f:
        json.dump(traj, f, indent=2)
    print(f"[Saved] {out_dir / 'trajectory.json'}")


if __name__ == "__main__":
    main()
