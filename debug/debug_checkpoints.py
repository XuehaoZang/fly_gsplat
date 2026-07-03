"""
读取 debug_checkpoints/{exp}/step_*_means.npy，
对每个 checkpoint 生成重投影图 + 汇总 extent/n 曲线到一张图。
不需要 dataparser_transforms.json 的 rescale——因为 dump_means 时机在
模型内部，此时 self.means 就是训练用的 rescale 坐标，需要 unrescale
回物理坐标才能和 hull 对比，这一步和 debug_splat_ply.py 里的逻辑一致，直接复用。
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.camera import CameraConfig
from utils.ply import unrescale, load_ply, clean_ply
from utils.viz import plot_reprojection


def debug_checkpoints(data_dir: str, splat_dir: str, checkpoint_dir: str) -> None:
    data_dir, splat_dir, checkpoint_dir = Path(data_dir), Path(splat_dir), Path(checkpoint_dir)

    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R_ns = np.array(dp["transform"])[:3, :3]
    t_ns = np.array(dp["transform"])[:3, 3]
    scale = float(dp["scale"])

    with open(data_dir / "transforms.json") as f:
        frames = json.load(f)["frames"]
    cameras = []
    for i, fr in enumerate(frames):
        cam = CameraConfig.from_opengl(fr)
        cam.cam_idx = i + 1
        cameras.append(cam)

    hull_pts = load_ply(data_dir / "init_points.ply")

    ckpts = sorted(checkpoint_dir.glob("step_*_means.npy"))
    steps, n_raw, extents = [], [], []

    out_dir = checkpoint_dir / "reproj"
    out_dir.mkdir(exist_ok=True)

    for ckpt_path in ckpts:
        step = int(ckpt_path.stem.split("_")[1])
        means_rescaled = np.load(ckpt_path)
        means_physical = unrescale(means_rescaled, R_ns, t_ns, scale)

        steps.append(step)
        n_raw.append(len(means_physical))
        extents.append(means_physical.max(0) - means_physical.min(0) if len(means_physical) else [0, 0, 0])

        # 每个 checkpoint 存一张重投影图，文件名带 step 号方便逐帧翻看
        plot_reprojection(data_dir, out_dir, cameras, hull_pts, means_physical)
        (out_dir / "debug_reproj.png").rename(out_dir / f"reproj_step{step:05d}.png")

    extents = np.array(extents)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, n_raw, marker='o')
    axes[0].set_title("gaussian count (raw, pre-clean)")
    axes[0].set_xlabel("step")

    hull_extent = hull_pts.max(0) - hull_pts.min(0)
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes[1].plot(steps, extents[:, i], marker='o', label=f"splat {label}")
        axes[1].axhline(hull_extent[i], linestyle='--', alpha=0.4, label=f"hull {label}")
    axes[1].set_title("extent vs step (dashed = hull reference)")
    axes[1].set_xlabel("step")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "convergence_curves.png", dpi=150)
    print(f"[Saved] {checkpoint_dir / 'convergence_curves.png'}")
    print(f"[Saved] {len(ckpts)} reprojection images -> {out_dir}")


if __name__ == "__main__":
    debug_checkpoints(
        data_dir="./data/ctrl_009_002_f10",
        splat_dir="./outputs/ctrl_009_002_f10/splatfacto-checkpoint/2026-07-03_112339",
        checkpoint_dir="./debug_checkpoints/ctrl_009_002_f10",
    )