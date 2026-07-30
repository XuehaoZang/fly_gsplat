"""
调参用诊断脚本(不是主deliverable): 给定一个测试帧，出三张图:
1. 体素帧计数值的直方图(看BODY_VOXEL_COUNT_THRESH选在哪里合理)
2. 累加点云的3D热图(按体素帧计数着色，前视+俯视两视角)
3. body候选体素(阈值后) vs 最终body体素(连通分量后) 的对比图

用法:
    python -m postprocessing.labeling.motion.diag.voxel_threshold_tuning --frame 300
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.labeling.motion import density as d  # noqa: E402
from postprocessing.labeling.motion.diag._viz3d import VIEWS, view_title  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"


def plot_histogram(counts_values: np.ndarray, frame_idx: int, thresh: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(counts_values, bins=np.arange(0.5, counts_values.max() + 1.5, 1), color="#4c72b0")
    ax.axvline(thresh, color="#d62728", linestyle="--",
               label=f"BODY_VOXEL_COUNT_THRESH={thresh}")
    ax.set_xlabel("voxel frame-count (n distinct frames hitting this voxel)")
    ax.set_ylabel("n voxels")
    ax.set_title(f"f{frame_idx:04d}: voxel frame-count distribution (n_voxels={len(counts_values)}, "
                 f"max_count={int(counts_values.max())})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(centers: np.ndarray, values: np.ndarray, frame_idx: int, out_path: Path,
                  title_extra: str = "") -> None:
    fig = plt.figure(figsize=(12, 5.5))
    for i, (name, view_kw) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=values, cmap="viridis",
                         s=14, alpha=0.9, depthshade=False)
        ax.view_init(**view_kw)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(view_title(name, view_kw), fontsize=10)
    fig.colorbar(sc, ax=fig.axes, shrink=0.6, label="voxel frame-count")
    fig.suptitle(f"f{frame_idx:04d}: accumulated point cloud 3D heatmap (colored by voxel frame-count)"
                 f"{title_extra}", fontsize=10)
    fig.subplots_adjust(left=0.03, right=0.9, top=0.85, bottom=0.05, wspace=0.15)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_body_voxel_compare(centers_all: np.ndarray, counts_all: np.ndarray,
                             cand_mask: np.ndarray, final_mask: np.ndarray,
                             frame_idx: int, thresh: int, out_path: Path) -> None:
    fig = plt.figure(figsize=(17, 5.5))
    panels = [
        ("all voxels (colored by count)", centers_all, counts_all, None),
        (f"candidate voxels (count>{thresh}, n={int(cand_mask.sum())})", centers_all[cand_mask], None, "#ff7f0e"),
        (f"final body voxels after CC (n={int(final_mask.sum())})", centers_all[final_mask], None, "#2ca02c"),
    ]
    for i, (title, pts, vals, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        if vals is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap="viridis", s=14, depthshade=False)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=14, depthshade=False)
        ax.view_init(elev=20, azim=-60)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"f{frame_idx:04d}: body candidate voxels (threshold) vs final body voxels "
                 f"(threshold + largest connected component)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(frame_idx: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = d.compute_body_voxels_for_frame(frame_idx)
    counts = info["voxel_counts"]
    voxel_keys_all = np.array(counts.index.tolist())
    centers_all = d.voxel_centers(voxel_keys_all)
    counts_values = counts.to_numpy()

    cand_mask = counts_values > d.BODY_VOXEL_COUNT_THRESH
    body_voxel_set = info["body_voxels"]
    final_mask = np.array([tuple(vk) in body_voxel_set for vk in voxel_keys_all])

    hist_path = OUT_DIR / f"voxel_hist_f{frame_idx:04d}.png"
    heatmap_path = OUT_DIR / f"voxel_heatmap_f{frame_idx:04d}.png"
    compare_path = OUT_DIR / f"voxel_body_compare_f{frame_idx:04d}.png"

    plot_histogram(counts_values, frame_idx, d.BODY_VOXEL_COUNT_THRESH, hist_path)
    plot_heatmap(centers_all, counts_values, frame_idx, heatmap_path,
                 title_extra=f"  n_frames_used={info['n_frames_used']}")
    plot_body_voxel_compare(centers_all, counts_values, cand_mask, final_mask, frame_idx,
                             d.BODY_VOXEL_COUNT_THRESH, compare_path)

    print(f"[{frame_idx}] n_frames_used={info['n_frames_used']}  n_voxels_hit={len(counts)}  "
          f"max_count={int(counts_values.max())}")
    print(f"  阈值候选体素(count>{d.BODY_VOXEL_COUNT_THRESH})={int(cand_mask.sum())}  "
          f"连通分量后最终body体素={int(final_mask.sum())}")
    print(f"  histogram -> {hist_path}")
    print(f"  heatmap   -> {heatmap_path}")
    print(f"  compare   -> {compare_path}")
    return {"frame_idx": frame_idx, "n_candidates": int(cand_mask.sum()), "n_final": int(final_mask.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, required=True, help="测试帧号(整数)")
    args = parser.parse_args()
    run(args.frame)


if __name__ == "__main__":
    main()
