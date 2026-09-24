"""
render_labeled_video.py

把某个(sweep_name, group)已跑完T3(_labeled.csv)的帧序列渲染成固定视角的mp4，纯目视
QC用，不是诊断图。跟`postprocessing/kinematics/simulate_gt/animate.py::
render_ground_truth_video`同一个思路(整段序列的xyz联合bbox算一次固定坐标轴范围，
不随帧抖动/缩放，保证苍蝇全程都在画面里；`cv2.VideoWriter`写mp4，不依赖系统ffmpeg)，
但读的是真实T3产出(`_labeled.csv`，走`postprocessing.viz._io.load_stage_csv`同款
"该帧最完整csv"发现逻辑)而不是simulate_gt的合成数据，颜色沿用
`postprocessing.viz._colors.PART_COLORS`同一色源(body灰/wing_L蓝/wing_R红)，只画
if_keep=True的点。

单一固定机位：45度侧-俯视(elev=45, azim=-60，可调)，不是simulate_gt那种front+top
两联视图。

用法(库函数，被run_round2_kinematics.py调用；也可单独跑):
    python -m gpu.schedule.analysis.render_labeled_video \\
        --dataset-root outputs/round2/ctrl_119_004_d1/D1_baseline \\
        --out outputs/round2/ctrl_119_004_d1/kinematics/D1_baseline_labeled_video.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.viz._colors import PART_COLORS, SINGLE_COLOR  # noqa: E402
from postprocessing.viz._io import load_stage_csv  # noqa: E402


def _load_labeled_frames(dataset_root: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """按frame目录扫描，只保留已经跑到T3(有part_label列)的帧，取if_keep=True的点。
    没有_labeled.csv(T3没跑过/该帧失败)的帧直接跳过、不报错——这是目视QC视频，
    个别缺帧不影响整体判断，也不该因为单帧问题让整个视频渲染失败。"""
    out = []
    frame_dirs = sorted(p for p in dataset_root.glob("f[0-9][0-9][0-9][0-9]") if p.is_dir())
    for frame_dir in frame_dirs:
        frame = frame_dir.name
        try:
            _, df = load_stage_csv(frame, dataset_root)
        except FileNotFoundError:
            continue
        if "part_label" not in df.columns or "if_keep" not in df.columns:
            continue
        kept = df[df["if_keep"].astype(bool)]
        if kept.empty:
            continue
        out.append((int(frame[1:]), kept[["x", "y", "z"]].to_numpy(), kept["part_label"].to_numpy()))
    return out


def render_labeled_video(dataset_root: Path, out_path: Path, fps: int = 15,
                          elev: float = 45.0, azim: float = -60.0,
                          figsize: tuple = (6, 6), label: str | None = None) -> Path | None:
    dataset_root = Path(dataset_root)
    frames = _load_labeled_frames(dataset_root)
    if not frames:
        print(f"[render_labeled_video] {dataset_root}: no _labeled.csv frames found, skipping")
        return None

    all_xyz = np.concatenate([xyz for _, xyz, _ in frames], axis=0)
    span = all_xyz.max(axis=0) - all_xyz.min(axis=0)
    pad = 0.05 * np.where(span > 0, span, 1.0)
    lims = tuple((all_xyz[:, i].min() - pad[i], all_xyz[:, i].max() + pad[i]) for i in range(3))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title_prefix = label if label is not None else f"{dataset_root.parent.name}/{dataset_root.name}"

    writer = None
    for frame_id, xyz, part_label in frames:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        for part in sorted(set(part_label)):
            mask = part_label == part
            color = PART_COLORS.get(part, SINGLE_COLOR)
            ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], s=4, color=color, depthshade=False)
        ax.set_title(f"{title_prefix}  frame {frame_id:04d}", fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(bgr)
    if writer is not None:
        writer.release()
    print(f"[render_labeled_video] {dataset_root} -> {out_path} ({len(frames)} frames)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--elev", type=float, default=45.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    args = ap.parse_args()
    render_labeled_video(args.dataset_root, args.out, fps=args.fps, elev=args.elev, azim=args.azim)


if __name__ == "__main__":
    main()
