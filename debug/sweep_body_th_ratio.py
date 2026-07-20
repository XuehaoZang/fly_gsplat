"""
body_th_ratio 参数sweep：诊断cam1"整只翅膀被吞"问题的根因(count长尾污染)，
对比不同body_th_ratio下4相机的body/wing分割效果。

对每个相机只跑一次 compute_motion_counts（最贵的部分），之后对每个
body_th_ratio候选值复用同一份(uniq_coords, counts)，只重跑threshold_counts
+形态学清理+wing分割（便宜），避免每个ratio都重新读±136帧数据。

输出：
  outputs/seg2d_debug/sweep/count_histogram_cam{cam}_frame{fr}.png  每相机1张，
      count分布直方图(log y)，标出5条候选阈值线
  outputs/seg2d_debug/sweep/ratio{ratio}_cam{cam}_frame{fr}.png     5x4=20张overlay图
  outputs/seg2d_debug/sweep/sweep_stats_frame{fr}.json              每(cam,ratio)的body_px/wing_px

用法示例:
    python -m debug.sweep_body_th_ratio \
        --sparse-dir "X:\\antenna\\control\\009_25052026\\Sparse\\Expr_009_mov_002" \
        --frame-idx 100
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.seg2d import (
    cleanup_body_mask,
    compute_motion_counts,
    load_sparse_frame,
    segment_wing,
    threshold_counts,
)

RATIOS = [0.6, 0.75, 0.85, 0.95, 1.0]

# BGR (cv2约定)，跟smoke_test_seg2d.py保持一致
LABEL_COLORS_BGR = {
    1: (0, 255, 0),    # body -> 绿
    2: (0, 0, 255),    # wing -> 红
    3: (0, 255, 255),  # 未分类 -> 黄
}
OVERLAY_ALPHA = 0.5


def make_overlay(gray: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    overlay = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for label_val, color in LABEL_COLORS_BGR.items():
        mask = label_map == label_val
        for c in range(3):
            overlay[..., c][mask] = (
                (1 - OVERLAY_ALPHA) * overlay[..., c][mask] + OVERLAY_ALPHA * color[c]
            )
    return overlay.astype(np.uint8)


def plot_count_histogram(counts: np.ndarray, window_size: int, cam: int, frame_idx: int,
                          out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(1, window_size + 2)
    ax.hist(counts, bins=bins, log=True, color="steelblue", edgecolor="black", linewidth=0.3)
    colors = plt.cm.autumn(np.linspace(0, 0.8, len(RATIOS)))
    for ratio, c in zip(RATIOS, colors):
        th = ratio * window_size
        ax.axvline(th, color=c, linestyle="--", linewidth=1.5,
                   label=f"ratio={ratio} (count>{th:.1f})")
    ax.set_xlabel("count (窗口内重复出现次数)")
    ax.set_ylabel("像素数 (log scale)")
    ax.set_title(f"cam{cam} frame{frame_idx}: 对齐窗口count分布 (window_size={window_size})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-dir", type=str, required=True)
    parser.add_argument("--frame-idx", type=int, default=100)
    parser.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--out-dir", type=str, default="outputs/seg2d_debug/sweep")
    parser.add_argument("--leg-th", type=int, default=100)
    parser.add_argument("--open-radius", type=int, default=5)
    parser.add_argument("--delta", type=int, default=36)
    parser.add_argument("--fit-scope", type=int, default=100)
    parser.add_argument("--cm-poly-degree", type=int, default=2)
    args = parser.parse_args()

    sparse_dir = Path(args.sparse_dir.replace("X:", "/mnt/x").replace("\\", "/"))
    sparse_files = sorted(sparse_dir.glob("Camera*_sparse.mat"))
    if not sparse_files:
        print(f"Error: No sparse files found under {sparse_dir}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    t_total0 = time.time()
    for cam in args.cams:
        sparse_path = sparse_files[cam - 1]
        gray = load_sparse_frame(sparse_path, args.frame_idx)

        t0 = time.time()
        uniq, counts, ok, info = compute_motion_counts(
            sparse_path, args.frame_idx, delta=args.delta, fit_scope=args.fit_scope,
            cm_poly_degree=args.cm_poly_degree, frame_size=gray.shape,
        )
        elapsed = time.time() - t0
        if not ok:
            print(f"cam{cam}: compute_motion_counts failed, reason={info.get('reason')}, "
                  f"跳过这个相机的sweep")
            continue

        window_size = info["window_size"]
        print(f"cam{cam}: compute_motion_counts done in {elapsed:.1f}s "
              f"(window_size={window_size}, n_valid_cm_frames={info['n_valid_cm_frames']}, "
              f"target_fg={info['target_fg']})")

        hist_path = out_dir / f"count_histogram_cam{cam}_frame{args.frame_idx}.png"
        plot_count_histogram(counts, window_size, cam, args.frame_idx, hist_path)

        for ratio in RATIOS:
            body_mask_raw = threshold_counts(uniq, counts, window_size, ratio, gray.shape)
            body_mask = cleanup_body_mask(body_mask_raw, open_radius=args.open_radius)
            wing_mask = segment_wing(gray, body_mask, leg_th=args.leg_th)

            label_map = np.zeros(gray.shape, dtype=np.uint8)
            label_map[gray > 0] = 3
            label_map[body_mask] = 1
            label_map[wing_mask] = 2

            overlay = make_overlay(gray, label_map)
            out_path = out_dir / f"ratio{ratio}_cam{cam}_frame{args.frame_idx}.png"
            cv2.imwrite(str(out_path), overlay)

            body_px_raw = int(body_mask_raw.sum())
            body_px = int((label_map == 1).sum())
            wing_px = int((label_map == 2).sum())
            print(f"  ratio={ratio}: body_px_raw(阈值判定后/清理前)={body_px_raw} "
                  f"body_px(清理后)={body_px} wing_px={wing_px}")

            records.append({
                "cam": cam, "ratio": ratio,
                "body_px_raw": body_px_raw, "body_px": body_px, "wing_px": wing_px,
                "unclassified_px": int((label_map == 3).sum()),
                "bg_px": int((label_map == 0).sum()),
                "window_size": window_size,
                "overlay_path": str(out_path),
            })

    t_total = time.time() - t_total0
    print(f"\ntotal_elapsed={t_total:.1f}s")

    with open(out_dir / f"sweep_stats_frame{args.frame_idx}.json", "w") as f:
        json.dump({"records": records, "total_elapsed_sec": round(t_total, 2)}, f, indent=2)


if __name__ == "__main__":
    main()
