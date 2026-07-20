"""
多帧检查：body_th_ratio固定0.85，对 frame_start~frame_end (含) × 4相机跑
segment_body_wing，输出统计表；重点看cam1这一列wing_px是持续异常还是
偶发异常。cam1的body/wing分割额外走一遍compute_motion_counts+threshold_counts
(而不是直接调segment_body_wing)，这样能顺手拿到(uniq_coords,counts)存一张
count直方图，不用为了画直方图再多跑一次Step A/B/C。

不是每帧都出overlay图（11帧x4相机=44张太多）。cam1的label_map/gray在内存里
缓存下来，等11帧统计都跑完、按wing_px挑出"正常"和"异常"帧后，只把选中的
那几帧从内存里的缓存写出overlay png，不重新计算。

CJK字体在这台机器上没装(fc-list :lang=zh为空)，直方图标签改用英文，避免
之前版本里中文标签缺字块的问题。

用法示例:
    python -m debug.sweep_multiframe_cam1 \
        --sparse-dir "X:\\antenna\\control\\009_25052026\\Sparse\\Expr_009_mov_002" \
        --frame-start 90 --frame-end 100 --body-th-ratio 0.85
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
    segment_body_intensity,
    segment_body_wing,
    segment_wing,
    threshold_counts,
)

LABEL_COLORS_BGR = {
    1: (0, 255, 0),    # body -> 绿
    2: (0, 0, 255),    # wing -> 红
    3: (0, 255, 255),  # 未分类 -> 黄
}
OVERLAY_ALPHA = 0.5
N_PICK_PER_SIDE = 2  # 正常/异常各挑几帧出overlay


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
                          body_th_ratio: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(1, window_size + 2)
    ax.hist(counts, bins=bins, log=True, color="steelblue", edgecolor="black", linewidth=0.3)
    th = body_th_ratio * window_size
    ax.axvline(th, color="red", linestyle="--", linewidth=1.5,
               label=f"body_th_ratio={body_th_ratio} (count>{th:.1f})")
    ax.set_xlabel("count (times pixel appears in aligned window)")
    ax.set_ylabel("pixel count (log scale)")
    ax.set_title(f"cam{cam} frame{frame_idx}: aligned-count distribution (window_size={window_size})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_cam1_frame(sparse_path: Path, frame_idx: int, body_th_ratio: float,
                    open_radius: int, leg_th: int, delta: int, fit_scope: int,
                    cm_poly_degree: int, hist_out_path: Path) -> dict:
    """cam1专用路径：手动走 compute_motion_counts -> threshold_counts，
    这样能拿到(uniq,counts)顺便画直方图，不用为了直方图再多跑一次Step A/B/C。
    退化检测/兜底逻辑跟 segment_body() 保持一致。
    """
    gray = load_sparse_frame(sparse_path, frame_idx)
    uniq, counts, ok, info = compute_motion_counts(
        sparse_path, frame_idx, delta=delta, fit_scope=fit_scope,
        cm_poly_degree=cm_poly_degree, frame_size=gray.shape,
    )

    if ok:
        window_size = info["window_size"]
        plot_count_histogram(counts, window_size, 1, frame_idx, body_th_ratio, hist_out_path)
        body_mask_raw = threshold_counts(uniq, counts, window_size, body_th_ratio, gray.shape)
        body_px_raw = int(body_mask_raw.sum())
        target_fg = info["target_fg"]
        degenerate = body_px_raw < 20 or (target_fg > 0 and body_px_raw > 0.9 * target_fg)
        if degenerate:
            body_bin = segment_body_intensity(gray)
            source = "intensity_fallback"
            fallback_reason = "degenerate_body_size"
        else:
            body_bin = body_mask_raw
            source = "motion"
            fallback_reason = None
    else:
        body_bin = segment_body_intensity(gray)
        source = "intensity_fallback"
        fallback_reason = info.get("reason")
        print(f"  [warn] cam1 frame{frame_idx}: compute_motion_counts failed "
              f"({fallback_reason})，没有count可画直方图")

    body_bin = cleanup_body_mask(body_bin, open_radius)
    wing_mask = segment_wing(gray, body_bin, leg_th)

    label_map = np.zeros(gray.shape, dtype=np.uint8)
    label_map[gray > 0] = 3
    label_map[body_bin] = 1
    label_map[wing_mask] = 2

    return {
        "cam": 1, "frame": frame_idx,
        "body_px": int((label_map == 1).sum()),
        "wing_px": int((label_map == 2).sum()),
        "unclassified_px": int((label_map == 3).sum()),
        "bg_px": int((label_map == 0).sum()),
        "body_source": source,
        "fallback_reason": fallback_reason,
        "_gray": gray,
        "_label_map": label_map,
    }


def run_other_cam_frame(sparse_path: Path, cam: int, frame_idx: int, body_th_ratio: float,
                         open_radius: int, leg_th: int, delta: int, fit_scope: int,
                         cm_poly_degree: int) -> dict:
    label_map, meta = segment_body_wing(
        sparse_path, cam, frame_idx, delta=delta, fit_scope=fit_scope,
        cm_poly_degree=cm_poly_degree, body_th_ratio=body_th_ratio,
        open_radius=open_radius, leg_th=leg_th,
    )
    body_info = meta["body_info"]
    return {
        "cam": cam, "frame": frame_idx,
        "body_px": int((label_map == 1).sum()),
        "wing_px": int((label_map == 2).sum()),
        "unclassified_px": int((label_map == 3).sum()),
        "bg_px": int((label_map == 0).sum()),
        "body_source": meta["body_source"],
        "fallback_reason": body_info.get("reason") if meta["body_source"] == "intensity_fallback" else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-dir", type=str, required=True)
    parser.add_argument("--frame-start", type=int, default=90)
    parser.add_argument("--frame-end", type=int, default=100)
    parser.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--body-th-ratio", type=float, default=0.85)
    parser.add_argument("--out-dir", type=str, default="outputs/seg2d_debug/multiframe")
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

    frames = list(range(args.frame_start, args.frame_end + 1))
    records = []
    cam1_cache = {}  # frame_idx -> {"gray":..., "label_map":...}

    t_total0 = time.time()
    for frame_idx in frames:
        for cam in args.cams:
            sparse_path = sparse_files[cam - 1]
            t0 = time.time()
            if cam == 1:
                hist_path = out_dir / f"count_histogram_cam1_frame{frame_idx}.png"
                r = run_cam1_frame(
                    sparse_path, frame_idx, args.body_th_ratio, args.open_radius,
                    args.leg_th, args.delta, args.fit_scope, args.cm_poly_degree, hist_path,
                )
                cam1_cache[frame_idx] = {"gray": r.pop("_gray"), "label_map": r.pop("_label_map")}
            else:
                r = run_other_cam_frame(
                    sparse_path, cam, frame_idx, args.body_th_ratio, args.open_radius,
                    args.leg_th, args.delta, args.fit_scope, args.cm_poly_degree,
                )
            elapsed = time.time() - t0
            r["elapsed_sec"] = round(elapsed, 2)
            records.append(r)
            print(f"frame{frame_idx} cam{cam}: body={r['body_px']} wing={r['wing_px']} "
                  f"unclassified={r['unclassified_px']} source={r['body_source']}"
                  + (f" fallback_reason={r['fallback_reason']}" if r["fallback_reason"] else "")
                  + f" ({elapsed:.1f}s)")

    t_total = time.time() - t_total0
    print(f"\ntotal_elapsed={t_total:.1f}s")

    # --- cam1列异常/正常帧挑选：按wing_px排序，最低的N_PICK_PER_SIDE帧记为"异常疑似"，
    # 最高的N_PICK_PER_SIDE帧记为"正常疑似"（wing被吞掉时wing_px会偏低、body_px偏高）---
    cam1_records = sorted([r for r in records if r["cam"] == 1], key=lambda r: r["wing_px"])
    abnormal_frames = [r["frame"] for r in cam1_records[:N_PICK_PER_SIDE]]
    normal_frames = [r["frame"] for r in cam1_records[-N_PICK_PER_SIDE:]]
    picked_frames = sorted(set(abnormal_frames + normal_frames))

    print(f"\ncam1 wing_px最低(疑似异常): {abnormal_frames}")
    print(f"cam1 wing_px最高(疑似正常): {normal_frames}")

    picked_overlay_paths = {}
    for frame_idx in picked_frames:
        cached = cam1_cache[frame_idx]
        overlay = make_overlay(cached["gray"], cached["label_map"])
        tag = "abnormal" if frame_idx in abnormal_frames else "normal"
        out_path = out_dir / f"cam1_{tag}_frame{frame_idx}.png"
        cv2.imwrite(str(out_path), overlay)
        picked_overlay_paths[frame_idx] = str(out_path)
        print(f"[saved] {out_path}")

    with open(out_dir / f"multiframe_stats_frame{args.frame_start}-{args.frame_end}.json", "w") as f:
        json.dump({
            "records": records,
            "total_elapsed_sec": round(t_total, 2),
            "body_th_ratio": args.body_th_ratio,
            "cam1_abnormal_frames": abnormal_frames,
            "cam1_normal_frames": normal_frames,
            "picked_overlay_paths": picked_overlay_paths,
        }, f, indent=2)


if __name__ == "__main__":
    main()
