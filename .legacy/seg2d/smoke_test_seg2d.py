"""
冒烟测试(v4)：对某一帧、4个相机分别跑 segment_body_wing(motion-based主算法 +
intensity兜底)，输出overlay png（灰度底图，body绿/wing红/label=3黄，半透明叠加）
+ 打印每相机统计（含body_source/fallback_reason，用于评估motion法成功率）。
数据路径未知，通过命令行参数传入，不在此处硬编码/探索 data/ 目录。

用法示例(archived under .legacy/, see .legacy/seg2d/seg2d.py):
    python .legacy/seg2d/smoke_test_seg2d.py \
        --sparse-dir "X:\\antenna\\control\\009_25052026\\Sparse\\Expr_009_mov_002" \
        --frame-idx 100 \
        --out-dir .legacy/seg2d/overlays
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

LEGACY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEGACY_ROOT))

from seg2d.seg2d import load_sparse_frame, segment_body_wing  # noqa: E402

# BGR (cv2约定)
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


def run_camera(sparse_path: Path, cam: int, frame_idx: int, out_dir: Path,
                leg_th: int, open_radius: int, delta: int, fit_scope: int,
                cm_poly_degree: int, body_th_ratio: float) -> dict:
    t0 = time.time()
    label_map, meta = segment_body_wing(
        sparse_path, cam, frame_idx,
        delta=delta, fit_scope=fit_scope, cm_poly_degree=cm_poly_degree,
        body_th_ratio=body_th_ratio, open_radius=open_radius, leg_th=leg_th,
    )
    elapsed = time.time() - t0

    gray = load_sparse_frame(sparse_path, frame_idx)  # 只用于画overlay底图
    overlay = make_overlay(gray, label_map)
    out_path = out_dir / f"seg2d_debug_cam{cam}_frame{frame_idx}.png"
    cv2.imwrite(str(out_path), overlay)

    body_info = meta["body_info"]
    return {
        "cam": cam,
        "body_px": int((label_map == 1).sum()),
        "wing_px": int((label_map == 2).sum()),
        "unclassified_px": int((label_map == 3).sum()),
        "bg_px": int((label_map == 0).sum()),
        "body_source": meta["body_source"],
        "fallback_reason": body_info.get("reason") if meta["body_source"] == "intensity_fallback" else None,
        "body_info": body_info,
        "elapsed_sec": round(elapsed, 2),
        "overlay_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-dir", type=str, required=True,
                        help="包含 Camera*_sparse.mat 的目录 (按文件名sorted后第i个对应cam i，同generate_dataset.py)")
    parser.add_argument("--frame-idx", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--out-dir", type=str, default=".legacy/seg2d/overlays")
    parser.add_argument("--leg-th", type=int, default=100)
    parser.add_argument("--open-radius", type=int, default=5)
    parser.add_argument("--delta", type=int, default=36)
    parser.add_argument("--fit-scope", type=int, default=100)
    parser.add_argument("--cm-poly-degree", type=int, default=2)
    parser.add_argument("--body-th-ratio", type=float, default=0.85)
    args = parser.parse_args()

    # 支持Windows风格路径 (X:\... -> /mnt/x/...)，与generate_dataset.py保持一致
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
        r = run_camera(sparse_path, cam, args.frame_idx, out_dir,
                        args.leg_th, args.open_radius, args.delta, args.fit_scope,
                        args.cm_poly_degree, args.body_th_ratio)
        records.append(r)
        print(f"cam{r['cam']}: body={r['body_px']} wing={r['wing_px']} "
              f"unclassified={r['unclassified_px']} bg={r['bg_px']} "
              f"source={r['body_source']}"
              + (f" fallback_reason={r['fallback_reason']}" if r["fallback_reason"] else "")
              + f" ({r['elapsed_sec']}s) -> {r['overlay_path']}")
    t_total = time.time() - t_total0

    n_fallback = sum(1 for r in records if r["body_source"] == "intensity_fallback")
    print(f"\ntotal_elapsed={t_total:.1f}s  fallback={n_fallback}/{len(records)}")

    with open(out_dir / f"seg2d_stats_frame{args.frame_idx}.json", "w") as f:
        json.dump({"records": records, "total_elapsed_sec": round(t_total, 2)}, f, indent=2)


if __name__ == "__main__":
    main()
