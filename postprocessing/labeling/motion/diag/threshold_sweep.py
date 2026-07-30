"""诊断任务4: BODY_VOXEL_COUNT_THRESH阈值敏感性分析。

背景: density.py::BODY_VOXEL_COUNT_THRESH=19来自f0300单帧90分位数一次性锁定的常量
(见density.py::BODY_VOXEL_COUNT_PERCENTILE注释——分布本身不是干净的双峰，90分位数只是
"取分布靠后一段"这个较弱的意义)。这个脚本量化两件事:
1. 阈值在明显偏松~明显偏严之间变化时，body/wing边界(重投影图上肉眼可见的范围)变化
   有多剧烈——如果差异很小说明当前固定值不敏感、调阈值没多大风险；如果差异很大说明
   固定阈值本身就不可靠。
2. 每帧各自的90分位数(动态阈值) vs 全局固定19，body候选点数差多少——如果跨帧差异大，
   说明"用一个全局常量"本身就不成立，需要per-frame相对阈值。

只做body/wing二分类(不细分wing_A/wing_B/L/R)，不重跑完整wing流程，不改
density.py/label.py。

用法:
    python -m postprocessing.labeling.motion.diag.threshold_sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import load_marked  # noqa: E402
from postprocessing.labeling.motion import density as d  # noqa: E402
from postprocessing.labeling.motion.diag.body_centroid_stability import ALL_DEV_FRAMES  # noqa: E402
from postprocessing.viz._colors import PART_COLORS  # noqa: E402
from utils.camera import CameraConfig  # noqa: E402
from utils.reproject import project_points  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"
RAW_DATA_DIR = REPO_ROOT / "data" / "ctrl_009_002"

SWEEP_FRAMES = [270, 291, 320, 377]
"""4个代表帧: 270/291是diag/reversal_frame_selection.py挑出的reversal相位帧(270=谷,
291=峰)，320/377是覆盖另外两种展开程度的补充(320=原8帧里的一个高展开幅度帧,
377=另一个reversal峰值帧)，见任务规格"包括第2项挑出的reversal帧"。"""

BASE_THRESH = d.BODY_VOXEL_COUNT_THRESH  # 19
THRESH_LEVELS = [
    ("-50%", round(BASE_THRESH * 0.5)),
    ("-30%", round(BASE_THRESH * 0.7)),
    ("baseline", BASE_THRESH),
    ("+30%", round(BASE_THRESH * 1.3)),
    ("+50%", round(BASE_THRESH * 1.5)),
]
"""覆盖"明显偏松"到"明显偏严"几档，见任务规格"±30%、±50%这几档"。"""

SWEEP_CAM_IDX = 1
"""单相机对比用第1个相机，减少图数量，见任务规格"简化成单相机多阈值对比也可以"。"""


def classify_body_binary(xyz_kept: np.ndarray, voxel_counts: pd.Series, thresh: int) -> np.ndarray:
    """跟label.py::classify_body_candidate同一套判据，但阈值可变、只返回body/wing二分类
    (不做wing连通分量拆L/R，见模块docstring)。复用density.py的公开函数，不重新实现CC逻辑。"""
    body_voxels = d.extract_body_voxels(voxel_counts, thresh=thresh)
    voxel_keys = d.points_to_voxel_keys(xyz_kept)
    is_body = np.array([tuple(vk) in body_voxels for vk in voxel_keys])
    return is_body


def sweep_one_frame(frame_idx: int) -> dict:
    frame = f"f{frame_idx:04d}"
    df_full, _ = load_marked(frame, data_root=d.DATASET_DIR)
    kept_mask = df_full["if_keep"].astype(bool).to_numpy()
    df_kept = df_full[kept_mask].reset_index(drop=True)
    xyz_kept = df_kept[["x", "y", "z"]].to_numpy()

    window_df, used_indices = d.load_window_points(frame_idx)
    voxel_counts = d.compute_voxel_frame_counts(window_df)

    per_thresh = {}
    for label, th in THRESH_LEVELS:
        is_body = classify_body_binary(xyz_kept, voxel_counts, th)
        per_thresh[label] = {"thresh": th, "is_body": is_body, "n_body": int(is_body.sum()),
                              "n_wing": int((~is_body).sum())}

    return {"frame": frame, "frame_idx": frame_idx, "xyz_kept": xyz_kept,
            "n_kept": len(xyz_kept), "voxel_counts": voxel_counts, "per_thresh": per_thresh}


def plot_single_camera_sweep(frame: str, xyz_kept: np.ndarray, per_thresh: dict, out_path: Path) -> None:
    frame_data_dir = RAW_DATA_DIR / frame
    import json
    with open(frame_data_dir / "transforms.json") as f:
        cam_frames = json.load(f)["frames"]
    cam_frame = cam_frames[SWEEP_CAM_IDX - 1]
    cam = CameraConfig.from_opengl(cam_frame)
    cam.cam_idx = SWEEP_CAM_IDX
    img_path = frame_data_dir / cam_frame["file_path"]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    uv, depth = project_points(xyz_kept, cam)
    us, vs = uv[:, 0], uv[:, 1]
    valid = depth > 0

    # the fly occupies a tiny corner of the full camera frame; crop to its bbox (+padding)
    # so individual body/wing points are actually distinguishable instead of collapsing
    # into one overlapping blob at full-frame zoom.
    pad = 0.25
    u_lo, u_hi = us[valid].min(), us[valid].max()
    v_lo, v_hi = vs[valid].min(), vs[valid].max()
    u_pad, v_pad = (u_hi - u_lo) * pad, (v_hi - v_lo) * pad

    n_cols = len(per_thresh)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
    for ax, (level_label, info) in zip(axes, per_thresh.items()):
        if img is not None:
            ax.imshow(img, cmap="gray")
        is_body = info["is_body"]
        # draw the minority class last so it isn't hidden under a fully-overlapping majority layer
        order = ["wing", "body"] if info["n_body"] >= info["n_wing"] else ["body", "wing"]
        draw = {"body": (is_body, PART_COLORS["body"], "body"),
                "wing": (~is_body, PART_COLORS["wing_L"], "wing (candidate, not split L/R)")}
        for key in order:
            mask, color, lab = draw[key]
            ax.scatter(us[valid & mask], vs[valid & mask], s=16, c=[color], marker="o",
                       alpha=0.9, label=lab)
        ax.set_xlim(u_lo - u_pad, u_hi + u_pad)
        ax.set_ylim(v_hi + v_pad, v_lo - v_pad)  # image v-axis points down
        ax.set_title(f"{level_label}\nthresh={info['thresh']}  n_body={info['n_body']}  "
                      f"n_wing={info['n_wing']}", fontsize=9)
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{frame}: BODY_VOXEL_COUNT_THRESH sweep (CAM {SWEEP_CAM_IDX}, gray=body, blue=wing candidate)",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def dynamic_vs_fixed_table(frame_indices: list[int] = ALL_DEV_FRAMES) -> pd.DataFrame:
    """每帧各自按自己的90分位数动态定阈值 vs 全局固定值19，body候选点数差多少(见任务规格)。"""
    rows = []
    for frame_idx in frame_indices:
        frame = f"f{frame_idx:04d}"
        try:
            df_full, _ = load_marked(frame, data_root=d.DATASET_DIR)
        except FileNotFoundError:
            continue
        kept_mask = df_full["if_keep"].astype(bool).to_numpy()
        df_kept = df_full[kept_mask].reset_index(drop=True)
        xyz_kept = df_kept[["x", "y", "z"]].to_numpy()

        window_df, _ = d.load_window_points(frame_idx)
        voxel_counts = d.compute_voxel_frame_counts(window_df)
        dynamic_thresh = int(np.percentile(voxel_counts.to_numpy(), d.BODY_VOXEL_COUNT_PERCENTILE))

        n_body_fixed = int(classify_body_binary(xyz_kept, voxel_counts, d.BODY_VOXEL_COUNT_THRESH).sum())
        n_body_dynamic = int(classify_body_binary(xyz_kept, voxel_counts, dynamic_thresh).sum())
        pct_diff = (n_body_dynamic - n_body_fixed) / n_body_fixed * 100 if n_body_fixed else np.nan

        rows.append({"frame": frame, "frame_idx": frame_idx, "n_kept": len(xyz_kept),
                     "fixed_thresh": d.BODY_VOXEL_COUNT_THRESH, "n_body_fixed": n_body_fixed,
                     "dynamic_thresh_this_frame": dynamic_thresh, "n_body_dynamic": n_body_dynamic,
                     "pct_diff_dynamic_vs_fixed": pct_diff})
    return pd.DataFrame(rows)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[threshold_sweep] 阈值档位(baseline={BASE_THRESH}): "
          + "  ".join(f"{lbl}={th}" for lbl, th in THRESH_LEVELS))
    summary_rows = []
    for frame_idx in SWEEP_FRAMES:
        result = sweep_one_frame(frame_idx)
        out_path = OUT_DIR / f"threshold_sweep_{result['frame']}.png"
        plot_single_camera_sweep(result["frame"], result["xyz_kept"], result["per_thresh"], out_path)
        print(f"\n[{result['frame']}] n_kept={result['n_kept']}")
        for level_label, info in result["per_thresh"].items():
            frac = info["n_body"] / result["n_kept"] * 100
            print(f"    {level_label:9s} thresh={info['thresh']:2d}  n_body={info['n_body']:4d} "
                  f"({frac:5.1f}% of kept)  n_wing={info['n_wing']:4d}")
            summary_rows.append({"frame": result["frame"], "frame_idx": frame_idx, "level": level_label,
                                 "thresh": info["thresh"], "n_body": info["n_body"], "n_wing": info["n_wing"],
                                 "body_frac_of_kept": frac / 100})
        n_body_base = result["per_thresh"]["baseline"]["n_body"]
        n_body_m50 = result["per_thresh"]["-50%"]["n_body"]
        n_body_p50 = result["per_thresh"]["+50%"]["n_body"]
        swing = (n_body_p50 - n_body_m50) / n_body_base * 100 if n_body_base else np.nan
        print(f"    body候选点数从-50%到+50%阈值的摆动幅度 = {swing:+.1f}% (相对baseline)")
        print(f"    sweep image -> {out_path}")

    sweep_summary_df = pd.DataFrame(summary_rows)
    sweep_summary_csv = OUT_DIR / "threshold_sweep_summary.csv"
    sweep_summary_df.to_csv(sweep_summary_csv, index=False)
    print(f"\n[threshold_sweep] sweep汇总csv -> {sweep_summary_csv}")

    dyn_df = dynamic_vs_fixed_table()
    dyn_csv = OUT_DIR / "dynamic_vs_fixed_threshold.csv"
    dyn_df.to_csv(dyn_csv, index=False)
    print(f"\n[threshold_sweep] 动态(每帧90分位数) vs 固定(19) 阈值对比({len(dyn_df)}帧):")
    print(dyn_df[["frame", "fixed_thresh", "n_body_fixed", "dynamic_thresh_this_frame", "n_body_dynamic",
                  "pct_diff_dynamic_vs_fixed"]].to_string(index=False))
    print(f"  pct_diff_dynamic_vs_fixed: mean={dyn_df['pct_diff_dynamic_vs_fixed'].mean():.1f}%  "
          f"std={dyn_df['pct_diff_dynamic_vs_fixed'].std():.1f}%  "
          f"max_abs={dyn_df['pct_diff_dynamic_vs_fixed'].abs().max():.1f}%")
    print(f"  csv -> {dyn_csv}")


if __name__ == "__main__":
    run()
