"""诊断任务1: 跨帧body质心稳定性。

背景: motion v0(density.py+label.py)不做刚性运动对齐(见density.py模块docstring
TODO)，如果body本身跨帧有整体小幅位移，或者body/wing判据在某些相位上把wing点漏判/
误判进body，都会表现为"不同帧算出来的body质心不重合"。这个脚本只读已经产出的
_labeled.csv，算body质心，跟同一帧内"body-to-wingtip"的典型尺度做对比——如果质心
漂移量级远小于这个尺度，说明v0至少在"body位置"这个维度上是自洽的；如果某帧质心
明显跳出其他帧的范围，直接标出来，不只是画图。

前提: 先跑完label.py::main()(8个原始dev帧)和
diag/reversal_frame_selection.py::run_reversal_frames()(5个补充帧)，
motion/eda_outputs/下要有全部13帧的_labeled.csv。

用法:
    python -m postprocessing.labeling.motion.diag.body_centroid_stability
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.labeling.motion import label as L  # noqa: E402
from postprocessing.labeling.motion.diag._viz3d import VIEWS, view_title  # noqa: E402
from postprocessing.labeling.motion.diag.reversal_frame_selection import REVERSAL_FRAMES_MOTION  # noqa: E402

LABELED_CSV_DIR = L.OUT_DIR
"""_labeled.csv所在目录: motion/eda_outputs(label.py::run_batch的落盘位置)，不是本
diag脚本自己的输出目录，见下面OUT_DIR。"""

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"
"""本diag脚本自己的图/csv输出目录: motion/diag/eda_outputs，跟_labeled.csv所在目录分开。"""

ALL_DEV_FRAMES = sorted(L.DEV_FRAMES_MOTION + REVERSAL_FRAMES_MOTION)
"""8个原始dev帧 + 5个reversal补充帧，一共13帧，见任务规格"1.跨帧body质心稳定性诊断"
要求把两批帧并在一起看。"""

DRIFT_FLAG_RATIO = 1.0
"""判据: 某帧质心到全部帧质心均值的距离，若超过 DRIFT_FLAG_RATIO * 全部帧body_extent
的中位数，标记为异常。取1.0(=跳变幅度达到body自身尺度量级)作为"肉眼一看就不对"的
粗略下限，不是精细标定值，见任务规格"跳变超过body自身extent的量级"。"""


def bbox_diagonal(xyz: np.ndarray) -> float:
    """点集的bbox对角线长度，作为该点集空间延展的粗略尺度(不用精确定义，见任务规格)。"""
    return float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))


def load_frame_stats(frame_idx: int) -> dict | None:
    frame = f"f{frame_idx:04d}"
    csv_path = LABELED_CSV_DIR / f"gaussian_features_{frame}_labeled.csv"
    if not csv_path.exists():
        print(f"[body_centroid_stability] {frame}: 找不到{csv_path}，跳过(需要先跑motion分类)")
        return None
    df = pd.read_csv(csv_path)
    body_xyz = df.loc[df["part_label"] == "body", ["x", "y", "z"]].to_numpy()
    wing_xyz = df.loc[df["part_label"].isin(["wing_L", "wing_R"]), ["x", "y", "z"]].to_numpy()
    if len(body_xyz) == 0:
        print(f"[body_centroid_stability] {frame}: body点数=0，跳过")
        return None

    centroid = body_xyz.mean(axis=0)
    body_extent = bbox_diagonal(body_xyz)
    wingtip_dist = float(np.linalg.norm(wing_xyz - centroid, axis=1).max()) if len(wing_xyz) > 0 else np.nan

    return {
        "frame": frame, "frame_idx": frame_idx,
        "cx": centroid[0], "cy": centroid[1], "cz": centroid[2],
        "n_body_points": len(body_xyz), "body_extent": body_extent,
        "wingtip_dist_max": wingtip_dist,
    }


def build_stats_table(frame_indices: list[int] = ALL_DEV_FRAMES) -> pd.DataFrame:
    rows = [r for idx in frame_indices if (r := load_frame_stats(idx)) is not None]
    return pd.DataFrame(rows)


def flag_drift(stats_df: pd.DataFrame) -> pd.DataFrame:
    centroids = stats_df[["cx", "cy", "cz"]].to_numpy()
    mean_centroid = centroids.mean(axis=0)
    dist_to_mean = np.linalg.norm(centroids - mean_centroid, axis=1)
    median_body_extent = float(stats_df["body_extent"].median())
    median_wingtip_dist = float(stats_df["wingtip_dist_max"].median())

    stats_df = stats_df.copy()
    stats_df["dist_to_mean_centroid"] = dist_to_mean
    stats_df["dist_frac_of_body_extent"] = dist_to_mean / median_body_extent
    stats_df["dist_frac_of_wingtip_dist"] = dist_to_mean / median_wingtip_dist
    stats_df["flagged_drift"] = dist_to_mean > DRIFT_FLAG_RATIO * median_body_extent
    return stats_df


def plot_lines(stats_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    mean_centroid = stats_df[["cx", "cy", "cz"]].to_numpy().mean(axis=0)
    for ax, col, label, mean_val in zip(axes, ["cx", "cy", "cz"], ["x", "y", "z"], mean_centroid):
        ax.plot(stats_df["frame_idx"], stats_df[col], "o-", color="#4c72b0")
        ax.axhline(mean_val, color="#999999", linestyle="--", lw=1, label="mean over all frames")
        for _, row in stats_df.iterrows():
            if row["flagged_drift"]:
                ax.plot(row["frame_idx"], row[col], "o", color="#d62728", ms=10, mfc="none", mew=2)
        ax.set_ylabel(f"body centroid {label} (m)")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("frame_idx")
    fig.suptitle("body centroid position vs frame_idx (per axis)  "
                 "red circle = flagged as drifted (see console output)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_3d_scatter(stats_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 5.5))
    centroids = stats_df[["cx", "cy", "cz"]].to_numpy()
    colors = ["#d62728" if f else "#4c72b0" for f in stats_df["flagged_drift"]]
    for i, (name, view_kw) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=colors, s=40, depthshade=False)
        for (x, y, z), frame in zip(centroids, stats_df["frame"]):
            ax.text(x, y, z, frame[1:], fontsize=7)
        ax.view_init(**view_kw)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(view_title(name, view_kw), fontsize=10)
    fig.suptitle("body centroid per frame, 3D  (red = flagged drift, label = frame idx)", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.2)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_linear_trend(stats_df: pd.DataFrame) -> dict:
    """检验质心是否随frame_idx做近似线性漂移(而不是围绕一个固定位置随机抖动)——
    如果R^2很高，说明body在这120帧跨度里有系统性的整体平动(比如实验动物本身在走动/
    漂移，不是"钉死+小幅晃动")，这比单帧跳变更能解释density.py不做刚性对齐这个简化
    有多大代价：按此线性外推，一个HALF_WINDOW=36(73帧)累加窗口内body自身能挪动多少，
    直接跟VOXEL_SIZE_M/body_extent对比。"""
    fidx = stats_df["frame_idx"].to_numpy(dtype=float)
    slopes = {}
    for col in ["cx", "cy", "cz"]:
        y = stats_df[col].to_numpy()
        slope, intercept = np.polyfit(fidx, y, 1)
        pred = slope * fidx + intercept
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        slopes[col] = {"slope_per_frame": float(slope), "r2": float(r2)}

    slope_vec = np.array([slopes["cx"]["slope_per_frame"], slopes["cy"]["slope_per_frame"],
                           slopes["cz"]["slope_per_frame"]])
    from postprocessing.labeling.motion import density as d
    disp_per_window = float(np.linalg.norm(slope_vec) * (2 * d.HALF_WINDOW + 1))
    return {"per_axis": slopes, "disp_per_window_m": disp_per_window,
            "voxel_size_m": d.VOXEL_SIZE_M, "window_frames": 2 * d.HALF_WINDOW + 1}


def run() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_df = build_stats_table()
    if len(stats_df) == 0:
        print("[body_centroid_stability] 没有可用的_labeled.csv，先跑motion分类。")
        return stats_df
    stats_df = flag_drift(stats_df)

    csv_path = OUT_DIR / "body_centroid_stability.csv"
    stats_df.to_csv(csv_path, index=False)
    plot_lines(stats_df, OUT_DIR / "body_centroid_stability_lines.png")
    plot_3d_scatter(stats_df, OUT_DIR / "body_centroid_stability_3d.png")

    median_extent = float(stats_df["body_extent"].median())
    median_wingtip = float(stats_df["wingtip_dist_max"].median())
    max_dev = float(stats_df["dist_to_mean_centroid"].max())
    max_dev_frame = stats_df.loc[stats_df["dist_to_mean_centroid"].idxmax(), "frame"]

    print(f"\n[body_centroid_stability] {len(stats_df)}帧汇总:")
    print(f"  body_extent(bbox对角线)中位数={median_extent:.3e} m  "
          f"wingtip_dist_max(body质心到最远wing点)中位数={median_wingtip:.3e} m")
    print(f"  质心到全帧均值的最大偏离={max_dev:.3e} m (frame={max_dev_frame})  "
          f"= body_extent的{max_dev / median_extent * 100:.1f}%  "
          f"= wingtip_dist的{max_dev / median_wingtip * 100:.1f}%")

    flagged = stats_df[stats_df["flagged_drift"]]
    if len(flagged) > 0:
        print(f"  [标记] {len(flagged)}帧质心偏离超过body_extent中位数x{DRIFT_FLAG_RATIO}:")
        for _, row in flagged.iterrows():
            print(f"    {row['frame']}: 偏离={row['dist_to_mean_centroid']:.3e} m "
                  f"(body_extent的{row['dist_frac_of_body_extent'] * 100:.1f}%, "
                  f"wingtip_dist的{row['dist_frac_of_wingtip_dist'] * 100:.1f}%)")
    else:
        print(f"  [结论] 没有单帧质心跳变超过body_extent中位数x{DRIFT_FLAG_RATIO}(没有离群跳变帧)。")

    trend = analyze_linear_trend(stats_df)
    r2_vals = {ax: v["r2"] for ax, v in trend["per_axis"].items()}
    print(f"  [线性趋势检验] 质心vs frame_idx线性拟合 R^2: x={r2_vals['cx']:.3f} y={r2_vals['cy']:.3f} "
          f"z={r2_vals['cz']:.3f}  (>0.9说明不是随机抖动，是近似匀速的系统性整体平动)")
    print(f"  按此线性趋势外推，单个累加窗口({trend['window_frames']}帧)内body自身预期挪动"
          f"={trend['disp_per_window_m']:.3e} m = VOXEL_SIZE_M的{trend['disp_per_window_m'] / trend['voxel_size_m']:.1f}倍"
          f" = body_extent中位数的{trend['disp_per_window_m'] / median_extent * 100:.1f}%")
    if min(r2_vals.values()) > 0.9:
        print("  [结论-重要] 质心漂移不是噪声，是跨帧近似线性的系统性平动，且单窗口内的预期位移"
              "远超VOXEL_SIZE_M——直接支持density.py模块docstring里"
              "未做刚性对齐会稀释体素命中帧数\"这个假设，不只是理论上的TODO。")

    import json
    trend_path = OUT_DIR / "body_centroid_linear_trend.json"
    trend_path.write_text(json.dumps(trend, indent=2))

    print(f"  csv -> {csv_path}")
    print(f"  line plot -> {OUT_DIR / 'body_centroid_stability_lines.png'}")
    print(f"  3d  plot -> {OUT_DIR / 'body_centroid_stability_3d.png'}")
    print(f"  trend json -> {trend_path}")
    return stats_df


if __name__ == "__main__":
    run()
