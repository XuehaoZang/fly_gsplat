"""诊断任务2: 挑选stroke reversal附近的帧，补进motion v0的dev帧集合。

背景: 原8帧(label.py::DEV_FRAMES_MOTION)按if_keep点云bbox extent振荡粗选，覆盖的是
"展开/折叠程度不同"，不专门覆盖"翅尖速度趋近于0的相位"(stroke顶点附近，wingtip在
折返点停留)。这里用现成的T1产出列dist_to_principal_axis(全身PCA主轴的径向距离，
翅膀展开时值大，见utils/gaussian_features.py/binary_split.py同一列的用法)作为
"wingtip展开程度"的粗代理: 单帧取if_keep点里该列的95分位数(比取max更抗floater离群点)。
这条曲线随frame振荡，局部极值(极大=接近全展开折返点，极小=接近收拢折返点)对应
翅尖速度趋近于0的相位，不需要精确的wingbeat相位标定。

覆盖范围限制: T2产出(_marked.csv)目前只覆盖f0224~f0416(见density.py DATASET_DIR
注释)，而motion v0单帧分类需要±HALF_WINDOW(36帧)的完整窗口，所以候选帧必须落在
[224+36, 416-36]=[260,380]——跟原8帧同一个区间，不是本脚本的选择，是T2覆盖范围
决定的硬约束，如实record。

用法:
    python -m postprocessing.labeling.motion.diag.reversal_frame_selection
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import load_marked  # noqa: E402
from postprocessing.labeling.motion import density as d  # noqa: E402
from postprocessing.labeling.motion import label as L  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"

PROXY_SCAN_RANGE = range(224, 417)
"""T2实际覆盖的连续帧范围(含端点)，见模块docstring，代理曲线只能在这个范围内算。"""

T2_COVERAGE_START, T2_COVERAGE_END = 224, 416
"""T2产出(_marked.csv)实际覆盖的连续帧范围，见模块docstring。"""

CANDIDATE_RANGE = range(T2_COVERAGE_START + d.HALF_WINDOW, T2_COVERAGE_END - d.HALF_WINDOW + 1)
"""窗口安全范围[260,380] = [224+HALF_WINDOW, 416-HALF_WINDOW]，跟label.py::DEV_FRAMES_MOTION
同一个区间，见模块docstring"覆盖范围限制"。"""

PROXY_PERCENTILE = 95.0
"""单帧wingtip展开代理: if_keep点dist_to_principal_axis列的95分位数(不用max，避免单个
floater/离群点主导曲线)。"""

SMOOTH_WINDOW = 5
"""代理曲线先做窗口=5的滚动中位数轻度平滑，再找局部极值，减少帧间噪声导致的假极值。"""

EXTREMA_ORDER = 3
"""scipy.signal.argrelextrema的order: 一个点左右各EXTREMA_ORDER帧内都不如它高/低才算局部极值。"""

MIN_DIST_TO_EXISTING = 3
"""候选极值帧跟label.py::DEV_FRAMES_MOTION已有帧的最小frame_idx间隔，太近视为重复相位，不选。"""

MIN_PICK_GAP = 8
"""最终选中的补充帧互相之间的最小frame_idx间隔，避免同一个宽平台内的极值被重复计入
(比如同一个trough的左右两侧都过argrelextrema判据，实际是同一个折返相位)。"""

N_TARGET = 5
"""目标补充帧数(3~5，见任务规格，取上限5，若极值不够则少选)。"""

REVERSAL_FRAMES_MOTION = [270, 291, 304, 324, 377]
"""select_reversal_frames()跑一次后锁定的结果(见wingtip_proxy_curve.png/
reversal_extrema_candidates.csv): 270/324为代理曲线局部谷值(翅膀收拢折返)，291/377为
局部峰值(翅膀展开折返)，304是300附近谷值右侧一个较小的局部隆起。跟DEV_FRAMES_MOTION
同一种"跑一次锁定为常量"的约定(见label.py::DEV_FRAMES_MOTION)，不是每次运行都重新算，
run_batch/motion_dev_summary.csv等下游脚本直接用这个列表，不重跑select_reversal_frames。
"""


def compute_proxy_curve(frame_range: range = PROXY_SCAN_RANGE,
                         dataset_dir: Path = d.DATASET_DIR) -> pd.DataFrame:
    """逐帧算wingtip展开代理(见模块docstring)，跳过_marked.csv缺失的帧(如实跳过不补算)。"""
    rows = []
    for idx in frame_range:
        frame = f"f{idx:04d}"
        try:
            df, _ = load_marked(frame, data_root=dataset_dir)
        except FileNotFoundError:
            continue
        kept = df[df["if_keep"].astype(bool)]
        if len(kept) == 0:
            continue
        proxy = float(np.percentile(kept["dist_to_principal_axis"].to_numpy(), PROXY_PERCENTILE))
        rows.append({"frame_idx": idx, "proxy": proxy, "n_kept": len(kept)})
    return pd.DataFrame(rows)


def find_local_extrema(curve_df: pd.DataFrame, candidate_range: range = CANDIDATE_RANGE,
                        existing_frames: list[int] = None) -> pd.DataFrame:
    """在candidate_range内找代理曲线(轻度平滑后)的局部极大值/极小值帧，剔除跟existing_frames
    太近的重复相位，返回按|frame_idx-最近已有帧距离|降序(优先补相位差异大的)排序的候选表。"""
    from scipy.signal import argrelextrema

    existing_frames = existing_frames or []
    smoothed = curve_df["proxy"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
    curve_df = curve_df.copy()
    curve_df["proxy_smoothed"] = smoothed

    values = smoothed
    maxima_local_idx = argrelextrema(values, np.greater_equal, order=EXTREMA_ORDER)[0]
    minima_local_idx = argrelextrema(values, np.less_equal, order=EXTREMA_ORDER)[0]

    def dedup_flat(idx_array: np.ndarray) -> list[int]:
        # argrelextrema with *_equal on a flat plateau returns every point on the plateau;
        # collapse consecutive runs to their midpoint so a plateau counts as one extremum.
        if len(idx_array) == 0:
            return []
        groups, cur = [], [idx_array[0]]
        for i in idx_array[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        return [g[len(g) // 2] for g in groups]

    extrema_positions = sorted(set(dedup_flat(maxima_local_idx) + dedup_flat(minima_local_idx)))

    records = []
    for pos in extrema_positions:
        row = curve_df.iloc[pos]
        fidx = int(row["frame_idx"])
        if fidx not in candidate_range:
            continue
        min_dist = min((abs(fidx - e) for e in existing_frames), default=10 ** 9)
        if min_dist < MIN_DIST_TO_EXISTING:
            continue
        kind = "max" if pos in maxima_local_idx else "min"
        records.append({"frame_idx": fidx, "proxy": row["proxy"], "proxy_smoothed": row["proxy_smoothed"],
                         "kind": kind, "min_dist_to_existing_dev_frame": min_dist})

    result = pd.DataFrame(records)
    if len(result) > 0:
        result = result.sort_values("min_dist_to_existing_dev_frame", ascending=False).reset_index(drop=True)
    return result


def select_reversal_frames(n_target: int = N_TARGET) -> list[int]:
    curve_df = compute_proxy_curve()
    extrema_df = find_local_extrema(curve_df, existing_frames=L.DEV_FRAMES_MOTION)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(OUT_DIR / "wingtip_proxy_curve.csv", index=False)
    extrema_df.to_csv(OUT_DIR / "reversal_extrema_candidates.csv", index=False)

    # alternate max/min so the picked set covers both stroke extremes, not just one side;
    # skip a candidate if it lands within MIN_PICK_GAP of an already-picked frame (same
    # broad plateau counted twice, not a genuinely distinct phase).
    maxima = extrema_df[extrema_df["kind"] == "max"]["frame_idx"].tolist()
    minima = extrema_df[extrema_df["kind"] == "min"]["frame_idx"].tolist()
    picked: list[int] = []

    def try_pick(pool: list[int]) -> None:
        while pool:
            cand = pool.pop(0)
            if all(abs(cand - p) >= MIN_PICK_GAP for p in picked):
                picked.append(cand)
                return

    while len(picked) < n_target and (maxima or minima):
        before = len(picked)
        if maxima:
            try_pick(maxima)
        if len(picked) < n_target and minima:
            try_pick(minima)
        if len(picked) == before:  # neither pool yielded a usable pick this round
            break
    picked = sorted(picked)

    plot_proxy_curve(curve_df, extrema_df, picked, OUT_DIR / "wingtip_proxy_curve.png")

    print(f"[reversal_frame_selection] 代理曲线覆盖frame_idx={curve_df['frame_idx'].min()}~"
          f"{curve_df['frame_idx'].max()}({len(curve_df)}帧)，候选窗口安全范围="
          f"[{CANDIDATE_RANGE.start},{CANDIDATE_RANGE.stop - 1}]")
    print(f"[reversal_frame_selection] 局部极值候选(去重+过滤太近已有帧后)={len(extrema_df)}个: "
          f"{extrema_df['frame_idx'].tolist() if len(extrema_df) else '(none)'}")
    print(f"[reversal_frame_selection] 最终选中{len(picked)}帧(交替max/min，覆盖两侧折返): {picked}")
    return picked


def plot_proxy_curve(curve_df: pd.DataFrame, extrema_df: pd.DataFrame, picked: list[int],
                      out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(curve_df["frame_idx"], curve_df["proxy"], color="#4c72b0", alpha=0.4, lw=1,
            label="raw (95th pct dist_to_principal_axis)")
    smoothed = curve_df["proxy"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()
    ax.plot(curve_df["frame_idx"], smoothed, color="#4c72b0", lw=1.8, label=f"smoothed (median, w={SMOOTH_WINDOW})")
    ax.axvspan(CANDIDATE_RANGE.start, CANDIDATE_RANGE.stop - 1, color="gray", alpha=0.08,
               label="window-safe candidate range [260,380]")
    for f in L.DEV_FRAMES_MOTION:
        ax.axvline(f, color="#999999", linestyle=":", lw=1)
    if len(extrema_df) > 0:
        ax.scatter(extrema_df["frame_idx"], extrema_df["proxy_smoothed"], color="#2ca02c", s=25, zorder=5,
                   label="local extrema candidates")
    for f in picked:
        ax.axvline(f, color="#d62728", linestyle="--", lw=1.5)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("wingtip spread proxy (m)")
    ax.set_title("wingtip spread proxy vs frame_idx  (dotted gray = existing 8 dev frames, "
                 "dashed red = newly picked reversal-candidate frames)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_reversal_frames(frames: list[int] = REVERSAL_FRAMES_MOTION) -> None:
    """对已锁定的REVERSAL_FRAMES_MOTION跑完整motion v0分类(label.py::run_batch)，
    _labeled.csv+reprojection落盘到motion/eda_outputs(跟原8帧同一个目录，同一套命名)，
    并把结果行并入motion_dev_summary.csv(按frame_idx去重、排序后整体覆盖写回)。"""
    print(f"\n[reversal_frame_selection] 对REVERSAL_FRAMES_MOTION跑motion v0分类: {frames}")
    results, failures = L.run_batch(frames)
    new_summary = L.build_summary_df(results, failures)

    if L.SUMMARY_CSV.exists():
        old_summary = pd.read_csv(L.SUMMARY_CSV)
        combined = pd.concat([old_summary, new_summary], ignore_index=True)
        combined = combined.drop_duplicates(subset="frame_id", keep="last")
    else:
        combined = new_summary
    combined["_frame_idx"] = combined["frame_id"].str[1:].astype(int)
    combined = combined.sort_values("_frame_idx").drop(columns="_frame_idx").reset_index(drop=True)
    combined.to_csv(L.SUMMARY_CSV, index=False)
    print(f"[reversal_frame_selection] motion_dev_summary.csv 合并更新(共{len(combined)}帧) -> {L.SUMMARY_CSV}")


def main() -> None:
    picked = select_reversal_frames()
    if picked != REVERSAL_FRAMES_MOTION:
        print(f"[reversal_frame_selection] 本次现算结果{picked}跟锁定的REVERSAL_FRAMES_MOTION="
              f"{REVERSAL_FRAMES_MOTION}不同(数据/参数变了才会这样)，如实提示，仍按锁定值跑分类。")
    run_reversal_frames(REVERSAL_FRAMES_MOTION)


if __name__ == "__main__":
    main()
