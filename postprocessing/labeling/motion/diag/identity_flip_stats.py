"""诊断任务3: 量化跨帧body朝向翻转 / wing_L-wing_R身份对调的规模和位置。

背景: 怀疑motion v0(density.py+label.py)产出的_labeled.csv在时间序列上存在两类
不连续性——(1) body PCA主轴x_body的符号在相邻帧间翻转(compute_body_axes用UP=+z
disambiguate x_body方向，而果蝇body长轴近似水平、跟UP接近垂直，dot(x_body,UP)
本身就接近0，PCA噪声容易把这个符号判反)；(2) 两个wing簇的L/R锚定(基于质心投影
right_axis的符号)在相邻帧间对调。这两类不连续会在下游角度序列里表现成瞬时尖峰。

本脚本只读已经产出的_labeled.csv做统计，不碰density.py/label.py的分割逻辑，不做
任何修正——按任务规格"本轮只做量化统计...结果我看完再定下一步"。

判据:
1. body翻转: 两帧body PCA主轴x_body(复用labeling.py::compute_body_axes，只依赖
   part_label=="body"的点，不要求wing_A/wing_B簇标签，直接吃part_label列)的点积，
   点积<0记为一次疑似翻转。
2. wing L/R对调: 分别取两帧wing_L/wing_R质心，比较"不交换"(f的L配f+1的L，R配R)
   和"交换"(f的L配f+1的R，反之)两种配对方式的位移平方和；交换配对的位移明显更小
   (< WING_SWAP_RATIO_THRESHOLD 倍不交换配对，默认0.5)记为一次疑似对调。

注意: "相邻帧对"指按frame_idx排序后在已产出_labeled.csv的帧集合里前后相邻的两帧，
不要求frame_idx严格差1——13个dev/reversal帧本身就不连续(见label.py::
DEV_FRAMES_MOTION + reversal_frame_selection.py::REVERSAL_FRAMES_MOTION)，如实
在输出里带上frame_idx_gap，解释时要考虑跨度大的帧对翻转判据的物理意义打了折扣。
跑整个数据集(如ratio3_sh0_dense的640帧连续输出)时frame_idx_gap基本恒为1。

支持两种_labeled.csv目录布局:
  1) 平铺: <root>/gaussian_features_f####_labeled.csv (motion/eda_outputs下13个
     dev/reversal帧的默认落盘位置，label.py::run_batch直接写这里)。
  2) 按帧嵌套: <root>/f####/<splatfacto|splatfacto-checkpoint>/<时间戳>/
     gaussian_features_f####_labeled.csv (calc_kinematics.py跑T1-T4整个数据集时
     的落盘位置，如outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense，
     该数据集T3就是本模块label.py::run_batch产出的，见calc_kinematics.py
     run_cleaning_and_labeling文档)。glob模式跟calc_kinematics.py::
     LABELED_FRAME_GLOB="f*/*/*/*_labeled.csv"一致，不写死method目录名。

用法:
    python -m postprocessing.labeling.motion.diag.identity_flip_stats
    python -m postprocessing.labeling.motion.diag.identity_flip_stats \\
        outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.labeling.labeling import compute_body_axes  # noqa: E402
from postprocessing.labeling.motion import label as L  # noqa: E402

LABELED_CSV_DIR = L.OUT_DIR
"""默认_labeled.csv所在目录: motion/eda_outputs(label.py::run_batch的落盘位置，13个
dev/reversal帧)。传其它数据集根目录(见NESTED_FRAME_GLOB)可覆盖，不限于这13帧。"""

NESTED_FRAME_GLOB = "f*/*/*/*_labeled.csv"
"""按帧嵌套布局的glob，跟calc_kinematics.py::LABELED_FRAME_GLOB同一个模式，兼容
splatfacto/splatfacto-checkpoint两种method目录名。"""

OUT_DIR = Path(__file__).resolve().parent

FRAME_CSV_RE = re.compile(r"gaussian_features_f(\d+)_labeled\.csv$")

WING_SWAP_RATIO_THRESHOLD = 0.5
"""交换配对位移平方和 < 此比例 x 不交换配对位移平方和 时，记为一次疑似wing L/R对调。
可调，见任务规格"给个阈值,比如小于50%,可调"。"""


def stats_csv_path(root: Path) -> Path:
    """输出csv路径: 默认13帧dev集用固定名identity_flip_stats.csv(原任务规格路径)，
    传其它数据集根目录时按数据集名区分，避免覆盖前一次结果。"""
    if root == LABELED_CSV_DIR:
        return OUT_DIR / "identity_flip_stats.csv"
    return OUT_DIR / f"identity_flip_stats_{root.name}.csv"


def discover_labeled_frames(root: Path = LABELED_CSV_DIR) -> dict[int, Path]:
    """扫描root下所有_labeled.csv(平铺+按帧嵌套两种布局都试)，返回frame_idx->csv路径。
    嵌套布局里同一帧若有多个时间戳目录(重跑过)，按时间戳字符串排序取最新一份
    (时间戳格式YYYY-MM-DD_HHMMSS，字典序=时间序)。"""
    candidates = list(root.glob("gaussian_features_f*_labeled.csv")) + list(root.glob(NESTED_FRAME_GLOB))
    by_frame: dict[int, list[Path]] = {}
    for p in candidates:
        m = FRAME_CSV_RE.search(p.name)
        if m:
            by_frame.setdefault(int(m.group(1)), []).append(p)
    return {frame_idx: sorted(paths)[-1] for frame_idx, paths in by_frame.items()}


def load_labeled(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def frame_body_axis(df: pd.DataFrame) -> np.ndarray | None:
    """该帧(仅if_keep=True点)的x_body方向，复用compute_body_axes——只需要part_label
    列里"body"这一类，不要求wing_A/wing_B簇标签(compute_body_axes只用semantic=="body"
    这一个比较)。body点数为0时返回None(不该发生，但防御式处理，不让整个统计崩掉)。"""
    df_kept = df[df["if_keep"].astype(bool)]
    if (df_kept["part_label"] == "body").sum() == 0:
        return None
    xyz_kept = df_kept[["x", "y", "z"]].to_numpy()
    semantic_kept = df_kept["part_label"].to_numpy()
    x_body, _right_axis, _body_cm = compute_body_axes(xyz_kept, semantic_kept)
    return x_body


def frame_wing_centroids(df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    """该帧(仅if_keep=True点)wing_L/wing_R质心，某一侧点数为0时对应返回None。"""
    df_kept = df[df["if_keep"].astype(bool)]
    centroids = {}
    for side in ("wing_L", "wing_R"):
        xyz = df_kept.loc[df_kept["part_label"] == side, ["x", "y", "z"]].to_numpy()
        centroids[side] = xyz.mean(axis=0) if len(xyz) > 0 else None
    return centroids["wing_L"], centroids["wing_R"]


def frame_confidence(df: pd.DataFrame) -> str:
    vals = df["confidence"].unique()
    if len(vals) != 1:
        print(f"  [警告] confidence列在该帧内不唯一: {vals}，取第一个值")
    return str(vals[0])


def compute_pair_stats(frame_i: int, frame_j: int, df_i: pd.DataFrame, df_j: pd.DataFrame) -> dict:

    conf_i = frame_confidence(df_i)
    conf_j = frame_confidence(df_j)

    row = {
        "frame_i": f"f{frame_i:04d}", "frame_j": f"f{frame_j:04d}",
        "frame_idx_i": frame_i, "frame_idx_j": frame_j, "frame_idx_gap": frame_j - frame_i,
        "confidence_i": conf_i, "confidence_j": conf_j,
        "body_dot_product": np.nan, "body_flip": False,
        "swap_vs_noswap_displacement_ratio": np.nan, "wing_swap": False,
    }

    x_body_i = frame_body_axis(df_i)
    x_body_j = frame_body_axis(df_j)
    if x_body_i is not None and x_body_j is not None:
        dot = float(np.dot(x_body_i, x_body_j))
        row["body_dot_product"] = dot
        row["body_flip"] = dot < 0
    else:
        print(f"  [警告] {row['frame_i']}-{row['frame_j']}: body点数为0，跳过body翻转判据")

    l_i, r_i = frame_wing_centroids(df_i)
    l_j, r_j = frame_wing_centroids(df_j)
    if l_i is not None and r_i is not None and l_j is not None and r_j is not None:
        noswap_sq = float(np.sum((l_i - l_j) ** 2) + np.sum((r_i - r_j) ** 2))
        swap_sq = float(np.sum((l_i - r_j) ** 2) + np.sum((r_i - l_j) ** 2))
        if noswap_sq > 0:
            ratio = swap_sq / noswap_sq
            row["swap_vs_noswap_displacement_ratio"] = ratio
            row["wing_swap"] = ratio < WING_SWAP_RATIO_THRESHOLD
        else:
            print(f"  [警告] {row['frame_i']}-{row['frame_j']}: 不交换位移平方和为0(质心完全"
                  f"重合)，无法算比值，跳过wing对调判据")
    else:
        print(f"  [警告] {row['frame_i']}-{row['frame_j']}: wing_L/wing_R至少一侧点数为0，"
              f"跳过wing对调判据")

    return row


def build_stats_table(root: Path = LABELED_CSV_DIR) -> pd.DataFrame:
    frame_paths = discover_labeled_frames(root)
    frame_idxs = sorted(frame_paths)
    if len(frame_idxs) <= 20:
        print(f"[identity_flip_stats] 发现{len(frame_idxs)}帧已产出_labeled.csv: "
              f"{[f'f{i:04d}' for i in frame_idxs]}")
    else:
        print(f"[identity_flip_stats] 发现{len(frame_idxs)}帧已产出_labeled.csv: "
              f"f{frame_idxs[0]:04d}..f{frame_idxs[-1]:04d}")

    rows = []
    # 相邻帧对共享一帧(f的df既是上一对的frame_j又是下一对的frame_i)，缓存上一次load
    # 的df，避免640帧规模下每个文件被读两遍。
    prev_idx, prev_df = None, None
    for frame_i, frame_j in zip(frame_idxs[:-1], frame_idxs[1:]):
        df_i = prev_df if frame_i == prev_idx else load_labeled(frame_paths[frame_i])
        df_j = load_labeled(frame_paths[frame_j])
        rows.append(compute_pair_stats(frame_i, frame_j, df_i, df_j))
        prev_idx, prev_df = frame_j, df_j
    return pd.DataFrame(rows)


def print_summary(stats_df: pd.DataFrame) -> None:
    n_pairs = len(stats_df)
    if n_pairs == 0:
        print("[identity_flip_stats] 相邻帧对数为0(需要至少2帧_labeled.csv)，无法统计。")
        return

    n_body_flip = int(stats_df["body_flip"].sum())
    n_wing_swap = int(stats_df["wing_swap"].sum())
    any_flip_mask = stats_df["body_flip"] | stats_df["wing_swap"]
    n_any_flip = int(any_flip_mask.sum())

    low_mask_any_frame = (stats_df["confidence_i"] == "low") | (stats_df["confidence_j"] == "low")
    base_low_rate = float(low_mask_any_frame.mean())
    flip_low_rate = float(low_mask_any_frame[any_flip_mask].mean()) if n_any_flip > 0 else float("nan")

    def frame_pair_list(mask: pd.Series, max_show: int = 30) -> str:
        pairs = [f"{row.frame_i}-{row.frame_j}" for row in stats_df[mask].itertuples()]
        if len(pairs) <= max_show:
            return str(pairs)
        return f"{pairs[:max_show]} ... (+{len(pairs) - max_show}个，完整列表见csv)"

    print(f"\n{'=' * 70}\nidentity_flip_stats 汇总 ({n_pairs}个相邻帧对)\n{'=' * 70}")
    print(f"  body翻转: {n_body_flip}/{n_pairs} ({100 * n_body_flip / n_pairs:.1f}%)")
    if n_body_flip > 0:
        print(f"    帧对: {frame_pair_list(stats_df['body_flip'])}")
    print(f"  wing L/R对调: {n_wing_swap}/{n_pairs} ({100 * n_wing_swap / n_pairs:.1f}%)"
          f"  (阈值 ratio < {WING_SWAP_RATIO_THRESHOLD})")
    if n_wing_swap > 0:
        print(f"    帧对: {frame_pair_list(stats_df['wing_swap'])}")
    print(f"  两者任一触发: {n_any_flip}/{n_pairs} ({100 * n_any_flip / n_pairs:.1f}%)")
    if n_any_flip > 0:
        print(f"    帧对: {frame_pair_list(any_flip_mask)}")

    print(f"\n  confidence=low 覆盖率对比:")
    print(f"    全体相邻帧对中,至少一帧confidence=low的基础比例: "
          f"{100 * base_low_rate:.1f}% ({int(low_mask_any_frame.sum())}/{n_pairs})")
    if n_any_flip > 0:
        print(f"    翻转/对调发生的帧对中,至少一帧confidence=low的比例: "
              f"{100 * flip_low_rate:.1f}% ({int(low_mask_any_frame[any_flip_mask].sum())}/{n_any_flip})")
        if flip_low_rate > base_low_rate:
            print(f"    -> 翻转/对调更多集中在confidence=low附近(高于基础比例)。")
        elif flip_low_rate < base_low_rate:
            print(f"    -> 翻转/对调更多发生在confidence=high帧对上(低于基础比例)，"
                  f"说明现有confidence标记没有覆盖到这类不连续性。")
        else:
            print(f"    -> 翻转/对调帧对里的low比例跟基础比例相同，看不出关联。")
    else:
        print(f"    没有发生翻转/对调的帧对，无法比较。")

    n_large_gap_flip = int((any_flip_mask & (stats_df["frame_idx_gap"] > 1)).sum())
    if n_large_gap_flip > 0:
        print(f"\n  [提示] {n_large_gap_flip}个触发翻转/对调的帧对frame_idx_gap>1(源_labeled.csv"
              f"帧集合本身不连续时才会出现，比如13个dev/reversal帧)，判据的物理意义打了折扣，"
              f"复查时优先看frame_idx_gap=1的帧对。")


def run(root: Path = LABELED_CSV_DIR) -> pd.DataFrame:
    stats_df = build_stats_table(root)
    if len(stats_df) > 0:
        out_csv = stats_csv_path(root)
        stats_df.to_csv(out_csv, index=False)
        print(f"\n  csv -> {out_csv}")
    print_summary(stats_df)
    return stats_df


if __name__ == "__main__":
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else LABELED_CSV_DIR
    run(dataset_root)
