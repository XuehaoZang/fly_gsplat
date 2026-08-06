"""诊断任务: 验证body符号翻转的根因——是否跟|dot(x_body,UP)|接近0(body朝向接近水平)相关。

背景: identity_flip_stats.py已产出640帧(ratio3_sh0_dense数据集)相邻帧对统计，39次
body翻转(dot(x_body_i, x_body_j) < 0)，成段连续出现(如f0158~f0164)。假设1: body长轴
接近水平(跟UP=+z夹角接近90°，即dot(x_body,UP)接近0)时，PCA主轴符号对噪声极度敏感，
orient_to_reference的符号修正容易翻错。备选假设2: 不是朝向问题，是body点云本身接近
球形/退化(PCA最大两个特征值比值接近1，长轴方向不再显著占优)，导致主轴方向本身(不只
是符号)不稳定，跟UP无关。

本脚本只做验证性统计，复用compute_body_axes同一套PCA计算(weighted_pca)，不改
compute_body_axes/identity_flip_stats.py的任何逻辑，不产出修复方案。

|dot(x_body,UP)|说明: weighted_pca返回的主轴(eigvecs[:,-1])符号本身是任意的
(orient_to_reference修正前)，但绝对值|dot(v,UP)|对符号翻转不变，所以这里不需要也不
调用orient_to_reference，直接对PCA原始输出取绝对值点积，反映轴本身跟UP的正交程度。

eigval_ratio说明: weighted_pca按特征值从小到大排列，eigvals[-1]是长轴(最大方差)、
eigvals[-2]是次大方差方向。eigval_ratio = eigvals[-2] / eigvals[-1]，比值接近1说明
点云在前两个主方向上方差相当，长轴方向对噪声敏感(细长程度低/接近扁圆盘或球)；比值
接近0说明长轴显著占优，方向稳定(细长程度高)。

判据(见任务规格"给个简单判据"): 翻转组|dot|中位数是否落在正常组|dot|分布的后10%
分位数(P10)以内 => 判定主因是body朝向接近水平。若不满足，看eigval_ratio翻转组中位数
是否落在正常组分布的前10%分位数(P90)以上 => 判定主因是PCA形状退化。两者都不满足 =>
两者都不是，需要进一步排查。

上游点云排查(run_point_cloud_diag): 已确认翻转组(=eigval_ratio高的"退化组")跟正常组
在eigval_ratio上分布分离，本节接着排查这个退化是否伴随更上游的问题——motion方法
(density.py+label.py)在这些帧提取出的body候选点云本身点数变少/覆盖不全(比如遮挡)。
复用同一套body_flip分组(不重新定阈值)，额外对比n_body_points(点数)、bbox_diag(空间
extent)、sqrt_eigval_max(第一主轴方差的绝对量级，不是比值)、相邻帧对质心位移(简单看
质心是否也偏离常态)。判据沿用同一套P10/P90分位数规则，只是"退化"的方向相反(点数/
extent变小、质心位移变大才是问题信号)。

用法:
    python -m postprocessing.labeling.motion.diag.flip_root_cause_check
    python -m postprocessing.labeling.motion.diag.flip_root_cause_check \\
        outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense
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

from postprocessing.calc_kinematics import DEFAULT_DATASET_ROOT  # noqa: E402
from postprocessing.kinematics.geometry import weighted_pca  # noqa: E402
from postprocessing.labeling.labeling import UP  # noqa: E402
from postprocessing.labeling.motion.diag import identity_flip_stats as IFS  # noqa: E402
from postprocessing.labeling.motion.diag.body_centroid_stability import bbox_diagonal  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"

LOW_PERCENTILE = 10.0
"""判据用的分位数: 翻转组|dot|中位数落在正常组分布后LOW_PERCENTILE%以内 => 判定为
朝向接近水平导致；翻转组eigval_ratio中位数落在正常组分布前LOW_PERCENTILE%以上(即
>=(100-LOW_PERCENTILE)分位数) => 判定为PCA形状退化导致。见任务规格"比如翻转组中位数
是否落在正常组的后10%分位数以内"。"""


def per_frame_pca_diag(root: Path) -> pd.DataFrame:
    """对root下所有已产出_labeled.csv的帧，各算一次body点云的PCA，返回
    frame_idx, abs_dot_up(|dot(长轴,UP)|,符号修正前后不影响), eigval_ratio
    (次大/最大特征值比值), n_body(body点数)，另外为上游点云排查任务(见模块docstring
    "排查PCA形状退化帧的body候选点云是否本身有问题"这段)补充: bbox_diag(body点云bbox
    对角线，body_centroid_stability.py同款定义，复用不重写)、sqrt_eigval_max(第一主轴
    方差的平方根，PCA尺度的绝对量级，不只是eigval_ratio这个比值)、cx/cy/cz(body质心，
    供跨帧质心位移对比用)。body点数为0的帧跳过(不该发生，防御式处理，对齐
    identity_flip_stats.py::frame_body_axis同样的空集处理方式)。"""
    frame_paths = IFS.discover_labeled_frames(root)
    rows = []
    for frame_idx in sorted(frame_paths):
        df = IFS.load_labeled(frame_paths[frame_idx])
        df_kept = df[df["if_keep"].astype(bool)]
        body_xyz = df_kept.loc[df_kept["part_label"] == "body", ["x", "y", "z"]].to_numpy()
        if len(body_xyz) == 0:
            print(f"  [警告] f{frame_idx:04d}: body点数为0，跳过PCA诊断")
            continue
        eigvals, eigvecs, centroid = weighted_pca(body_xyz)
        x_body_raw = eigvecs[:, -1]
        abs_dot = float(abs(np.dot(x_body_raw, UP)))
        eigval_ratio = float(eigvals[-2] / eigvals[-1]) if eigvals[-1] > 0 else float("nan")
        sqrt_eigval_max = float(np.sqrt(max(eigvals[-1], 0.0)))
        rows.append({"frame_idx": frame_idx, "abs_dot_up": abs_dot,
                     "eigval_ratio": eigval_ratio, "n_body": len(body_xyz),
                     "bbox_diag": bbox_diagonal(body_xyz), "sqrt_eigval_max": sqrt_eigval_max,
                     "cx": centroid[0], "cy": centroid[1], "cz": centroid[2]})
    return pd.DataFrame(rows).set_index("frame_idx")


def load_flip_pairs(root: Path) -> pd.DataFrame:
    """复用identity_flip_stats.py已产出的_labeled.csv翻转统计(body_flip列)；csv不存在
    时现算一遍(逻辑完全复用IFS.build_stats_table，不重新实现)。"""
    csv_path = IFS.stats_csv_path(root)
    if csv_path.exists():
        print(f"[flip_root_cause_check] 复用已产出的翻转统计: {csv_path}")
        return pd.read_csv(csv_path)
    print(f"[flip_root_cause_check] {csv_path}不存在，现算一遍identity_flip_stats")
    return IFS.build_stats_table(root)


def build_group_values(stats_df: pd.DataFrame, pca_df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """按body_flip把相邻帧对切成翻转组/正常组，每对贡献两帧各自的col值(不去重——同一帧
    在相邻两对里各出现一次是预期行为，见任务规格"取其中两帧各自的...值")。"""
    flip_vals, normal_vals = [], []
    for row in stats_df.itertuples():
        if row.frame_idx_i not in pca_df.index or row.frame_idx_j not in pca_df.index:
            continue
        vi = pca_df.loc[row.frame_idx_i, col]
        vj = pca_df.loc[row.frame_idx_j, col]
        target = flip_vals if row.body_flip else normal_vals
        target.append(vi)
        target.append(vj)
    return np.array(flip_vals, dtype=float), np.array(normal_vals, dtype=float)


def pair_centroid_distance(stats_df: pd.DataFrame, pca_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """质心位移是相邻帧对(i,j)之间的量，不是单帧属性，跟build_group_values的"每对拆两个
    单帧值"不是一回事——这里每对只产出一个||centroid_i - centroid_j||值，按body_flip切
    成翻转组/正常组，对应任务规格"跟前后相邻正常帧的质心距离"：body_flip=True的帧对本身
    就是"退化帧跟其相邻帧"的位移，body_flip=False的帧对是"正常帧跟其相邻帧"的位移，
    两组直接对比。"""
    flip_vals, normal_vals = [], []
    for row in stats_df.itertuples():
        if row.frame_idx_i not in pca_df.index or row.frame_idx_j not in pca_df.index:
            continue
        ci = pca_df.loc[row.frame_idx_i, ["cx", "cy", "cz"]].to_numpy(dtype=float)
        cj = pca_df.loc[row.frame_idx_j, ["cx", "cy", "cz"]].to_numpy(dtype=float)
        dist = float(np.linalg.norm(ci - cj))
        target = flip_vals if row.body_flip else normal_vals
        target.append(dist)
    return np.array(flip_vals, dtype=float), np.array(normal_vals, dtype=float)


def summarize_group(name: str, vals: np.ndarray) -> dict:
    return {"group": name, "n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def plot_comparison(flip_dot: np.ndarray, normal_dot: np.ndarray,
                     flip_ratio: np.ndarray, normal_ratio: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    bins = np.linspace(0, 1, 31)
    ax.hist(normal_dot, bins=bins, alpha=0.6, label=f"normal (n={len(normal_dot)})", color="tab:blue", density=True)
    ax.hist(flip_dot, bins=bins, alpha=0.6, label=f"flipped (n={len(flip_dot)})", color="tab:red", density=True)
    ax.set_xlabel("|dot(x_body_raw, UP)|")
    ax.set_ylabel("density")
    ax.set_title("Histogram: |dot(x_body, UP)| by group")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.boxplot([normal_dot, flip_dot], tick_labels=["normal", "flipped"])
    ax.set_ylabel("|dot(x_body_raw, UP)|")
    ax.set_title("Boxplot: |dot(x_body, UP)| by group")

    ax = axes[1, 0]
    bins_r = np.linspace(0, 1, 31)
    ax.hist(normal_ratio, bins=bins_r, alpha=0.6, label=f"normal (n={len(normal_ratio)})", color="tab:blue", density=True)
    ax.hist(flip_ratio, bins=bins_r, alpha=0.6, label=f"flipped (n={len(flip_ratio)})", color="tab:red", density=True)
    ax.set_xlabel("eigval_ratio (2nd largest / largest)")
    ax.set_ylabel("density")
    ax.set_title("Histogram: PCA eigval_ratio by group")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.boxplot([normal_ratio, flip_ratio], tick_labels=["normal", "flipped"])
    ax.set_ylabel("eigval_ratio (2nd largest / largest)")
    ax.set_title("Boxplot: PCA eigval_ratio by group")

    fig.suptitle("body_flip root-cause check: orientation (|dot·UP|) vs PCA shape degeneracy (eigval_ratio)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot -> {out_path}")


def plot_point_cloud_diag(flip_n: np.ndarray, normal_n: np.ndarray,
                           flip_extent: np.ndarray, normal_extent: np.ndarray,
                           flip_scale: np.ndarray, normal_scale: np.ndarray,
                           flip_cdist: np.ndarray, normal_cdist: np.ndarray, out_path: Path) -> None:
    metrics = [
        ("n_body_points", "body candidate point count", normal_n, flip_n, None),
        ("bbox_diag", "body bbox diagonal (m)", normal_extent, flip_extent, None),
        ("sqrt_eigval_max", "sqrt(largest PCA eigenvalue) (m)", normal_scale, flip_scale, None),
        ("centroid_dist", "adjacent-pair body centroid displacement (m)", normal_cdist, flip_cdist, None),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11, 16))
    for row_idx, (_key, xlabel, normal_vals, flip_vals, _unused) in enumerate(metrics):
        ax_hist, ax_box = axes[row_idx]
        lo = min(normal_vals.min(), flip_vals.min())
        hi = max(normal_vals.max(), flip_vals.max())
        bins = np.linspace(lo, hi, 31)
        ax_hist.hist(normal_vals, bins=bins, alpha=0.6, label=f"normal (n={len(normal_vals)})",
                     color="tab:blue", density=True)
        ax_hist.hist(flip_vals, bins=bins, alpha=0.6, label=f"flipped (n={len(flip_vals)})",
                     color="tab:red", density=True)
        ax_hist.set_xlabel(xlabel)
        ax_hist.set_ylabel("density")
        ax_hist.set_title(f"Histogram: {xlabel} by group")
        ax_hist.legend(fontsize=8)

        ax_box.boxplot([normal_vals, flip_vals], tick_labels=["normal", "flipped"])
        ax_box.set_ylabel(xlabel)
        ax_box.set_title(f"Boxplot: {xlabel} by group")

    fig.suptitle("body_flip point-cloud diagnostics: point count / extent / centroid displacement by group",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot -> {out_path}")


def run_point_cloud_diag(stats_df: pd.DataFrame, pca_df: pd.DataFrame) -> None:
    """排查PCA形状退化(=body_flip)帧的body候选点云本身是否有问题(点数偏少/覆盖收缩/
    质心异常跳变)，还是点云跟正常帧差不多、纯粹是PCA数值鲁棒性问题。只读上一步已经
    算出的stats_df(body_flip分组)和pca_df(每帧PCA+点数+bbox+质心)，不改任何分割/PCA
    逻辑，见模块docstring本节说明。"""
    flip_n, normal_n = build_group_values(stats_df, pca_df, "n_body")
    flip_extent, normal_extent = build_group_values(stats_df, pca_df, "bbox_diag")
    flip_scale, normal_scale = build_group_values(stats_df, pca_df, "sqrt_eigval_max")
    flip_cdist, normal_cdist = pair_centroid_distance(stats_df, pca_df)

    summary_rows = [summarize_group("flipped_frames_n_body_points", flip_n),
                     summarize_group("normal_frames_n_body_points", normal_n),
                     summarize_group("flipped_frames_bbox_diag", flip_extent),
                     summarize_group("normal_frames_bbox_diag", normal_extent),
                     summarize_group("flipped_frames_sqrt_eigval_max", flip_scale),
                     summarize_group("normal_frames_sqrt_eigval_max", normal_scale),
                     summarize_group("flipped_pairs_centroid_dist", flip_cdist),
                     summarize_group("normal_pairs_centroid_dist", normal_cdist)]
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'=' * 78}\n上游点云排查: 点数/extent/质心位移 统计汇总\n{'=' * 78}")
    print(summary_df.to_string(index=False))

    out_csv = OUT_DIR / "flip_point_cloud_diag_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n  csv -> {out_csv}")

    plot_point_cloud_diag(flip_n, normal_n, flip_extent, normal_extent,
                           flip_scale, normal_scale, flip_cdist, normal_cdist,
                           OUT_DIR / "flip_point_cloud_diag.png")

    # 判据: 退化组中位数是否落在正常组分布后10%分位数以内(点数/extent明显收缩)，
    # 跟flip_root_cause_check的LOW_PERCENTILE判据同一套逻辑，只是这里"退化"是量变小
    # (点数/extent变小是问题信号)，跟之前"eigval_ratio变大是问题信号"方向相反。
    def shrink_hit(flip_vals: np.ndarray, normal_vals: np.ndarray) -> tuple[bool, float, float, float]:
        p10 = float(np.percentile(normal_vals, LOW_PERCENTILE))
        flip_med = float(np.median(flip_vals))
        normal_med = float(np.median(normal_vals))
        return flip_med <= p10, flip_med, normal_med, p10

    n_hit, n_flip_med, n_normal_med, n_p10 = shrink_hit(flip_n, normal_n)
    extent_hit, extent_flip_med, extent_normal_med, extent_p10 = shrink_hit(flip_extent, normal_extent)
    scale_hit, scale_flip_med, scale_normal_med, scale_p10 = shrink_hit(flip_scale, normal_scale)
    cdist_p90 = float(np.percentile(normal_cdist, 100 - LOW_PERCENTILE))
    cdist_flip_med = float(np.median(flip_cdist))
    cdist_normal_med = float(np.median(normal_cdist))
    cdist_hit = cdist_flip_med >= cdist_p90

    print(f"\n{'=' * 78}\n判据结果(退化组 vs 正常组, P{LOW_PERCENTILE:.0f}/P{100 - LOW_PERCENTILE:.0f}分位线)\n{'=' * 78}")
    print(f"  [点数] 退化组中位数={n_flip_med:.1f}  正常组中位数={n_normal_med:.1f}  "
          f"正常组P{LOW_PERCENTILE:.0f}={n_p10:.1f}  => {'命中(点数明显偏少)' if n_hit else '不命中'}")
    print(f"  [bbox extent] 退化组中位数={extent_flip_med:.3e}  正常组中位数={extent_normal_med:.3e}  "
          f"正常组P{LOW_PERCENTILE:.0f}={extent_p10:.3e}  => {'命中(extent明显收缩)' if extent_hit else '不命中'}")
    print(f"  [sqrt(eigval_max)] 退化组中位数={scale_flip_med:.3e}  正常组中位数={scale_normal_med:.3e}  "
          f"正常组P{LOW_PERCENTILE:.0f}={scale_p10:.3e}  => {'命中(PCA绝对尺度明显收缩)' if scale_hit else '不命中'}")
    print(f"  [质心位移] 退化组中位数={cdist_flip_med:.3e}  正常组中位数={cdist_normal_med:.3e}  "
          f"正常组P{100 - LOW_PERCENTILE:.0f}={cdist_p90:.3e}  "
          f"=> {'命中(质心异常跳变)' if cdist_hit else '不命中'}")

    n_ratio = n_flip_med / n_normal_med if n_normal_med > 0 else float("nan")
    extent_ratio = extent_flip_med / extent_normal_med if extent_normal_med > 0 else float("nan")
    scale_ratio = scale_flip_med / scale_normal_med if scale_normal_med > 0 else float("nan")

    print(f"\n{'=' * 78}\n结论: 上游点云是否有问题\n{'=' * 78}")
    print(f"  退化组/正常组中位数比值: 点数={n_ratio:.2f}x  bbox_extent={extent_ratio:.2f}x  "
          f"sqrt_eigval_max={scale_ratio:.2f}x")
    if n_hit or extent_hit or scale_hit:
        hit_names = [name for name, hit in [("点数", n_hit), ("bbox extent", extent_hit),
                                             ("PCA绝对尺度sqrt(eigval_max)", scale_hit)] if hit]
        print(f"  [结论] 退化组伴随{'/'.join(hit_names)}明显偏低(落进正常组P{LOW_PERCENTILE:.0f}分位数"
              f"以内)，说明问题部分在上游: motion方法在这些帧提取出的body候选点云本身"
              f"点数变少/覆盖收缩(比如遮挡)，PCA主轴方向本身就是在退化的点云上算出来的，"
              f"不是纯粹的PCA数值鲁棒性问题。")
    else:
        print(f"  [结论] 退化组的点数({n_ratio:.2f}x)、bbox extent({extent_ratio:.2f}x)、"
              f"PCA绝对尺度({scale_ratio:.2f}x)都没有明显低于正常组(均未落进正常组P"
              f"{LOW_PERCENTILE:.0f}分位数以内)，说明上游body候选点云本身规模/覆盖度正常，"
              f"eigval_ratio(第二/最大特征值比值)变大纯粹是点云在两个主方向上方差变得接近"
              f"(形状更接近扁圆盘/球)——一个数值鲁棒性问题，不是点数/覆盖度不足导致的。")
    if cdist_hit:
        print(f"  [附加] 退化组相邻帧对质心位移中位数({cdist_flip_med:.3e})落进正常组P"
              f"{100 - LOW_PERCENTILE:.0f}以上，说明这些帧body质心位置也偏离常态(可能是"
              f"部分点被误判进/漏出body导致质心跳变)，跟上面点云规模的结论互相印证。")
    else:
        print(f"  [附加] 退化组相邻帧对质心位移中位数({cdist_flip_med:.3e})没有明显高于正常组，"
              f"body质心位置本身没有异常跳变。")


def run(root: Path = DEFAULT_DATASET_ROOT) -> None:
    print(f"[flip_root_cause_check] 数据集根目录: {root}")
    stats_df = load_flip_pairs(root)
    n_flip_pairs = int(stats_df["body_flip"].sum())
    print(f"[flip_root_cause_check] {len(stats_df)}个相邻帧对，其中{n_flip_pairs}个body_flip=True")

    pca_df = per_frame_pca_diag(root)
    print(f"[flip_root_cause_check] 完成{len(pca_df)}帧PCA诊断计算")

    flip_dot, normal_dot = build_group_values(stats_df, pca_df, "abs_dot_up")
    flip_ratio, normal_ratio = build_group_values(stats_df, pca_df, "eigval_ratio")

    summary_rows = [summarize_group("flipped_pairs_frames_|dot_up|", flip_dot),
                     summarize_group("normal_pairs_frames_|dot_up|", normal_dot),
                     summarize_group("flipped_pairs_frames_eigval_ratio", flip_ratio),
                     summarize_group("normal_pairs_frames_eigval_ratio", normal_ratio)]
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'=' * 78}\nflip_root_cause_check 统计汇总\n{'=' * 78}")
    print(summary_df.to_string(index=False))

    out_csv = OUT_DIR / "flip_root_cause_check_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n  csv -> {out_csv}")

    plot_comparison(flip_dot, normal_dot, flip_ratio, normal_ratio, OUT_DIR / "flip_root_cause_check.png")

    # 判据1: 朝向假设——翻转组|dot|中位数是否落在正常组分布后10%分位数以内
    dot_p10 = float(np.percentile(normal_dot, LOW_PERCENTILE))
    flip_dot_median = float(np.median(flip_dot))
    normal_dot_median = float(np.median(normal_dot))
    orientation_hit = flip_dot_median <= dot_p10

    # 判据2(备选): 形状退化假设——翻转组eigval_ratio中位数是否落在正常组分布前10%分位数以上
    ratio_p90 = float(np.percentile(normal_ratio, 100 - LOW_PERCENTILE))
    flip_ratio_median = float(np.median(flip_ratio))
    normal_ratio_median = float(np.median(normal_ratio))
    shape_hit = flip_ratio_median >= ratio_p90

    print(f"\n{'=' * 78}\n判据结果\n{'=' * 78}")
    print(f"  [朝向假设] 翻转组|dot(x_body,UP)|中位数={flip_dot_median:.4f}  "
          f"vs 正常组中位数={normal_dot_median:.4f}, 正常组P{LOW_PERCENTILE:.0f}={dot_p10:.4f}")
    print(f"    翻转组中位数{'<=' if orientation_hit else '>'}正常组P{LOW_PERCENTILE:.0f} "
          f"=> {'命中' if orientation_hit else '不命中'}")
    print(f"  [形状退化假设] 翻转组eigval_ratio中位数={flip_ratio_median:.4f}  "
          f"vs 正常组中位数={normal_ratio_median:.4f}, 正常组P{100 - LOW_PERCENTILE:.0f}={ratio_p90:.4f}")
    print(f"    翻转组中位数{'>=' if shape_hit else '<'}正常组P{100 - LOW_PERCENTILE:.0f} "
          f"=> {'命中' if shape_hit else '不命中'}")

    print(f"\n{'=' * 78}\n结论\n{'=' * 78}")
    if orientation_hit:
        print(f"  根因: body朝向接近水平 (orientation)。"
              f"翻转组|dot(x_body,UP)|中位数={flip_dot_median:.4f} <= "
              f"正常组P{LOW_PERCENTILE:.0f}={dot_p10:.4f}(正常组中位数{normal_dot_median:.4f})，"
              f"说明翻转确实集中发生在body长轴接近跟UP正交(接近水平)的帧对上。")
    elif shape_hit:
        print(f"  根因: PCA形状退化 (eigval ratio close to 1)。"
              f"翻转组eigval_ratio中位数={flip_ratio_median:.4f} >= "
              f"正常组P{100 - LOW_PERCENTILE:.0f}={ratio_p90:.4f}(正常组中位数{normal_ratio_median:.4f})，"
              f"朝向假设未命中(翻转组|dot|中位数{flip_dot_median:.4f} > 正常组P{LOW_PERCENTILE:.0f}"
              f"={dot_p10:.4f})，说明翻转跟朝向无关，而是body点云本身接近球形/退化导致主轴方向不稳定。")
    else:
        print(f"  根因: 两者都不是，需要进一步排查。朝向假设未命中(翻转组|dot|中位数"
              f"{flip_dot_median:.4f} > 正常组P{LOW_PERCENTILE:.0f}={dot_p10:.4f})，形状退化假设也未命中"
              f"(翻转组eigval_ratio中位数{flip_ratio_median:.4f} < 正常组P{100 - LOW_PERCENTILE:.0f}"
              f"={ratio_p90:.4f})，翻转组跟正常组在这两个指标上分布重叠，需要看别的信号"
              f"(如body种子点数/confidence/其它帧间不连续性)。")

    run_point_cloud_diag(stats_df, pca_df)


if __name__ == "__main__":
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_ROOT
    run(dataset_root)
