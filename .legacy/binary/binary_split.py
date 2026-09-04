"""
T3 第一步: body/wing 二分类，规则阈值法(不做无监督聚类)。

判据 (两特征"或"组合，均来自 T1 utils/gaussian_features.py 的既有列，不新算特征):
- dist_to_principal_axis: 点到全身主轴的径向距离。翅膀展开时在主轴的横向(展向)
  伸出很远，是最直接的空间判据。
- planarity: (lam2-lam3)/lam1，形状"扁平程度"。翅膀是膜状结构，对应的高斯团
  法向那一维尺度远小于面内两维，planarity偏高；躯干更接近实心/线状(细长毛刺
  除外)，planarity偏低。这一项用来补上"翅根附近离主轴不远、但形状已经明显扁平"
  的点——纯空间判据在这个区域会漏判。

    is_wing = (dist_to_principal_axis > axis_dist_th) | (planarity > planarity_th)

用"或"而非"与": 翅根/翼尖附近的点经常只强烈满足其中一个条件，同时要求两者会漏判
太多真实翅膀点(尤其是根部)。判据本身粗糙，边界误差留给后续步骤兜底(见任务说明:
根部划给body是预期行为)。

阈值参数化: 不写死绝对值，改用"分位数"(quantile)——不同帧点云的绝对尺度(姿态/
展开程度)有差异，绝对阈值在帧间不稳定，分位数阈值对这种帧间漂移更鲁棒。分位数
在 if_keep=True(排除floater)的点里计算，避免floater尾部把分位数拉偏。

默认分位数 DEFAULT_PLANARITY_Q / DEFAULT_AXIS_DIST_Q 是在
.legacy/kmeans/diag/select_dev_frames.DEV_FRAMES 6帧上跑网格搜索
(见 scan_thresholds())，结合 eda_outputs/ 下的多视角散点图目测选定的一版，
不代表理论最优。

TODO: 无监督聚类(KMeans/GMM k=2)作为对照方法，本阶段不实现，
待规则阈值法验证后再评估是否需要

Archived under .legacy/ -- superseded by postprocessing/labeling/motion/ as the T3 default,
kept for reference only (still imported by postprocessing/kinematics/simulate_gt/segment.py
for comparison). See the repo README for the current status.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LEGACY_ROOT))

# 网格搜索 + 目测eda_outputs/图后选定的默认分位数，见模块docstring。
DEFAULT_PLANARITY_Q = 0.82
DEFAULT_AXIS_DIST_Q = 0.82


def thresholds_from_quantiles(df: pd.DataFrame, planarity_q: float = DEFAULT_PLANARITY_Q,
                               axis_dist_q: float = DEFAULT_AXIS_DIST_Q) -> tuple[float, float]:
    """把分位数参数换算成该帧的具体阈值。df 应该只含 if_keep=True 的点(调用方负责过滤，
    本函数不做if_keep相关的过滤/传播逻辑，那是S2/S3的事)。"""
    planarity_th = float(np.quantile(df["planarity"], planarity_q))
    axis_dist_th = float(np.quantile(df["dist_to_principal_axis"], axis_dist_q))
    return planarity_th, axis_dist_th


def classify_body_wing(df: pd.DataFrame, planarity_th: float, axis_dist_th: float) -> pd.Series:
    """规则阈值判定: dist_to_principal_axis 或 planarity 任一超过各自阈值即判 wing。
    返回与 df 同索引的 bool Series，列名建议记为 is_wing(不是最终 part_label)。"""
    is_wing = (df["dist_to_principal_axis"] > axis_dist_th) | (df["planarity"] > planarity_th)
    return is_wing.rename("is_wing")


# TODO: 无监督聚类(KMeans/GMM k=2)作为对照方法，本阶段不实现，
# 待规则阈值法验证后再评估是否需要
def classify_body_wing_quantile(df: pd.DataFrame, planarity_q: float = DEFAULT_PLANARITY_Q,
                                 axis_dist_q: float = DEFAULT_AXIS_DIST_Q) -> pd.Series:
    """thresholds_from_quantiles + classify_body_wing 的便捷组合，帧间用同一组分位数
    但各帧换算出各自的具体阈值。"""
    planarity_th, axis_dist_th = thresholds_from_quantiles(df, planarity_q, axis_dist_q)
    return classify_body_wing(df, planarity_th, axis_dist_th)


def print_color_diagnostics(df: pd.DataFrame, is_wing: pd.Series) -> None:
    """颜色特征(R/G/B/opacity)不进主判据，只在这里打印wing/body两组均值差异供参考诊断。"""
    cols = ["R", "G", "B", "opacity"]
    wing_mean = df.loc[is_wing, cols].mean()
    body_mean = df.loc[~is_wing, cols].mean()
    print(f"    [颜色诊断，不参与决策] wing均值: "
          f"R={wing_mean['R']:.3f} G={wing_mean['G']:.3f} B={wing_mean['B']:.3f} "
          f"opacity={wing_mean['opacity']:.3f}")
    print(f"    [颜色诊断，不参与决策] body均值: "
          f"R={body_mean['R']:.3f} G={body_mean['G']:.3f} B={body_mean['B']:.3f} "
          f"opacity={body_mean['opacity']:.3f}")


# ---------------------------------------------------------------------------
# 默认分位数网格搜索 (仅用于人工选定 DEFAULT_PLANARITY_Q / DEFAULT_AXIS_DIST_Q，
# 不是正式pipeline的一部分)
# ---------------------------------------------------------------------------

def scan_thresholds(q_candidates: list[float] = (0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95)) -> None:
    from kmeans.diag.select_dev_frames import DEV_FRAMES, DATASET_DIR
    from postprocessing.cleaning.viz_floater_check import find_features_csv

    data_root = DATASET_DIR
    frames_kept = {}
    for f in DEV_FRAMES:
        csv_path = find_features_csv(f, data_root).with_name(
            find_features_csv(f, data_root).stem + "_marked.csv"
        )
        df = pd.read_csv(csv_path)
        frames_kept[f] = df[df["if_keep"]].reset_index(drop=True)

    print(f"{'planarity_q':>12} {'axis_dist_q':>12}  " + "  ".join(f"{f:>8}" for f in DEV_FRAMES))
    for pq in q_candidates:
        for aq in q_candidates:
            ratios = []
            for f in DEV_FRAMES:
                df = frames_kept[f]
                is_wing = classify_body_wing_quantile(df, pq, aq)
                ratios.append(100 * is_wing.mean())
            print(f"{pq:>12.2f} {aq:>12.2f}  " + "  ".join(f"{r:>7.1f}%" for r in ratios))


if __name__ == "__main__":
    scan_thresholds()
