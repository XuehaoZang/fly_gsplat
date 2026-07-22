"""
验证 T3 body/wing v2聚类(种子初始化 + FEATURES_V2=[x,y,z,opacity,R]) 在
aux_weight=1x(w1)配置下，是否真的像之前6帧目测印象那样全面优于v1(旧版本，无种子
初始化、FEATURES六维随机初始化)。不再只凭读图判断，复用kmeans_split.py里已经写好
的两套量化诊断逻辑，对全部DEV_FRAMES跑一遍：

1. 稳定性检验: 复用kmeans_split.py最初版本(无种子初始化)的ARI稳定性逻辑
   (stability_check())，换成w1配置——同一帧固定aux_weight=1，seed质心本身不随
   random_state变，但build_seed_init()里另外两个初始质心用kmeans_plusplus挑选，
   这里跑STABILITY_SEEDS(5个)不同random_state重复run_kmeans_v2，两两算ARI，检验
   种子之外的初始化随机性是否会带来不同的最终划分。

2. 簇间/簇内kNN距离对比: 复用kmeans_split.py/eda_body_wing_features.py里的
   cross_vs_intra_table()，分别对w1(种子初始化)和v1(旧6维随机初始化)的最终聚类
   结果(都用MAIN_RANDOM_STATE)算这张表，统计"疑似同一结构被硬切"(note里标出的
   ratio<1.5)的簇对数量，两版本直接对比。

3. 汇总表: 每帧 [ARI均值, ARI最小值, 硬切对数(w1), 硬切对数(v1)]。

不改动kmeans_split.py里种子初始化/聚类逻辑本身，只复用现成函数加验证和打印。若
某帧的数字结果和"w1全帧最优"的目测印象明显矛盾(w1硬切对数不小于v1，或稳定性
min_ari偏低)，额外出一张图核实，并在打印里注明。

用法:
    python -m postprocessing.labeling.diag.verify_w1_config
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.labeling.kmeans_split import (  # noqa: E402
    OUT_DIR, STABILITY_SEEDS, MAIN_RANDOM_STATE, cross_vs_intra_table, load_kept,
    plot_kmeans_clusters, run_kmeans, run_kmeans_v2, seed_mask, standardize,
    standardize_v2,
)
from postprocessing.labeling.diag.select_dev_frames import DEV_FRAMES  # noqa: E402

AUX_WEIGHT_W1 = 1
FEATURES_V2_LABEL = ["x", "y", "z", "opacity", "R"]
STABILITY_MIN_ARI_OK = 0.9   # 同kmeans_split.py里的"稳定/不稳定"判据，仅用于打印提示
# 触发"额外出图核实"的判据比STABILITY_MIN_ARI_OK更严格：hard_cut计数持平不算矛盾
# (阈值化计数本身粗糙，同计数下ratio可能已有明显改善)，只有w1严格更差，或
# 稳定性差到接近"随机"(ARI<0.5)才算真的和"w1全帧最优"的目测印象矛盾。
CONTRADICTION_ARI_THRESH = 0.5


def count_hard_cut_pairs(dist_df: pd.DataFrame) -> int:
    """疑似"同一结构被硬切"的簇对数：直接数cross_vs_intra_table()里note列已经标出
    的"接近(疑似同一结构被硬切)"，不重复定义ratio阈值。"""
    return int(dist_df["note"].str.contains("疑似同一结构被硬切").sum())


def stability_check_w1(df: pd.DataFrame, seeds: np.ndarray,
                        seeds_list: list[int] = STABILITY_SEEDS) -> dict:
    """复用kmeans_split.stability_check()同款ARI稳定性逻辑，换成w1(种子初始化)配置。"""
    X = standardize_v2(df, AUX_WEIGHT_W1)
    runs = {s: run_kmeans_v2(X, seeds, s) for s in seeds_list}
    pairwise = [(s1, s2, adjusted_rand_score(runs[s1], runs[s2]))
                for s1, s2 in combinations(seeds_list, 2)]
    aris = [a for _, _, a in pairwise]
    return {"pairwise": pairwise, "mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris))}


def verify_frame(frame: str) -> dict:
    df = load_kept(frame)
    seeds = seed_mask(df)

    # v1: 旧版本，无种子初始化，FEATURES六维随机初始化
    labels_v1 = run_kmeans(standardize(df), MAIN_RANDOM_STATE)
    dist_v1 = cross_vs_intra_table(df, labels_v1)
    hard_cut_v1 = count_hard_cut_pairs(dist_v1)

    # w1: 种子初始化 + FEATURES_V2=[x,y,z,opacity,R], aux_weight=1
    labels_w1 = run_kmeans_v2(standardize_v2(df, AUX_WEIGHT_W1), seeds, MAIN_RANDOM_STATE)
    dist_w1 = cross_vs_intra_table(df, labels_w1)
    hard_cut_w1 = count_hard_cut_pairs(dist_w1)

    stability = stability_check_w1(df, seeds)

    contradicts_visual = (hard_cut_w1 > hard_cut_v1) or (stability["min_ari"] < CONTRADICTION_ARI_THRESH)

    print(f"\n[{frame}] n_kept={len(df)}")
    print(f"  稳定性(w1, 5个random_state两两ARI): mean={stability['mean_ari']:.3f} "
          f"min={stability['min_ari']:.3f}  "
          f"{'稳定' if stability['min_ari'] > STABILITY_MIN_ARI_OK else '[警告]不同初始化的聚类结果有明显差异'}")
    print("  簇间/簇内kNN距离 -- w1(种子初始化):")
    for _, row in dist_w1.iterrows():
        print(f"    cluster{row['cluster_pair']}: ratio={row['ratio_cross_over_min_intra']:.2f}  {row['note']}")
    print("  簇间/簇内kNN距离 -- v1(旧6维随机初始化):")
    for _, row in dist_v1.iterrows():
        print(f"    cluster{row['cluster_pair']}: ratio={row['ratio_cross_over_min_intra']:.2f}  {row['note']}")
    print(f"  疑似硬切簇对数: w1={hard_cut_w1}  v1={hard_cut_v1}  " + (
        "w1更少(更优)" if hard_cut_w1 < hard_cut_v1 else
        ("持平" if hard_cut_w1 == hard_cut_v1 else "w1更多(反而更差!)")))

    if contradicts_visual:
        print(f"  [注意] 本帧数字结果和\"w1全帧最优\"的目测印象矛盾"
              f"(hard_cut_w1={hard_cut_w1} vs hard_cut_v1={hard_cut_v1}, "
              f"stability_min_ari={stability['min_ari']:.3f})，额外出图核实。")
        extra_path = OUT_DIR / f"verify_w1_recheck_{frame}.png"
        plot_kmeans_clusters(
            df, labels_w1, frame, extra_path, features_label=FEATURES_V2_LABEL,
            extra_info="[核实用] w1(aux_weight=1) 数字诊断与目测印象矛盾，详见终端打印")
        print(f"    核实图 -> {extra_path}")

    return {
        "frame": frame, "ari_mean": stability["mean_ari"], "ari_min": stability["min_ari"],
        "hard_cut_w1": hard_cut_w1, "hard_cut_v1": hard_cut_v1,
        "contradicts_visual": contradicts_visual,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [verify_frame(f) for f in DEV_FRAMES]

    summary = pd.DataFrame(results)[["frame", "ari_mean", "ari_min", "hard_cut_w1", "hard_cut_v1"]]

    print(f"\n{'=' * 70}\n汇总: w1(种子初始化+精简特征) vs v1(旧6维随机初始化)，{len(DEV_FRAMES)}帧\n{'=' * 70}")
    print(summary.to_string(index=False))

    n_w1_better = int((summary["hard_cut_w1"] < summary["hard_cut_v1"]).sum())
    n_w1_worse = int((summary["hard_cut_w1"] > summary["hard_cut_v1"]).sum())
    n_tie = len(summary) - n_w1_better - n_w1_worse
    print(f"\nw1硬切对数 < v1 的帧数: {n_w1_better}/{len(summary)}  "
          f"持平: {n_tie}/{len(summary)}  w1反而更多的帧数: {n_w1_worse}/{len(summary)}")
    print(f"稳定性(min ARI)均值: {summary['ari_min'].mean():.3f}  "
          f"({int((summary['ari_min'] > STABILITY_MIN_ARI_OK).sum())}/{len(summary)}帧 min_ari>{STABILITY_MIN_ARI_OK})")

    if len(summary) > 12:
        csv_path = OUT_DIR / "verify_w1_summary.csv"
        summary.to_csv(csv_path, index=False)
        print(f"\n表格较长，已存 -> {csv_path}")


if __name__ == "__main__":
    main()
