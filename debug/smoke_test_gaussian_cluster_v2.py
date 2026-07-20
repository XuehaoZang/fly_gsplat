"""
冒烟测试v2：修复RGB三通道重复计权bug + 探索更合适的cluster粒度。

任务A: R,G,B在这批数据里几乎共线(corr>0.99999, PCA主成分解释99.9999%方差,
因为是灰度相机)，直接塞3个高度相关的通道进欧氏距离等于把"亮度"权重放大3倍。
用 brightness = mean(R,G,B) 替代，比PCA第一主成分更简单且几乎等价，
color_oob不进聚类特征，只做诊断字段保留。

任务B: 对KMeans k=3..6跑silhouette score，同时跑一版HDBSCAN
(min_cluster_size=15, 对应"占比>=1.3%才算一个真实簇"的经验值，
不用太大以免把小的真实结构如"翼脉/关节"也吞并成噪声)，
用上一轮同样的验证锚点(top5尖刺点 + 点172)交叉检查。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLAT_DIR = Path("outputs/ctrl_009_002/f0100/splatfacto-checkpoint/2026-07-04_022821")
FEATURES_CSV = SPLAT_DIR / "gaussian_features_f0100.csv"

RAW_COLS = ["x", "y", "z", "brightness", "opacity", "linearity", "planarity", "sphericity"]
LOG_COLS = ["scale_phys_0", "scale_phys_1", "scale_phys_2", "local_density"]
CLUSTER_COLS = RAW_COLS + LOG_COLS

TOP5_IDX = [694, 1017, 435, 974, 1073]
MID_SPECIAL_IDX = 172

STAT_COLS = ["linearity", "planarity", "sphericity", "opacity",
             "scale_phys_0", "scale_phys_1", "scale_phys_2", "brightness", "local_density"]


def cluster_report(df: pd.DataFrame, label_col: str, title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    n = len(df)
    counts = df[label_col].value_counts().sort_index()
    for cid, cnt in counts.items():
        tag = " (噪声)" if cid == -1 else ""
        print(f"  cluster {cid}{tag}: n={cnt}  ({100*cnt/n:.1f}%)")

    mean_table = df.groupby(label_col)[STAT_COLS].mean()
    with pd.option_context("display.float_format", lambda x: f"{x:.6g}", "display.width", 220, "display.max_columns", 30):
        print()
        print("均值 (行=cluster, 列=字段):")
        print(mean_table)

    print()
    top5 = df.loc[TOP5_IDX, ["scale_ratio", "linearity", label_col]]
    print("top5尖刺点归属:")
    print(top5.to_string())
    print(f"  分布: {df.loc[TOP5_IDX, label_col].value_counts().to_dict()}")

    print()
    row172 = df.loc[[MID_SPECIAL_IDX], ["scale_ratio", "linearity", "planarity", label_col]]
    print("点172(疑似真实扁平翼面)归属:")
    print(row172.to_string())
    lbl172 = df.loc[MID_SPECIAL_IDX, label_col]
    is_noise = (lbl172 == -1)
    print(f"  是否被标为噪声: {is_noise}")


def main():
    df = pd.read_csv(FEATURES_CSV)
    n = len(df)
    df["brightness"] = df[["R", "G", "B"]].mean(axis=1)

    X_raw = df[RAW_COLS].values
    X_log = np.log(df[LOG_COLS].values)
    X = np.concatenate([X_raw, X_log], axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    print("=" * 70)
    print("B1. KMeans k=3..6 silhouette score")
    print("=" * 70)
    kmeans_labels = {}
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        kmeans_labels[k] = labels
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        print(f"  k={k}: silhouette={sil:.4f}  cluster_sizes={sizes}")

    # 把k=3..6的label都存下来，用于后面挑一个做深入报告
    for k in [3, 4, 5, 6]:
        df[f"cluster_kmeans_k{k}"] = kmeans_labels[k]

    # ---- B2. HDBSCAN ----
    hdb = HDBSCAN(min_cluster_size=15)
    hdb_labels = hdb.fit_predict(X_scaled)
    df["cluster_hdbscan"] = hdb_labels
    n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise = int((hdb_labels == -1).sum())
    print()
    print("=" * 70)
    print(f"B2. HDBSCAN (min_cluster_size=15): {n_clusters_hdb}个cluster, 噪声点 n={n_noise} ({100*n_noise/n:.1f}%)")
    print("=" * 70)

    # ---- B3/B4: 详细报告，挑一个silhouette较高且可解释的k，以及全部k供你对比 ----
    for k in [3, 4, 5, 6]:
        cluster_report(df, f"cluster_kmeans_k{k}", f"KMeans k={k} 详细报告")

    cluster_report(df, "cluster_hdbscan", "HDBSCAN 详细报告")

    out_csv = SPLAT_DIR / "gaussian_features_f0100_clustered_v2.csv"
    df.to_csv(out_csv, index=False)
    print()
    print(f"完整特征表(含所有k的KMeans标签+HDBSCAN标签)已保存: {out_csv} (行数={len(df)}, 未做任何过滤)")


if __name__ == "__main__":
    main()
