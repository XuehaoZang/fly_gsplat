"""
冒烟测试：在 gaussian_features_f0100.csv 基础上跑一版最基础的无监督聚类
(KMeans, k=3, 对应body/wing/伪影的粗略假设)，只做打标签，不做任何过滤/删点。

特征预处理:
- scale_phys_0/1/2、local_density 是天然的正值/跨数量级字段(lognormal形状)，
  先取log再标准化，避免长尾极端值主导距离度量。
- x,y,z / R,G,B / opacity / linearity,planarity,sphericity 量纲相对温和，
  直接标准化(StandardScaler, 零均值单位方差)。
- 用StandardScaler而不是MinMaxScaler: KMeans基于欧氏距离，MinMax对个别
  极端值(min/max)很敏感，一个离群点会把其他点全部压缩到很窄的区间；
  StandardScaler用均值/标准差，对长尾更鲁棒一些(尽管log变换已经处理了主要的偏度问题)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLAT_DIR = Path("outputs/ctrl_009_002/f0100/splatfacto-checkpoint/2026-07-04_022821")
FEATURES_CSV = SPLAT_DIR / "gaussian_features_f0100.csv"

RAW_COLS = ["x", "y", "z", "R", "G", "B", "opacity", "linearity", "planarity", "sphericity"]
LOG_COLS = ["scale_phys_0", "scale_phys_1", "scale_phys_2", "local_density"]
CLUSTER_COLS = RAW_COLS + LOG_COLS

TOP5_IDX = [694, 1017, 435, 974, 1073]  # 上轮 scale_ratio 最高5点
MID_SPECIAL_IDX = 172  # 上轮对照组里疑似真实扁平翼面点


def main():
    df = pd.read_csv(FEATURES_CSV)
    n = len(df)

    X_raw = df[RAW_COLS].values
    X_log = np.log(df[LOG_COLS].values)  # 全部是严格正值，可以直接log
    X = np.concatenate([X_raw, X_log], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X_scaled)
    df["cluster_id"] = labels

    print("=" * 70)
    print("聚类特征列:", CLUSTER_COLS)
    print(f"n_points = {n}, k = 3, random_state=0, n_init=10")
    print("=" * 70)

    print()
    print("2a. 每个cluster点数/占比")
    print("=" * 70)
    counts = df["cluster_id"].value_counts().sort_index()
    for cid, cnt in counts.items():
        print(f"  cluster {cid}: n={cnt}  ({100*cnt/n:.1f}%)")

    print()
    print("2b. 每个cluster关键字段均值/中位数")
    print("=" * 70)
    stat_cols = ["x", "y", "z", "linearity", "planarity", "sphericity",
                 "opacity", "scale_phys_0", "scale_phys_1", "scale_phys_2", "local_density"]
    mean_table = df.groupby("cluster_id")[stat_cols].mean()
    median_table = df.groupby("cluster_id")[stat_cols].median()
    print("均值:")
    with pd.option_context("display.float_format", lambda x: f"{x:.6g}", "display.width", 220, "display.max_columns", 30):
        print(mean_table.T)
    print()
    print("中位数:")
    with pd.option_context("display.float_format", lambda x: f"{x:.6g}", "display.width", 220, "display.max_columns", 30):
        print(median_table.T)

    print()
    print("3c. scale_ratio最高5点的cluster归属")
    print("=" * 70)
    top5 = df.loc[TOP5_IDX, ["scale_ratio", "linearity", "cluster_id"]]
    print(top5.to_string())
    print(f"这5点cluster分布: {df.loc[TOP5_IDX, 'cluster_id'].value_counts().to_dict()}")

    print()
    print("3d. 对照点172(疑似真实扁平翼面点)的cluster归属")
    print("=" * 70)
    print(df.loc[[MID_SPECIAL_IDX], ["scale_ratio", "linearity", "planarity", "cluster_id"]].to_string())
    same_cluster = df.loc[MID_SPECIAL_IDX, "cluster_id"]
    n_in_same_cluster = (df["cluster_id"] == same_cluster).sum()
    print(f"点172所在cluster={same_cluster}, 该cluster总点数={n_in_same_cluster} ({100*n_in_same_cluster/n:.1f}%)")

    out_csv = SPLAT_DIR / "gaussian_features_f0100_clustered.csv"
    df.to_csv(out_csv, index=False)
    print()
    print(f"带cluster_id的完整特征表已保存: {out_csv}  (行数={len(df)}, 未做任何过滤)")


if __name__ == "__main__":
    main()
