"""
冒烟测试：patch_size(连通分量大小) + patch_extent(邻域PCA前两轴延展)
两个连通性特征，复用k=10近邻图，只看判别力(不聚类)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLAT_DIR = Path("outputs/ctrl_009_002/f0100/splatfacto-checkpoint/2026-07-04_022821")
FEATURES_CSV = SPLAT_DIR / "gaussian_features_f0100.csv"

K = 10
TOP5_IDX = [694, 1017, 435, 974, 1073]
MID_SPECIAL_IDX = 172


def main():
    df = pd.read_csv(FEATURES_CSV)
    xyz = df[["x", "y", "z"]].values
    n = len(df)

    tree = cKDTree(xyz)
    dists, idxs = tree.query(xyz, k=K + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]  # 去掉自身

    # ---- patch_size: 用邻域距离p75做阈值断开较远连接，连通分量分析 ----
    all_nbr_dists = dists.flatten()
    p75 = np.percentile(all_nbr_dists, 75)
    print(f"邻域距离(k={K})p75阈值 = {p75:.6f} m")

    rows, cols = [], []
    for i in range(n):
        for j_idx, d in zip(idxs[i], dists[i]):
            if d <= p75:
                rows.append(i)
                cols.append(j_idx)
    data = np.ones(len(rows))
    adj = coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)  # 对称化(无向图)
    n_components, labels = connected_components(adj, directed=False)
    comp_sizes = pd.Series(labels).value_counts()
    patch_size = comp_sizes[labels].values
    df["patch_size"] = patch_size
    print(f"连通分量数 = {n_components}, 最大分量占比 = {comp_sizes.max()}/{n} ({100*comp_sizes.max()/n:.1f}%)")
    print(f"分量大小分布: {comp_sizes.sort_values(ascending=False).values[:10]} ...(前10大)")

    # ---- patch_extent: 邻域点在PCA前两主轴方向的延展(取极差 range) ----
    patch_extent = np.zeros(n)
    for i in range(n):
        nbr_pts = xyz[idxs[i]]
        nbr_pts = np.vstack([xyz[i], nbr_pts])  # 包含自身
        centered = nbr_pts - nbr_pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        top2 = eigvecs[:, order[:2]]
        proj = centered @ top2  # (k+1, 2)
        extent = proj.max(axis=0) - proj.min(axis=0)  # 每根轴的极差
        patch_extent[i] = np.linalg.norm(extent)  # 合成成一个标量(两轴延展的合向量长度)
    df["patch_extent"] = patch_extent

    print()
    print("=" * 70)
    print("patch_size / patch_extent 分布")
    print("=" * 70)
    for col in ["patch_size", "patch_extent"]:
        arr = df[col].values
        print(f"{col:15s} min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g} "
              f"median={np.median(arr):.6g} p90={np.percentile(arr,90):.6g}")

    print()
    print("=" * 70)
    print("三组对比: top5尖刺 vs 点172 vs 中位数对照组(5点)")
    print("=" * 70)
    median_ratio = df["scale_ratio"].median()
    df["_d"] = (df["scale_ratio"] - median_ratio).abs()
    mid5_idx = df.nsmallest(5, "_d").index.tolist()
    df.drop(columns=["_d"], inplace=True)

    cols_show = ["scale_ratio", "linearity", "patch_size", "patch_extent"]
    print("top5尖刺点:")
    print(df.loc[TOP5_IDX, cols_show].to_string())
    print(f"  均值: patch_size={df.loc[TOP5_IDX,'patch_size'].mean():.2f}  patch_extent={df.loc[TOP5_IDX,'patch_extent'].mean():.6g}")

    print()
    print(f"点172(疑似真实翼面):")
    print(df.loc[[MID_SPECIAL_IDX], cols_show].to_string())

    print()
    print(f"中位数对照组(idx={mid5_idx}):")
    print(df.loc[mid5_idx, cols_show].to_string())
    print(f"  均值: patch_size={df.loc[mid5_idx,'patch_size'].mean():.2f}  patch_extent={df.loc[mid5_idx,'patch_extent'].mean():.6g}")

    print()
    print("=" * 70)
    print("patch_size/patch_extent 与 linearity 的相关系数(判断是否独立信息)")
    print("=" * 70)
    for col in ["patch_size", "patch_extent"]:
        pear_r, pear_p = pearsonr(df[col], df["linearity"])
        spear_r, spear_p = spearmanr(df[col], df["linearity"])
        print(f"{col} vs linearity: Pearson r={pear_r:.4f} (p={pear_p:.2e})  Spearman rho={spear_r:.4f} (p={spear_p:.2e})")

    pear_r, pear_p = pearsonr(df["patch_size"], df["patch_extent"])
    print(f"patch_size vs patch_extent: Pearson r={pear_r:.4f} (p={pear_p:.2e})")

    for col in ["patch_size", "patch_extent"]:
        pear_r, pear_p = pearsonr(df[col], df["local_density"])
        print(f"{col} vs local_density: Pearson r={pear_r:.4f} (p={pear_p:.2e})")

    out_csv = SPLAT_DIR / "gaussian_features_f0100_patch.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n已保存: {out_csv} (行数={len(df)})")


if __name__ == "__main__":
    main()
