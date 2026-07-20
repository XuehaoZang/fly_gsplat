"""
冒烟测试：去噪前后主轴角度敏感度测试。
用f0100完整点云(1162点) vs 去掉HDBSCAN噪声点(-1)和纯伪影簇(color_oob≈100%)后的
点云(仅保留cluster 0)，分别算第一主轴(PCA)，比较两次主轴方向的夹角。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLAT_DIR = Path("outputs/ctrl_009_002/f0100/splatfacto-checkpoint/2026-07-04_022821")
CLUSTERED_CSV = SPLAT_DIR / "gaussian_features_f0100_clustered_v2.csv"


def principal_axis(xyz: np.ndarray) -> np.ndarray:
    centroid = xyz.mean(axis=0)
    cov = np.cov((xyz - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axis = eigvecs[:, order[0]]
    return axis, eigvals[order]


def main():
    df = pd.read_csv(CLUSTERED_CSV)
    n_total = len(df)

    print("=" * 70)
    print("HDBSCAN cluster构成 (来自上一轮结果)")
    print("=" * 70)
    counts = df["cluster_hdbscan"].value_counts().sort_index()
    oob_rate = df.groupby("cluster_hdbscan")["color_oob"].mean()
    for cid, cnt in counts.items():
        tag = " <- 去除(噪声)" if cid == -1 else ""
        if cid != -1 and oob_rate[cid] > 0.95:
            tag = " <- 去除(纯伪影簇, color_oob≈100%)"
        print(f"  cluster {cid}: n={cnt} ({100*cnt/n_total:.1f}%)  color_oob率={oob_rate[cid]:.4f}{tag}")

    drop_clusters = [-1] + [cid for cid in counts.index if cid != -1 and oob_rate[cid] > 0.95]
    df_clean = df[~df["cluster_hdbscan"].isin(drop_clusters)]
    print(f"\n去除cluster {drop_clusters} 后剩余点数 = {len(df_clean)}/{n_total} ({100*len(df_clean)/n_total:.1f}%)")

    xyz_full = df[["x", "y", "z"]].values
    xyz_clean = df_clean[["x", "y", "z"]].values

    axis_full, eig_full = principal_axis(xyz_full)
    axis_clean, eig_clean = principal_axis(xyz_clean)

    # 保证方向一致(特征向量符号任意，取和原方向点积为正的那个)
    if np.dot(axis_full, axis_clean) < 0:
        axis_clean = -axis_clean

    cos_angle = np.clip(np.dot(axis_full, axis_clean), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))

    print()
    print("=" * 70)
    print("主轴方向对比")
    print("=" * 70)
    print(f"完整点云(n={n_total}): 主轴 = {axis_full}")
    print(f"  特征值(降序, 反映各方向延展): {eig_full}")
    print(f"去噪后点云(n={len(df_clean)}): 主轴 = {axis_clean}")
    print(f"  特征值(降序): {eig_clean}")
    print()
    print(f"两次主轴夹角 = {angle_deg:.4f} 度  (cos={cos_angle:.8f})")

    # 用第二主轴也算一下，交叉核验(第一主轴在体型接近对称/延展比不悬殊时可能不稳定)
    def second_axis(xyz):
        centroid = xyz.mean(axis=0)
        cov = np.cov((xyz - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        return eigvecs[:, order[1]], eigvals[order]

    ax2_full, _ = second_axis(xyz_full)
    ax2_clean, _ = second_axis(xyz_clean)
    if np.dot(ax2_full, ax2_clean) < 0:
        ax2_clean = -ax2_clean
    angle2_deg = np.degrees(np.arccos(np.clip(np.dot(ax2_full, ax2_clean), -1, 1)))
    print(f"第二主轴夹角(交叉核验) = {angle2_deg:.4f} 度")

    print()
    print("=" * 70)
    print("延展比(特征值比, 反映主轴稳定性)")
    print("=" * 70)
    print(f"完整点云: lambda1/lambda2 = {eig_full[0]/eig_full[1]:.4f}, lambda1/lambda3 = {eig_full[0]/eig_full[2]:.4f}")
    print(f"去噪点云: lambda1/lambda2 = {eig_clean[0]/eig_clean[1]:.4f}, lambda1/lambda3 = {eig_clean[0]/eig_clean[2]:.4f}")


if __name__ == "__main__":
    main()
