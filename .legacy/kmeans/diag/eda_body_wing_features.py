"""
T3 body/wing 二分类特征诊断：kmeans_split.py(6维 [x,y,z,planarity,scale_ratio,opacity]
KMeans k=3) 在 DEV_FRAMES 上结果不稳定(部分帧疑似整条翼被并入body簇，或一条翼被切成
两段)，先在特征层面做诊断，再决定怎么调整。

参考 postprocessing/cleaning/eda_features.py 的代码结构和输出风格(basic_stats/
df_to_md/直方图画法/汇总md)，但这是全新的、针对body/wing判别的诊断脚本，跑的是
G2b_G9 的 DEV_FRAMES(select_dev_frames.py)，不复用/修改它的floater分组逻辑，也不
改动 kmeans_split.py / binary/binary_split.py。

纯诊断脚本：不产出任何标签列(if_keep/cluster_id/body_wing等)，只读取
kmeans_split.load_kept() 已有的 if_keep=True 点集。

在 DEV_FRAMES 每一帧上做:
1. 候选特征([planarity, linearity, sphericity, scale_ratio, opacity, R, G, B])
   两两皮尔逊相关系数矩阵(csv+热力图)，标出 |r|>0.8 的特征对。
2. opacity/R/G/B/planarity/scale_ratio 的原始分布直方图(不分组)，附偏度(skewness)
   和 Sarle's bimodality coefficient，辅助判断哪些特征看起来双峰可分。
3. 特征消融: 复用 kmeans_split 的标准化+KMeans(k=3, 同一 random_state)流程，
   额外跑 xyz-only 和 shape-only([planarity,scale_ratio,opacity]) 两组对照，
   跟原有6维结果两两算 Adjusted Rand Index，看6维结果的簇分配主要像谁。
4. 簇间/簇内 kNN 距离对比: 对当前6维KMeans结果，任意两簇之间的
   cross-cluster nearest-neighbor distance(单链接，最近点对距离) vs 各自簇内部
   典型kNN距离(复用 utils/ply.py 里 tree.query(xyz, k=k+1) 再去掉自身列这套
   kNN距离计算逻辑)，两者接近说明可能是同一结构被硬切，明显更大说明是真正分开
   的结构。

输出落在 .legacy/kmeans/diag/eda_outputs/(文件名前缀 bw_ 以区分
kmeans_split.py 自己的 kmeans_split_*.png，那份存在 .legacy/
kmeans/eda_outputs/)，汇总成 .legacy/kmeans/diag/eda_outputs/Label_EDA.md。

用法(archived under .legacy/, see .legacy/kmeans/kmeans_split.py):
    python .legacy/kmeans/diag/eda_body_wing_features.py
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LEGACY_ROOT))

from kmeans.kmeans_split import (  # noqa: E402
    FEATURES, MAIN_RANDOM_STATE, K, load_kept, run_kmeans, standardize,
)
from kmeans.diag.select_dev_frames import DEV_FRAMES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"

CORR_COLS = ["planarity", "linearity", "sphericity", "scale_ratio", "opacity", "R", "G", "B"]
RAW_DIST_COLS = ["opacity", "R", "G", "B", "planarity", "scale_ratio"]
STRONG_CORR_THRESH = 0.8
BIMODALITY_THRESH = 5.0 / 9.0  # Sarle's 临界值(均匀分布的BC)，超过视为疑似双峰

ABLATION_SETS = {
    "full6": FEATURES,                                   # kmeans_split.py 当前实际使用的6维
    "xyz_only": ["x", "y", "z"],
    "shape_only": ["planarity", "scale_ratio", "opacity"],
}

K_NN_INTRA = 5  # 簇内"典型"kNN距离用的近邻数(见 typical_intra_knn_dist)


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return float_fmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------- 1. 候选特征相关系数矩阵 ----------

def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[CORR_COLS].corr(method="pearson")


def strong_pairs(corr: pd.DataFrame, thresh: float = STRONG_CORR_THRESH) -> list[tuple[str, str, float]]:
    pairs = []
    for a, b in combinations(CORR_COLS, 2):
        r = float(corr.loc[a, b])
        if abs(r) > thresh:
            pairs.append((a, b, r))
    return sorted(pairs, key=lambda t: -abs(t[2]))


def corr_heatmap(corr: pd.DataFrame, frame: str, path: Path) -> None:
    n = len(CORR_COLS)
    fig, ax = plt.subplots(figsize=(1.1 * n + 1, 1.1 * n))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(CORR_COLS, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(CORR_COLS)
    for i in range(n):
        for j in range(n):
            r = corr.iloc[i, j]
            strong = abs(r) > STRONG_CORR_THRESH and i != j
            ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if strong else "normal",
                    color="black" if abs(r) < 0.6 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(f"{frame}: candidate feature correlation matrix\n(bold = |r|>{STRONG_CORR_THRESH})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 2. 原始特征分布 + 偏度/双峰性 ----------

def bimodality_coefficient(x: np.ndarray) -> tuple[float, float, float]:
    """Sarle's bimodality coefficient (样本修正版)。BC > 5/9 提示疑似双峰/多峰，
    是经验性判据，不是分组依据。"""
    n = len(x)
    g1 = float(stats.skew(x, bias=False))
    g2 = float(stats.kurtosis(x, bias=False, fisher=True))  # excess kurtosis
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    bc = (g1 ** 2 + 1) / (g2 + correction)
    return bc, g1, g2


def raw_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in RAW_DIST_COLS:
        x = df[col].to_numpy(dtype=float)
        bc, g1, g2 = bimodality_coefficient(x)
        rows.append({
            "feature": col, "skew": g1, "excess_kurtosis": g2, "bimodality_coef": bc,
            "looks_bimodal": bc > BIMODALITY_THRESH,
        })
    return pd.DataFrame(rows)


def raw_hist_grid(df: pd.DataFrame, stats_df: pd.DataFrame, frame: str, path: Path) -> None:
    ncols = 3
    nrows = int(np.ceil(len(RAW_DIST_COLS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    stats_by_feat = stats_df.set_index("feature")
    for i, feat in enumerate(RAW_DIST_COLS):
        ax = axes[i]
        ax.hist(df[feat], bins=25, color="#4c72b0", edgecolor="white")
        row = stats_by_feat.loc[feat]
        tag = "looks bimodal" if row["looks_bimodal"] else "unimodal/no clear split"
        ax.set_title(f"{feat}  (skew={row['skew']:.2f}, BC={row['bimodality_coef']:.2f}, {tag})",
                     fontsize=9)
    for j in range(len(RAW_DIST_COLS), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{frame}: raw feature distributions (ungrouped, n={len(df)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 3. 特征消融: full6 vs xyz-only vs shape-only ----------

def standardize_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(df[cols].to_numpy(dtype=float))


def run_ablation(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """复用 kmeans_split 的 standardize+run_kmeans 流程(full6 用它本来的 standardize()，
    ablation 组用同样的单维standardize手法)，三组统一用 MAIN_RANDOM_STATE，隔离特征集
    本身的影响，不引入种子差异。"""
    labels = {}
    for name, cols in ABLATION_SETS.items():
        X = standardize(df) if name == "full6" else standardize_cols(df, cols)
        labels[name] = run_kmeans(X, MAIN_RANDOM_STATE)
    return labels


def ablation_ari_table(labels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(ABLATION_SETS.keys(), 2):
        rows.append({"pair": f"{a} vs {b}", "ari": adjusted_rand_score(labels[a], labels[b])})
    return pd.DataFrame(rows)


def ablation_verdict(ari_df: pd.DataFrame) -> str:
    ari_xyz = float(ari_df.loc[ari_df["pair"] == "full6 vs xyz_only", "ari"].iloc[0])
    ari_shape = float(ari_df.loc[ari_df["pair"] == "full6 vs shape_only", "ari"].iloc[0])
    if abs(ari_xyz - ari_shape) < 0.1:
        return f"xyz(ARI={ari_xyz:.3f})和形状特征(ARI={ari_shape:.3f})跟full6都接近，两者共同起作用，无明显单一主导"
    winner = "xyz" if ari_xyz > ari_shape else "形状特征(planarity/scale_ratio/opacity)"
    return f"full6的簇分配更像{winner}主导 (ARI vs xyz={ari_xyz:.3f}, ARI vs shape={ari_shape:.3f})"


# ---------- 4. 簇间/簇内 kNN 距离对比 ----------

def typical_intra_knn_dist(xyz_cluster: np.ndarray, k: int = K_NN_INTRA) -> float:
    """簇内部'典型'局部近邻距离：复用 utils/ply.py connected_component_sizes/
    local_pca_extent 里同一套 tree.query(xyz, k=k+1) 再丢弃自身列的kNN距离计算逻辑，
    取全簇kNN距离的中位数。"""
    n = len(xyz_cluster)
    kk = min(k, n - 1)
    if kk < 1:
        return float("nan")
    tree = cKDTree(xyz_cluster)
    dists, idxs = tree.query(xyz_cluster, k=kk + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]  # 去掉自身(距离0)，同 utils/ply.py 写法
    return float(np.median(dists))


def cross_cluster_nn_dist(xyz_a: np.ndarray, xyz_b: np.ndarray) -> float:
    """两簇之间的最近点对距离(单链接)：a中每点在b里查最近邻，取全局最小值。"""
    tree = cKDTree(xyz_b)
    dists, _ = tree.query(xyz_a, k=1)
    return float(dists.min())


def cross_vs_intra_table(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    xyz = df[["x", "y", "z"]].to_numpy()
    intra = {c: typical_intra_knn_dist(xyz[labels == c]) for c in range(K)}
    rows = []
    for i, j in combinations(range(K), 2):
        cross = cross_cluster_nn_dist(xyz[labels == i], xyz[labels == j])
        ref = min(intra[i], intra[j])
        ratio = cross / ref if ref > 0 else float("inf")
        note = "接近(疑似同一结构被硬切)" if ratio < 1.5 else "明显更大(像是真正分开的结构)"
        rows.append({
            "cluster_pair": f"{i}-{j}", "cross_nn_dist": cross,
            f"intra_typical_k{K_NN_INTRA}_i": intra[i], f"intra_typical_k{K_NN_INTRA}_j": intra[j],
            "ratio_cross_over_min_intra": ratio, "note": note,
        })
    return pd.DataFrame(rows)


# ---------- main ----------

def process_frame(frame: str, md_parts: list[str]) -> None:
    df = load_kept(frame)
    md_parts.append(f"## {frame} (n_kept={len(df)})\n")

    # 1. correlation matrix
    corr = corr_matrix(df)
    corr.to_csv(OUT_DIR / f"bw_corr_matrix_{frame}.csv")
    heatmap_path = OUT_DIR / f"bw_corr_heatmap_{frame}.png"
    corr_heatmap(corr, frame, heatmap_path)
    pairs = strong_pairs(corr)
    planarity_scale_r = float(corr.loc["planarity", "scale_ratio"])
    md_parts.append("### 1. 候选特征相关系数矩阵\n")
    md_parts.append(f"矩阵: `{heatmap_path.name}` / `bw_corr_matrix_{frame}.csv`\n")
    md_parts.append(f"planarity vs scale_ratio: r={planarity_scale_r:.3f}\n")
    if pairs:
        pair_str = ", ".join(f"{a}~{b}(r={r:.2f})" for a, b, r in pairs)
        md_parts.append(f"|r|>{STRONG_CORR_THRESH} 的特征对: {pair_str}\n")
    else:
        md_parts.append(f"没有 |r|>{STRONG_CORR_THRESH} 的特征对。\n")

    # 2. raw distributions
    stats_df = raw_feature_stats(df)
    stats_df.to_csv(OUT_DIR / f"bw_raw_feature_stats_{frame}.csv", index=False)
    hist_path = OUT_DIR / f"bw_raw_hist_{frame}.png"
    raw_hist_grid(df, stats_df, frame, hist_path)
    md_parts.append("### 2. 原始特征分布(不分组) + 偏度/双峰性\n")
    md_parts.append(f"直方图: `{hist_path.name}`\n")
    md_parts.append(df_to_md(stats_df) + "\n")
    bimodal_feats = stats_df.loc[stats_df["looks_bimodal"], "feature"].tolist()
    md_parts.append(f"疑似双峰(BC>{BIMODALITY_THRESH:.3f})的特征: "
                     f"{bimodal_feats if bimodal_feats else '无'}\n")

    # 3. ablation ARI
    ablation_labels = run_ablation(df)
    ari_df = ablation_ari_table(ablation_labels)
    ari_df.to_csv(OUT_DIR / f"bw_ablation_ari_{frame}.csv", index=False)
    md_parts.append("### 3. 特征消融: full6 / xyz-only / shape-only 两两 ARI\n")
    md_parts.append(df_to_md(ari_df) + "\n")
    md_parts.append(ablation_verdict(ari_df) + "\n")

    # 4. cross vs intra cluster kNN distance
    dist_df = cross_vs_intra_table(df, ablation_labels["full6"])
    dist_df.to_csv(OUT_DIR / f"bw_cross_intra_dist_{frame}.csv", index=False)
    md_parts.append("### 4. 簇间 vs 簇内 kNN 距离(基于当前6维KMeans结果)\n")
    md_parts.append(df_to_md(dist_df) + "\n")

    print(f"\n[{frame}] n_kept={len(df)}")
    print(f"  |r|>{STRONG_CORR_THRESH} 特征对: {pairs}")
    print(f"  planarity vs scale_ratio r={planarity_scale_r:.3f}")
    print(f"  疑似双峰特征: {bimodal_feats}")
    print(f"  {ablation_verdict(ari_df)}")
    print("  簇间/簇内kNN距离:")
    for _, row in dist_df.iterrows():
        print(f"    cluster{row['cluster_pair']}: cross={row['cross_nn_dist']:.5f}  "
              f"intra_i={row[f'intra_typical_k{K_NN_INTRA}_i']:.5f}  "
              f"intra_j={row[f'intra_typical_k{K_NN_INTRA}_j']:.5f}  "
              f"ratio={row['ratio_cross_over_min_intra']:.2f}  {row['note']}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_parts = [
        "# T3 body/wing 特征诊断 (DEV_FRAMES)\n",
        "纯诊断记录，不产出任何标签列，不改动 kmeans_split.py / "
        "binary/binary_split.py。\n",
        f"DEV_FRAMES = {DEV_FRAMES}\n",
    ]
    for frame in DEV_FRAMES:
        process_frame(frame, md_parts)

    summary_path = OUT_DIR / "Label_EDA.md"
    summary_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n[done] summary written to {summary_path}")


if __name__ == "__main__":
    main()
