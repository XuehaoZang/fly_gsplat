"""
纯观察性 EDA：对 T1(utils/gaussian_features.py) 输出的逐点特征表做统计/可视化，
不做任何floater判定，不产出 if_keep 列，不复用/修改 mark_floaters.py。

覆盖 f0090/f0091/f0092 三帧：
1) 逐列基础统计
2) 3D散点图 + 三帧叠加对比
3) k近邻图连通分量分析（分量size列表、直方图、10~17 gap区间检查）
4) 分量size<=10 vs >10 两组的现成特征分布对比
5) (k, dist_percentile) 参数网格下的分量统计敏感性

输出全部落在 postprocessing/cleaning/eda_outputs/，并汇总成一份 markdown 文档。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "postprocessing" / "cleaning" / "eda_outputs"
DATA_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_scale_reg_ratio3"

FRAME_NAMES = ["f0090", "f0091", "f0092"]

STAT_COLS = [
    "x", "y", "z", "R", "G", "B", "opacity",
    "scale_phys_0", "scale_phys_1", "scale_phys_2", "scale_ratio",
    "linearity", "planarity", "sphericity",
    "orientation_x", "orientation_y", "orientation_z", "local_density",
]

CAND_THRESHOLD = 10   # 候选floater阈值 (仅用于本EDA的分组观察，非判定)
STRUCT_MIN = 17        # 已知真实结构分量的经验下限 (来自 mark_floaters.py 记录，仅作参照线)

K_DEFAULT = 10
PCTL_DEFAULT = 75.0

K_GRID = [8, 10, 12, 15]
PCTL_GRID = [70, 75, 80]

FRAME_COLORS = {"f0090": "#1f77b4", "f0091": "#ff7f0e", "f0092": "#2ca02c"}
COLOR_SMALL = "#d62728"
COLOR_LARGE = "#1f77b4"


def find_csv(frame_name: str) -> Path:
    matches = sorted(DATA_ROOT.glob(f"{frame_name}/**/gaussian_features_{frame_name}.csv"))
    if not matches:
        raise FileNotFoundError(f"no gaussian_features csv found for {frame_name}")
    return matches[0]


def load_frames() -> dict:
    dfs = {}
    for name in FRAME_NAMES:
        path = find_csv(name)
        df = pd.read_csv(path)
        dfs[name] = df
        print(f"[load] {name}: {len(df)} points  <- {path}")
    return dfs


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


# ---------- 1. 基础统计 ----------

def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in STAT_COLS:
        s = df[col]
        rows.append({
            "column": col,
            "min": s.min(), "max": s.max(),
            "mean": s.mean(), "median": s.median(), "std": s.std(),
        })
    return pd.DataFrame(rows)


# ---------- 2. 空间分布可视化 ----------

def scatter3d(df: pd.DataFrame, color_col: str, title: str, path: Path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df.x, df.y, df.z, c=df[color_col], cmap="viridis", s=8, alpha=0.85)
    fig.colorbar(sc, ax=ax, shrink=0.6, label=color_col)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter3d_overlay(dfs: dict, path: Path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for name, df in dfs.items():
        ax.scatter(df.x, df.y, df.z, s=6, alpha=0.5, label=f"{name} (n={len(df)})",
                   color=FRAME_COLORS[name])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Overlay: f0090 vs f0091 vs f0092")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 3. k近邻图连通分量 ----------

def knn_component_labels(xyz: np.ndarray, k: int, dist_percentile: float) -> np.ndarray:
    """构建k近邻图(按点对距离分位数截断边)，返回每点所属连通分量的label。
    纯图结构计算，不含任何floater判定逻辑。"""
    n = len(xyz)
    tree = cKDTree(xyz)
    dists, idxs = tree.query(xyz, k=k + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]
    threshold = np.percentile(dists, dist_percentile)
    mask = dists <= threshold
    rows = np.repeat(np.arange(n), k)[mask.ravel()]
    cols = idxs.ravel()[mask.ravel()]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)
    _, labels = connected_components(adj, directed=False)
    return labels


def component_hist(sizes: np.ndarray, frame_name: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(1, sizes.max() + 2) - 0.5
    ax.hist(sizes, bins=bins, color="#4c72b0", edgecolor="white")
    ax.set_yscale("log")
    ax.axvline(CAND_THRESHOLD, color=COLOR_SMALL, linestyle="--",
               label=f"candidate threshold = {CAND_THRESHOLD}")
    ax.axvline(STRUCT_MIN, color="#2ca02c", linestyle="--",
               label=f"real-structure min = {STRUCT_MIN}")
    ax.set_xlabel("connected component size")
    ax.set_ylabel("number of components (log scale)")
    ax.set_title(f"{frame_name}: component size distribution (k={K_DEFAULT}, pctl={PCTL_DEFAULT:.0f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gap_component_scatter(df: pd.DataFrame, sizes_per_point: np.ndarray, frame_name: str, path: Path):
    mask = (sizes_per_point > CAND_THRESHOLD) & (sizes_per_point < STRUCT_MIN)
    if not mask.any():
        return mask
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df.x[~mask], df.y[~mask], df.z[~mask], s=4, alpha=0.12, color="gray", label="other points")
    ax.scatter(df.x[mask], df.y[mask], df.z[mask], s=30, alpha=0.95, color=COLOR_SMALL,
               label=f"gap components ({CAND_THRESHOLD}<size<{STRUCT_MIN}), n={mask.sum()}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{frame_name}: points in gap-size components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return mask


# ---------- 4. 特征分布对比 ----------

FEATURE_DIST_COLS = ["local_density", "planarity", "scale_ratio", "linearity", "sphericity", "opacity"]


def feature_dist_grid(df: pd.DataFrame, sizes_per_point: np.ndarray, frame_name: str, path: Path):
    small_mask = sizes_per_point <= CAND_THRESHOLD
    large_mask = ~small_mask
    ncols = 3
    nrows = int(np.ceil(len(FEATURE_DIST_COLS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, feat in enumerate(FEATURE_DIST_COLS):
        ax = axes[i]
        vals_large = df.loc[large_mask, feat]
        vals_small = df.loc[small_mask, feat]
        ax.hist(vals_large, bins=20, alpha=0.55, color=COLOR_LARGE, density=True,
                label=f"size>{CAND_THRESHOLD} (n={int(large_mask.sum())})")
        ax.hist(vals_small, bins=20, alpha=0.55, color=COLOR_SMALL, density=True,
                label=f"size<={CAND_THRESHOLD} (n={int(small_mask.sum())})")
        ax.set_title(feat)
        ax.legend(fontsize=7)
    for j in range(len(FEATURE_DIST_COLS), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{frame_name}: feature distributions by component-size group "
                f"(k={K_DEFAULT}, pctl={PCTL_DEFAULT:.0f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 5. 参数敏感性网格 ----------

def param_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    xyz = df[["x", "y", "z"]].to_numpy()
    n = len(df)
    rows = []
    for k in K_GRID:
        for pctl in PCTL_GRID:
            labels = knn_component_labels(xyz, k, pctl)
            sizes = np.bincount(labels)
            n_components = len(sizes)
            small_sizes = sizes[sizes <= CAND_THRESHOLD]
            n_small_components = len(small_sizes)
            n_points_in_small = int(small_sizes.sum())
            rows.append({
                "k": k, "dist_percentile": pctl,
                "n_components": n_components,
                "n_small_components": n_small_components,
                "n_points_in_small": n_points_in_small,
                "pct_points_in_small": 100 * n_points_in_small / n,
            })
    return pd.DataFrame(rows)


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_frames()

    md_parts = ["# T1 特征表 EDA 总结 (f0090 / f0091 / f0092)\n",
                "纯观察记录，不含floater判定逻辑。\n"]

    # --- 1. basic stats ---
    md_parts.append("## 1. 基础统计\n")
    for name, df in dfs.items():
        md_parts.append(f"### {name} (总点数 = {len(df)})\n")
        stats = basic_stats(df)
        stats.to_csv(OUT_DIR / f"basic_stats_{name}.csv", index=False)
        md_parts.append(df_to_md(stats) + "\n")
        n_oob = int(df["color_oob"].sum()) if "color_oob" in df.columns else None
        if n_oob is not None:
            md_parts.append(f"color_oob=True 的点数: {n_oob} ({100*n_oob/len(df):.1f}%)\n")

    # --- 2. spatial scatter ---
    md_parts.append("## 2. 空间分布可视化\n")
    for name, df in dfs.items():
        p1 = OUT_DIR / f"scatter3d_{name}_opacity.png"
        p2 = OUT_DIR / f"scatter3d_{name}_density.png"
        scatter3d(df, "opacity", f"{name}: 3D scatter colored by opacity", p1)
        scatter3d(df, "local_density", f"{name}: 3D scatter colored by local_density", p2)
        md_parts.append(f"- {name}: `{p1.name}`, `{p2.name}`\n")
    overlay_path = OUT_DIR / "scatter3d_overlay_frames.png"
    scatter3d_overlay(dfs, overlay_path)
    md_parts.append(f"- 三帧叠加对比: `{overlay_path.name}`\n")

    # --- 3. connected components ---
    md_parts.append("## 3. k近邻图连通分量分布 (k=10, dist_percentile=75)\n")
    comp_labels_by_frame = {}
    comp_sizes_per_point_by_frame = {}
    for name, df in dfs.items():
        xyz = df[["x", "y", "z"]].to_numpy()
        labels = knn_component_labels(xyz, K_DEFAULT, PCTL_DEFAULT)
        comp_labels_by_frame[name] = labels
        sizes = np.bincount(labels)
        sizes_per_point = sizes[labels]
        comp_sizes_per_point_by_frame[name] = sizes_per_point

        sizes_sorted = np.sort(sizes)[::-1]
        md_parts.append(f"### {name}\n")
        md_parts.append(f"连通分量总数: {len(sizes)}\n")
        md_parts.append(f"所有分量size列表 (从大到小): {sizes_sorted.tolist()}\n")

        hist_path = OUT_DIR / f"component_hist_{name}.png"
        component_hist(sizes, name, hist_path)
        md_parts.append(f"分量size直方图: `{hist_path.name}`\n")

        gap_path = OUT_DIR / f"gap_components_{name}.png"
        gap_mask = gap_component_scatter(df, sizes_per_point, name, gap_path)
        gap_sizes = sorted(set(sizes_per_point[gap_mask].tolist())) if gap_mask.any() else []
        n_gap_components = sum(1 for s in sizes if CAND_THRESHOLD < s < STRUCT_MIN)
        md_parts.append(
            f"{CAND_THRESHOLD}~{STRUCT_MIN} 区间(不含端点)的分量数: {n_gap_components}"
            f"，对应size取值: {gap_sizes}，涉及点数: {int(gap_mask.sum())}\n"
        )
        if gap_mask.any():
            md_parts.append(f"这些分量的点已单独可视化: `{gap_path.name}`\n")
        else:
            md_parts.append("该帧在此区间没有分量（候选阈值与真实结构下限之间存在空隙）。\n")

    # --- 4. feature distributions by group ---
    md_parts.append("## 4. 现成特征分布: 分量size<=10 vs >10\n")
    for name, df in dfs.items():
        sizes_per_point = comp_sizes_per_point_by_frame[name]
        fig_path = OUT_DIR / f"feature_dist_{name}.png"
        feature_dist_grid(df, sizes_per_point, name, fig_path)
        n_small = int((sizes_per_point <= CAND_THRESHOLD).sum())
        n_large = int((sizes_per_point > CAND_THRESHOLD).sum())
        md_parts.append(f"- {name}: size<=10 点数={n_small}, size>10 点数={n_large} -> `{fig_path.name}`\n")

    # --- 5. parameter sensitivity grid ---
    md_parts.append("## 5. 参数敏感性网格 (k x dist_percentile)\n")
    for name, df in dfs.items():
        sens = param_sensitivity(df)
        sens.to_csv(OUT_DIR / f"param_sensitivity_{name}.csv", index=False)
        md_parts.append(f"### {name}\n")
        md_parts.append(df_to_md(sens) + "\n")

        pivot = sens.pivot(index="k", columns="dist_percentile", values="pct_points_in_small")
        pivot.to_csv(OUT_DIR / f"param_sensitivity_pivot_{name}.csv")
        pivot_disp = pivot.reset_index()
        pivot_disp.columns = ["k"] + [f"pctl={c}" for c in pivot.columns]
        md_parts.append(f"\n占比矩阵 (行=k, 列=dist_percentile, 值=size<=10分量里总点数占比 %):\n\n")
        md_parts.append(df_to_md(pivot_disp) + "\n")

    summary_path = OUT_DIR / "EDA_SUMMARY.md"
    summary_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n[done] summary written to {summary_path}")


if __name__ == "__main__":
    main()
