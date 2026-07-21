"""
T3 body/wing 二分类，第二版：无监督聚类(KMeans)方法，作为规则阈值法
(binary_split.py，现归档在 binary_split_NO_USE/ 供对照)的替代方案。

背景(阈值法失败的根本原因): body -> wing 是连续渐变(尤其翼根附近)，不是双峰分布，
硬阈值(dist_to_principal_axis / planarity 任一超阈值)只能抓住翼尖附近的极值点，
漏掉根部/中段的wing点。改用无监督聚类，让多维特征的联合分布决定分界，而不是
单特征硬切。

方法:
1. 特征: [x, y, z, planarity, scale_ratio, opacity] 六维(先用这六维；R/G/B/
   color_oob不用，历史上这批数据颜色信号不稳定)。每维单独standardize
   (zero mean unit variance)后再做kmeans，避免xyz的米级尺度和0~1的形状特征
   混在一起时被某一类特征暗中主导。
2. k=3(body/wing_L/wing_R一次性聚类，不再分两步做body-vs-all)，sklearn KMeans
   (n_init>=10，固定random_state)。
3. cluster_id -> 语义标签，两条规则都跑、都打印，不自动选一个:
   - 规则A: 点数最多的簇 = body；其余两簇按质心投影到"次主轴"(第二大方差方向，
     猜测对应展翅时的左右展向)的正负暂称wing_A/wing_B(不代表真实L/R语义，
     那是ST3的事，这里不做)。
   - 规则B: mean planarity最低的簇 = body(翅膀是膜状结构，planarity该更高)。
   两规则如果对body的判定一致最好；不一致时都打印，供人工判断哪个更可信，
   不在代码里自动拍板选一个。
4. 出图: 前视+俯视两视角散点图(排版同binary_split_NO_USE/check_binary_split.py，
   方便直接对比)，三簇三色(按raw cluster_id着色，不是body/wing语义)，存到
   eda_outputs/，文件名带"kmeans"以区分threshold版的图。
5. 稳定性检查: kmeans对初始化敏感，不能假设它天然稳定——同一帧用5个不同
   random_state重跑，两两算adjusted_rand_score(ARI，对簇编号permutation不敏感，
   能正确处理"同一划分但簇编号换了"的情况)比较cluster分配一致性。

不做的事(留给之后的步骤): L/R真实语义判断(ST3)、颜色特征加入判据、跨帧对齐簇号。

用法:
    python -m postprocessing.labeling.kmeans_split
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import find_features_csv  # noqa: E402
from postprocessing.kinematics.geometry import weighted_pca  # noqa: E402
from postprocessing.labeling.select_dev_frames import DEV_FRAMES, DATASET_DIR  # noqa: E402

OUT_DIR = REPO_ROOT / "postprocessing" / "labeling" / "eda_outputs"

FEATURES = ["x", "y", "z", "planarity", "scale_ratio", "opacity"]
K = 3
N_INIT = 10
MAIN_RANDOM_STATE = 0              # 用来出图/定语义标签的那一次kmeans
STABILITY_SEEDS = [0, 1, 2, 3, 4]  # 稳定性检查用的5个种子(含MAIN_RANDOM_STATE)

CLUSTER_COLORS = [to_rgb("#1f77b4"), to_rgb("#ff7f0e"), to_rgb("#2ca02c")]


def load_kept(frame: str) -> pd.DataFrame:
    """加载该帧的_marked表，只取if_keep=True的点(跟binary_split_NO_USE同样的口径)。"""
    features_csv = find_features_csv(frame, DATASET_DIR)
    marked_csv = features_csv.with_name(features_csv.stem + "_marked.csv")
    df = pd.read_csv(marked_csv)
    return df[df["if_keep"]].reset_index(drop=True)


def standardize(df: pd.DataFrame) -> np.ndarray:
    """FEATURES 每维单独zero-mean/unit-variance后再喂给kmeans，见模块docstring。"""
    return StandardScaler().fit_transform(df[FEATURES].to_numpy(dtype=float))


def run_kmeans(X: np.ndarray, random_state: int) -> np.ndarray:
    km = KMeans(n_clusters=K, n_init=N_INIT, random_state=random_state)
    return km.fit_predict(X)


def cluster_sizes(labels: np.ndarray) -> np.ndarray:
    return np.array([int((labels == c).sum()) for c in range(K)])


def secondary_axis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """次主轴(第二大方差方向)+质心，猜测对应展翅时的左右展向，仅用来给wing_A/wing_B
    一个暂时的、带点物理意义的排序方向；不代表真实L/R语义(ST3的事，这里不做)。"""
    _, eigvecs, centroid = weighted_pca(xyz)
    return eigvecs[:, -2], centroid


def _split_others_by_axis(df: pd.DataFrame, labels: np.ndarray, others: list[int],
                           axis: np.ndarray, centroid: np.ndarray) -> tuple[int, int]:
    """把body之外的两个簇按质心在axis上的投影排序，返回(wing_A簇id, wing_B簇id)
    (投影较大的一侧记作wing_A)。规则A/B共用同一条排序规则，方便直接对比。"""
    xyz = df[["x", "y", "z"]].to_numpy()
    projs = {c: float(np.dot(xyz[labels == c].mean(axis=0) - centroid, axis)) for c in others}
    ordered = sorted(others, key=lambda c: projs[c], reverse=True)
    return ordered[0], ordered[1]


def label_by_rule_a(df: pd.DataFrame, labels: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> dict[int, str]:
    """规则A: 点数最多的簇 = body；其余两簇按质心在次主轴上的投影正负分wing_A/wing_B。"""
    body_id = int(np.argmax(cluster_sizes(labels)))
    others = [c for c in range(K) if c != body_id]
    wing_a, wing_b = _split_others_by_axis(df, labels, others, axis, centroid)
    return {body_id: "body", wing_a: "wing_A", wing_b: "wing_B"}


def label_by_rule_b(df: pd.DataFrame, labels: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> dict[int, str]:
    """规则B: mean planarity最低的簇 = body；其余两簇的wing_A/wing_B排序方式同规则A，
    方便直接对比两条规则是否选出同一个body簇。"""
    planarity_means = {c: float(df.loc[labels == c, "planarity"].mean()) for c in range(K)}
    body_id = min(planarity_means, key=planarity_means.get)
    others = [c for c in range(K) if c != body_id]
    wing_a, wing_b = _split_others_by_axis(df, labels, others, axis, centroid)
    return {body_id: "body", wing_a: "wing_A", wing_b: "wing_B"}


def stability_check(X: np.ndarray, seeds: list[int] = STABILITY_SEEDS) -> dict:
    """kmeans对初始化敏感，不能假设天然稳定：同一帧用多个不同random_state重跑，
    两两算adjusted_rand_score(ARI，对簇编号permutation不敏感)比较cluster分配的
    一致性。"""
    runs = {s: run_kmeans(X, s) for s in seeds}
    pairwise = [(s1, s2, adjusted_rand_score(runs[s1], runs[s2]))
                for s1, s2 in combinations(seeds, 2)]
    aris = [ari for _, _, ari in pairwise]
    return {"pairwise": pairwise, "mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris))}


def plot_kmeans_clusters(df: pd.DataFrame, labels: np.ndarray, frame: str, out_path: Path) -> None:
    xyz = df[["x", "y", "z"]].to_numpy()
    colors = np.array([CLUSTER_COLORS[c] for c in labels])
    sizes = cluster_sizes(labels)
    n_total = len(df)

    views = [
        ("front (elev=0, azim=-90)", dict(elev=0, azim=-90)),
        ("top (elev=90, azim=-90)", dict(elev=90, azim=-90)),
    ]

    fig = plt.figure(figsize=(12, 5.5))
    for i, (title, view_kw) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=14, alpha=0.9, depthshade=False)
        ax.view_init(**view_kw)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)

    size_str = "  ".join(
        f"cluster{c}={sizes[c]}({100 * sizes[c] / n_total:.1f}%)" for c in range(K))
    fig.suptitle(
        f"{frame}: KMeans k={K} on standardized {FEATURES}  |  n_total={n_total}  |  {size_str}\n"
        f"colors = raw cluster_id (blue=0 orange=1 green=2, NOT body/wing semantics "
        f"-- see printed rule A/B mapping)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(frame: str) -> dict:
    df = load_kept(frame)
    X = standardize(df)

    labels = run_kmeans(X, MAIN_RANDOM_STATE)
    sizes = cluster_sizes(labels)
    n_total = len(df)

    xyz = df[["x", "y", "z"]].to_numpy()
    axis, centroid = secondary_axis(xyz)
    mapping_a = label_by_rule_a(df, labels, axis, centroid)
    mapping_b = label_by_rule_b(df, labels, axis, centroid)
    body_a = next(c for c, lab in mapping_a.items() if lab == "body")
    body_b = next(c for c, lab in mapping_b.items() if lab == "body")
    agree = body_a == body_b

    stability = stability_check(X)

    out_path = OUT_DIR / f"kmeans_split_{frame}.png"
    plot_kmeans_clusters(df, labels, frame, out_path)

    print(f"\n[{frame}] n_total={n_total}")
    print("  簇点数占比: " + "  ".join(
        f"cluster{c}={sizes[c]}({100 * sizes[c] / n_total:.1f}%)" for c in range(K)))
    print("  规则A(点数最多=body): " + "  ".join(
        f"cluster{c}={lab}" for c, lab in sorted(mapping_a.items())))
    print("  规则B(mean planarity最低=body): " + "  ".join(
        f"cluster{c}={lab}" for c, lab in sorted(mapping_b.items())))
    print(f"  规则A/B对body簇的判定{'一致' if agree else '不一致，需要人工判断'}: "
          f"A选cluster{body_a}, B选cluster{body_b}")
    if not agree:
        planarity_means = {c: float(df.loc[labels == c, "planarity"].mean()) for c in range(K)}
        print("    [人工判断参考] 各簇mean planarity: " +
              "  ".join(f"cluster{c}={planarity_means[c]:.3f}" for c in range(K)) +
              f"  (点数最多的是cluster{int(np.argmax(sizes))})")
    print(f"  稳定性检查(5个random_state两两ARI): mean={stability['mean_ari']:.3f} "
          f"min={stability['min_ari']:.3f}  "
          f"{'稳定' if stability['min_ari'] > 0.9 else '[警告]不同初始化的聚类结果有明显差异，需留意'}")
    print(f"  plot -> {out_path}")

    return {
        "frame": frame, "sizes": sizes, "mapping_a": mapping_a, "mapping_b": mapping_b,
        "agree": agree, "stability": stability,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run(f) for f in DEV_FRAMES]

    n_agree = sum(r["agree"] for r in results)
    mean_min_ari = float(np.mean([r["stability"]["min_ari"] for r in results]))
    print(f"\n{'=' * 70}\n{len(results)}帧汇总\n{'=' * 70}")
    print(f"  规则A/B对body簇判定一致的帧数: {n_agree}/{len(results)}")
    print(f"  各帧稳定性(min ARI)的均值: {mean_min_ari:.3f}")


if __name__ == "__main__":
    main()
