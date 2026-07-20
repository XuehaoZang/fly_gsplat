import subprocess
from pathlib import Path
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

def export_splat(splat_dir: Path) -> None:
    config_path = splat_dir / "config.yml"
    if not (splat_dir / "splat.ply").exists():
        subprocess.run([
            "ns-export", "gaussian-splat",
            "--load-config", str(config_path),
            "--output-dir", str(splat_dir)
        ], check=True)


def load_ply(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)

from plyfile import PlyData

def load_ply_with_attrs(path: Path) -> dict:
    """
    读取 gaussian-splat ply 的 xyz + opacity + scale，并做激活函数还原
    （splat.ply 里存的是训练用的原始参数：opacity 是 sigmoid 之前的 logit，
    scale 是 log 之前的值，需要 sigmoid/exp 还原成真实物理量）。
    缺失字段时优雅降级为 None，调用方需要检查。
    """
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)

    opacity = None
    if "opacity" in v.data.dtype.names:
        opacity = 1.0 / (1.0 + np.exp(-v["opacity"]))  # sigmoid

    scale = None
    scale_names = [f"scale_{i}" for i in range(3)]
    if all(n in v.data.dtype.names for n in scale_names):
        scale_raw = np.stack([v[n] for n in scale_names], axis=-1)
        scale = np.exp(scale_raw).mean(axis=-1)  # exp还原，三轴取平均作为标量尺度

    rgb = None
    SH_C0 = 0.28209479177387814
    dc_names = [f"f_dc_{i}" for i in range(3)]
    if all(n in v.data.dtype.names for n in dc_names):
        dc = np.stack([v[n] for n in dc_names], axis=-1)
        rgb = np.clip(0.5 + SH_C0 * dc, 0, 1)  # 球谐0阶系数还原成0~1 RGB

    return {"xyz": xyz, "opacity": opacity, "scale": scale, "rgb": rgb}

def print_stats(label: str, pts: np.ndarray) -> None:
    if len(pts) == 0:
        print(f"[{label}] EMPTY")
        return
    print(f"[{label}]")
    print(f"  n       = {len(pts)}")
    print(f"  center  = {pts.mean(axis=0)}")
    print(f"  min     = {pts.min(axis=0)}")
    print(f"  max     = {pts.max(axis=0)}")
    print(f"  extent  = {pts.max(axis=0) - pts.min(axis=0)}")

def unrescale(pts: np.ndarray, R_ns: np.ndarray, t_ns: np.ndarray, scale: float) -> np.ndarray:
    """把 dataparser 归一化坐标变换回原始物理坐标（transforms.json 空间）。"""
    return (R_ns.T @ (pts.T / scale - t_ns[:, None])).T

def characterize_sphere(pts: np.ndarray, expected_radius: float) -> None:
    center = pts.mean(axis=0)
    r = np.linalg.norm(pts - center, axis=1)
    print(f"[sphere check] n={len(pts)}")
    print(f"  radius: mean={r.mean():.6f}  std={r.std():.6f}  expected={expected_radius:.6f}")
    print(f"  radius min/max = {r.min():.6f} / {r.max():.6f}")

    cov = np.cov((pts - center).T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    axis_len = np.sqrt(eigvals) * 2  # 近似轴长
    print(f"  PCA axis lengths (long->short) = {axis_len}")
    print(f"  anisotropy ratio (max/min) = {axis_len[0]/axis_len[2]:.3f}")

    hist, edges = np.histogram(r, bins=10, range=(0, expected_radius*1.2))
    print(f"  radius histogram (10 bins, 0~{expected_radius*1.2:.4f}):")
    for h, e in zip(hist, edges):
        print(f"    {e:.5f}: ({h})")

def clean_ply(pts: np.ndarray, eps: float, min_samples: int = 5,
              min_cluster_frac: float = 0.02) -> tuple:
    """
    DBSCAN 密度聚类去噪：保留所有"足够大"的簇（可能是主体+翅膀，多个连通分量），
    丢弃孤立小簇（floaters）。返回 (kept_pts, removed_pts)。

    eps              : DBSCAN 邻域半径（米）。建议从 hull 点间平均最近邻距离的 2~3 倍开始试。
    min_samples      : DBSCAN 核心点最小邻居数。
    min_cluster_frac : 簇大小占总点数的最小比例，低于此比例的簇视为 floaters。
    """
    if len(pts) == 0:
        return pts, np.empty((0, 3))

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    n_total = len(pts)

    keep_mask = np.zeros(n_total, dtype=bool)
    unique_labels = [l for l in set(labels) if l != -1]  # -1 = DBSCAN 噪声点，直接丢弃
    for l in unique_labels:
        cluster_mask = labels == l
        if cluster_mask.sum() / n_total >= min_cluster_frac:
            keep_mask |= cluster_mask

    print(f"[clean_ply] eps={eps:.5f} clusters={len(unique_labels)} "
          f"kept={keep_mask.sum()}/{n_total} ({100*keep_mask.sum()/n_total:.1f}%)")

    return pts[keep_mask], pts[~keep_mask]

def analyze_scale_ratio(splat_path: Path) -> dict:
    """统计每个高斯 max(scale)/min(scale) 的比值分布，衡量"尖刺程度"。
    比值=1 是完美球形，比值越大说明这个高斯被拉得越细长（尖刺）。"""
    ply = PlyData.read(str(splat_path))
    v = ply["vertex"]
    scale_names = [f"scale_{i}" for i in range(3)]
    if not all(n in v.data.dtype.names for n in scale_names):
        return {"error": "no scale fields found"}

    scales = np.exp(np.stack([v[n] for n in scale_names], axis=-1))  # 还原log-scale
    ratios = scales.max(axis=-1) / np.clip(scales.min(axis=-1), 1e-12, None)

    return {
        "n": len(ratios),
        "median": float(np.median(ratios)),
        "p90": float(np.percentile(ratios, 90)),
        "p95": float(np.percentile(ratios, 95)),
        "max": float(ratios.max()),
        "frac_over_10": float((ratios > 10).mean()),  # 超过官方默认阈值的比例
    }

def connected_component_sizes(xyz: np.ndarray, k: int = 10, dist_percentile: float = 75.0) -> np.ndarray:
    """
    构建 k-近邻图，只保留邻距不超过全局 dist_percentile 分位数的边，做连通分量分析，
    返回每个点所在连通分量的大小(patch_size)。孤立的小分量是"离群噪点"的强信号，
    且比逐点的形状特征(scale_ratio/linearity等)更能把噪点和贴着结构边缘的真实薄片/尖刺
    区分开——后者形状上同样细长，但在空间上仍连着主体，因此连通分量大。
    """
    n = len(xyz)
    tree = cKDTree(xyz)
    dists, idxs = tree.query(xyz, k=k + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]  # 去掉自身(距离0)

    threshold = np.percentile(dists, dist_percentile)
    mask = dists <= threshold
    rows = np.repeat(np.arange(n), k)[mask.ravel()]
    cols = idxs.ravel()[mask.ravel()]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)  # 对称化(无向图)
    _, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)
    return comp_sizes[labels]

def local_pca_extent(xyz: np.ndarray, k: int = 10) -> np.ndarray:
    """
    每个点 k 近邻(含自身)在局部PCA前两主轴方向的延展(极差合成范数)。
    连续片状/线状结构的延展明显大于孤立噪点，可作为connected_component_sizes之外
    的补充判据。
    """
    n = len(xyz)
    tree = cKDTree(xyz)
    _, idxs = tree.query(xyz, k=k + 1)

    extent = np.zeros(n)
    for i in range(n):
        nbr_pts = xyz[idxs[i]]  # 含自身
        centered = nbr_pts - nbr_pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        top2 = eigvecs[:, order[:2]]
        proj = centered @ top2
        rng = proj.max(axis=0) - proj.min(axis=0)
        extent[i] = np.linalg.norm(rng)
    return extent