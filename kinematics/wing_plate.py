"""单翅"平板"拟合：用每个高斯的完整协方差，把翅膀当成椭球并集里的一块平板。

1. 连通：两高斯 Δᵀ(Σᵢ+Σⱼ)⁻¹Δ < CONNECT_MAHAL2 即相连；只保留不透明度质量最大的连通块（去掉漂浮块）。
2. 平面投票：假设 = 每个较扁高斯的（中心, 自身最短轴）+ 若干随机三点平面。高斯 j 支持平面 (p, n) 当且仅当
     中心距离 |n·(μⱼ−p)| < τ_d，且沿法向的 2σ 半厚度 2√(nᵀΣⱼn) < τ_t；
   得分 = 支持者权重（不透明度 × 面积，p90 截断）之和。最佳平面用内点的混合矩 PCA 精化，迭代 REFINE_ITERS 次。
   τ_d、τ_t 按本翅连通块的半厚度中位数自适应（系数在 ctrl_009_003 上按距离/厚度分布标定）。
3. 并集 PCA：不透明度 >= UNION_MIN_OPACITY 的内点，把各自 2σ 椭球体素化后取并集，对占用体素做 PCA，
   最小特征向量 = 翅面法向。并集不重复计数重叠区域，避免中心点密度带来的偏差。
   翅展默认（span_source="comp3"）取整个连通块（不按不透明度筛选）3σ 并集的主轴、投影到平板平面内：
   平板内点常只覆盖翅膀的一条带，其主轴可能是弦向（合成数据上翅展误差可达 50–85°）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .io import Gaussians

SIGMA = 2.0
"""椭球截断在 2σ。"""
CONNECT_MAHAL2 = 18.0
"""连通判据 Δᵀ(Σᵢ+Σⱼ)⁻¹Δ 的阈值：等大球在 3σ 处相接时为 2·3² = 18。2σ（≈9）会把合成数据里的翅膀切成 3–11 块，
最大块只占 38–69% 质量。"""
TAU_D_FACTOR = 1.0
"""τ_d = 该系数 × 连通块 2σ 半厚度中位数（ctrl_009_003 上约 70 µm）。"""
TAU_T_FACTOR = 1.75
"""τ_t = 该系数 × 连通块 2σ 半厚度中位数（约 120 µm；分布主体 35–110 µm，长尾为厚团）。"""
HYPOTHESIS_MAX_FLATNESS = 0.6
"""s_min/s_mid 小于该值的高斯用自身最短轴提出平面假设（翅膀高斯该比值中位数约 0.55）。"""
N_RANDOM_HYPOTHESES = 200
REFINE_ITERS = 2
WEIGHT_CLIP_QUANTILE = 0.9
UNION_MIN_OPACITY = 0.1
"""低于该不透明度的高斯不进体素并集（仍参与连通判断和投票）。"""
VOXEL_FACTOR = 0.6
"""体素边长 = 该系数 × 内点 s_min 中位数（约 18 µm）。"""
MIN_INLIERS = 5
SPAN_SOURCES = ("plate", "comp2", "comp3", "all2")
"""翅展来源：plate = 平板内点并集的主轴；compK = 整个最大连通块（不按不透明度筛选）的 Kσ 椭球并集主轴，
再投影到平板平面内；allK = 该翅全部高斯（不做连通筛选）的 Kσ 并集。法向始终来自平板。"""
SPAN_VOXEL_FACTOR = 2.0
"""compK 并集的体素边长 = 该系数 × 平板体素边长（翅展是粗量，体素放粗以控制大椭球的开销）。"""


@dataclass
class PlateFit:
    span_axis: np.ndarray
    """翅展方向（未定号）。"""
    normal: np.ndarray
    """翅面法向（未定号）。"""
    centroid: np.ndarray
    """并集体素质心。"""
    inlier: np.ndarray
    """(N,) bool：参与并集的高斯。"""
    component: np.ndarray
    """(N,) bool：最大连通块。"""
    flatness: float
    """并集 PCA λ3/λ2，越小越像平板。"""
    inlier_mass_frac: float
    """内点权重 / 全部权重。"""
    tau_d: float
    tau_t: float


def _components(g: Gaussians) -> np.ndarray:
    d = g.xyz[:, None, :] - g.xyz[None, :, :]
    S = g.cov[:, None] + g.cov[None, :]
    m2 = np.einsum("ijk,ijk->ij", d, np.linalg.solve(S, d[..., None])[..., 0])
    _, labels = connected_components(coo_matrix(m2 < CONNECT_MAHAL2), directed=False)
    return labels


def _half_thickness(cov: np.ndarray, n: np.ndarray) -> np.ndarray:
    return SIGMA * np.sqrt(np.einsum("i,nij,j->n", n, cov, n))


def _support(g: Gaussians, p: np.ndarray, n: np.ndarray, tau_d: float, tau_t: float) -> np.ndarray:
    return (np.abs((g.xyz - p) @ n) < tau_d) & (_half_thickness(g.cov, n) < tau_t)


def _mixture_pca(xyz: np.ndarray, cov: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C = Σ wᵢ[Σᵢ + ΔᵢΔᵢᵀ] / Σ wᵢ；返回 (特征值升序, 特征向量列, 质心)。"""
    mu = (w[:, None] * xyz).sum(axis=0) / w.sum()
    D = xyz - mu
    C = (w[:, None, None] * (cov + D[:, :, None] * D[:, None, :])).sum(axis=0) / w.sum()
    lam, V = np.linalg.eigh(C)
    return lam, V, mu


def _union_voxels(xyz: np.ndarray, cov: np.ndarray, h: float, sigma: float = SIGMA) -> np.ndarray:
    """各 sigma 椭球覆盖的体素中心取并集，(M,3)。每个椭球至少占它中心所在的体素。"""
    keys = []
    for mu, S in zip(xyz, cov):
        half = sigma * np.sqrt(np.diag(S))
        lo = np.floor((mu - half) / h).astype(np.int64)
        hi = np.floor((mu + half) / h).astype(np.int64)
        grid = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        rel = (grid + 0.5) * h - mu
        inside = np.einsum("ni,ij,nj->n", rel, np.linalg.inv(S), rel) <= sigma**2
        keys.append(grid[inside])
        keys.append(np.floor(mu / h).astype(np.int64)[None])
    return (np.unique(np.concatenate(keys), axis=0) + 0.5) * h


def fit_wing_plate(g: Gaussians, rng: int = 0, span_source: str = "comp3") -> PlateFit:
    n_g = g.xyz.shape[0]
    eigvals, eigvecs = np.linalg.eigh(g.cov)
    s = np.sqrt(np.clip(eigvals, 0.0, None))  # 升序：s_min, s_mid, s_max
    own_normal = eigvecs[:, :, 0]

    area = s[:, 1] * s[:, 2]
    w = g.opacity * area
    w = np.minimum(w, np.quantile(w, WEIGHT_CLIP_QUANTILE))

    labels = _components(g)
    component = labels == int(np.argmax(np.bincount(labels, weights=w)))
    comp_idx = np.nonzero(component)[0]
    if comp_idx.size < MIN_INLIERS:
        raise ValueError(f"wing_plate: largest connected component has {comp_idx.size} gaussians")
    w_comp = np.where(component, w, 0.0)

    # 阈值：连通块内高斯沿各自最短轴的 2σ 半厚度中位数
    t_ref = float(np.median(SIGMA * s[comp_idx, 0]))
    tau_d, tau_t = TAU_D_FACTOR * t_ref, TAU_T_FACTOR * t_ref

    hyps: list[tuple[np.ndarray, np.ndarray]] = [
        (g.xyz[i], own_normal[i]) for i in comp_idx if s[i, 0] / max(s[i, 1], 1e-12) < HYPOTHESIS_MAX_FLATNESS
    ]
    gen = np.random.default_rng(rng)
    for _ in range(N_RANDOM_HYPOTHESES if comp_idx.size >= 3 else 0):
        a, b, c = g.xyz[gen.choice(comp_idx, size=3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) > 1e-15:
            hyps.append((a, n / np.linalg.norm(n)))
    if not hyps:
        raise ValueError("wing_plate: no plane hypotheses")

    scores = [float(w_comp[_support(g, p, n, tau_d, tau_t)].sum()) for p, n in hyps]
    p, n = hyps[int(np.argmax(scores))]

    inlier = component & _support(g, p, n, tau_d, tau_t)
    for _ in range(REFINE_ITERS):
        if inlier.sum() < MIN_INLIERS:
            break
        _, V, p = _mixture_pca(g.xyz[inlier], g.cov[inlier], w[inlier])
        n = V[:, 0]
        inlier = component & _support(g, p, n, tau_d, tau_t)

    union = inlier & (g.opacity >= UNION_MIN_OPACITY)
    if union.sum() < MIN_INLIERS:
        raise ValueError(f"wing_plate: only {int(union.sum())} inliers above opacity {UNION_MIN_OPACITY}")
    h = VOXEL_FACTOR * float(np.median(s[union, 0]))
    vox = _union_voxels(g.xyz[union], g.cov[union], h)
    lam, V = np.linalg.eigh(np.cov((vox - vox.mean(axis=0)).T))
    normal, span_axis, centroid = V[:, 0], V[:, 2], vox.mean(axis=0)

    if span_source != "plate":
        k = float(span_source[-1])
        sel = component if span_source.startswith("comp") else np.ones(n_g, dtype=bool)
        vox_c = _union_voxels(g.xyz[sel], g.cov[sel], SPAN_VOXEL_FACTOR * h, sigma=k)
        _, Vc = np.linalg.eigh(np.cov((vox_c - vox_c.mean(axis=0)).T))
        span_axis = Vc[:, 2] - normal * float(np.dot(Vc[:, 2], normal))
        span_axis = span_axis / np.linalg.norm(span_axis)
        centroid = vox_c.mean(axis=0)

    return PlateFit(
        span_axis=span_axis, normal=normal, centroid=centroid,
        inlier=union, component=component,
        flatness=float(lam[0] / lam[1]),
        inlier_mass_frac=float(w[union].sum() / w.sum()),
        tau_d=tau_d, tau_t=tau_t,
    )
