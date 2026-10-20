"""点云的公共工具函数。"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def knn_component_labels(xyz: np.ndarray, k: int = 10, dist_percentile: float = 75.0) -> np.ndarray:
    """k 近邻图的连通分量编号（0..n_components-1）。

    只保留邻距不超过全体邻距 dist_percentile 分位数的边，再做无向连通分量。
    """
    n = len(xyz)
    dists, idxs = cKDTree(xyz).query(xyz, k=k + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]  # 去掉自身

    mask = dists <= np.percentile(dists, dist_percentile)
    rows = np.repeat(np.arange(n), k)[mask.ravel()]
    cols = idxs.ravel()[mask.ravel()]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)
    _, labels = connected_components(adj, directed=False)
    return labels
