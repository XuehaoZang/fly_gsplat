"""去 floater：k 近邻图里所在连通分量不超过 MIN_PATCH_SIZE 个点的高斯记为 floater。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.pointcloud import knn_component_labels

MIN_PATCH_SIZE = 10
K_NEIGHBORS = 10
DIST_PERCENTILE = 75.0


def mark_floaters(df: pd.DataFrame) -> pd.DataFrame:
    """在特征表后面加 if_keep 列（True = 保留），不删点。"""
    labels = knn_component_labels(df[["x", "y", "z"]].to_numpy(), k=K_NEIGHBORS, dist_percentile=DIST_PERCENTILE)
    out = df.copy()
    out["if_keep"] = np.bincount(labels)[labels] > MIN_PATCH_SIZE
    return out
