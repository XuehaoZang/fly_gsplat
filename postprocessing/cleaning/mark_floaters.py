"""
T2: 在T1(utils/gaussian_features.py)输出的逐点特征表基础上标记明显孤立的floater点。
原样保留所有行/列，只新增一列 if_keep(bool)；不删点、不加权、不分类原因。

不处理贴着翼缘的尖刺——那部分在单帧特征空间和真实薄片结构连续分布，分不开，已验证
多种方法失败，本阶段搁置。

判据: 点所在的 k-近邻连通分量大小(patch_size, 见 utils.ply.connected_component_sizes)
<= min_patch_size 视为floater。在 G2b_scale_reg_ratio3 的 f0090/f0091/f0092 上验证:
真实解剖结构(躯干/翅膀/腿)对应的连通分量都 >= 17 点，孤立噪点的连通分量在 1~6 点之间
且和其它点没有邻接边，中间有清晰的gap，对 k(8~15) 和 dist_percentile(70~80) 的选取
不敏感。不用逐点的形状特征(scale_ratio/linearity等)判floater，因为真实尖刺形状上同样
细长但在空间上仍连着主体，用连通性可以正确保留它们；color_oob按连通分量分组后在这个
数据集里也不再是有效的区分信号(大分量oob_rate普遍接近0)，故不再需要。
"""
import argparse
from pathlib import Path

import pandas as pd

from utils.ply import connected_component_sizes

MIN_PATCH_SIZE = 10
K_NEIGHBORS = 10
DIST_PERCENTILE = 75.0


def mark_floaters(df: pd.DataFrame, min_patch_size: int = MIN_PATCH_SIZE,
                   k: int = K_NEIGHBORS, dist_percentile: float = DIST_PERCENTILE) -> pd.DataFrame:
    xyz = df[["x", "y", "z"]].to_numpy()
    patch_size = connected_component_sizes(xyz, k=k, dist_percentile=dist_percentile)
    out = df.copy()
    out["if_keep"] = patch_size > min_patch_size
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="T1输出的逐点特征表路径 (gaussian_features_*.csv)")
    parser.add_argument("--out", type=str, default=None, help="输出路径，默认在原文件名后加 _marked")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    marked = mark_floaters(df)

    out_path = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_marked.csv")
    marked.to_csv(out_path, index=False)

    n_total = len(marked)
    n_floater = int((~marked["if_keep"]).sum())
    print(f"[{csv_path.name}] n_total={n_total}  n_floater={n_floater} "
          f"({100 * n_floater / n_total:.1f}%)  saved -> {out_path}")


if __name__ == "__main__":
    main()
