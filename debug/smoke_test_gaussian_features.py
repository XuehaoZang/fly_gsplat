"""
冒烟测试：计算 utils/gaussian_features.py 的逐点特征表，
针对 outputs/ctrl_009_002/f0100/splatfacto-checkpoint/2026-07-04_022821/splat.ply
做完整性检查 + 分布统计 + 抽样核验，不做任何过滤/删点。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.gaussian_features import compute_gaussian_features
from plyfile import PlyData

SPLAT_DIR = Path("outputs/ctrl_009_002/f0100/splatfacto-checkpoint/2026-07-04_022821")
SPLAT_PATH = SPLAT_DIR / "splat.ply"
TRANSFORM_PATH = SPLAT_DIR / "dataparser_transforms.json"


def main():
    n_ply = len(PlyData.read(str(SPLAT_PATH))["vertex"].data)
    df = compute_gaussian_features(SPLAT_PATH, TRANSFORM_PATH, k=10)

    print("=" * 70)
    print("1. 完整性检查")
    print("=" * 70)
    print(f"ply原始点数 = {n_ply}")
    print(f"DataFrame shape = {df.shape}")
    print(f"columns = {list(df.columns)}")
    n_nan = df.isna().sum()
    n_inf = df.select_dtypes(include=[np.number]).apply(lambda c: np.isinf(c).sum())
    print("每列 NaN 数:")
    print(n_nan[n_nan > 0] if n_nan.sum() else "  (无)")
    print("每列 Inf 数:")
    print(n_inf[n_inf > 0] if n_inf.sum() else "  (无)")
    print(f"行数是否等于ply点数: {len(df) == n_ply}")

    print()
    print("=" * 70)
    print("2. 每个新增字段分布统计")
    print("=" * 70)
    numeric_cols = [c for c in df.columns if df[c].dtype != bool]
    stats_rows = []
    for c in numeric_cols:
        arr = df[c].values
        stats_rows.append({
            "field": c, "min": arr.min(), "max": arr.max(), "mean": arr.mean(),
            "median": np.median(arr), "p90": np.percentile(arr, 90), "p95": np.percentile(arr, 95),
        })
    stats_df = pd.DataFrame(stats_rows).set_index("field")
    with pd.option_context("display.float_format", lambda x: f"{x:.6g}"):
        print(stats_df)

    lps_sum = df["linearity"] + df["planarity"] + df["sphericity"]
    print()
    print(f"linearity+planarity+sphericity: min={lps_sum.min():.8f} max={lps_sum.max():.8f} "
          f"mean={lps_sum.mean():.8f} (期望≈1)")

    pear_r, pear_p = pearsonr(df["local_density"], df["scale_ratio"])
    spear_r, spear_p = spearmanr(df["local_density"], df["scale_ratio"])
    print()
    print(f"local_density vs scale_ratio: Pearson r={pear_r:.4f} (p={pear_p:.2e}), "
          f"Spearman rho={spear_r:.4f} (p={spear_p:.2e})")
    print(f"color_oob 比例: {df['color_oob'].mean():.4f} ({df['color_oob'].sum()}/{len(df)})")

    print()
    print("=" * 70)
    print("3. 抽样人工核验")
    print("=" * 70)
    cols_show = ["scale_ratio", "linearity", "planarity", "sphericity",
                 "local_density", "dist_to_principal_axis", "dist_to_centroid", "opacity"]
    top5 = df.nlargest(5, "scale_ratio")[cols_show]
    print("scale_ratio 最高的5个点(最像尖刺):")
    print(top5.to_string())

    median_ratio = df["scale_ratio"].median()
    df["_dist_to_median_ratio"] = (df["scale_ratio"] - median_ratio).abs()
    mid5 = df.nsmallest(5, "_dist_to_median_ratio")[cols_show]
    print()
    print(f"scale_ratio 接近中位数({median_ratio:.3f})的5个点(对照):")
    print(mid5.to_string())
    df.drop(columns=["_dist_to_median_ratio"], inplace=True)

    print()
    print("对比均值:")
    print(f"  top5 (尖刺候选):  scale_ratio={top5['scale_ratio'].mean():.2f}  "
          f"linearity={top5['linearity'].mean():.4f}  local_density={top5['local_density'].mean():.2f}  "
          f"dist_to_principal_axis={top5['dist_to_principal_axis'].mean():.6f}")
    print(f"  mid5 (中位数对照): scale_ratio={mid5['scale_ratio'].mean():.2f}  "
          f"linearity={mid5['linearity'].mean():.4f}  local_density={mid5['local_density'].mean():.2f}  "
          f"dist_to_principal_axis={mid5['dist_to_principal_axis'].mean():.6f}")

    print()
    print("=" * 70)
    print("4. orientation 正确性自检 (随机3点)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(df), size=3, replace=False)
    for i in sample_idx:
        row = df.iloc[i]
        o = np.array([row["orientation_x"], row["orientation_y"], row["orientation_z"]])
        scales = [row["scale_phys_0"], row["scale_phys_1"], row["scale_phys_2"]]
        min_axis = int(np.argmin(scales))
        print(f"point {i}: orientation=({o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f})  "
              f"norm={np.linalg.norm(o):.6f}  scale_phys={np.array(scales)}  "
              f"min_scale_axis_idx={min_axis}")

    print()
    print("=" * 70)
    print("5. 行数硬性检查")
    print("=" * 70)
    print(f"输出行数 = {len(df)}, ply原始点数 = {n_ply}, 是否一致 = {len(df) == n_ply}")

    out_csv = SPLAT_DIR / "gaussian_features_f0100.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n特征表已保存: {out_csv}")


if __name__ == "__main__":
    main()
