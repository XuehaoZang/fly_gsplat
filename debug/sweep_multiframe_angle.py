"""
T1续接八：多帧验证去噪前后主轴角度偏差的稳定性。
在 f0000/f0025/f0050/f0075/f0100 五帧上重复：
gaussian_features -> HDBSCAN(min_cluster_size=15) -> 去掉噪声点(-1)和
color_oob>=0.95的纯伪影小簇 -> 去噪前后各算一次PCA主轴 -> 记录夹角。

选帧理由：默认配置(outputs/ctrl_009_002/)下只有f0000-f0100(101帧)全部
训练到10000步收敛，等间隔取5帧(0/25/50/75/100)覆盖约1/4周期跨度
(225Hz拍打频率下，100帧@16000fps对应6.25ms，约1.4个拍打周期)，
不针对具体拍打相位挑帧，避免选帧偏好影响结论。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.gaussian_features import compute_gaussian_features

FRAMES = {
    "f0000": "2026-07-03_174430",
    "f0025": "2026-07-03_195519",
    "f0050": "2026-07-03_220648",
    "f0075": "2026-07-04_001659",
    "f0100": "2026-07-04_022821",
}
BASE = Path("outputs/ctrl_009_002")

RAW_COLS = ["x", "y", "z", "brightness", "opacity", "linearity", "planarity", "sphericity"]
LOG_COLS = ["scale_phys_0", "scale_phys_1", "scale_phys_2", "local_density"]


def principal_axes(xyz: np.ndarray):
    centroid = xyz.mean(axis=0)
    cov = np.cov((xyz - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[0]], eigvecs[:, order[1]], eigvals[order]


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    if np.dot(a, b) < 0:
        b = -b
    cos_a = np.clip(np.dot(a, b), -1, 1)
    return np.degrees(np.arccos(cos_a))


def process_frame(frame_name: str, ts: str) -> dict:
    splat_dir = BASE / frame_name / "splatfacto-checkpoint" / ts
    splat_path = splat_dir / "splat.ply"
    transform_path = splat_dir / "dataparser_transforms.json"

    df = compute_gaussian_features(splat_path, transform_path, k=10)
    n_total = len(df)
    df["brightness"] = df[["R", "G", "B"]].mean(axis=1)

    X_raw = df[RAW_COLS].values
    X_log = np.log(df[LOG_COLS].values)
    X = np.concatenate([X_raw, X_log], axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    hdb = HDBSCAN(min_cluster_size=15)
    labels = hdb.fit_predict(X_scaled)
    df["cluster_hdbscan"] = labels

    oob_rate = df.groupby("cluster_hdbscan")["color_oob"].mean()
    drop_clusters = [-1] + [cid for cid in oob_rate.index if cid != -1 and oob_rate[cid] >= 0.95]
    df_clean = df[~df["cluster_hdbscan"].isin(drop_clusters)]

    n_floater = n_total - len(df_clean)
    floater_frac = n_floater / n_total

    xyz_full = df[["x", "y", "z"]].values
    xyz_clean = df_clean[["x", "y", "z"]].values

    ax1_full, ax2_full, eig_full = principal_axes(xyz_full)
    ax1_clean, ax2_clean, eig_clean = principal_axes(xyz_clean)

    angle1 = angle_between(ax1_full, ax1_clean)
    angle2 = angle_between(ax2_full, ax2_clean)

    ratio_full = eig_full[0] / eig_full[1]
    ratio_clean = eig_clean[0] / eig_clean[1]

    print(f"[{frame_name}] n_total={n_total}  cluster构成: "
          f"{dict(df['cluster_hdbscan'].value_counts().sort_index())}  "
          f"drop_clusters={drop_clusters}  n_clean={len(df_clean)}")

    return {
        "frame": frame_name,
        "n_total": n_total,
        "n_floater": n_floater,
        "floater_frac": floater_frac,
        "angle1_deg": angle1,
        "angle2_deg": angle2,
        "lambda1_lambda2_full": ratio_full,
        "lambda1_lambda2_clean": ratio_clean,
    }


def main():
    rows = []
    for frame_name, ts in FRAMES.items():
        rows.append(process_frame(frame_name, ts))

    result_df = pd.DataFrame(rows)
    print()
    print("=" * 100)
    print("5帧汇总")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}", "display.width", 200):
        print(result_df.to_string(index=False))

    print()
    print(f"angle1 (第一主轴夹角): min={result_df['angle1_deg'].min():.2f}  max={result_df['angle1_deg'].max():.2f}  "
          f"mean={result_df['angle1_deg'].mean():.2f}  std={result_df['angle1_deg'].std():.2f}")
    print(f"angle2 (第二主轴夹角): min={result_df['angle2_deg'].min():.2f}  max={result_df['angle2_deg'].max():.2f}  "
          f"mean={result_df['angle2_deg'].mean():.2f}  std={result_df['angle2_deg'].std():.2f}")
    print(f"floater_frac: min={result_df['floater_frac'].min():.4f}  max={result_df['floater_frac'].max():.4f}")

    from scipy.stats import pearsonr
    r, p = pearsonr(result_df["floater_frac"], result_df["angle1_deg"])
    print(f"\nfloater_frac vs angle1_deg 相关系数: Pearson r={r:.4f} (p={p:.4f}, n=5)")

    out_csv = Path("debug") / "sweep_multiframe_angle_summary.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\n已保存: {out_csv}")


if __name__ == "__main__":
    main()
