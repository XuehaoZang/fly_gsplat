"""
把 compare_train_steps_f0000_more_groups.py 里 baseline vs G2b_scale_reg_ratio3 的
单帧(f0000)对比逻辑扩展到全部可用帧，判断 ratio3 能否作为后续 pipeline 的新基线数据集。

沿用同一套字段/口径，不重新定义指标：
- floater 代理: dist_to_centroid > 该帧p90 且 local_density < 该帧p10 的点占比
  (定义来自 compare_train_steps_f0000_more_groups.py 的 print_floater_check)
- scale_ratio / linearity / planarity / sphericity: 来自 utils/gaussian_features.py
- "高planarity占比"(翼面结构是否解出的代理): 全局 planarity>0.3 占比，
  以及远端(dist_to_centroid前10%候选翼尖/肢端)点里 planarity>0.3 占比
  (阈值 PLANARITY_HIGH_TH=0.3、FAR_PCTL=90 与之前单帧脚本保持一致)

新增(之前单帧脚本没测过):
- color_oob 占比 (f_dc 还原RGB后越界的点比例)，同帧 baseline vs ratio3 对比
- 跨帧稳定性: floater代理占比、color_oob占比在100帧上的分布
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.gaussian_features import compute_gaussian_features

BASELINE_ROOT = Path("outputs/ctrl_009_002")
RATIO3_ROOT = Path("outputs/ctrl_009_002_8groups_100frames/G2b_scale_reg_ratio3")
MAX_FRAME = 100  # 覆盖范围 f0000-f0099

PLANARITY_HIGH_TH = 0.3
FAR_PCTL = 90

OUT_DIR = Path(__file__).resolve().parent / "_compare_train_steps_ratio3_allframes"
OUT_DIR.mkdir(exist_ok=True)


def find_frame_dir(root: Path, frame_idx: int) -> Path | None:
    frame_dir = root / f"f{frame_idx:04d}" / "splatfacto-checkpoint"
    if not frame_dir.exists():
        return None
    timestamps = sorted(frame_dir.iterdir())
    if not timestamps:
        return None
    splat_dir = timestamps[-1]
    return splat_dir if (splat_dir / "splat.ply").exists() else None


def list_available_frames(root: Path) -> dict:
    out = {}
    for i in range(MAX_FRAME):
        d = find_frame_dir(root, i)
        if d is not None:
            out[i] = d
    return out


def load_or_compute(splat_dir: Path, frame_idx: int) -> pd.DataFrame:
    csv_path = splat_dir / f"gaussian_features_f{frame_idx:04d}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    df = compute_gaussian_features(splat_dir / "splat.ply", splat_dir / "dataparser_transforms.json", k=10)
    df.to_csv(csv_path, index=False)
    return df


def frame_metrics(df: pd.DataFrame) -> dict:
    dist_th = np.percentile(df["dist_to_centroid"], FAR_PCTL)
    density_th = np.percentile(df["local_density"], 10)
    floater_mask = (df["dist_to_centroid"] > dist_th) & (df["local_density"] < density_th)
    far_df = df[df["dist_to_centroid"] > dist_th]
    return {
        "n": len(df),
        "floater_frac": floater_mask.mean(),
        "scale_ratio_median": df["scale_ratio"].median(),
        "linearity_median": df["linearity"].median(),
        "planarity_median": df["planarity"].median(),
        "sphericity_median": df["sphericity"].median(),
        "planarity_high_frac": (df["planarity"] > PLANARITY_HIGH_TH).mean(),
        "far_planarity_high_frac": (far_df["planarity"] > PLANARITY_HIGH_TH).mean() if len(far_df) else np.nan,
        "opacity_median": df["opacity"].median(),
        "color_oob_frac": df["color_oob"].mean(),
    }


def main():
    base_frames = list_available_frames(BASELINE_ROOT)
    ratio3_frames = list_available_frames(RATIO3_ROOT)
    common = sorted(set(base_frames) & set(ratio3_frames))

    print(f"baseline 可用帧数 = {len(base_frames)}  ratio3 可用帧数 = {len(ratio3_frames)}  "
          f"交集帧数 = {len(common)}")
    missing_in_ratio3 = sorted(set(base_frames) - set(ratio3_frames))
    missing_in_base = sorted(set(ratio3_frames) - set(base_frames))
    if missing_in_ratio3:
        print(f"仅baseline有、ratio3缺失的帧: {missing_in_ratio3}")
    if missing_in_base:
        print(f"仅ratio3有、baseline缺失的帧: {missing_in_base}")

    rows = []
    for idx in common:
        df_base = load_or_compute(base_frames[idx], idx)
        df_ratio3 = load_or_compute(ratio3_frames[idx], idx)
        m_base = frame_metrics(df_base)
        m_ratio3 = frame_metrics(df_ratio3)

        row = {"frame": idx}
        for k, v in m_base.items():
            row[f"{k}_baseline"] = v
        for k, v in m_ratio3.items():
            row[f"{k}_ratio3"] = v
        row["n_ratio"] = m_ratio3["n"] / m_base["n"] if m_base["n"] else np.nan
        row["floater_ratio"] = (m_ratio3["floater_frac"] / m_base["floater_frac"]
                                 if m_base["floater_frac"] > 0 else np.nan)
        row["planarity_high_frac_diff"] = m_ratio3["planarity_high_frac"] - m_base["planarity_high_frac"]
        row["far_planarity_high_frac_diff"] = (m_ratio3["far_planarity_high_frac"]
                                                - m_base["far_planarity_high_frac"])
        row["color_oob_frac_diff"] = m_ratio3["color_oob_frac"] - m_base["color_oob_frac"]
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("frame")
    csv_path = OUT_DIR / "summary_allframes.csv"
    summary.to_csv(csv_path)
    print(f"\n逐帧汇总表已保存: {csv_path}  (shape={summary.shape})")

    print("\n" + "=" * 70)
    print("1. 关键列预览 (前5帧 + 后5帧)")
    print("=" * 70)
    key_cols = ["n_baseline", "n_ratio3", "n_ratio",
                "floater_frac_baseline", "floater_frac_ratio3", "floater_ratio",
                "planarity_median_baseline", "planarity_median_ratio3",
                "sphericity_median_baseline", "sphericity_median_ratio3",
                "color_oob_frac_baseline", "color_oob_frac_ratio3"]
    with pd.option_context("display.float_format", lambda x: f"{x:.4g}", "display.width", 200):
        print(summary[key_cols].head(5))
        print("...")
        print(summary[key_cols].tail(5))

    print("\n" + "=" * 70)
    print("2. 跨帧稳定性 —— floater代理占比 (baseline vs ratio3, 100帧)")
    print("=" * 70)
    for tag in ["baseline", "ratio3"]:
        s = summary[f"floater_frac_{tag}"]
        print(f"[{tag}] min={s.min():.4f} p10={s.quantile(.1):.4f} median={s.median():.4f} "
              f"p90={s.quantile(.9):.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}")

    print("\n" + "=" * 70)
    print("3. 颜色收敛质量 —— color_oob占比 (baseline vs ratio3, 100帧)")
    print("=" * 70)
    for tag in ["baseline", "ratio3"]:
        s = summary[f"color_oob_frac_{tag}"]
        print(f"[{tag}] min={s.min():.4f} p10={s.quantile(.1):.4f} median={s.median():.4f} "
              f"p90={s.quantile(.9):.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}")

    print("\n" + "=" * 70)
    print("4. Wing结构解出情况 —— planarity_high_frac / far_planarity_high_frac (100帧统计)")
    print("=" * 70)
    for col_prefix in ["planarity_high_frac", "far_planarity_high_frac"]:
        for tag in ["baseline", "ratio3"]:
            s = summary[f"{col_prefix}_{tag}"]
            print(f"[{col_prefix}_{tag}] median={s.median():.4f} mean={s.mean():.4f} "
                  f"min={s.min():.4f} max={s.max():.4f}")

    print("\n" + "=" * 70)
    print("5. 异常帧检测 (相对ratio3自身100帧分布的z-score, |z|>2 标记)")
    print("=" * 70)
    flag_specs = [
        ("floater_frac_ratio3", "floater代理占比偏高", 2.0, "high"),
        ("sphericity_median_ratio3", "sphericity中位数偏高(疑似坍缩成球)", 2.0, "high"),
        ("color_oob_frac_ratio3", "color_oob占比偏高", 2.0, "high"),
        ("n_ratio", "点数相对baseline异常偏低", 2.0, "low"),
        ("far_planarity_high_frac_ratio3", "远端薄片占比异常偏低(翼面疑似未解出)", 2.0, "low"),
    ]
    anomalies = {}
    for col, desc, z_th, direction in flag_specs:
        s = summary[col]
        mu, sigma = s.mean(), s.std()
        if sigma == 0 or np.isnan(sigma):
            continue
        z = (s - mu) / sigma
        if direction == "high":
            flagged = summary.index[z > z_th].tolist()
        else:
            flagged = summary.index[z < -z_th].tolist()
        if flagged:
            anomalies[col] = (desc, flagged)
            print(f"[{col}] {desc}: 帧 {flagged}")
            for f in flagged:
                print(f"    f{f:04d}: {col}={summary.loc[f, col]:.4f}  (整体mean={mu:.4f}, std={sigma:.4f})")
        else:
            print(f"[{col}] {desc}: 无异常帧")

    # 跨帧稳定性可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(summary.index, summary["floater_frac_baseline"], marker="o", ms=3, label="baseline", alpha=0.7)
    axes[0].plot(summary.index, summary["floater_frac_ratio3"], marker="o", ms=3, label="ratio3", alpha=0.7)
    axes[0].set_ylabel("floater_frac")
    axes[0].legend()
    axes[0].set_title("floater代理占比 —— 逐帧对比")

    axes[1].plot(summary.index, summary["color_oob_frac_baseline"], marker="o", ms=3, label="baseline", alpha=0.7)
    axes[1].plot(summary.index, summary["color_oob_frac_ratio3"], marker="o", ms=3, label="ratio3", alpha=0.7)
    axes[1].set_ylabel("color_oob_frac")
    axes[1].set_xlabel("frame")
    axes[1].legend()
    axes[1].set_title("color_oob占比 —— 逐帧对比")

    fig.tight_layout()
    out_path = OUT_DIR / "cross_frame_stability.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n跨帧稳定性图已保存: {out_path}")

    fig2, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(summary.index, summary["planarity_median_baseline"], marker="o", ms=3, label="baseline", alpha=0.7)
    ax[0].plot(summary.index, summary["planarity_median_ratio3"], marker="o", ms=3, label="ratio3", alpha=0.7)
    ax[0].set_title("planarity_median 逐帧对比")
    ax[0].legend()
    ax[1].plot(summary.index, summary["sphericity_median_baseline"], marker="o", ms=3, label="baseline", alpha=0.7)
    ax[1].plot(summary.index, summary["sphericity_median_ratio3"], marker="o", ms=3, label="ratio3", alpha=0.7)
    ax[1].set_title("sphericity_median 逐帧对比")
    ax[1].legend()
    fig2.tight_layout()
    out_path2 = OUT_DIR / "cross_frame_shape_metrics.png"
    fig2.savefig(out_path2, dpi=140)
    plt.close(fig2)
    print(f"形状指标逐帧对比图已保存: {out_path2}")


if __name__ == "__main__":
    main()
