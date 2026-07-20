"""
诊断性检查（非新判定算法）：对比现有 if_keep=False 点(mark_floaters.py, 连通分量法)
和"视觉上怀疑的白色高opacity点"的重合度。

"白色高opacity点"没有现成列，本数据集里 R=G=B(灰度)，故用 R 通道代表亮度。
由于不知道用户目视判断的具体阈值，这里用多组阈值/分位数定义做敏感性对比，
而不是挑一个阈值下结论；同时输出散点图让重合关系可以直接目视核对。
不新增 if_keep 判据，不修改 mark_floaters.py 或 _marked.csv。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "postprocessing" / "cleaning" / "eda_outputs"
DATA_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_scale_reg_ratio3"
FRAME_NAMES = ["f0090", "f0091", "f0092"]

# 绝对阈值定义 (R即亮度, R=G=B)
ABS_DEFS = [
    {"label": "R>=0.7 & opacity>=0.7", "r_thresh": 0.7, "op_thresh": 0.7},
    {"label": "R>=0.8 & opacity>=0.8", "r_thresh": 0.8, "op_thresh": 0.8},
]
# 帧内分位数定义 (两个条件都取该帧内的top-K%)
PCTL_DEFS = [10, 15, 20]


def find_marked_csv(frame_name: str) -> Path:
    matches = sorted(DATA_ROOT.glob(f"{frame_name}/**/gaussian_features_{frame_name}_marked.csv"))
    if not matches:
        raise FileNotFoundError(f"no _marked.csv found for {frame_name}")
    return matches[0]


def load_frames() -> dict:
    dfs = {}
    for name in FRAME_NAMES:
        path = find_marked_csv(name)
        df = pd.read_csv(path)
        dfs[name] = df
        print(f"[load] {name}: {len(df)} points, if_keep=False count={int((~df.if_keep).sum())} <- {path}")
    return dfs


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.3g}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return float_fmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def white_high_opacity_mask_abs(df, r_thresh, op_thresh):
    return (df["R"] >= r_thresh) & (df["opacity"] >= op_thresh)


def white_high_opacity_mask_pctl(df, pctl):
    r_thresh = np.percentile(df["R"], 100 - pctl)
    op_thresh = np.percentile(df["opacity"], 100 - pctl)
    return (df["R"] >= r_thresh) & (df["opacity"] >= op_thresh)


def overlap_row(df, mask_white, label):
    floater_mask = ~df["if_keep"]
    n_white = int(mask_white.sum())
    n_floater = int(floater_mask.sum())
    n_inter = int((mask_white & floater_mask).sum())
    recall = 100 * n_inter / n_floater if n_floater else float("nan")
    precision = 100 * n_inter / n_white if n_white else float("nan")
    return {
        "definition": label,
        "n_white_high_opacity": n_white,
        "n_if_keep_false": n_floater,
        "n_overlap": n_inter,
        "pct_of_if_keep_false_that_are_white": recall,
        "pct_of_white_that_are_if_keep_false": precision,
    }


def scatter_overlap(df, mask_white, frame_name, path):
    floater_mask = (~df["if_keep"]).to_numpy()
    mask_white = mask_white.to_numpy()
    both = floater_mask & mask_white
    floater_only = floater_mask & ~mask_white
    white_only = mask_white & ~floater_mask
    neither = ~floater_mask & ~mask_white

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df.x[neither], df.y[neither], df.z[neither], s=4, alpha=0.12, color="gray",
               label=f"neither (n={neither.sum()})")
    ax.scatter(df.x[white_only], df.y[white_only], df.z[white_only], s=20, alpha=0.85, color="#d4af00",
               marker="^", label=f"white-high-opacity only (n={white_only.sum()})")
    ax.scatter(df.x[floater_only], df.y[floater_only], df.z[floater_only], s=20, alpha=0.85, color="#1f77b4",
               marker="s", label=f"if_keep=False only (n={floater_only.sum()})")
    ax.scatter(df.x[both], df.y[both], df.z[both], s=45, alpha=0.95, color="#d62728",
               marker="o", label=f"overlap: both (n={both.sum()})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{frame_name}: if_keep=False vs white-high-opacity overlap\n(white def: R>=0.7 & opacity>=0.7)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_frames()

    md_parts = ["# 诊断: if_keep=False (连通分量法) vs 白色高opacity点 重合度\n",
                "诊断性检查，非新判定算法。\"白色\"用R通道代表(本数据集R=G=B灰度)。\n",
                "阈值定义未知目视标准来源，故给出多组绝对阈值+帧内分位数定义做敏感性对比，\n",
                "并附散点图供直接目视核对重合关系。\n"]

    for name, df in dfs.items():
        md_parts.append(f"## {name} (总点数={len(df)}, if_keep=False点数={int((~df.if_keep).sum())})\n")

        rows = []
        for d in ABS_DEFS:
            mask = white_high_opacity_mask_abs(df, d["r_thresh"], d["op_thresh"])
            rows.append(overlap_row(df, mask, d["label"]))
        for pctl in PCTL_DEFS:
            mask = white_high_opacity_mask_pctl(df, pctl)
            rows.append(overlap_row(df, mask, f"per-frame top {pctl}% (R & opacity)"))
        overlap_df = pd.DataFrame(rows)
        overlap_df.to_csv(OUT_DIR / f"diag_white_opacity_overlap_{name}.csv", index=False)
        md_parts.append(df_to_md(overlap_df) + "\n")

        # scatter using the first absolute definition (R>=0.7 & opacity>=0.7) as the illustrative case
        mask_illustrative = white_high_opacity_mask_abs(df, ABS_DEFS[0]["r_thresh"], ABS_DEFS[0]["op_thresh"])
        fig_path = OUT_DIR / f"diag_white_opacity_scatter_{name}.png"
        scatter_overlap(df, mask_illustrative, name, fig_path)
        md_parts.append(f"散点图 (白色定义=R>=0.7 & opacity>=0.7): `{fig_path.name}`\n")

    summary_path = OUT_DIR / "DIAG_WHITE_OPACITY_OVERLAP.md"
    summary_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n[done] summary written to {summary_path}")


if __name__ == "__main__":
    main()
