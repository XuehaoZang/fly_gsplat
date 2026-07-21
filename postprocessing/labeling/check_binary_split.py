"""
T3 第一步 binary_split.classify_body_wing 的内部自查脚本(非最终重投影验收图)。

对 select_dev_frames.DEV_FRAMES 里的每一帧:
- 只在 if_keep=True(排除floater)的点里跑 classify_body_wing_quantile
- 打印 wing/body 点数和占比
- 出一张多角度3D散点图(正视 + 俯视，画法参考
  postprocessing/cleaning/viz_floater_check.py 的 plot_multiview): body一种颜色，
  wing另一种颜色
- 打印颜色诊断(仅供参考，不参与判据)

图存到 postprocessing/labeling/eda_outputs/ 下。

用法:
    python -m postprocessing.labeling.check_binary_split
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from postprocessing.cleaning.viz_floater_check import find_features_csv
from postprocessing.labeling.binary_split import (
    classify_body_wing_quantile, print_color_diagnostics,
    DEFAULT_PLANARITY_Q, DEFAULT_AXIS_DIST_Q,
)
from postprocessing.labeling.select_dev_frames import DEV_FRAMES, DATASET_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "postprocessing" / "labeling" / "eda_outputs"

COLOR_BODY = to_rgb("#1f77b4")
COLOR_WING = to_rgb("#ff7f0e")


def load_kept(frame: str) -> pd.DataFrame:
    """加载该帧的_marked表，只取if_keep=True的点(body/wing二分类只在真实结构点上做，
    不碰if_keep=False点的传播逻辑，那是S2/S3的事)。"""
    features_csv = find_features_csv(frame, DATASET_DIR)
    marked_csv = features_csv.with_name(features_csv.stem + "_marked.csv")
    df = pd.read_csv(marked_csv)
    return df[df["if_keep"]].reset_index(drop=True)


def plot_body_wing(df: pd.DataFrame, is_wing: pd.Series, frame: str, out_path: Path) -> None:
    xyz = df[["x", "y", "z"]].to_numpy()
    colors = np.where(is_wing.to_numpy()[:, None], COLOR_WING, COLOR_BODY)
    n_total = len(df)
    n_wing = int(is_wing.sum())

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

    fig.suptitle(
        f"{frame}: orange = is_wing (planarity_q={DEFAULT_PLANARITY_Q}, "
        f"axis_dist_q={DEFAULT_AXIS_DIST_Q}), blue = body  |  "
        f"n_total={n_total}  n_wing={n_wing} ({100 * n_wing / n_total:.1f}%)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(frame: str) -> dict:
    df = load_kept(frame)
    is_wing = classify_body_wing_quantile(df)
    n_total = len(df)
    n_wing = int(is_wing.sum())

    out_path = OUT_DIR / f"binary_split_check_{frame}.png"
    plot_body_wing(df, is_wing, frame, out_path)

    print(f"[{frame}] n_total={n_total}  n_wing={n_wing} ({100 * n_wing / n_total:.1f}%)  "
          f"n_body={n_total - n_wing} ({100 * (n_total - n_wing) / n_total:.1f}%)")
    print_color_diagnostics(df, is_wing)
    print(f"[{frame}] plot -> {out_path}")

    return {"frame": frame, "n_total": n_total, "n_wing": n_wing}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in DEV_FRAMES:
        run(f)


if __name__ == "__main__":
    main()
