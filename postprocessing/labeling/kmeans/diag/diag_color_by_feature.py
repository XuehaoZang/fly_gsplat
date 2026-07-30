"""
T3 body/wing 特征选型诊断：planarity/scale_ratio/opacity 里，planarity 倾向保留，
scale_ratio/sphericity 已确定暂不用，opacity 是否可用还需要肉眼核实——尤其
opacity 直方图里 1.0 附近的尖峰，不确定是 sigmoid 饱和的 artifact 还是真实的
body/wing 区分信号。

本脚本纯出图诊断，不做任何判定/阈值化/统计检验，不产出标签列，不改动
kmeans_split.py / binary/binary_split.py。画法复用
binary/diag/diag_principal_axis.py 的 plot_axis_diag 同款风格
(前视 elev=0,azim=-90 + 俯视 elev=90,azim=-90 双视角，连续值colormap叠加3D散点)。

在 select_dev_frames.DEV_FRAMES 每一帧上，用 kmeans_split.load_kept (if_keep=True
的点，同一份数据口径) 出三张图:
1. color = planarity，额外把 planarity >= PLANARITY_HIGHLIGHT_TH 的疑点单独
   加大 + 描边高亮出来(不改变颜色映射本身，只是显式标出这批点)，方便肉眼看
   它们在空间上到底落在 body 还是 wing 区域。
2. color = opacity，额外把 opacity >= OPACITY_HIGHLIGHT_TH 的点单独加大 + 描边
   高亮出来(不改变颜色映射本身，只是显式标出这批疑似"饱和"的点)，方便肉眼看
   它们在空间上落在 body 还是 wing 区域。
3. color = 该点自己的真实(R,G,B)颜色(不是colormap映射的代理值，是marked表里的
   真实颜色属性)，跟上面两张平行对照。marked表里 R==G==B(灰度贴图)，这张图看
   起来会是灰阶的，但用的是真实颜色通道而不是colormap。额外把 R < R_HIGHLIGHT_TH
   (偏暗/疑似阴影或翼膜半透明处)的点单独加大 + 描边高亮出来(颜色仍是真实RGB，
   只是显式标出这批点)。

图存到 eda_outputs/color_by_feature/ 子目录。

用法:
    python -m postprocessing.labeling.kmeans.diag.diag_color_by_feature
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.labeling.kmeans.kmeans_split import load_kept  # noqa: E402
from postprocessing.labeling.kmeans.diag.select_dev_frames import DEV_FRAMES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs" / "color_by_feature"

VIEWS = [
    ("front (elev=0, azim=-90)", dict(elev=0, azim=-90)),
    ("top (elev=90, azim=-90)", dict(elev=90, azim=-90)),
]

OPACITY_HIGHLIGHT_TH = 0.98    # 单纯用于在图上标出"疑似饱和"的点，不是分类阈值
PLANARITY_HIGHLIGHT_TH = 0.45  # 单纯用于在图上标出高planarity疑点，不是分类阈值
R_HIGHLIGHT_TH = 0.2           # 单纯用于在rgb图上标出R偏暗的疑点，不是分类阈值


def plot_color_by_with_highlight(xyz: np.ndarray, values: np.ndarray, feature_name: str,
                                  frame: str, out_path: Path, highlight_th: float) -> int:
    """连续值colormap散点图(前视+俯视)，额外把 values>=highlight_th 的点用更大尺寸+红色
    描边叠加一层高亮层(颜色映射本身不变)，方便肉眼看这批点落在body还是wing区域。"""
    highlight_mask = values >= highlight_th
    vmin, vmax = float(values.min()), float(values.max())

    fig = plt.figure(figsize=(12, 5.5))
    sc = None
    for i, (title, view_kw) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values, cmap="viridis",
                         vmin=vmin, vmax=vmax, s=14, alpha=0.9, depthshade=False)
        if highlight_mask.any():
            ax.scatter(xyz[highlight_mask, 0], xyz[highlight_mask, 1], xyz[highlight_mask, 2],
                       c=values[highlight_mask], cmap="viridis", vmin=vmin, vmax=vmax,
                       s=70, alpha=1.0, depthshade=False,
                       edgecolors="red", linewidths=1.3)
        ax.view_init(**view_kw)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)
    fig.colorbar(sc, ax=fig.axes, shrink=0.7, label=feature_name)
    n_highlight = int(highlight_mask.sum())
    fig.suptitle(
        f"{frame}: color = {feature_name} (if_keep=True points, n={len(xyz)})  |  "
        f"red-ring = {feature_name}>={highlight_th} (n={n_highlight})"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n_highlight


def plot_color_by_rgb(xyz: np.ndarray, rgb: np.ndarray, frame: str, out_path: Path,
                       highlight_th: float) -> int:
    """跟plot_color_by_with_highlight同款双视角布局，但用点自己的真实(R,G,B)做散点
    facecolor，不经过colormap映射，也没有colorbar(颜色本身就是真值)。额外把
    R < highlight_th 的点用更大尺寸+红色描边叠加一层高亮(facecolor仍是真实RGB，
    只是显式标出这批点)。"""
    rgb_clipped = np.clip(rgb, 0.0, 1.0)
    highlight_mask = rgb[:, 0] < highlight_th

    fig = plt.figure(figsize=(12, 5.5))
    for i, (title, view_kw) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb_clipped, s=14, alpha=0.9,
                   depthshade=False)
        if highlight_mask.any():
            ax.scatter(xyz[highlight_mask, 0], xyz[highlight_mask, 1], xyz[highlight_mask, 2],
                       c=rgb_clipped[highlight_mask], s=70, alpha=1.0, depthshade=False,
                       edgecolors="red", linewidths=1.3)
        ax.view_init(**view_kw)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)
    n_highlight = int(highlight_mask.sum())
    fig.suptitle(
        f"{frame}: color = true point (R,G,B) (if_keep=True points, n={len(xyz)})  |  "
        f"red-ring = R<{highlight_th} (n={n_highlight})"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n_highlight


def process_frame(frame: str) -> None:
    df = load_kept(frame)
    xyz = df[["x", "y", "z"]].to_numpy()

    planarity = df["planarity"].to_numpy()
    planarity_path = OUT_DIR / f"diag_color_by_planarity_{frame}.png"
    n_hi_planarity = plot_color_by_with_highlight(
        xyz, planarity, "planarity", frame, planarity_path, PLANARITY_HIGHLIGHT_TH)

    opacity = df["opacity"].to_numpy()
    opacity_path = OUT_DIR / f"diag_color_by_opacity_{frame}.png"
    n_hi_opacity = plot_color_by_with_highlight(
        xyz, opacity, "opacity", frame, opacity_path, OPACITY_HIGHLIGHT_TH)

    rgb = df[["R", "G", "B"]].to_numpy()
    rgb_path = OUT_DIR / f"diag_color_by_rgb_{frame}.png"
    n_hi_rgb = plot_color_by_rgb(xyz, rgb, frame, rgb_path, R_HIGHLIGHT_TH)

    print(f"[{frame}] n={len(df)}  "
          f"planarity>={PLANARITY_HIGHLIGHT_TH}: n={n_hi_planarity} "
          f"({100 * n_hi_planarity / len(df):.1f}%)  "
          f"opacity>={OPACITY_HIGHLIGHT_TH}: n={n_hi_opacity} "
          f"({100 * n_hi_opacity / len(df):.1f}%)  "
          f"R<{R_HIGHLIGHT_TH}: n={n_hi_rgb} ({100 * n_hi_rgb / len(df):.1f}%)")
    print(f"  -> {planarity_path}")
    print(f"  -> {opacity_path}")
    print(f"  -> {rgb_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame in DEV_FRAMES:
        process_frame(frame)
    print(f"\n[done] {len(DEV_FRAMES)} 帧 x 3 图，纯诊断出图，肉眼判断请看 "
          f"{OUT_DIR}/diag_color_by_{{planarity,opacity,rgb}}_*.png")


if __name__ == "__main__":
    main()
