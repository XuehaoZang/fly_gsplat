"""
可复用的floater标记人工核查渲染脚本。不改动mark_floaters.py的判据逻辑，只做可视化。

输入任意一帧号，自动定位(或按mark_floaters.py同样的规则现算并落盘)该帧的_marked表，
输出:
1) 多角度3D点云图 (正视 + 俯视 + 翼缘特写)，if_keep=False的点标红，其余保持原RGB
2) 翼缘区域局部放大图 (x-y / y-z 两个投影，聚焦在离质心最远的一批点所在的bbox内)

翼缘区域取法: 在if_keep=True的点里找离质心最远的一个点(翼尖候选)，再取其最近的N个
近邻点框定局部bbox(非精确分割，见mark_floaters.py顶部说明: 真实的翼缘尖刺在单帧特征
空间/精确分割上和floater分不开，这里只是给人工核查提供一个"大概率能看到翼缘"的取景框)。

用法:
    python -m postprocessing.cleaning.viz_floater_check --frame f0090
    python -m postprocessing.cleaning.viz_floater_check --frame 90
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.spatial import cKDTree

from postprocessing.cleaning.mark_floaters import mark_floaters

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_scale_reg_ratio3"
DEFAULT_OUT_DIR = REPO_ROOT / "postprocessing" / "cleaning" / "eda_outputs"

COLOR_DROP = to_rgb("#d62728")
WING_TIP_NEIGHBORS = 40   # 翼缘特写取"离质心最远的点"周围最近的N个点作为局部窗口
WING_PAD_FRAC = 0.25      # 翼缘bbox外扩比例，避免边界点被裁掉


def normalize_frame_name(frame: str) -> str:
    frame = str(frame)
    if frame.startswith("f"):
        return frame
    return f"f{int(frame):04d}"


def find_features_csv(frame: str, data_root: Path) -> Path:
    matches = sorted(
        p for p in data_root.glob(f"{frame}/**/gaussian_features_{frame}.csv")
        if not p.stem.endswith("_marked")
    )
    if not matches:
        raise FileNotFoundError(f"no gaussian_features csv found for {frame} under {data_root}")
    return matches[0]


def load_marked(frame: str, data_root: Path = DEFAULT_DATA_ROOT) -> tuple[pd.DataFrame, Path]:
    """加载某帧的_marked表；不存在就用mark_floaters()现算(判据不变)并按同样命名落盘。"""
    features_csv = find_features_csv(frame, data_root)
    marked_csv = features_csv.with_name(features_csv.stem + "_marked.csv")
    if marked_csv.exists():
        df = pd.read_csv(marked_csv)
    else:
        df = mark_floaters(pd.read_csv(features_csv))
        df.to_csv(marked_csv, index=False)
        print(f"[mark] {frame}: 未找到已有_marked.csv，现算并保存 -> {marked_csv}")
    return df, marked_csv


def point_colors(df: pd.DataFrame) -> np.ndarray:
    rgb = np.clip(df[["R", "G", "B"]].to_numpy(dtype=float), 0.0, 1.0)
    drop_mask = ~df["if_keep"].to_numpy()
    rgb[drop_mask] = COLOR_DROP
    return rgb


def wing_edge_bbox(df: pd.DataFrame, n_neighbors: int = WING_TIP_NEIGHBORS,
                    pad_frac: float = WING_PAD_FRAC):
    """粗略框出翼缘区域: 在if_keep=True(真实结构)的点里取离质心最远的一个点作为翼尖候选
    (不用全体点的最远点，因为那经常就是孤立噪点本身，会把局部窗口拉得很大)，
    再从全体点(含floater)里取它周围最近的n_neighbors个点，用这批点的空间bbox(留边距)
    近似翼缘局部窗口。这样窗口里既有真实翼缘结构也能看到贴着它的floater点。
    只锁定一侧翼缘，不做精确分割。"""
    xyz = df[["x", "y", "z"]].to_numpy()
    kept_mask = df["if_keep"].to_numpy()
    kept_idx = np.flatnonzero(kept_mask)
    tip_idx = kept_idx[df.loc[kept_mask, "dist_to_centroid"].to_numpy().argmax()]
    tree = cKDTree(xyz)
    k = min(n_neighbors, len(df))
    _, nbr_idx = tree.query(xyz[tip_idx], k=k)
    edge_mask = np.zeros(len(df), dtype=bool)
    edge_mask[nbr_idx] = True
    edge_pts = xyz[edge_mask]
    mins, maxs = edge_pts.min(axis=0), edge_pts.max(axis=0)
    pad = (maxs - mins) * pad_frac
    pad = np.where(pad < 1e-9, (xyz.max(axis=0) - xyz.min(axis=0)) * 0.02, pad)
    return mins - pad, maxs + pad, edge_mask


def points_in_bbox(df: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    xyz = df[["x", "y", "z"]].to_numpy()
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


def plot_multiview(df: pd.DataFrame, colors: np.ndarray, frame: str, out_path: Path,
                    lo: np.ndarray, hi: np.ndarray, edge_mask: np.ndarray) -> None:
    n_total = len(df)
    n_drop = int((~df["if_keep"]).sum())
    xyz = df[["x", "y", "z"]].to_numpy()

    views = [
        ("front (elev=0, azim=-90)", dict(elev=0, azim=-90), None),
        ("top (elev=90, azim=-90)", dict(elev=90, azim=-90), None),
        (f"wing-edge close-up (n_in_box={int(points_in_bbox(df, lo, hi).sum())})",
         dict(elev=20, azim=-60), (lo, hi)),
    ]

    fig = plt.figure(figsize=(17, 5.5))
    for i, (title, view_kw, lims) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=14, alpha=0.9, depthshade=False)
        ax.view_init(**view_kw)
        if lims is not None:
            lo_, hi_ = lims
            ax.set_xlim(lo_[0], hi_[0]); ax.set_ylim(lo_[1], hi_[1]); ax.set_zlim(lo_[2], hi_[2])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        f"{frame}: red = if_keep=False (floater), else original RGB  |  "
        f"n_total={n_total}  n_floater={n_drop} ({100 * n_drop / n_total:.1f}%)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wing_zoom(df: pd.DataFrame, colors: np.ndarray, frame: str, out_path: Path,
                    lo: np.ndarray, hi: np.ndarray, n_neighbors: int) -> None:
    in_box = points_in_bbox(df, lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (xa, ya) in zip(axes, [("x", "y"), ("y", "z")]):
        xv, yv = df[xa].to_numpy(), df[ya].to_numpy()
        ax.scatter(xv[~in_box], yv[~in_box], s=6, alpha=0.10, color="gray", label="outside wing-edge box")
        ax.scatter(xv[in_box], yv[in_box], c=colors[in_box], s=32, alpha=0.95,
                    edgecolor="k", linewidth=0.3, label="inside wing-edge box")
        ax.set_xlabel(xa); ax.set_ylabel(ya)
        ax.set_title(f"{xa}-{ya} projection")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"{frame}: wing-edge close-up  "
        f"(farthest-from-centroid point + {n_neighbors} nearest neighbors, n_in_box={int(in_box.sum())}, "
        f"n_floater_in_box={int((in_box & ~df['if_keep'].to_numpy()).sum())})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(frame: str, data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR,
        wing_tip_neighbors: int = WING_TIP_NEIGHBORS) -> dict:
    frame = normalize_frame_name(frame)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, marked_csv = load_marked(frame, data_root)
    colors = point_colors(df)
    lo, hi, edge_mask = wing_edge_bbox(df, n_neighbors=wing_tip_neighbors)

    multiview_path = out_dir / f"floater_check_{frame}_multiview.png"
    zoom_path = out_dir / f"floater_check_{frame}_wingzoom.png"
    plot_multiview(df, colors, frame, multiview_path, lo, hi, edge_mask)
    plot_wing_zoom(df, colors, frame, zoom_path, lo, hi, wing_tip_neighbors)

    n_total = len(df)
    n_floater = int((~df["if_keep"]).sum())
    print(f"[{frame}] n_total={n_total}  n_floater={n_floater} ({100 * n_floater / n_total:.1f}%)")
    print(f"[{frame}] marked csv   -> {marked_csv}")
    print(f"[{frame}] multiview    -> {multiview_path}")
    print(f"[{frame}] wing zoom    -> {zoom_path}")

    return {
        "frame": frame, "df": df, "marked_csv": marked_csv,
        "multiview_path": multiview_path, "zoom_path": zoom_path,
        "n_total": n_total, "n_floater": n_floater,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=str, required=True, help="帧号，如 f0090 或 90")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                         help="存放各帧gaussian_features_*.csv的根目录")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="图像输出目录")
    parser.add_argument("--wing-tip-neighbors", type=int, default=WING_TIP_NEIGHBORS,
                         help="翼缘特写窗口大小：取离质心最远点周围最近的N个点")
    args = parser.parse_args()

    run(args.frame, Path(args.data_root), Path(args.out_dir), args.wing_tip_neighbors)


if __name__ == "__main__":
    main()
