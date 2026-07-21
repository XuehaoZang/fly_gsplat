"""
G2b_G9 阈值校准：复用 floater_census_100frames.py (component_multiset /
latest_checkpoint_dir / load_or_compute_features) 和
postprocessing/cleaning/eda_features.py (knn_component_labels) 里已有的逻辑，
不新开发判定/连通分量算法，只对选定的6帧跑一遍对比。

样本帧:
- 正常帧 x3: floater_ratio_pct 最接近100帧均值(9.51%)且 10~17 gap为空的帧
  -> f0058, f0069, f0009 (来自 floater_census_100frames_G2b_G9.csv)
- gap非空帧 x3: 21帧gap非空普查里 size 最接近10(=11)的帧
  -> f0002, f0007, f0014
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from postprocessing.cleaning.floater_census_100frames import (  # noqa: E402
    K, DIST_PERCENTILE, MIN_PATCH_SIZE,
    latest_checkpoint_dir, load_or_compute_features, component_multiset,
)
from postprocessing.cleaning.eda_features import knn_component_labels  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"

NORMAL_FRAMES = ["f0058", "f0069", "f0009"]
GAP_FRAMES = ["f0002", "f0007", "f0014"]
ALL_FRAMES = NORMAL_FRAMES + GAP_FRAMES

K_GRID = [8, 10, 12, 15]
PCTL_GRID = [70, 75, 80]


def load_xyz(frame_name: str):
    frame_idx = int(frame_name[1:])
    frame_dir = DATASET_DIR / frame_name
    splat_dir = latest_checkpoint_dir(frame_dir)
    df = load_or_compute_features(splat_dir, frame_idx)
    return df[["x", "y", "z"]].to_numpy()


def report_component_sizes(frame_name: str, xyz):
    """当前参数(k=10, pctl=75)下的完整分量size列表，标出10/17参考线位置。"""
    labels = knn_component_labels(xyz, K, DIST_PERCENTILE)
    sizes = pd.Series(labels).value_counts().to_numpy()
    sizes_sorted = sorted(sizes.tolist(), reverse=True)
    n_le_10 = sum(1 for s in sizes_sorted if s <= 10)
    n_gap = sum(1 for s in sizes_sorted if 10 < s < 17)
    n_ge_17 = sum(1 for s in sizes_sorted if s >= 17)
    print(f"\n[{frame_name}] n_components={len(sizes_sorted)}  "
          f"(<=10: {n_le_10} 个, 10~17 gap: {n_gap} 个, >=17: {n_ge_17} 个)")
    print(f"  sizes (desc): {sizes_sorted}")
    return sizes_sorted


def param_grid_for_frame(frame_name: str, xyz) -> pd.DataFrame:
    n = len(xyz)
    rows = []
    for k in K_GRID:
        for pctl in PCTL_GRID:
            labels = knn_component_labels(xyz, k, pctl)
            comp_sizes = pd.Series(labels).value_counts().to_numpy()
            n_components = len(comp_sizes)
            small = comp_sizes[comp_sizes <= MIN_PATCH_SIZE]
            n_points_in_small = int(small.sum())
            gap_sizes = sorted(comp_sizes[(comp_sizes > 10) & (comp_sizes < 17)].tolist())
            rows.append({
                "frame": frame_name, "k": k, "dist_percentile": pctl,
                "n_components": n_components,
                "pct_points_in_small": round(100 * n_points_in_small / n, 3),
                "gap_10_17_sizes": gap_sizes,
            })
    return pd.DataFrame(rows)


def main():
    xyz_by_frame = {f: load_xyz(f) for f in ALL_FRAMES}

    print("=" * 70)
    print("1) 当前参数 (k=10, pctl=75) 下的完整分量size分布")
    print("=" * 70)
    for f in ALL_FRAMES:
        report_component_sizes(f, xyz_by_frame[f])

    print("\n" + "=" * 70)
    print("2) 参数网格 k x dist_percentile")
    print("=" * 70)
    all_grid = pd.concat([param_grid_for_frame(f, xyz_by_frame[f]) for f in ALL_FRAMES],
                          ignore_index=True)
    out_csv = Path(__file__).resolve().parent / "calibrate_G2b_G9_param_grid.csv"
    all_grid.to_csv(out_csv, index=False)
    print(all_grid.to_string(index=False))
    print(f"\n[Saved] {out_csv}")


if __name__ == "__main__":
    main()
