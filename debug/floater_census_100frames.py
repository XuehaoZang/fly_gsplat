"""
对指定 group(100帧)做 mark_floaters 判据(k=10, dist_percentile=75, min_patch_size=10)的全量普查，
只统计不出图、不做可视化核查。复用 utils.gaussian_features.compute_gaussian_features (T1) 和
utils.ply.connected_component_sizes (mark_floaters 内部同一份逻辑)，不修改这两个文件。

每帧取 splatfacto-checkpoint 下按目录名排序最新的一次训练结果(与 batch_8groups_100frames.py
run_group_frame 里 splat_dir = sorted(...)[-1] 同一约定)。若该目录下没有 gaussian_features_fXXXX.csv
就用 T1 现算(不落盘覆盖已有结果之外的任何文件之外的东西)，已有就直接复用。

第10~17区间分量计数用了一个不需要改 connected_component_sizes 返回值形状的技巧：该函数对每个点
返回它所在分量的大小(同分量的点值相同)，所以某个 size=s 的值在全体点里出现的次数 c 必是 s 的整数倍，
c/s 就是恰好有多少个不同的分量是这个大小；把每个 size 展开 c/s 次即可还原完整的“分量大小多重集”，
在不touch源码、不用labels的前提下拿到 n_components / largest / second_largest / 10~17区间明细。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.gaussian_features import compute_gaussian_features
from utils.ply import connected_component_sizes

K = 10
DIST_PERCENTILE = 75.0
MIN_PATCH_SIZE = 10


def latest_checkpoint_dir(frame_dir: Path) -> Path:
    ckpt_root = frame_dir / "splatfacto-checkpoint"
    return sorted(ckpt_root.iterdir())[-1]


def load_or_compute_features(splat_dir: Path, frame_idx: int) -> pd.DataFrame:
    csv_path = splat_dir / f"gaussian_features_f{frame_idx:04d}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    df = compute_gaussian_features(splat_dir / "splat.ply", splat_dir / "dataparser_transforms.json", k=K)
    df.to_csv(csv_path, index=False)
    return df


def component_multiset(patch_size: np.ndarray) -> np.ndarray:
    """把逐点的 patch_size 数组还原成分量大小的多重集(每个分量恰好出现一次)。"""
    sizes, counts = np.unique(patch_size, return_counts=True)
    assert np.all(counts % sizes == 0), "size 的出现次数不是自身的整数倍，说明假设不成立"
    n_comp_per_size = counts // sizes
    comp_sizes = np.repeat(sizes, n_comp_per_size)
    return np.sort(comp_sizes)[::-1]


def census_frame(frame_dir: Path, frame_idx: int) -> dict:
    splat_dir = latest_checkpoint_dir(frame_dir)
    df = load_or_compute_features(splat_dir, frame_idx)
    xyz = df[["x", "y", "z"]].to_numpy()

    patch_size = connected_component_sizes(xyz, k=K, dist_percentile=DIST_PERCENTILE)
    n_total = len(patch_size)
    n_floater = int((patch_size <= MIN_PATCH_SIZE).sum())

    comp_sizes = component_multiset(patch_size)
    n_components = len(comp_sizes)
    largest = int(comp_sizes[0])
    second_largest = int(comp_sizes[1]) if n_components > 1 else largest

    mid_mask = (comp_sizes > 10) & (comp_sizes < 17)
    mid_sizes = sorted(comp_sizes[mid_mask].tolist())

    return {
        "frame": f"f{frame_idx:04d}",
        "n_total": n_total,
        "n_floater": n_floater,
        "floater_ratio_pct": round(100 * n_floater / n_total, 3),
        "n_components": n_components,
        "largest_comp": largest,
        "second_largest_comp": second_largest,
        "n_mid_10_17": len(mid_sizes),
        "mid_10_17_sizes": mid_sizes,
        "mid_10_17_points": int(sum(mid_sizes)),
    }


def run_census(dataset_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    for frame_idx in range(100):
        frame_dir = dataset_dir / f"f{frame_idx:04d}"
        rows.append(census_frame(frame_dir, frame_idx))
        print(f"[{rows[-1]['frame']}] n_total={rows[-1]['n_total']} "
              f"floater_ratio={rows[-1]['floater_ratio_pct']}% "
              f"n_components={rows[-1]['n_components']} "
              f"mid_10_17={rows[-1]['mid_10_17_sizes']}")
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")
    return out


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    group = sys.argv[1] if len(sys.argv) > 1 else "G2b_G9"
    dataset_dir = repo / "outputs" / "ctrl_009_002_8groups_100frames" / group
    out_csv = repo / "debug" / f"floater_census_100frames_{group}.csv"
    run_census(dataset_dir, out_csv)
