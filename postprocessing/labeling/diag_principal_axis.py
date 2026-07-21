"""
T3诊断: 检验"dist_to_principal_axis假设反了"的疑虑。

背景: binary_split.py 的 body/wing 判据里，dist_to_principal_axis 是相对
"全体点(body+两翼)"算出的全局第一主轴的径向距离，原假设"离轴远=wing"。但如果
翅膀展开造成的方差比body本身更大，这条全局主轴可能顺着"翼尖到翼尖"方向而不是
"body头尾"方向——这样离轴远的点反而是紧凑的body而不是wing，和原假设正好相反。

本脚本只做诊断，不改动T1/T2/binary_split.py的判据，不产出新标签列。
在 select_dev_frames.DEV_FRAMES 上跑，每帧:

1. 读取该帧marked表全部点(if_keep不筛，先看全貌)，重新算一次全局PCA(算法与
   utils/gaussian_features.py compute_gaussian_features一致: 对xyz做cov+eigh，
   取最大特征值对应的特征向量作principal_axis)，打印方向向量、第一主轴解释方差
   占比(eigval1/sum(eigvals))，并跟该帧marked表里既有的dist_to_principal_axis列
   做一致性检查(算法/数据一样理应高度吻合)。
2. 用binary_split.classify_body_wing_quantile在if_keep=True点上跑出的初步is_wing
   标签分组(仅供参考，is_wing本身可能不准，不作为下面核心判断的依据)，对比body
   候选组/wing候选组各自的空间延展(bbox对角线、平均质心距离)。
3. 核心证据: 不做二值化，把dist_to_principal_axis当连续值，用colormap画在3D散点图
   上(前视+俯视，画法参考check_binary_split.py的风格)，图存到eda_outputs/，肉眼
   判断颜色深(离轴远)的点落在"两翼"还是"body团块"上。
4. 量化代理指标(不依赖is_wing标签，避免跟判据本身循环论证): 取
   dist_to_principal_axis最高的一批点("远点"，分位数复用binary_split里实际生效的
   DEFAULT_AXIS_DIST_Q作参照，不代表其kept-only总体的精确复现)，看它们在主轴方向
   上的投影proj_len是集中在轴中段(centroid附近)还是贴着两端极值——集中在中段支持
   "轴被wingspan带偏"假设(远点是居中的body团块，因为身体在垂直轴的方向上有厚度)，
   贴两端则反驳该假设(远点是沿轴延伸到两端的翼尖，符合原设计)。6帧打印汇总结论。

核心证据是图，其余(is_wing组延展对比、eigval占比)只是辅助支撑，最终判断以肉眼看
eda_outputs/axis_diag_*.png为准。

用法:
    python -m postprocessing.labeling.diag_principal_axis
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.floater_census_100frames import latest_checkpoint_dir  # noqa: E402
from postprocessing.labeling.binary_split import (  # noqa: E402
    classify_body_wing_quantile, DEFAULT_AXIS_DIST_Q,
)
from postprocessing.labeling.select_dev_frames import DEV_FRAMES, DATASET_DIR  # noqa: E402

OUT_DIR = REPO_ROOT / "postprocessing" / "labeling" / "eda_outputs"

FAR_Q = DEFAULT_AXIS_DIST_Q  # "远点"分位数，借用binary_split里实际生效的量级作参照


def load_full(frame_name: str) -> pd.DataFrame:
    """该帧marked表全部点，if_keep不筛(先看全貌)。"""
    frame_idx = int(frame_name[1:])
    splat_dir = latest_checkpoint_dir(DATASET_DIR / frame_name)
    marked_csv = splat_dir / f"gaussian_features_f{frame_idx:04d}_marked.csv"
    return pd.read_csv(marked_csv)


def compute_principal_axis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与utils/gaussian_features.py compute_gaussian_features同款算法: 全部点(未加权)
    做cov+eigh，取最大特征值对应的特征向量作principal_axis。
    返回(axis, eigvals_desc, centroid)。"""
    centroid = xyz.mean(axis=0)
    cov = np.cov((xyz - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axis = eigvecs[:, order[0]]
    return axis, eigvals[order], centroid


def bbox_diag(xyz: np.ndarray) -> float:
    return float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))


def mean_dist_to_own_centroid(xyz: np.ndarray) -> float:
    centroid = xyz.mean(axis=0)
    return float(np.linalg.norm(xyz - centroid, axis=1).mean())


def plot_axis_diag(xyz: np.ndarray, dist: np.ndarray, frame: str, out_path: Path) -> None:
    views = [
        ("front (elev=0, azim=-90)", dict(elev=0, azim=-90)),
        ("top (elev=90, azim=-90)", dict(elev=90, azim=-90)),
    ]
    fig = plt.figure(figsize=(12, 5.5))
    sc = None
    for i, (title, view_kw) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=dist, cmap="viridis",
                         s=14, alpha=0.9, depthshade=False)
        ax.view_init(**view_kw)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)
    fig.colorbar(sc, ax=fig.axes, shrink=0.7, label="dist_to_principal_axis")
    fig.suptitle(
        f"{frame}: color = dist_to_principal_axis (global PCA, all points, if_keep not filtered)  "
        f"n_total={len(xyz)}"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def diagnose_frame(frame: str) -> dict:
    df = load_full(frame)
    xyz = df[["x", "y", "z"]].to_numpy()

    axis, eigvals_desc, centroid = compute_principal_axis(xyz)
    var_ratio1 = float(eigvals_desc[0] / eigvals_desc.sum())

    rel = xyz - centroid
    proj_len = rel @ axis
    proj_vec = np.outer(proj_len, axis)
    dist = np.linalg.norm(rel - proj_vec, axis=1)

    # 与marked表里既有的dist_to_principal_axis列做一致性检查(算法/数据一样理应高度吻合)
    existing_dist = df["dist_to_principal_axis"].to_numpy()
    max_abs_diff = float(np.max(np.abs(dist - existing_dist)))

    print(f"\n{'=' * 70}\n[{frame}]\n{'=' * 70}")
    print(f"  principal_axis = {np.round(axis, 4).tolist()}")
    print(f"  eigval方差占比(降序): {np.round(eigvals_desc / eigvals_desc.sum(), 4).tolist()}  "
          f"(第一主轴占比={var_ratio1:.3f})")
    print(f"  与既有dist_to_principal_axis列一致性检查: max_abs_diff={max_abs_diff:.3e} "
          f"{'OK' if max_abs_diff < 1e-6 else '[警告]算法/数据不一致，需要检查'}")

    # ---- 2) is_wing候选组 spatial extent 对比(仅参考，is_wing本身可能不准) ----
    kept = df[df["if_keep"]].reset_index(drop=True)
    is_wing = classify_body_wing_quantile(kept)
    body_xyz = kept.loc[~is_wing, ["x", "y", "z"]].to_numpy()
    wing_xyz = kept.loc[is_wing, ["x", "y", "z"]].to_numpy()
    body_bbox, wing_bbox = bbox_diag(body_xyz), bbox_diag(wing_xyz)
    body_mdist, wing_mdist = mean_dist_to_own_centroid(body_xyz), mean_dist_to_own_centroid(wing_xyz)
    print(f"  [参考,依赖is_wing初步标签] body候选(n={len(body_xyz)}): "
          f"bbox对角线={body_bbox:.4f}  平均质心距离={body_mdist:.4f}")
    print(f"  [参考,依赖is_wing初步标签] wing候选(n={len(wing_xyz)}): "
          f"bbox对角线={wing_bbox:.4f}  平均质心距离={wing_mdist:.4f}  "
          f"({'wing候选延展更大(符合常规预期)' if wing_bbox > body_bbox else 'body候选延展更大(反常，需留意)'})")

    # ---- 3) 核心证据: dist_to_principal_axis连续值colormap散点图 ----
    out_path = OUT_DIR / f"axis_diag_{frame}.png"
    plot_axis_diag(xyz, dist, frame, out_path)
    print(f"  核心证据图 -> {out_path}")

    # ---- 4) 量化代理指标: 远点在主轴上的投影是居中还是贴两端 ----
    far_th = float(np.quantile(dist, FAR_Q))
    far_mask = dist > far_th
    proj_abs_max = float(np.abs(proj_len).max())
    extreme_ratio = float(np.abs(proj_len[far_mask]).mean() / proj_abs_max) if proj_abs_max > 0 else float("nan")

    supports_bent_axis = extreme_ratio < 0.5
    verdict = ("支持'轴被wingspan带偏'假设(远点集中在轴中段，像居中的body团块)"
               if supports_bent_axis else
               "反驳'轴被wingspan带偏'假设(远点集中在轴两端，像伸展的翼尖)")
    print(f"  远点(dist>{FAR_Q:.2f}分位, n={int(far_mask.sum())})在轴上的投影极值化程度 "
          f"extreme_ratio={extreme_ratio:.3f} (0=居中/团块状, 1=贴两端/翼尖状) -> {verdict}")

    return {
        "frame": frame, "var_ratio1": var_ratio1, "extreme_ratio": extreme_ratio,
        "supports_bent_axis": supports_bent_axis,
        "body_bbox": body_bbox, "wing_bbox": wing_bbox,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [diagnose_frame(f) for f in DEV_FRAMES]

    print(f"\n{'=' * 70}\n6帧汇总\n{'=' * 70}")
    for r in results:
        print(f"  [{r['frame']}] extreme_ratio={r['extreme_ratio']:.3f}  "
              f"var_ratio1={r['var_ratio1']:.3f}  "
              f"{'支持带偏假设' if r['supports_bent_axis'] else '反驳带偏假设'}")

    n_support = sum(r["supports_bent_axis"] for r in results)
    mean_extreme = float(np.mean([r["extreme_ratio"] for r in results]))
    mean_var_ratio1 = float(np.mean([r["var_ratio1"] for r in results]))
    print(f"\n  {n_support}/{len(results)} 帧的量化代理指标支持'轴被wingspan带偏'假设 "
          f"(mean extreme_ratio={mean_extreme:.3f}, mean 第一主轴方差占比={mean_var_ratio1:.3f})")
    print("  最终判断仍以核心证据图(eda_outputs/axis_diag_*.png)目测为准，"
          "以上量化代理指标(尤其是依赖is_wing初步标签的部分)只是辅助支撑，见模块docstring。")


if __name__ == "__main__":
    main()
