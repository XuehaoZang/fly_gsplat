"""
T3 (body/wing/L-R 标注) dev帧选择：从 G2b_G9 100帧里选一小组代表帧，先在少量帧上
验证标注方法，覆盖不同的身体朝向以避免"看起来对但只是巧合"。

只读取T2既有产出物，不重新计算判据/连通分量逻辑：
- `debug/floater_census_100frames_G2b_G9.csv`
  (`postprocessing/cleaning/floater_census_100frames.py` 的输出，逐帧
  floater_ratio_pct / n_mid_10_17 统计)
- `calibrate_G2b_G9_threshold.py` 里已手工确认的 NORMAL_FRAMES / GAP_FRAMES
- 每帧 splatfacto-checkpoint 下缓存的 `gaussian_features_f*_marked.csv`
  (`postprocessing/cleaning/mark_floaters.py` 的输出，`if_keep` 列)

边缘帧判据：直接复用 CLEAN.md §4 里已经文档化的边缘帧定义——分量size落在
10~17 gap 区间(`n_mid_10_17 > 0`)的帧，不新开发判据。GAP_FRAMES 是这批边缘帧
里手工核查过的3帧子集。

body点云用 `if_keep=True` 的全部点粗略近似(T3 还没做 body/wing 分割)，主轴方向
复用 `postprocessing/kinematics/geometry.weighted_pca`，不重新实现PCA。

DEV_FRAMES 是跑一次 main() 里的选帧算法后锁定的结果(做法和 NORMAL_FRAMES /
GAP_FRAMES 一致)：NORMAL_FRAMES 3帧 + 在非边缘候选帧里用"贪心最大化与已选帧集合
主轴夹角的最小值"选出的3帧(f0076/f0075/f0061)，覆盖不同朝向。main() 会重新跑一遍
同样的算法并跟这个常量做一致性检查。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from postprocessing.cleaning.calibrate_G2b_G9_threshold import (  # noqa: E402
    NORMAL_FRAMES, GAP_FRAMES,
)
from postprocessing.cleaning.floater_census_100frames import latest_checkpoint_dir  # noqa: E402
from postprocessing.kinematics.geometry import weighted_pca  # noqa: E402

DATASET_DIR = REPO / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
CENSUS_CSV = REPO / "debug" / "floater_census_100frames_G2b_G9.csv"

N_ADDITIONAL = 3  # 额外挑选的帧数(在 NORMAL_FRAMES 3帧之外)

# 锁定结果：NORMAL_FRAMES + 3帧朝向差异帧，见上方 docstring。
DEV_FRAMES = NORMAL_FRAMES + ["f0076", "f0075", "f0061"]


def edge_frame_set(census: pd.DataFrame) -> set[str]:
    """CLEAN.md §4 里已文档化的"边缘帧"(10~17 gap 分量非空)集合，直接从census读。"""
    return set(census.loc[census["n_mid_10_17"] > 0, "frame"])


def load_body_xyz(frame_name: str) -> np.ndarray:
    """body点云的粗略近似：该帧 if_keep=True 的全部点(还没有body/wing分割)。"""
    frame_idx = int(frame_name[1:])
    splat_dir = latest_checkpoint_dir(DATASET_DIR / frame_name)
    marked_csv = splat_dir / f"gaussian_features_f{frame_idx:04d}_marked.csv"
    df = pd.read_csv(marked_csv)
    kept = df[df["if_keep"]]
    return kept[["x", "y", "z"]].to_numpy()


def principal_axis(xyz: np.ndarray) -> np.ndarray:
    _, eigvecs, _ = weighted_pca(xyz)
    return eigvecs[:, -1]  # 最大特征值对应的主轴，方向符号任意


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """两条无符号轴线间的夹角(0~90度)，用绝对值折叠PCA轴的符号歧义。"""
    c = np.clip(abs(float(np.dot(a, b))), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def select_additional_frames(
    candidates: dict[str, np.ndarray],
    base_axes: dict[str, np.ndarray],
    n_additional: int,
) -> list[tuple[str, float]]:
    """贪心最大化-最小夹角选帧：每一步选一个候选帧，使其到"已选帧集合(含base)"
    里最近那个的夹角尽量大，目的是让每一帧新增的朝向都跟已经覆盖的朝向明显不同，
    而不是都朝同一个"不同"的方向扎堆。返回 [(frame, min_angle_to_selected), ...]。
    """
    selected = dict(base_axes)
    remaining = dict(candidates)
    picked: list[tuple[str, float]] = []
    for _ in range(n_additional):
        best_f, best_score = None, -1.0
        for f, ax in remaining.items():
            score = min(axis_angle_deg(ax, sax) for sax in selected.values())
            if score > best_score:
                best_f, best_score = f, score
        picked.append((best_f, best_score))
        selected[best_f] = remaining.pop(best_f)
    return picked


def main() -> None:
    census = pd.read_csv(CENSUS_CSV)
    edge_frames = edge_frame_set(census)

    print("=" * 70)
    print("3) G2b_G9 全部帧里“两翼贴一起/边缘”类帧统计 (给ST2用的参考)")
    print("=" * 70)
    print(f"  边缘帧(10~17 gap 分量非空) 数量: {len(edge_frames)} / {len(census)} "
          f"({100 * len(edge_frames) / len(census):.1f}%)")
    print(f"  其中人工核查过的 GAP_FRAMES 子集: {GAP_FRAMES}")
    assert set(GAP_FRAMES) <= edge_frames, "GAP_FRAMES 应该是edge_frames的子集，判据对不上"

    print("\n" + "=" * 70)
    print("1) 选帧: NORMAL_FRAMES 基础 + 朝向多样性额外帧")
    print("=" * 70)
    for f in NORMAL_FRAMES:
        assert f not in edge_frames, f"{f} 是 NORMAL_FRAMES 但落在边缘帧里，判据矛盾"

    normal_axes = {f: principal_axis(load_body_xyz(f)) for f in NORMAL_FRAMES}

    candidate_pool = [
        f"f{i:04d}" for i in range(100)
        if f"f{i:04d}" not in NORMAL_FRAMES and f"f{i:04d}" not in edge_frames
    ]
    cand_axes = {f: principal_axis(load_body_xyz(f)) for f in candidate_pool}
    picked = select_additional_frames(cand_axes, normal_axes, N_ADDITIONAL)
    additional_frames = [f for f, _ in picked]

    selected_frames = NORMAL_FRAMES + additional_frames
    if selected_frames != DEV_FRAMES:
        print(f"  [警告] 本次重跑选出 {selected_frames}，跟锁定的 DEV_FRAMES={DEV_FRAMES} 不一致，"
              f"需要检查上游数据(census csv / marked csv)是否变了并更新常量。")
    else:
        print(f"  重跑选帧算法结果和锁定的 DEV_FRAMES 一致: {DEV_FRAMES}")

    census_by_frame = census.set_index("frame")
    print("\n每帧诊断 (floater_ratio_pct / 边缘帧状态 / 入选原因):")
    for f in DEV_FRAMES:
        row = census_by_frame.loc[f]
        reason = "NORMAL_FRAMES 基础帧(floater_ratio接近均值,无gap)" if f in NORMAL_FRAMES else None
        if reason is None:
            angle = next(a for ff, a in picked if ff == f)
            reason = f"朝向多样性额外帧, 与已选帧集合的最小夹角={angle:.1f}°"
        print(f"  [{f}] floater_ratio_pct={row['floater_ratio_pct']:.3f}%  "
              f"是否边缘帧={f in edge_frames}  原因: {reason}")

    print("\n" + "=" * 70)
    print("2) 额外帧 vs NORMAL_FRAMES 的主轴夹角 (覆盖度说明)")
    print("=" * 70)
    for f in additional_frames:
        ax = cand_axes[f]
        angles = {nf: axis_angle_deg(ax, naxis) for nf, naxis in normal_axes.items()}
        detail = ", ".join(f"{nf}={a:.1f}°" for nf, a in angles.items())
        print(f"  [{f}] 与NORMAL_FRAMES夹角: {detail}  "
              f"(min={min(angles.values()):.1f}°, mean={np.mean(list(angles.values())):.1f}°)")

    print(f"\nDEV_FRAMES = {DEV_FRAMES}  (共 {len(DEV_FRAMES)} 帧)")


if __name__ == "__main__":
    main()
