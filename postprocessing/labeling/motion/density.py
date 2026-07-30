"""
T3 新方法 v0 —— 跨帧运动累加密度分割 body/wing，第一步：累加窗口 + 体素帧计数密度场
+ body体素提取。

背景: 现有T3(kmeans_split.py + labeling.py)在单帧特征空间(xyz+opacity+R)上聚类，
body/wing边界靠"点数最多的簇"或"高置信度种子"这类单帧线索决定，翼根附近的连续渐变
区域容易被切错(见labeling.py顶部docstring)。这里换一个跟单帧特征无关的信号：body
在时间上几乎不动(相对翅膀)，如果把很多帧的if_keep点叠在同一个体素网格里，body所在
的体素会被"来自不同帧的点"反复命中，命中的帧数远高于翅膀扫过的体素(翅膀在振翅一圈
里只经过每个体素一次或很少几次)。这是 postprocessing/reference/seg2d/seg2d_spec.md
v4 的核心思路(2D版本按像素做，这里搬到3D体素)。

本模块只做"体素帧计数密度场 -> body候选体素 -> 最大连通分量"这一段。单帧body/wing
判定、wing连通分量拆L/R、confidence、落盘见 label.py。

v0 已知简化/TODO(不在本轮做):
- **不做刚性运动对齐**: 窗口内各帧的if_keep点直接用原始xyz叠加，不对齐body的
  刚性运动(平移/旋转)。如果body本身有整体的小幅位移(实验动物不是钉死的)，累加出来
  的body密度峰会被这部分运动"抹宽"，边界比理想情况模糊。下一步候选方案:
  每帧用if_keep点质心做刚性平移对齐(只对齐质心，不估计旋转)，再重新累加看密度峰
  是否更干净——现在不做，先看v0(无对齐)的密度场够不够用。
- **不处理边界帧**: idx < HALF_WINDOW 或 idx > 639-HALF_WINDOW 的帧本轮跳过，
  不做截断窗口(比如只用[0, f+36])兜底，见 check_t2_coverage / valid_frame_range。
- **VOXEL_SIZE_M / BODY_VOXEL_COUNT_THRESH 不做自动调参**: 都是从一次性人工检查
  锁定的常量，见各自定义处的注释。
- **不处理身体自身小幅晃动导致的腹部/头部误判进wing**: body候选体素阈值判据只看
  "命中帧数"，如果body表面某一小片区域(比如腹部尖端)因为姿态晃动，命中帧数没有
  达到阈值，会被误判成wing候选点，交给label.py的wing连通分量/最近距离合并兜底，
  这里不特殊处理。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import load_marked  # noqa: E402
from utils.ply import connected_component_labels  # noqa: E402

DATASET_DIR = REPO_ROOT / "outputs" / "ctrl_009_002_ratio3_sh0_dense" / "ratio3_sh0_dense"
"""640帧数据集(f0000~f0639)，训练时没开checkpoint-dump，T1/T2产出是本轮现算落盘的，
见开发过程记录：只对累加窗口覆盖的帧范围(f0224~f0416)补跑过，不是全量640帧。"""

HALF_WINDOW = 36
"""累加窗口半宽(帧)，窗口=[f-HALF_WINDOW, f+HALF_WINDOW]共73帧。任务规格锁定值，不调参。"""

FIRST_VALID_IDX = HALF_WINDOW           # 36
LAST_VALID_IDX = 639 - HALF_WINDOW      # 603
"""本轮只处理窗口不越界的帧，边界帧(idx<36或idx>603)跳过不做截断窗口兜底，见模块docstring。"""

VOXEL_SIZE_M = 2.5e-4
"""体素边长(米)。取值依据: 在候选帧(f0260~f0380，8帧)的if_keep点云上算10近邻平均距离
(跟T1 local_density列同一个kNN距离概念，k=10)，中位数≈1.17e-4 m，25~75分位数区间
≈[6.4e-5, 1.68e-4] m——这是该数据集单帧点云的"典型点间距"量级。VOXEL_SIZE_M取约2倍
中位点间距(≈2.5e-4 m)，让同一物理位置在不同帧因重建噪声/量化造成的小幅点位抖动能落进
同一个体素，同时仍比body/wing的物理尺度(翅长~2.5~3mm，见io_schema.py UNITS注释)小一
个数量级以上，不会把body和wing的体素混在一起。不做自动调参，固定用这个值。"""

BODY_VOXEL_COUNT_PERCENTILE = 90.0
"""BODY_VOXEL_COUNT_THRESH的选取依据: 在f0300(73帧窗口)的体素帧计数场上，取全部
"被至少1帧命中"的体素的count分布，第90分位数。**如实记录一个和最初假设不符的现象**:
这个分布不是"翅膀路径体素(低计数) vs body体素(计数接近73)"的干净双峰——实测(见
diag/voxel_threshold_tuning.py出的直方图)max_count只有42(满窗口73帧的57%)，
整体分布更像单调递减、没有第二个高计数簇。可能原因(候选，未验证): v0没做刚性运动
对齐(见模块docstring开头TODO)，body本身哪怕只有很小的整体位移/转动，73帧累加下来
也会把body表面同一物理点分散到好几个相邻体素，稀释了单体素的命中帧数。90分位数
(=19)不是"两簇之间的自然分界"，只是"取分布靠后的一段"这个更弱的意义——人工核查
(diag/voxel_threshold_tuning.py的body_compare图)确认这个阈值下的候选体素连通分量后
是一个空间上连续、跟真实body位置大致吻合的团块，暂时够用，但不是理想的清晰阈值，
下一步如果要改进应该优先验证"刚性对齐后分布会不会变干净"这个假设。仅记录取值依据，
运行时不重新计算百分位数——见BODY_VOXEL_COUNT_THRESH。"""

BODY_VOXEL_COUNT_THRESH = 19
"""body候选体素判据: 体素帧计数 > 此值 视为body候选体素。数值来源: 见
BODY_VOXEL_COUNT_PERCENTILE的注释，在f0300窗口上90分位数实测=19(不是最初设想的
"接近满窗口帧数"的高值，分布本身没有那么干净——如实记录，见上一条注释)。人工核查
该阈值下body候选体素连通分量后在3D热图上呈现一个空间连续的团块(diag/
voxel_threshold_tuning.py)后锁定为常量，不随帧变化、不自动调参。"""

BODY_CC_K = 10
BODY_CC_PERCENTILE = 75.0
"""body候选体素做连通分量分析用的k-近邻图参数，复用全仓库连通分量分析的默认值
(postprocessing/cleaning/mark_floaters.py的K_NEIGHBORS/DIST_PERCENTILE、
postprocessing/labeling/labeling.py的WING_CC_K/WING_CC_PERCENTILE都是同一组默认值)，
不为体素网格重新调参。"""


def valid_frame_range() -> range:
    """本轮处理的帧号范围(含端点)，见HALF_WINDOW/FIRST_VALID_IDX/LAST_VALID_IDX。"""
    return range(FIRST_VALID_IDX, LAST_VALID_IDX + 1)


def check_t2_coverage(frame_indices: list[int], dataset_dir: Path = DATASET_DIR) -> dict[int, bool]:
    """检查给定帧号列表里，每一帧的T2产出(_marked.csv)是否已存在。不齐备的帧
    不报错，只如实返回False，调用方据此跳过并打印，不在这里静默补跑。"""
    coverage = {}
    for idx in frame_indices:
        frame = f"f{idx:04d}"
        try:
            load_marked(frame, data_root=dataset_dir)
            coverage[idx] = True
        except FileNotFoundError:
            coverage[idx] = False
    return coverage


def load_window_points(frame_idx: int, dataset_dir: Path = DATASET_DIR,
                        half_window: int = HALF_WINDOW) -> tuple[pd.DataFrame, list[int]]:
    """加载 [frame_idx-half_window, frame_idx+half_window] 窗口内所有帧的if_keep=True点，
    合并成一张表，新增frame_idx整数列标记来源帧号(供compute_voxel_frame_counts按帧去重)。
    T2产出缺失的帧跳过并打印(不报错、不做任何补算)，返回(合并后的DataFrame, 实际用到的
    帧号列表)。跟kmeans_split.py::load_kept同样口径: 复用load_marked现算或读取_marked.csv，
    只取if_keep=True的点。"""
    window_indices = list(range(frame_idx - half_window, frame_idx + half_window + 1))
    coverage = check_t2_coverage(window_indices, dataset_dir)
    missing = [i for i, ok in coverage.items() if not ok]
    if missing:
        print(f"[density] frame_idx={frame_idx}: 窗口内{len(missing)}/{len(window_indices)}帧"
              f"缺少T2产出(_marked.csv)，跳过不参与累加: {[f'f{i:04d}' for i in missing]}")

    frames_dfs = []
    used_indices = []
    for idx in window_indices:
        if not coverage[idx]:
            continue
        frame = f"f{idx:04d}"
        df_full, _ = load_marked(frame, data_root=dataset_dir)
        kept = df_full[df_full["if_keep"].astype(bool)].copy()
        kept["frame_idx"] = idx
        frames_dfs.append(kept)
        used_indices.append(idx)

    window_df = pd.concat(frames_dfs, ignore_index=True) if frames_dfs else pd.DataFrame(
        columns=["x", "y", "z", "frame_idx"])
    return window_df, used_indices


def compute_voxel_frame_counts(window_df: pd.DataFrame, voxel_size: float = VOXEL_SIZE_M) -> pd.Series:
    """体素帧计数密度场: 体素化窗口内累加的全部点(voxel_size见常量注释)，每个体素统计
    "有多少个不同帧号的点落入此体素"(0~窗口帧数)，同一帧同一体素里的多个点只算一次。
    这是复刻seg2d_spec.md v4的核心思路(body在对齐后重复出现的帧数高，不是点数量高)，
    不是简单点密度/KDE。返回pd.Series，MultiIndex=(vx,vy,vz)，值=命中帧数，
    只包含被至少1帧命中的体素。"""
    xyz = window_df[["x", "y", "z"]].to_numpy(dtype=float)
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    keys = pd.DataFrame(voxel_idx, columns=["vx", "vy", "vz"])
    keys["frame_idx"] = window_df["frame_idx"].to_numpy()
    dedup = keys.drop_duplicates()  # 同一帧同一体素只保留一条，即"该体素被这一帧命中"这件事只算一次
    return dedup.groupby(["vx", "vy", "vz"]).size()


def voxel_centers(voxel_keys: np.ndarray, voxel_size: float = VOXEL_SIZE_M) -> np.ndarray:
    """(vx,vy,vz)整数体素坐标(N,3) -> 体素中心点的世界坐标(N,3)，供connected_component_labels
    在"点集"上跑(该函数本身不区分点云点还是体素中心点，只要是(N,3)坐标)。"""
    return (voxel_keys.astype(float) + 0.5) * voxel_size


def extract_body_voxels(voxel_counts: pd.Series, thresh: int = BODY_VOXEL_COUNT_THRESH,
                         voxel_size: float = VOXEL_SIZE_M, k: int = BODY_CC_K,
                         dist_percentile: float = BODY_CC_PERCENTILE) -> set[tuple[int, int, int]]:
    """body候选体素提取: (1)帧计数 > thresh 的体素 = body候选体素；(2)对候选体素中心点集
    做3D连通分量分析(复用utils.ply.connected_component_labels)，取最大连通分量作为最终
    body体素集合，丢弃游离的高计数体素碎块(比如某个翅膀drupe short-term停留造成的假阳性)。
    返回最终body体素的(vx,vy,vz)元组集合。"""
    candidates = voxel_counts[voxel_counts > thresh]
    if len(candidates) == 0:
        return set()
    voxel_keys = np.array(candidates.index.tolist())  # (N,3) int
    if len(voxel_keys) == 1:
        return {tuple(voxel_keys[0])}

    centers = voxel_centers(voxel_keys, voxel_size)
    k_use = min(k, len(centers) - 1)
    comp_labels = connected_component_labels(centers, k=k_use, dist_percentile=dist_percentile)
    comp_sizes = np.bincount(comp_labels)
    main_comp = int(np.argmax(comp_sizes))
    is_main = comp_labels == main_comp

    n_dropped = int((~is_main).sum())
    if n_dropped > 0:
        print(f"[density] body候选体素连通分量分析: 丢弃{n_dropped}/{len(voxel_keys)}个"
              f"非最大连通分量的碎块体素")

    return {tuple(vk) for vk in voxel_keys[is_main]}


def points_to_voxel_keys(xyz: np.ndarray, voxel_size: float = VOXEL_SIZE_M) -> np.ndarray:
    """(N,3)点坐标 -> (N,3)所在体素的整数坐标，跟compute_voxel_frame_counts同一套体素化。"""
    return np.floor(xyz / voxel_size).astype(np.int64)


def compute_body_voxels_for_frame(frame_idx: int, dataset_dir: Path = DATASET_DIR) -> dict:
    """给定测试帧号，跑完整的"累加窗口 -> 体素帧计数 -> body候选体素 -> 最大连通分量"流程，
    返回诊断+结果dict，供label.py单帧分类和diag/出图复用，避免重复计算。"""
    window_df, used_indices = load_window_points(frame_idx, dataset_dir)
    voxel_counts = compute_voxel_frame_counts(window_df)
    body_voxels = extract_body_voxels(voxel_counts)

    return {
        "frame_idx": frame_idx,
        "window_df": window_df,
        "used_frame_indices": used_indices,
        "n_frames_used": len(used_indices),
        "voxel_counts": voxel_counts,
        "body_voxels": body_voxels,
    }
