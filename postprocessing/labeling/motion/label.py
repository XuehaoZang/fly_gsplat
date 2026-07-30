"""
T3 新方法 v0 —— 跨帧运动累加密度分割，第二步: 单帧body/wing判定 + wing连通分量拆
L/R + 落盘_labeled.csv。编排逻辑参考 postprocessing/labeling/labeling.py::process_frame
的结构(先分part再L/R锚定再落盘)，但body/wing的判据换成了density.py算出的跨帧累加
体素场，不是单帧kmeans聚类。

跟labeling.py的复用边界(任务规格锁定):
- compute_body_axes / finalize_part_labels: 通用几何函数(body PCA定轴+L/R锚定+
  if_keep=False点1-NN标签传播)，不依赖kmeans，直接从labeling.py import。
- fix_wing_connectivity / check_wing_merged / forced_wing_split 的**思路**复用，
  但不import私有实现——这里从头改写成一份适配"body/wing二分类(没有kmeans给的
  wing_A/wing_B簇标签)"输入形式的独立版本，任务规格要求独立维护，不改labeling.py。

confidence规则(v0简化版，比labeling.py简单——没有kmeans那套ARI稳定性判据，因为
本方法没有随机初始化，不存在"多次跑结果不一致"这个不确定性来源):
  - wing_merged_forced_split触发 => low
  - body候选点数 < MIN_BODY_POINTS 或 > BODY_FRAC_MAX*n_kept(body候选点占比异常，
    说明density.py那步大概率把这一帧判串了) => low
  - 否则 => high
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import load_marked  # noqa: E402
from postprocessing.kinematics.io_schema import PART_LABELS  # noqa: E402
from postprocessing.labeling.labeling import compute_body_axes, finalize_part_labels  # noqa: E402
from postprocessing.labeling.motion import density as d  # noqa: E402
from postprocessing.viz.reprojection_viewer import plot_reprojection_overlay  # noqa: E402
from utils.ply import connected_component_labels  # noqa: E402

RAW_DATA_DIR = REPO_ROOT / "data" / "ctrl_009_002"
OUT_DIR = REPO_ROOT / "postprocessing" / "labeling" / "motion" / "eda_outputs"
REPROJ_DIR = OUT_DIR / "reprojection"
SUMMARY_CSV = OUT_DIR / "motion_dev_summary.csv"

WING_CC_K = 10
WING_CC_PERCENTILE = 75.0
"""wing候选点连通分量分析用的k-近邻图参数，复用T2/labeling.py的通用默认值，不重新调，
见任务规格"参数先复用T2默认k=10/percentile=75,不重新调"。"""

WING_MERGE_MIN_FRAC = 0.05
WING_MERGE_MIN_ABS = 5
"""判断"两翼贴一起无法连通分离"用的显著分量阈值，跟labeling.py::check_wing_merged同一组
取值(WING_MERGE_MIN_FRAC/WING_MERGE_MIN_ABS)，思路复用，独立维护一份，见模块docstring。"""

MIN_BODY_POINTS = 20
BODY_FRAC_MAX = 0.9
"""confidence判据用: body候选点数 < MIN_BODY_POINTS(明显太少，density.py那步大概率漏判
了大部分body体素) 或 > BODY_FRAC_MAX*n_kept(明显太多，大概率把wing也吞进了body候选)
时，本帧confidence强制为low。这两个值参照labeling.py::MIN_BODY_SEED同一类"数量级明显
异常"判据风格给的粗略下限/上限，不是从这个数据集专门标定的精确值。"""


def classify_body_candidate(df_kept: pd.DataFrame, frame_idx: int, dataset_dir: Path) -> tuple[np.ndarray, dict]:
    """单帧body/wing候选点判定: 该帧每个if_keep点，查其所在体素是否属于
    density.compute_body_voxels_for_frame算出的body体素集合 -> body候选点；其余
    -> wing候选点。返回(is_body布尔数组, density诊断dict)。"""
    info = d.compute_body_voxels_for_frame(frame_idx, dataset_dir)
    xyz_kept = df_kept[["x", "y", "z"]].to_numpy()
    voxel_keys = d.points_to_voxel_keys(xyz_kept)
    body_voxels = info["body_voxels"]
    is_body = np.array([tuple(vk) in body_voxels for vk in voxel_keys])
    return is_body, info


def split_wing_candidates(xyz_kept: np.ndarray, is_body: np.ndarray,
                           k: int = WING_CC_K, dist_percentile: float = WING_CC_PERCENTILE
                           ) -> tuple[np.ndarray, np.ndarray, dict]:
    """对wing候选点(~is_body)做3D连通分量分析，取最大的两个连通分量记为wing_A/wing_B，
    其余碎块按最近距离并入body/wing_A/wing_B三者之一。返回(semantic数组, wing子连通分量
    大小数组comp_sizes, 诊断dict)。comp_sizes另外交给check_wing_merged判断两翼是否贴一起。

    这里的wing_A/wing_B是从"wing候选点"里现分出来的两个最大连通分量，不是像kmeans版本
    那样已经有3个簇标签——本方法只有body/wing二分类，没有天然的3簇结构，见模块docstring。
    """
    n = len(xyz_kept)
    semantic = np.where(is_body, "body", "wing_candidate").astype(object)
    wing_idx = np.where(~is_body)[0]

    if len(wing_idx) < 2:
        semantic[wing_idx] = "wing_A"
        return semantic, np.array([len(wing_idx)]), {"degenerate_wing_split": True,
                                                       "n_fragments": 0, "reassigned": []}

    xyz_wing = xyz_kept[wing_idx]
    k_use = min(k, len(wing_idx) - 1)
    comp_labels = connected_component_labels(xyz_wing, k=k_use, dist_percentile=dist_percentile)
    comp_sizes = np.bincount(comp_labels)
    order = np.argsort(comp_sizes)[::-1]  # 分量大小从大到小排序
    main_a_comp = order[0]
    main_b_comp = order[1] if len(order) > 1 else order[0]

    is_main_a = comp_labels == main_a_comp
    is_main_b = (comp_labels == main_b_comp) & ~is_main_a
    semantic[wing_idx[is_main_a]] = "wing_A"
    semantic[wing_idx[is_main_b]] = "wing_B"

    frag_local = ~is_main_a & ~is_main_b
    frag_idx = wing_idx[frag_local]
    n_fragment_points = len(frag_idx)
    reassigned = []
    if n_fragment_points > 0:
        candidate_blocks = {
            "body": xyz_kept[semantic == "body"],
            "wing_A": xyz_kept[semantic == "wing_A"],
            "wing_B": xyz_kept[semantic == "wing_B"],
        }
        candidate_trees = {name: cKDTree(pts) for name, pts in candidate_blocks.items() if len(pts) > 0}
        for pt_idx in frag_idx:
            pt = xyz_kept[pt_idx]
            dists = {name: tree.query(pt)[0] for name, tree in candidate_trees.items()}
            nearest = min(dists, key=dists.get)
            reassigned.append((int(pt_idx), "wing_candidate", nearest))
            semantic[pt_idx] = nearest

    assert n == len(semantic)
    return semantic, comp_sizes, {"degenerate_wing_split": False,
                                   "n_fragments": n_fragment_points, "reassigned": reassigned}


def check_wing_merged(comp_sizes: np.ndarray, n_wing_total: int,
                       min_frac: float = WING_MERGE_MIN_FRAC, min_abs: int = WING_MERGE_MIN_ABS) -> bool:
    """两翼是否物理上贴一起、无法连通分离: wing候选点的连通分量里，"显著分量"
    (点数占比>=min_frac 且绝对点数>=min_abs，过滤掉离群噪点分量)的个数<=1。
    跟labeling.py::check_wing_merged同一个判据，独立维护一份(见模块docstring)。"""
    if n_wing_total == 0:
        return False
    n_significant = int(np.sum((comp_sizes >= min_abs) & (comp_sizes >= min_frac * n_wing_total)))
    return n_significant <= 1


def forced_wing_split(xyz_kept: np.ndarray, semantic: np.ndarray, right_axis: np.ndarray,
                       body_cm: np.ndarray) -> np.ndarray:
    """两翼贴一起、连通分量无法物理拆分时的降级处理: wing_A∪wing_B全部点按
    x_body×up(right_axis)投影的中位数为界强行切成两半，替换原wing_A/wing_B归属
    (body不变)。跟labeling.py::forced_wing_split同一个思路，独立维护一份。"""
    semantic = semantic.copy()
    idx = np.where((semantic == "wing_A") | (semantic == "wing_B"))[0]
    proj_vals = (xyz_kept[idx] - body_cm) @ right_axis
    median = np.median(proj_vals)
    semantic[idx] = np.where(proj_vals > median, "wing_A", "wing_B")
    return semantic


def compute_confidence(n_body_candidate: int, n_kept: int, is_wing_merged: bool) -> str:
    if is_wing_merged:
        return "low"
    if n_body_candidate < MIN_BODY_POINTS or n_body_candidate > BODY_FRAC_MAX * n_kept:
        return "low"
    return "high"


def process_frame(frame_idx: int, dataset_dir: Path = d.DATASET_DIR) -> dict:
    frame = f"f{frame_idx:04d}"
    df_full, marked_csv = load_marked(frame, data_root=dataset_dir)
    kept_mask = df_full["if_keep"].astype(bool).to_numpy()
    df_kept = df_full[kept_mask].reset_index(drop=True)
    n_total, n_kept = len(df_full), len(df_kept)
    xyz_kept = df_kept[["x", "y", "z"]].to_numpy()

    is_body, density_info = classify_body_candidate(df_kept, frame_idx, dataset_dir)
    n_body_candidate = int(is_body.sum())

    semantic, comp_sizes, split_diag = split_wing_candidates(xyz_kept, is_body)
    n_wing_total = int((~is_body).sum())
    is_wing_merged = check_wing_merged(comp_sizes, n_wing_total)

    if is_wing_merged:
        print(f"  [警告][{frame}] wing_merged_forced_split: wing候选点的连通分量分析后"
              f"只有1个显著分量(贴一起无法物理拆分)，强行按x_body×up投影中位数切成两半，"
              f"强制confidence=low")
        _x_body, right_axis, body_cm = compute_body_axes(xyz_kept, semantic)
        semantic = forced_wing_split(xyz_kept, semantic, right_axis, body_cm)

    confidence = compute_confidence(n_body_candidate, n_kept, is_wing_merged)

    part_label_full, lr_map = finalize_part_labels(df_full, kept_mask, semantic)

    bad = set(part_label_full) - set(PART_LABELS)
    if bad:
        raise ValueError(f"{frame}: part_label 出现不在 io_schema.PART_LABELS 里的值: {bad}")

    df_out = df_full.copy()
    df_out["part_label"] = part_label_full
    df_out["confidence"] = confidence

    labeled_csv = marked_csv.with_name(marked_csv.name.replace("_marked.csv", "_labeled.csv"))
    df_out.to_csv(labeled_csv, index=False)

    print(f"\n[{frame}] n_total={n_total} n_kept={n_kept} ({100 * n_kept / n_total:.1f}%)  "
          f"n_frames_in_window={density_info['n_frames_used']}")
    print(f"  body候选体素数(阈值+连通分量后)={len(density_info['body_voxels'])}  "
          f"body候选点数={n_body_candidate}({100 * n_body_candidate / n_kept:.1f}% of kept)")
    if split_diag.get("degenerate_wing_split"):
        print(f"  [警告] wing候选点数={n_wing_total}太少(<2)，无法做连通分量拆分，"
              f"全部归入wing_A，wing_B为空")
    elif split_diag["n_fragments"] > 0:
        print(f"  wing连通性: 碎块点={split_diag['n_fragments']}  实际改判={len(split_diag['reassigned'])}")
    else:
        print("  wing连通性: 两个wing分量均为单一连通整体，无碎块")
    print(f"  confidence={confidence}  (wing_merged强制low; body候选点数<{MIN_BODY_POINTS}或"
          f">{BODY_FRAC_MAX}*n_kept强制low; 否则high)")
    print(f"  L/R锚定: wing_A -> {lr_map['wing_A']}  wing_B -> {lr_map['wing_B']}")
    print("  part_label分布(全部点，含if_keep=False的1-NN传播): " + "  ".join(
        f"{lab}={int((part_label_full == lab).sum())}" for lab in sorted(PART_LABELS)))
    print(f"  labeled csv -> {labeled_csv}")

    return {
        "frame": frame, "frame_idx": frame_idx, "confidence": confidence,
        "n_total": n_total, "n_kept": n_kept, "n_body_candidate": n_body_candidate,
        "n_wing_total": n_wing_total, "is_wing_merged_forced": is_wing_merged,
        "n_frames_in_window": density_info["n_frames_used"],
        "labeled_csv": labeled_csv, "df_out": df_out,
    }


def plot_labeled_reprojection(frame: str, df_out: pd.DataFrame, confidence: str, out_path: Path) -> None:
    """薄封装，通用重投影/画序逻辑复用 postprocessing/viz/reprojection_viewer.py，
    跟labeling.py::plot_labeled_reprojection同一份复用方式。"""
    plot_reprojection_overlay(frame, df_out, out_path, raw_data_dir=RAW_DATA_DIR,
                               title_suffix=f"  [motion-v0, confidence={confidence}]")


def run_batch(frames: list[str], data_root: Path = d.DATASET_DIR,
              save_reprojection: bool = True) -> tuple[list[dict], list[dict]]:
    """逐帧处理，单帧异常catch住、跳过、记录帧号，不中断整个批处理(同
    postprocessing/labeling/labeling.py::run_batch的约定)。frames/data_root命名和签名
    对齐labeling.py::run_batch，供calc_kinematics.py直接替换导入使用。

    save_reprojection=False时跳过每帧一张的重投影图落盘(eda_outputs/reprojection)——
    calc_kinematics.py跑整个数据集的T3时帧数可能上百，不需要每帧一张诊断图，最终验收
    图已由calc_kinematics.py自己在kinematics/reprojection/下等距挑N_FRAMES张画。
    默认True保留独立调本模块(如本文件main()的开发用途)时逐帧出图诊断的行为。"""
    results = []
    failures = []
    for frame in frames:
        frame_idx = int(frame[1:])
        try:
            r = process_frame(frame_idx, data_root)
            if save_reprojection:
                reproj_path = REPROJ_DIR / f"motion_labeled_reproj_{frame}.png"
                plot_labeled_reprojection(frame, r["df_out"], r["confidence"], reproj_path)
                print(f"  reprojection plot -> {reproj_path}")
            results.append(r)
        except Exception as e:
            failures.append({"frame": frame, "error": f"{type(e).__name__}: {e}"})
            print(f"[{frame}] FAILED: {type(e).__name__}: {e}")
    return results, failures


def build_summary_df(results: list[dict], failures: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "frame_id": r["frame"], "status": "ok", "confidence": r["confidence"],
            "n_total": r["n_total"], "n_kept": r["n_kept"],
            "n_body_candidate": r["n_body_candidate"], "n_wing_total": r["n_wing_total"],
            "is_wing_merged_forced": r["is_wing_merged_forced"],
            "n_frames_in_window": r["n_frames_in_window"], "error": "",
        })
    for f in failures:
        rows.append({
            "frame_id": f["frame"], "status": "failed", "confidence": np.nan,
            "n_total": np.nan, "n_kept": np.nan, "n_body_candidate": np.nan, "n_wing_total": np.nan,
            "is_wing_merged_forced": np.nan, "n_frames_in_window": np.nan, "error": f["error"],
        })
    df = pd.DataFrame(rows)
    df["_frame_idx"] = df["frame_id"].str[1:].astype(int)
    return df.sort_values("_frame_idx").drop(columns="_frame_idx").reset_index(drop=True)


DEV_FRAMES_MOTION = [260, 280, 300, 320, 330, 340, 360, 380]
"""开发阶段选的8个测试帧(<10帧, 见任务规格)。该数据集没有现成的边缘帧census/floater
统计(见density.py DATASET_DIR的说明: 这640帧数据集本轮才第一次补跑T1/T2)，不强求正式
判据——用一次粗扫(每2帧算一次if_keep点云bbox对角线extent，f0036~f0602)大致定位
extent振荡的位置，在一段约120帧的连续区间(f0260~f0380，落在640帧中段，避免碰到
HALF_WINDOW=36边界)里挑了extent高低交替的8帧，覆盖不同的振翅相位(展开/折叠的粗略
代理)，不是严格的wingbeat周期相位标定。"""


def main() -> None:
    results, failures = run_batch([f"f{i:04d}" for i in DEV_FRAMES_MOTION])

    summary_df = build_summary_df(results, failures)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    n_requested, n_ok, n_failed = len(DEV_FRAMES_MOTION), len(results), len(failures)
    high = [r["frame"] for r in results if r["confidence"] == "high"]
    low = [r["frame"] for r in results if r["confidence"] == "low"]
    merged = [r["frame"] for r in results if r["is_wing_merged_forced"]]

    print(f"\n{'=' * 70}\n{n_requested}帧请求汇总(motion v0)\n{'=' * 70}")
    print(f"  成功: {n_ok}  失败: {n_failed}")
    if failures:
        print(f"  失败帧: {[f['frame'] for f in failures]}")
    if n_ok > 0:
        print(f"  high confidence: {len(high)}/{n_ok} ({100 * len(high) / n_ok:.1f}%)")
        print(f"  low  confidence: {len(low)}/{n_ok} ({100 * len(low) / n_ok:.1f}%)")
        print(f"  触发wing_merged_forced_split帧数: {len(merged)}/{n_ok}"
              + (f"  {merged}" if merged else ""))
    print(f"  汇总表 -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
