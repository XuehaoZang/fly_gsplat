"""
T3 Step1 定稿: 用 kmeans_split.py 的 v2(body种子初始化, aux_weight=1x) 产出
body/wing_L/wing_R 三分标签，落盘为 _labeled.csv，打通到 T4 的端到端链路。

流程 (对应此前讨论的 ST1~ST4):
1. v2 聚类 + confidence 标签 (复用 n_hardcut_pairs + 5种子ARI稳定性两个既有诊断)。
2. 两个 wing 簇各自做连通分量检查，碎块合并到最近的主块(body/wing_A/wing_B 三者之一)。
3. body PCA 定 x_body, right_axis = x_body x up，两翼质心投影定 wing_L/wing_R；
   if_keep=False 的点用 1-NN 从已标点传播 part_label。
4. 存 _labeled.csv (不改动 _marked.csv)，新增 confidence 列。
5. 重投影可视化(body灰、wing_L/wing_R两色、if_keep=False叉号标记)，2x2四相机。

用法:
    python -m postprocessing.labeling.labeling
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import load_marked  # noqa: E402
from postprocessing.kinematics.geometry import orient_to_reference, weighted_pca  # noqa: E402
from postprocessing.kinematics.io_schema import PART_LABELS  # noqa: E402
from postprocessing.labeling.kmeans_split import (  # noqa: E402
    K, MAIN_RANDOM_STATE, build_seed_init, cluster_sizes, cross_vs_intra_table,
    label_by_rule_a, n_hardcut_pairs, run_kmeans, run_kmeans_v2, secondary_axis,
    seed_mask, stability_check, stability_check_with_init, standardize_v2,
)
from postprocessing.labeling.diag.select_dev_frames import DATASET_DIR, DEV_FRAMES  # noqa: E402
from postprocessing.viz._colors import PART_COLORS  # noqa: E402
from postprocessing.viz.reprojection_viewer import plot_reprojection_overlay  # noqa: E402
from utils.ply import connected_component_labels  # noqa: E402

RAW_DATA_DIR = REPO_ROOT / "data" / "ctrl_009_002"
LABEL_REPROJ_DIR = REPO_ROOT / "postprocessing" / "labeling" / "eda_outputs" / "reprojection"
SUMMARY_CSV = REPO_ROOT / "postprocessing" / "labeling" / "eda_outputs" / "confidence_summary_G2b_G9.csv"

AUX_WEIGHT_FINAL = 1          # 定版配置: v2, aux_weight=1x(w1), 见 verify_w1_config.py
CONF_N_HARDCUT_MAX = 1        # confidence 规则: n_hardcut_pairs<=1 且 ari_min>0.8 => high
CONF_ARI_MIN_THRESH = 0.8
WING_CC_K = 10                 # wing簇连通性检查用的k/percentile，同utils.ply默认
WING_CC_PERCENTILE = 75.0
UP = np.array([0.0, 0.0, 1.0])  # 实验室 up = +z (calc_kinematics.md §0)

MIN_BODY_SEED = 5              # body种子(opacity>=0.98或R<0.2)点数<此值视为该帧种子退化
WING_MERGE_MIN_FRAC = 0.05     # 两翼合并连通分量检查: 显著分量的最小点数占比
WING_MERGE_MIN_ABS = 5         # 显著分量的最小绝对点数(过滤离群噪点分量)


def compute_confidence(n_hardcut: int, ari_min: float) -> str:
    if n_hardcut <= CONF_N_HARDCUT_MAX and ari_min > CONF_ARI_MIN_THRESH:
        return "high"
    return "low"


def fix_wing_connectivity(df_kept: pd.DataFrame, labels: np.ndarray, mapping: dict[int, str],
                           k: int = WING_CC_K, dist_percentile: float = WING_CC_PERCENTILE
                           ) -> tuple[np.ndarray, dict]:
    """对mapping里语义为wing_A/wing_B的两个簇分别做连通分量检查；每个簇里的非最大
    分量(碎块)重新分配到最近的主块(body全部点 / wing_A主分量 / wing_B主分量三者之一)。
    返回(更新后的semantic数组, 诊断dict)。"""
    xyz = df_kept[["x", "y", "z"]].to_numpy()
    semantic = np.array([mapping[c] for c in labels], dtype=object)

    main_idx = {"wing_A": None, "wing_B": None}
    fragments: list[tuple[str, np.ndarray]] = []
    for wname in ("wing_A", "wing_B"):
        idx = np.where(semantic == wname)[0]
        if len(idx) == 0:
            main_idx[wname] = idx
            continue
        k_use = min(k, len(idx) - 1)
        if k_use < 1:
            main_idx[wname] = idx
            continue
        comp_labels = connected_component_labels(xyz[idx], k=k_use, dist_percentile=dist_percentile)
        comp_sizes = np.bincount(comp_labels)
        main_comp = int(np.argmax(comp_sizes))
        is_main = comp_labels == main_comp
        main_idx[wname] = idx[is_main]
        frag = idx[~is_main]
        if len(frag) > 0:
            fragments.append((wname, frag))

    n_fragment_points = sum(len(f) for _, f in fragments)
    if n_fragment_points == 0:
        return semantic, {"n_fragments": 0, "reassigned": []}

    candidate_blocks = {
        "body": xyz[semantic == "body"],
        "wing_A": xyz[main_idx["wing_A"]],
        "wing_B": xyz[main_idx["wing_B"]],
    }
    candidate_trees = {name: cKDTree(pts) for name, pts in candidate_blocks.items() if len(pts) > 0}

    reassigned = []
    for _, frag_idx in fragments:
        for pt_idx in frag_idx:
            pt = xyz[pt_idx]
            dists = {name: tree.query(pt)[0] for name, tree in candidate_trees.items()}
            nearest = min(dists, key=dists.get)
            if nearest != semantic[pt_idx]:
                reassigned.append((int(pt_idx), str(semantic[pt_idx]), nearest))
                semantic[pt_idx] = nearest

    return semantic, {"n_fragments": n_fragment_points, "reassigned": reassigned}


def compute_body_axes(xyz_kept: np.ndarray, semantic_kept: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """body PCA 定 x_body, right_axis = x_body x up。返回(x_body, right_axis, body_cm)，
    供L/R锚定(finalize_part_labels)和两翼合并时的强制切分(forced_wing_split)共用。"""
    body_xyz = xyz_kept[semantic_kept == "body"]
    _, eigvecs, _ = weighted_pca(body_xyz)
    x_body = orient_to_reference(eigvecs[:, -1], UP)
    right_axis = np.cross(x_body, UP)
    right_axis = right_axis / np.linalg.norm(right_axis)
    body_cm = body_xyz.mean(axis=0)
    return x_body, right_axis, body_cm


def check_wing_merged(xyz_kept: np.ndarray, semantic: np.ndarray,
                       k: int = WING_CC_K, dist_percentile: float = WING_CC_PERCENTILE) -> bool:
    """在wing_A∪wing_B全部点(不分簇边界)上做一次连通分量分析，判断两翼是否物理上
    只有1个显著分量(贴在一起，kmeans按特征空间切开了但空间上无法物理拆分)。
    显著分量 = 点数占比>=WING_MERGE_MIN_FRAC 且绝对点数>=WING_MERGE_MIN_ABS，
    用来过滤掉离群噪点分量。"""
    idx = np.where((semantic == "wing_A") | (semantic == "wing_B"))[0]
    if len(idx) < 2:
        return False
    xyz = xyz_kept[idx]
    k_use = min(k, len(xyz) - 1)
    if k_use < 1:
        return False
    comp_labels = connected_component_labels(xyz, k=k_use, dist_percentile=dist_percentile)
    comp_sizes = np.bincount(comp_labels)
    n_total = len(idx)
    n_significant = int(np.sum((comp_sizes >= WING_MERGE_MIN_ABS) & (comp_sizes >= WING_MERGE_MIN_FRAC * n_total)))
    return n_significant <= 1


def forced_wing_split(xyz_kept: np.ndarray, semantic: np.ndarray, right_axis: np.ndarray,
                       body_cm: np.ndarray) -> np.ndarray:
    """两翼贴一起、连通分量无法物理拆分时的降级处理: wing_A∪wing_B全部点按
    x_body×up(right_axis)投影的中位数为界强行切成两半，替换原kmeans给的wing_A/wing_B
    归属(body不变)。调用方需要另外把该帧confidence强制标记为low。"""
    semantic = semantic.copy()
    idx = np.where((semantic == "wing_A") | (semantic == "wing_B"))[0]
    proj_vals = (xyz_kept[idx] - body_cm) @ right_axis
    median = np.median(proj_vals)
    semantic[idx] = np.where(proj_vals > median, "wing_A", "wing_B")
    return semantic


def finalize_part_labels(df_full: pd.DataFrame, kept_mask: np.ndarray, semantic_kept: np.ndarray
                          ) -> tuple[np.ndarray, dict[str, str]]:
    """body PCA + right_axis 给两个wing簇定 wing_L/wing_R，再把if_keep=False的点用
    1-NN从已标点(kept)传播label。返回(全量part_label数组, {"wing_A":.., "wing_B":..})。"""
    xyz_kept = df_full.loc[kept_mask, ["x", "y", "z"]].to_numpy()

    _x_body, right_axis, body_cm = compute_body_axes(xyz_kept, semantic_kept)

    wing_a_xyz = xyz_kept[semantic_kept == "wing_A"]
    wing_b_xyz = xyz_kept[semantic_kept == "wing_B"]
    proj_a = float(np.dot(wing_a_xyz.mean(axis=0) - body_cm, right_axis))
    proj_b = float(np.dot(wing_b_xyz.mean(axis=0) - body_cm, right_axis))

    lr_map = {"body": "body"}
    if proj_a > proj_b:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_R", "wing_L"
    else:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_L", "wing_R"

    final_kept = np.array([lr_map[s] for s in semantic_kept], dtype=object)

    n_full = len(df_full)
    part_label_full = np.empty(n_full, dtype=object)
    kept_orig_idx = np.where(kept_mask)[0]
    dropped_orig_idx = np.where(~kept_mask)[0]
    part_label_full[kept_orig_idx] = final_kept

    if len(dropped_orig_idx) > 0:
        xyz_full = df_full[["x", "y", "z"]].to_numpy()
        tree = cKDTree(xyz_kept)
        _, nn_idx = tree.query(xyz_full[dropped_orig_idx], k=1)
        part_label_full[dropped_orig_idx] = final_kept[nn_idx]

    return part_label_full, {"wing_A": lr_map["wing_A"], "wing_B": lr_map["wing_B"]}


def process_frame(frame: str) -> dict:
    df_full, marked_csv = load_marked(frame, data_root=DATASET_DIR)
    kept_mask = df_full["if_keep"].astype(bool).to_numpy()
    df_kept = df_full[kept_mask].reset_index(drop=True)
    n_total, n_kept = len(df_full), len(df_kept)

    seeds = seed_mask(df_kept)
    n_body_seed = int(seeds.sum())
    degraded_seed = n_body_seed < MIN_BODY_SEED
    X = standardize_v2(df_kept, AUX_WEIGHT_FINAL)

    if degraded_seed:
        print(f"  [警告][{frame}] body种子点数={n_body_seed} < {MIN_BODY_SEED}，无法做种子初始化，"
              f"退化为普通kmeans++随机初始化，强制confidence=low")
        labels = run_kmeans(X, MAIN_RANDOM_STATE)
        stability = stability_check(X)
    else:
        labels = run_kmeans_v2(X, seeds, MAIN_RANDOM_STATE)
        stability = stability_check_with_init(X, lambda s: build_seed_init(X, seeds, s))

    xyz_kept = df_kept[["x", "y", "z"]].to_numpy()
    axis, centroid = secondary_axis(xyz_kept)
    mapping = label_by_rule_a(df_kept, labels, axis, centroid)

    dist_df = cross_vs_intra_table(df_kept, labels)
    n_hardcut = n_hardcut_pairs(dist_df)
    confidence = compute_confidence(n_hardcut, stability["min_ari"])
    if degraded_seed:
        confidence = "low"

    semantic_raw = np.array([mapping[c] for c in labels], dtype=object)
    is_wing_merged = check_wing_merged(xyz_kept, semantic_raw)
    if is_wing_merged:
        print(f"  [警告][{frame}] wing_merged_forced_split: 两翼连通分量分析后只有1个显著分量"
              f"(贴一起无法物理拆分)，强行按x_body×up投影中位数切成两半，强制confidence=low")
        _x_body, right_axis, body_cm = compute_body_axes(xyz_kept, semantic_raw)
        semantic = forced_wing_split(xyz_kept, semantic_raw, right_axis, body_cm)
        cc_diag = {"n_fragments": 0, "reassigned": []}
        confidence = "low"
    else:
        semantic, cc_diag = fix_wing_connectivity(df_kept, labels, mapping)

    part_label_full, lr_map = finalize_part_labels(df_full, kept_mask, semantic)

    bad = set(part_label_full) - set(PART_LABELS)
    if bad:
        raise ValueError(f"{frame}: part_label 出现不在 io_schema.PART_LABELS 里的值: {bad}")

    df_out = df_full.copy()
    df_out["part_label"] = part_label_full
    df_out["confidence"] = confidence

    labeled_csv = marked_csv.with_name(marked_csv.name.replace("_marked.csv", "_labeled.csv"))
    df_out.to_csv(labeled_csv, index=False)

    sizes = cluster_sizes(labels)
    print(f"\n[{frame}] n_total={n_total} n_kept={n_kept} ({100 * n_kept / n_total:.1f}%)  "
          f"n_body_seed={n_body_seed}{'  [退化:随机初始化]' if degraded_seed else ''}")
    print(f"  簇点数占比(KMeans, aux_weight={AUX_WEIGHT_FINAL}x): " + "  ".join(
        f"cluster{c}={sizes[c]}({100 * sizes[c] / n_kept:.1f}%)" for c in range(K)))
    print("  簇语义(规则A): " + "  ".join(f"cluster{c}={lab}" for c, lab in sorted(mapping.items())))
    print(f"  疑似硬切簇对数={n_hardcut}  稳定性ARI: mean={stability['mean_ari']:.3f} "
          f"min={stability['min_ari']:.3f}")
    print(f"  confidence={confidence}  (n_hardcut<={CONF_N_HARDCUT_MAX} 且 "
          f"ari_min>{CONF_ARI_MIN_THRESH} => high, 否则 low; 种子退化或两翼强制拆分时无条件low)")
    if is_wing_merged:
        print("  wing连通性: wing_merged_forced_split(两翼贴一起，按投影中位数强制切分)")
    elif cc_diag["n_fragments"] > 0:
        print(f"  wing连通性: 碎块点={cc_diag['n_fragments']}  实际改判={len(cc_diag['reassigned'])}")
        for pt_idx, old, new in cc_diag["reassigned"]:
            print(f"    point#{pt_idx}: {old} -> {new}")
    else:
        print("  wing连通性: 两个wing簇均为单一连通整体")
    print(f"  L/R锚定: wing_A -> {lr_map['wing_A']}  wing_B -> {lr_map['wing_B']}")
    print("  part_label分布(全部点，含if_keep=False的1-NN传播): " + "  ".join(
        f"{lab}={int((part_label_full == lab).sum())}" for lab in sorted(PART_LABELS)))
    print(f"  labeled csv -> {labeled_csv}")

    return {"frame": frame, "confidence": confidence, "n_hardcut": n_hardcut,
            "ari_min": stability["min_ari"], "ari_mean": stability["mean_ari"],
            "n_body_seed": n_body_seed, "degraded_seed_init": degraded_seed,
            "is_wing_merged_forced": is_wing_merged, "n_total": n_total, "n_kept": n_kept,
            "labeled_csv": labeled_csv, "df_out": df_out}


def plot_labeled_reprojection(frame: str, df_out: pd.DataFrame, confidence: str, out_path: Path) -> None:
    """body先画(灰)、wing_L/wing_R后画(两色)盖上，if_keep=False点用叉号标记，2x2四相机。
    薄封装，通用重投影/画图逻辑见 postprocessing/viz/reprojection_viewer.py::plot_reprojection_overlay。"""
    plot_reprojection_overlay(frame, df_out, out_path, raw_data_dir=RAW_DATA_DIR,
                               title_suffix=f"  [confidence={confidence}]")


def run_batch(frames: list[str]) -> tuple[list[dict], list[dict]]:
    """逐帧处理，单帧异常catch住、跳过、记录帧号，不中断整个批处理(同
    postprocessing/cleaning/mark_floaters.py run_batch的约定)。"""
    results = []
    failures = []
    for frame in frames:
        try:
            r = process_frame(frame)
            reproj_path = LABEL_REPROJ_DIR / f"labeled_reproj_{frame}.png"
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
            "n_hardcut_pairs": r["n_hardcut"], "ari_min": r["ari_min"], "ari_mean": r["ari_mean"],
            "is_wing_merged_forced": r["is_wing_merged_forced"],
            "degraded_seed_init": r["degraded_seed_init"], "n_body_seed": r["n_body_seed"],
            "n_total": r["n_total"], "n_kept": r["n_kept"], "error": "",
        })
    for f in failures:
        rows.append({
            "frame_id": f["frame"], "status": "failed", "confidence": np.nan,
            "n_hardcut_pairs": np.nan, "ari_min": np.nan, "ari_mean": np.nan,
            "is_wing_merged_forced": np.nan, "degraded_seed_init": np.nan, "n_body_seed": np.nan,
            "n_total": np.nan, "n_kept": np.nan, "error": f["error"],
        })
    df = pd.DataFrame(rows)
    df["_frame_idx"] = df["frame_id"].str[1:].astype(int)
    return df.sort_values("_frame_idx").drop(columns="_frame_idx").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="批处理起始帧号(含)")
    parser.add_argument("--end", type=int, default=99, help="批处理结束帧号(含)")
    parser.add_argument("--dev", action="store_true",
                         help="只跑select_dev_frames.DEV_FRAMES(6帧)，忽略--start/--end")
    args = parser.parse_args()

    frames = DEV_FRAMES if args.dev else [f"f{i:04d}" for i in range(args.start, args.end + 1)]

    results, failures = run_batch(frames)

    summary_df = build_summary_df(results, failures)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    n_requested, n_ok, n_failed = len(frames), len(results), len(failures)
    high = [r["frame"] for r in results if r["confidence"] == "high"]
    low = [r["frame"] for r in results if r["confidence"] == "low"]
    degraded = [r["frame"] for r in results if r["degraded_seed_init"]]
    merged = [r["frame"] for r in results if r["is_wing_merged_forced"]]

    print(f"\n{'=' * 70}\n{n_requested}帧请求汇总\n{'=' * 70}")
    print(f"  成功: {n_ok}  失败: {n_failed}")
    if failures:
        print(f"  失败帧: {[f['frame'] for f in failures]}")
    if n_ok > 0:
        print(f"  high confidence: {len(high)}/{n_ok} ({100 * len(high) / n_ok:.1f}%)")
        print(f"  low  confidence: {len(low)}/{n_ok} ({100 * len(low) / n_ok:.1f}%)")
        print(f"  触发body种子退化(n_body_seed<{MIN_BODY_SEED}, 随机初始化)帧数: "
              f"{len(degraded)}/{n_ok}" + (f"  {degraded}" if degraded else ""))
        print(f"  触发wing_merged_forced_split帧数: {len(merged)}/{n_ok}"
              + (f"  {merged}" if merged else ""))
    print(f"  汇总表 -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
