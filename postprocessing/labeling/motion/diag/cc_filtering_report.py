"""诊断任务3: CC(连通分量)步骤实际过滤效果报告。

背景: label.py::run_batch跑完的motion_dev_summary.csv里is_wing_merged_forced全是
False(见memory: 原8帧全部如此)，看上去CC这一步"从没起作用"。但这个印象只看了
"有没有触发wing_merged强制拆分"这一个最终判据，没看CC分析本身在中间到底丢没丢点/
体素——这个脚本直接调用density.py::extract_body_voxels和label.py::split_wing_candidates
内部已经算出来的候选数/主分量数/碎块数，如实报告，不改这两个函数的实现和输出schema
(只是从外部调用现成的公开函数，不需要改label.py/density.py代码)。

用法:
    python -m postprocessing.labeling.motion.diag.cc_filtering_report
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import load_marked  # noqa: E402
from postprocessing.labeling.motion import density as d  # noqa: E402
from postprocessing.labeling.motion import label as L  # noqa: E402
from postprocessing.labeling.motion.diag.body_centroid_stability import ALL_DEV_FRAMES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"


def compute_frame_cc_diagnostics(frame_idx: int, dataset_dir: Path = d.DATASET_DIR) -> dict | None:
    """对单帧重新跑一遍body候选/wing候选分类(复用label.py的公开函数，不重复实现CC逻辑)，
    抠出"候选 -> CC后"两边的计数，body/wing两段分开报告。"""
    frame = f"f{frame_idx:04d}"
    try:
        df_full, _ = load_marked(frame, data_root=dataset_dir)
    except FileNotFoundError:
        print(f"[cc_filtering_report] {frame}: 找不到_marked.csv，跳过")
        return None
    kept_mask = df_full["if_keep"].astype(bool).to_numpy()
    df_kept = df_full[kept_mask].reset_index(drop=True)
    xyz_kept = df_kept[["x", "y", "z"]].to_numpy()

    is_body, density_info = L.classify_body_candidate(df_kept, frame_idx)
    voxel_counts = density_info["voxel_counts"]
    n_body_candidate_voxels = int((voxel_counts > d.BODY_VOXEL_COUNT_THRESH).sum())
    n_body_final_voxels = len(density_info["body_voxels"])
    n_body_dropped_voxels = n_body_candidate_voxels - n_body_final_voxels

    semantic, comp_sizes, split_diag = L.split_wing_candidates(xyz_kept, is_body)
    n_wing_candidate_points = int((~is_body).sum())
    sorted_sizes = sorted(comp_sizes.tolist(), reverse=True)
    top2 = sorted_sizes[:2] + [0] * max(0, 2 - len(sorted_sizes))
    n_wing_fragment_points = split_diag["n_fragments"]
    n_wing_main_points = sum(top2)

    return {
        "frame": frame, "frame_idx": frame_idx,
        "n_body_candidate_voxels": n_body_candidate_voxels,
        "n_body_final_voxels": n_body_final_voxels,
        "n_body_dropped_voxels": n_body_dropped_voxels,
        "body_dropped_frac": n_body_dropped_voxels / n_body_candidate_voxels if n_body_candidate_voxels else 0.0,
        "n_wing_candidate_points": n_wing_candidate_points,
        "wing_main_comp_1": top2[0], "wing_main_comp_2": top2[1],
        "n_wing_fragment_points": n_wing_fragment_points,
        "wing_fragment_frac": n_wing_fragment_points / n_wing_candidate_points if n_wing_candidate_points else 0.0,
        "degenerate_wing_split": split_diag.get("degenerate_wing_split", False),
    }


def run(frame_indices: list[int] = ALL_DEV_FRAMES) -> pd.DataFrame:
    rows = [r for idx in frame_indices if (r := compute_frame_cc_diagnostics(idx)) is not None]
    report_df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "cc_filtering_report.csv"
    report_df.to_csv(csv_path, index=False)

    n_body_effective = int((report_df["n_body_dropped_voxels"] > 0).sum())
    n_wing_effective = int((report_df["n_wing_fragment_points"] > 0).sum())
    total_body_dropped = int(report_df["n_body_dropped_voxels"].sum())
    total_wing_fragment = int(report_df["n_wing_fragment_points"].sum())

    print(f"\n[cc_filtering_report] {len(report_df)}帧CC诊断汇总:")
    print(report_df[["frame", "n_body_candidate_voxels", "n_body_final_voxels", "n_body_dropped_voxels",
                      "n_wing_candidate_points", "wing_main_comp_1", "wing_main_comp_2",
                      "n_wing_fragment_points"]].to_string(index=False))

    print(f"\n  body CC: {n_body_effective}/{len(report_df)}帧实际丢弃了>=1个候选体素"
          f"(丢弃碎块体素总计{total_body_dropped}个，跨13帧)")
    print(f"  wing CC: {n_wing_effective}/{len(report_df)}帧wing候选点分连通分量后有碎块被按最近距离改判"
          f"(碎块点总计{total_wing_fragment}个，跨13帧)")
    if n_body_effective == 0 and n_wing_effective == 0:
        print("  [结论] 这13帧里CC步骤(body体素连通分量+wing点连通分量)完全没有丢弃/改判任何"
              "候选，从头到尾就是候选集合本身，CC分析这一步在当前数据/阈值下没有实际发挥过滤作用。")
    else:
        print("  [结论] CC步骤在部分帧里确实丢弃/改判了候选点/体素(不是完全没用)，"
              "但具体量级见上表，跟总候选规模比是否显著还需要结合body_dropped_frac/"
              "wing_fragment_frac列判断(已存入csv)。")
    print(f"  csv -> {csv_path}")
    return report_df


if __name__ == "__main__":
    run()
