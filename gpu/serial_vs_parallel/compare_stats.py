"""
compare_stats.py
G5第一层验证：对比新并行调度器(G4, gpu/schedule/)和旧串行脚本
(debug/batch_8groups_100frames.py)在相同参数(ratio3_sh0 == G2b_G9)、
相同帧范围(f0000-f0099)下产出的训练指标是否一致。

一次性分析脚本，只读 outputs/ 下已有的产出文件，不调用ns-train/generate_dataset/
generate_hull，不动 models/ 或 gpu/schedule/ 任何代码。可重复运行，幂等。

数据来源：
  serial:   outputs/ctrl_009_002_8groups_100frames/raw_records.json["G2b_G9"]
  parallel: outputs/ctrl_009_002_ratio3_sh0_full/_progress/*.jsonl
            (param_set=="ratio3_sh0" and frame<100)
两边densify窗口/正则化参数已核对一致(warmup-length=50, stop-split-at=1800,
use-scale-regularization=True, max-gauss-ratio=3.0, sh-degree=0)。
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

SERIAL_RAW = REPO / "outputs" / "ctrl_009_002_8groups_100frames" / "raw_records.json"
SERIAL_KEY = "G2b_G9"
PARALLEL_PROGRESS_DIR = REPO / "outputs" / "ctrl_009_002_ratio3_sh0_full" / "_progress"
PARALLEL_PARAM_SET = "ratio3_sh0"
N_FRAMES = 100

OUT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = OUT_DIR / "stats_diff_summary.json"
SCATTER_PNG = OUT_DIR / "scatter_key_metrics.png"

METRICS = [
    "n_gaussians",
    "scale_ratio_median",
    "scale_ratio_p95",
    "scale_ratio_frac_over_10",
    "opacity_median",
    "low_opacity_frac",
    "bbox_extent_max",
    "extent_overshoot",
    "dbscan_floater_frac",
]

REL_DIFF_FLAG_PCT = 5.0  # 仅用于肉眼标出，不作为通过/不通过判据


# --------------------------------------------------------------- 数据加载 --

def load_serial() -> dict:
    """-> {frame: record}，只取 status=='ok'。"""
    with open(SERIAL_RAW) as f:
        all_records = json.load(f)
    records = all_records[SERIAL_KEY]
    return {r["frame"]: r for r in records if r["status"] == "ok" and r["frame"] < N_FRAMES}


def load_parallel() -> dict:
    """-> {frame: record}，只取 param_set==PARALLEL_PARAM_SET and frame<N_FRAMES and status=='ok'。"""
    by_frame = {}
    for jsonl_path in sorted(PARALLEL_PROGRESS_DIR.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("param_set") != PARALLEL_PARAM_SET:
                    continue
                if rec.get("frame", -1) >= N_FRAMES:
                    continue
                if rec.get("status") != "ok":
                    continue
                by_frame[rec["frame"]] = rec
    return by_frame


# --------------------------------------------------------------- jitter --

def jitter_score(frame_to_record: dict) -> dict:
    """按frame排序后一阶差分的std，和debug/batch_8groups_100frames.py::jitter_score逻辑一致。"""
    ok = sorted(frame_to_record.values(), key=lambda r: r["frame"])
    if len(ok) < 2:
        return {"bbox_extent_jitter": float("nan"), "n_gaussians_jitter": float("nan")}
    extent_seq = np.array([r["bbox_extent_max"] for r in ok])
    ngauss_seq = np.array([r["n_gaussians"] for r in ok])
    return {
        "bbox_extent_jitter": float(np.std(np.diff(extent_seq))),
        "n_gaussians_jitter": float(np.std(np.diff(ngauss_seq))),
    }


# --------------------------------------------------------------- 主分析 --

def main():
    serial = load_serial()
    parallel = load_parallel()

    serial_frames = set(serial.keys())
    parallel_frames = set(parallel.keys())
    common_frames = sorted(serial_frames & parallel_frames)
    only_serial = sorted(serial_frames - parallel_frames)
    only_parallel = sorted(parallel_frames - serial_frames)

    print("=" * 70)
    print("帧覆盖情况")
    print("=" * 70)
    print(f"serial   status=='ok' 帧数: {len(serial_frames)}/{N_FRAMES}")
    print(f"parallel status=='ok' 帧数: {len(parallel_frames)}/{N_FRAMES}")
    print(f"两边共同(inner join) 帧数: {len(common_frames)}")
    if only_serial:
        print(f"仅serial有、parallel缺失的frame: {only_serial}")
    if only_parallel:
        print(f"仅parallel有、serial缺失的frame: {only_parallel}")
    if not only_serial and not only_parallel:
        print("两边帧集合完全一致。")
    print()

    # ----------------------------------------------------------- 逐帧diff --
    metric_stats = {}
    for metric in METRICS:
        diffs = []
        rel_diffs = []
        skipped_zero_or_nan = 0
        for frame in common_frames:
            sv = serial[frame][metric]
            pv = parallel[frame][metric]
            if sv is None or pv is None:
                skipped_zero_or_nan += 1
                continue
            sv, pv = float(sv), float(pv)
            if np.isnan(sv) or np.isnan(pv):
                skipped_zero_or_nan += 1
                continue
            diff = pv - sv
            diffs.append(diff)
            if sv == 0:
                skipped_zero_or_nan += 1
                continue
            rel_diffs.append(diff / sv)

        diffs_arr = np.array(diffs) if diffs else np.array([np.nan])
        rel_arr = np.array(rel_diffs) if rel_diffs else np.array([np.nan])

        metric_stats[metric] = {
            "n_common_frames": len(common_frames),
            "n_diff_computed": len(diffs),
            "n_rel_diff_computed": len(rel_diffs),
            "n_skipped_zero_or_nan": skipped_zero_or_nan,
            "mean_diff": float(np.mean(diffs_arr)),
            "std_diff": float(np.std(diffs_arr)),
            "median_diff": float(np.median(diffs_arr)),
            "max_abs_diff": float(np.max(np.abs(diffs_arr))),
            "mean_rel_diff_pct": float(np.mean(rel_arr) * 100),
            "max_abs_rel_diff_pct": float(np.max(np.abs(rel_arr)) * 100),
        }

    print("=" * 70)
    print("逐指标diff统计 (diff = parallel - serial)")
    print("=" * 70)
    header = f"{'metric':<28}{'mean_diff':>14}{'std_diff':>14}{'median_diff':>14}{'max_abs_diff':>14}{'mean_rel%':>12}{'max_abs_rel%':>14}"
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        s = metric_stats[metric]
        print(f"{metric:<28}{s['mean_diff']:>14.6g}{s['std_diff']:>14.6g}{s['median_diff']:>14.6g}"
              f"{s['max_abs_diff']:>14.6g}{s['mean_rel_diff_pct']:>12.4g}{s['max_abs_rel_diff_pct']:>14.4g}")
    print()

    # ----------------------------------------------------------- jitter --
    serial_jitter = jitter_score(serial)
    parallel_jitter = jitter_score(parallel)

    print("=" * 70)
    print("帧间稳定性(jitter) 对比：各组自己算，不是逐帧diff")
    print("=" * 70)
    print(f"{'':<20}{'serial':>16}{'parallel':>16}{'ratio(parallel/serial)':>26}")
    for key in ["bbox_extent_jitter", "n_gaussians_jitter"]:
        sv, pv = serial_jitter[key], parallel_jitter[key]
        ratio = pv / sv if sv not in (0, float("nan")) and not np.isnan(sv) else float("nan")
        print(f"{key:<20}{sv:>16.6g}{pv:>16.6g}{ratio:>26.4g}")
    print()

    # ----------------------------------------------------------- 存json --
    summary_out = {
        "n_serial_ok": len(serial_frames),
        "n_parallel_ok": len(parallel_frames),
        "n_common_frames": len(common_frames),
        "only_serial_frames": only_serial,
        "only_parallel_frames": only_parallel,
        "metric_diff_stats": metric_stats,
        "jitter": {
            "serial": serial_jitter,
            "parallel": parallel_jitter,
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"[Saved] {SUMMARY_JSON}")

    # ----------------------------------------------------------- 画图 --
    scatter_metrics = ["n_gaussians", "scale_ratio_median", "bbox_extent_max", "extent_overshoot"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, metric in zip(axes.flat, scatter_metrics):
        xs = np.array([serial[f][metric] for f in common_frames], dtype=float)
        ys = np.array([parallel[f][metric] for f in common_frames], dtype=float)
        ax.scatter(xs, ys, s=14, alpha=0.6, color="tab:blue")
        lo = min(xs.min(), ys.min())
        hi = max(xs.max(), ys.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1, label="y=x")
        ax.set_xlabel(f"serial: {metric}")
        ax.set_ylabel(f"parallel: {metric}")
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.suptitle(f"serial(G2b_G9) vs parallel(ratio3_sh0), f0000-f{N_FRAMES - 1:04d}, n={len(common_frames)}")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=150)
    plt.close(fig)
    print(f"[Saved] {SCATTER_PNG}")
    print()

    # ----------------------------------------------------------- 总结 --
    print("=" * 70)
    print("总结")
    print("=" * 70)
    flagged = [m for m in METRICS if abs(metric_stats[m]["mean_rel_diff_pct"]) > REL_DIFF_FLAG_PCT]
    if flagged:
        print(f"以下指标 mean_rel_diff_pct 超过 {REL_DIFF_FLAG_PCT}%，标出以供肉眼检查(不代表判定为不通过):")
        for m in flagged:
            print(f"  - {m}: mean_rel_diff_pct={metric_stats[m]['mean_rel_diff_pct']:.4g}%")
    else:
        print(f"没有指标的 mean_rel_diff_pct 超过 {REL_DIFF_FLAG_PCT}%。")
    print()
    print(f"jitter量级对比: bbox_extent_jitter serial={serial_jitter['bbox_extent_jitter']:.6g} "
          f"vs parallel={parallel_jitter['bbox_extent_jitter']:.6g}")
    print(f"jitter量级对比: n_gaussians_jitter serial={serial_jitter['n_gaussians_jitter']:.6g} "
          f"vs parallel={parallel_jitter['n_gaussians_jitter']:.6g}")


if __name__ == "__main__":
    main()
