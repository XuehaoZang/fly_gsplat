"""
analyze_g4_schedule.py
对G4双卡并行调度器(gpu/schedule/)一轮完整跑批的评估：
  outputs/ctrl_009_002_ratio3_sh0_full/ 640个任务(ratio3_sh0参数集,全部成功,
  schedule_summary.json记录总墙钟5814.77s)。

一次性只读分析脚本，不调用ns-train/generate_dataset/generate_hull，不动
gpu/schedule/或models/任何代码。可重复运行，幂等。

数据来源：
  - outputs/ctrl_009_002_ratio3_sh0_full/schedule_summary.json
      (n_ok/n_failed/wall_s 总墙钟)
  - outputs/ctrl_009_002_ratio3_sh0_full/_progress/gpu{0,1}_w{0..5}.jsonl
      (每个worker的独立进度日志，逐任务一行：worker_tag/gpu_index/frame/
      wall_s/scale_ratio_median/...)
  - 每个任务splat_dir下 debug_checkpoints/stats/step_*_stats.json
      (最终step的stats.json，嵌套结构 scale_ratio.median/p95/...)
    用于交叉校验progress jsonl里拍平的scale_ratio_median字段没有失真
    (逐任务核对，不只抽样)。

G3基线来源：gpu/parallel/PARALLEL_REPORT.md 第1节表格，N=6(单卡6并发，
G4每卡也是6个worker，档位对应)一行："聚合吞吐 223.0 frames/hour"，
换算 3600/223.0 = 16.14s/frame ≈ 用户给出的16.1s/frame。
注意：这个16.1s/frame是"单卡6个worker同时跑"这个系统级别的吞吐倒数，
不是单个worker自己处理一帧的墙钟时间(那个数字在同一张表里是94.28±1.21s，
因为6个worker并发，单worker耗时/6 ≈ 15.7s，量级上能对上16.1这个吞吐基线)。
所以第3张图里"各worker平均单帧耗时"(每个worker自己的墙钟均值，量级~100-160s)
和16.1s/frame参考线不是同一层级的量，图上会明显看到参考线远低于所有柱子——
这不是bug，而是必然的量级差，脚本里会同时打印/标注真正可比的"系统级"数字
(每卡有效帧率、全系统有效帧率)，避免柱状图被误读。
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

RUN_DIR = REPO / "outputs" / "ctrl_009_002_ratio3_sh0_full"
SCHEDULE_SUMMARY_JSON = RUN_DIR / "schedule_summary.json"
PROGRESS_DIR = RUN_DIR / "_progress"

OUT_DIR = Path(__file__).resolve().parent
EVAL_SUMMARY_JSON = OUT_DIR / "g4_schedule_eval_summary.json"
BOXPLOT_PNG = OUT_DIR / "g4_worker_frametime_boxplot.png"
SCALE_RATIO_PNG = OUT_DIR / "g4_scale_ratio_by_gpu.png"
BASELINE_BAR_PNG = OUT_DIR / "g4_worker_vs_g3_baseline.png"

# gpu/parallel/PARALLEL_REPORT.md 第1节, N=6行: 3600s / 223.0 frames/hour
G3_BASELINE_S_PER_FRAME = 16.1

# tab:blue / tab:orange: repo里既有分析脚本(compare_stats.py)已在用的配色，
# 两者色相/明度差异足够大，色觉障碍下也能靠形状/图例区分。
GPU_COLOR = {0: "tab:blue", 1: "tab:orange"}


# --------------------------------------------------------------- 数据加载 --

def load_progress_records() -> list[dict]:
    """-> 640条记录的list，每条来自_progress/*.jsonl的一行。"""
    records = []
    for jsonl_path in sorted(PROGRESS_DIR.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def verify_against_stats_json(records: list[dict]) -> list[tuple]:
    """逐任务核对progress jsonl里的scale_ratio_median和该任务splat_dir下
    最终step的stats.json里scale_ratio.median是否一致。返回不一致的列表
    (task_id, jsonl值, stats.json值)，正常应为空。"""
    mismatches = []
    missing = []
    for r in records:
        stats_dir = Path(r["splat_dir"]) / "debug_checkpoints" / "stats"
        stat_files = sorted(stats_dir.glob("step_*_stats.json"))
        if not stat_files:
            missing.append(r["task_id"])
            continue
        final_step = max(int(f.stem.split("_")[1]) for f in stat_files)
        final_file = stats_dir / f"step_{final_step:05d}_stats.json"
        d = json.load(open(final_file))
        v = d["scale_ratio"]["median"]
        if abs(v - r["scale_ratio_median"]) > 1e-6:
            mismatches.append((r["task_id"], v, r["scale_ratio_median"]))
    if missing:
        print(f"[警告] {len(missing)}个任务缺少stats.json: {missing[:10]}")
    return mismatches


# --------------------------------------------------------------- 主分析 --

def main():
    schedule_summary = json.load(open(SCHEDULE_SUMMARY_JSON))
    records = load_progress_records()

    print("=" * 70)
    print("总体情况 (schedule_summary.json)")
    print("=" * 70)
    print(f"n_ok={schedule_summary['n_ok']}  n_failed={schedule_summary['n_failed']}  "
          f"总墙钟={schedule_summary['wall_s']:.2f}s")
    print(f"_progress/*.jsonl 逐任务记录数: {len(records)}")
    assert len(records) == schedule_summary["n_ok"], "进度日志行数和schedule_summary.n_ok对不上"
    print()

    print("=" * 70)
    print("交叉校验: progress jsonl的scale_ratio_median vs 各任务stats.json(最终step)")
    print("=" * 70)
    mismatches = verify_against_stats_json(records)
    if mismatches:
        print(f"[异常] 发现{len(mismatches)}处不一致，前5条:")
        for task_id, stats_v, jsonl_v in mismatches[:5]:
            print(f"  {task_id}: stats.json={stats_v:.6g}  jsonl={jsonl_v:.6g}")
    else:
        print(f"全部{len(records)}条记录一致，未发现异常。")
    print()

    worker_tags = sorted({r["worker_tag"] for r in records},
                          key=lambda t: (int(t.split("_")[0][3:]), int(t.split("_w")[1])))
    by_worker = {w: [r for r in records if r["worker_tag"] == w] for w in worker_tags}
    gpu_of_worker = {w: by_worker[w][0]["gpu_index"] for w in worker_tags}

    # ----------------------------------------------------- 图1: 单帧耗时箱线图 --
    fig, ax = plt.subplots(figsize=(11, 6))
    wall_s_by_worker = [np.array([r["wall_s"] for r in by_worker[w]]) for w in worker_tags]
    bp = ax.boxplot(wall_s_by_worker, tick_labels=worker_tags, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5),
                     flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, w in zip(bp["boxes"], worker_tags):
        patch.set_facecolor(GPU_COLOR[gpu_of_worker[w]])
        patch.set_alpha(0.65)
    ax.set_ylabel("per-frame wall time, wall_s (s)")
    ax.set_xlabel("worker")
    ax.set_title(f"Per-worker per-frame wall time distribution (G4, {len(records)} tasks, 12 workers x 2 GPU)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GPU_COLOR[g], alpha=0.65, label=f"GPU{g}")
                      for g in (0, 1)]
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(BOXPLOT_PNG, dpi=150)
    plt.close(fig)
    print(f"[Saved] {BOXPLOT_PNG}")

    # ----------------------------------------------- 图2: scale_ratio.median分布 --
    scale_ratio_by_gpu = {
        g: np.array([r["scale_ratio_median"] for r in records if r["gpu_index"] == g])
        for g in (0, 1)
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(
        min(v.min() for v in scale_ratio_by_gpu.values()),
        max(v.max() for v in scale_ratio_by_gpu.values()),
        40,
    )
    for g in (0, 1):
        v = scale_ratio_by_gpu[g]
        ax.hist(v, bins=bins, alpha=0.55, color=GPU_COLOR[g], label=f"GPU{g} (n={len(v)})")
        ax.axvline(np.median(v), color=GPU_COLOR[g], linestyle="--", linewidth=1.5)
    ax.set_xlabel("scale_ratio.median")
    ax.set_ylabel("task count")
    ax.set_title("scale_ratio.median distribution (dashed = per-GPU median)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    for g in (0, 1):
        v = scale_ratio_by_gpu[g]
        xs = np.random.default_rng(0).normal(g, 0.05, size=len(v))
        ax.scatter(xs, v, s=10, alpha=0.5, color=GPU_COLOR[g], label=f"GPU{g}")
        ax.scatter([g], [np.median(v)], s=90, marker="_", color="black", linewidths=2, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GPU0", "GPU1"])
    ax.set_ylabel("scale_ratio.median")
    ax.set_title("scale_ratio.median by GPU, scatter (black tick = median)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("scale_ratio.median: GPU0 vs GPU1")
    plt.tight_layout()
    plt.savefig(SCALE_RATIO_PNG, dpi=150)
    plt.close(fig)
    print(f"[Saved] {SCALE_RATIO_PNG}")

    # ------------------------------------------- 图3: worker均值 vs G3基线 --
    means = np.array([wall_s_by_worker[i].mean() for i in range(len(worker_tags))])
    stds = np.array([wall_s_by_worker[i].std() for i in range(len(worker_tags))])
    colors = [GPU_COLOR[gpu_of_worker[w]] for w in worker_tags]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(worker_tags))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.75,
           error_kw=dict(elinewidth=1, ecolor="black"))
    ax.axhline(G3_BASELINE_S_PER_FRAME, color="red", linestyle="--", linewidth=1.5,
               label=f"G3 baseline (N=6/GPU throughput-derived) = {G3_BASELINE_S_PER_FRAME}s/frame")
    ax.set_xticks(x)
    ax.set_xticklabels(worker_tags, rotation=45)
    ax.set_ylabel("mean per-frame wall time ± std (s)")
    ax.set_title("Per-worker mean per-frame wall time vs G3 baseline")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GPU_COLOR[g], alpha=0.75, label=f"GPU{g}")
                      for g in (0, 1)]
    legend_handles.append(plt.Line2D([0], [0], color="red", linestyle="--",
                                      label=f"G3 baseline {G3_BASELINE_S_PER_FRAME}s/frame"))
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 系统级有效帧率(和G3基线同一层级、真正可比的数字)，标注在图上避免误读。
    total_wall_s = schedule_summary["wall_s"]
    n_ok = schedule_summary["n_ok"]
    n_frames_by_gpu = {g: sum(1 for r in records if r["gpu_index"] == g) for g in (0, 1)}
    per_gpu_effective = {g: total_wall_s / n_frames_by_gpu[g] for g in (0, 1)}
    system_effective = total_wall_s / n_ok
    note = (f"Note: bars are each worker's OWN mean wall time (serial per-worker),\n"
            f"not directly comparable to the G3 baseline (which is a throughput-\n"
            f"derived, 6-worker-concurrent system-level number). The genuinely\n"
            f"comparable numbers are: per-GPU effective rate "
            f"GPU0={per_gpu_effective[0]:.2f}s/frame, GPU1={per_gpu_effective[1]:.2f}s/frame\n"
            f"(whole-system 2-GPU/12-worker effective rate={system_effective:.2f}s/frame)")
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(BASELINE_BAR_PNG, dpi=150)
    plt.close(fig)
    print(f"[Saved] {BASELINE_BAR_PNG}")
    print()

    # ----------------------------------------------------------- 打印总结 --
    print("=" * 70)
    print("各worker单帧耗时均值±标准差 (s)")
    print("=" * 70)
    for w, m, s in zip(worker_tags, means, stds):
        print(f"  {w} (GPU{gpu_of_worker[w]}): n={len(by_worker[w])}  mean={m:.2f}  std={s:.2f}")
    print()
    print("=" * 70)
    print("系统级有效帧率 (真正和G3基线可比的数字)")
    print("=" * 70)
    print(f"  GPU0: {n_frames_by_gpu[0]}帧 / {total_wall_s:.2f}s总墙钟 = {per_gpu_effective[0]:.2f}s/frame")
    print(f"  GPU1: {n_frames_by_gpu[1]}帧 / {total_wall_s:.2f}s总墙钟 = {per_gpu_effective[1]:.2f}s/frame")
    print(f"  全系统: {n_ok}帧 / {total_wall_s:.2f}s总墙钟 = {system_effective:.2f}s/frame")
    print(f"  G3基线(单卡N=6): {G3_BASELINE_S_PER_FRAME}s/frame")
    print(f"  单卡有效帧率 vs G3基线: "
          f"GPU0 {(per_gpu_effective[0]/G3_BASELINE_S_PER_FRAME-1)*100:+.1f}%, "
          f"GPU1 {(per_gpu_effective[1]/G3_BASELINE_S_PER_FRAME-1)*100:+.1f}%")
    print()

    scale_ratio_gpu_stats = {
        f"gpu{g}": {
            "n": int(len(scale_ratio_by_gpu[g])),
            "mean": float(scale_ratio_by_gpu[g].mean()),
            "median": float(np.median(scale_ratio_by_gpu[g])),
            "std": float(scale_ratio_by_gpu[g].std()),
        }
        for g in (0, 1)
    }
    print("=" * 70)
    print("scale_ratio.median 按GPU分组统计")
    print("=" * 70)
    for g in (0, 1):
        s = scale_ratio_gpu_stats[f"gpu{g}"]
        print(f"  GPU{g}: n={s['n']}  mean={s['mean']:.4f}  median={s['median']:.4f}  std={s['std']:.4f}")
    print()

    eval_summary = {
        "schedule_summary": schedule_summary,
        "n_records": len(records),
        "stats_json_cross_check_mismatches": len(mismatches),
        "worker_wall_s_stats": {
            w: {"gpu_index": gpu_of_worker[w], "n": len(by_worker[w]),
                "mean_s": float(means[i]), "std_s": float(stds[i])}
            for i, w in enumerate(worker_tags)
        },
        "scale_ratio_median_by_gpu": scale_ratio_gpu_stats,
        "g3_baseline_s_per_frame": G3_BASELINE_S_PER_FRAME,
        "g3_baseline_source": "gpu/parallel/PARALLEL_REPORT.md 第1节 N=6行: 3600/223.0f/h=16.14s/frame",
        "per_gpu_effective_s_per_frame": per_gpu_effective,
        "system_effective_s_per_frame": system_effective,
    }
    with open(EVAL_SUMMARY_JSON, "w") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {EVAL_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
