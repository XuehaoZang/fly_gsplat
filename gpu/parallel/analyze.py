"""
analyze.py
读取 scan.py 产出的 results/scan_raw.json，汇总成:
  - results/scan_summary.csv          (每个并发档位: 聚合吞吐/单帧墙钟时间/拖慢比例/GPU-CPU利用率/显存峰值)
  - results/throughput_slowdown.png   (吞吐 vs 并发数 + 单帧墙钟时间 vs 并发数，2子图)
  - results/gpu_cpu_util_grid.png     (每个并发档位一个子图，GPU%/CPU%曲线，2次重复叠加)
不改动 scan.py/worker.py，独立后处理脚本。

重要说明(CPU采样器bug)：L1/L2两个档位是用有bug的CPU采样器(gpu/parallel/samplers_multi.py
修复前版本)跑的——新发现的子进程在被prime的同一轮就参与求和，导致分母(墙钟时间差)
趋近于0而分子(该进程可能已经用多线程跑了一阵子的CPU时间)不小，产生物理上不可能的
瞬时读数(单样本能到>10000%，而28线程机器物理上限是2800%)。samplers_multi.py已修复
(把新discover的pid在prime当轮的求和里跳过，从下一轮开始才计入)，L3/L4/L6三个档位
用的是修复后的采样器。为了让L1/L2的cpu_pct_mean/max仍然可用于跨档位比较，这里对所有
档位的CPU原始采样一律过滤掉 >2800% 的物理不可能样本再统计——这只是让展示的数字不被
个别bug artifact污染，不代表L1/L2的CPU利用率本身有什么特殊之处，L3/L4/L6的过滤基本
不会删除任何样本(本来就没有超过2800%的点)。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PHYSICAL_CPU_PCT_CEILING = 2800.0  # 28线程 x 100%


def load():
    with open(RESULTS_DIR / "scan_raw.json") as f:
        return json.load(f)


def group_by_level(records):
    levels = {}
    for r in records:
        levels.setdefault(r["n_conc"], []).append(r)
    return dict(sorted(levels.items()))


def frame_e2e_values(rep_record):
    vals = []
    for wr in rep_record["worker_records"]:
        for frame in wr:
            if "error" not in frame:
                vals.append(frame["e2e_s"])
    return vals


def filtered_cpu_stats(rep_record):
    """过滤掉物理上不可能(>2800%)的CPU%采样点后再算mean/max，见文件头说明。"""
    vals = [s[1] for s in rep_record["cpu_samples"] if s[1] <= PHYSICAL_CPU_PCT_CEILING]
    n_dropped = len(rep_record["cpu_samples"]) - len(vals)
    if not vals:
        return float("nan"), float("nan"), n_dropped
    return float(np.mean(vals)), float(np.max(vals)), n_dropped


def summarize(levels: dict) -> list:
    baseline_e2e = []
    for r in levels[1]:
        baseline_e2e.extend(frame_e2e_values(r))
    baseline_mean = float(np.mean(baseline_e2e))

    rows = []
    for n_conc, reps in levels.items():
        total_frames_ok = sum(r["n_frames_ok"] for r in reps)
        total_wall_s = sum(r["wall_s"] for r in reps)
        throughput_f_h = total_frames_ok / total_wall_s * 3600

        all_e2e = []
        for r in reps:
            all_e2e.extend(frame_e2e_values(r))
        e2e_mean, e2e_std = float(np.mean(all_e2e)), float(np.std(all_e2e))
        slowdown_vs_l1 = e2e_mean / baseline_mean

        gpu_util_means = [r["gpu_util_mean"] for r in reps]
        gpu_util_maxes = [r["gpu_util_max"] for r in reps]
        mem_maxes = [r["gpu_mem_max_mib"] for r in reps]

        cpu_means, cpu_maxes, cpu_dropped = [], [], 0
        for r in reps:
            m, mx, nd = filtered_cpu_stats(r)
            cpu_means.append(m)
            cpu_maxes.append(mx)
            cpu_dropped += nd

        rows.append({
            "n_conc": n_conc,
            "n_repeats": len(reps),
            "total_frames_ok": total_frames_ok,
            "total_frames_failed": sum(r["n_frames_failed"] for r in reps),
            "throughput_f_h": throughput_f_h,
            "e2e_mean_s": e2e_mean,
            "e2e_std_s": e2e_std,
            "slowdown_vs_l1": slowdown_vs_l1,
            "gpu_util_mean_pct": float(np.mean(gpu_util_means)),
            "gpu_util_max_pct": float(np.max(gpu_util_maxes)),
            "cpu_pct_mean_filtered": float(np.mean(cpu_means)),
            "cpu_pct_max_filtered": float(np.max(cpu_maxes)),
            "cpu_samples_dropped_as_artifact": cpu_dropped,
            "gpu_mem_max_mib": float(np.max(mem_maxes)),
        })
    return rows


def write_csv(rows: list):
    header = ["n_conc", "throughput_f_h", "e2e_mean_s", "e2e_std_s", "slowdown_vs_l1",
              "gpu_util_mean_pct", "gpu_util_max_pct", "cpu_pct_mean_filtered",
              "cpu_pct_max_filtered", "gpu_mem_max_mib", "cpu_samples_dropped_as_artifact"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(
            f"{row[k]:.2f}" if isinstance(row[k], float) else str(row[k])
            for k in header
        ))
    (RESULTS_DIR / "scan_summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_throughput_slowdown(rows: list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ns = [r["n_conc"] for r in rows]
    thpt = [r["throughput_f_h"] for r in rows]
    ax = axes[0]
    ax.plot(ns, thpt, marker="o", linewidth=2, color="tab:blue")
    for n, t in zip(ns, thpt):
        ax.annotate(f"{t:.0f}", (n, t), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("concurrent processes per GPU (N)")
    ax.set_ylabel("aggregate throughput (frames/hour)")
    ax.set_title("Throughput vs concurrency (single GPU)")
    ax.set_xticks(ns)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    e2e_mean = [r["e2e_mean_s"] for r in rows]
    e2e_std = [r["e2e_std_s"] for r in rows]
    slowdown = [r["slowdown_vs_l1"] for r in rows]
    ax2.errorbar(ns, e2e_mean, yerr=e2e_std, marker="o", linewidth=2, color="tab:red", capsize=4)
    for n, m, s in zip(ns, e2e_mean, slowdown):
        ax2.annotate(f"{m:.1f}s\n({(s-1)*100:+.1f}%)", (n, m), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)
    ax2.axhline(e2e_mean[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label="N=1 baseline mean")
    ax2.set_xlabel("concurrent processes per GPU (N)")
    ax2.set_ylabel("per-frame end-to-end wall time (s)")
    ax2.set_title("Per-frame wall time vs concurrency\n(% = slowdown relative to N=1 baseline)")
    ax2.set_xticks(ns)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "throughput_slowdown.png", dpi=150)
    print(f"[Saved] {RESULTS_DIR / 'throughput_slowdown.png'}")


def plot_util_grid(levels: dict):
    n_levels = len(levels)
    fig, axes = plt.subplots(n_levels, 2, figsize=(13, 3.2 * n_levels), squeeze=False)

    for row_idx, (n_conc, reps) in enumerate(levels.items()):
        ax_gpu, ax_cpu = axes[row_idx]
        for i, r in enumerate(reps):
            gs = r["gpu_samples"]
            ax_gpu.plot([s[0] for s in gs], [s[1] for s in gs], alpha=0.7, linewidth=1,
                        label=f"rep {i}")
            cs = [s for s in r["cpu_samples"] if s[1] <= PHYSICAL_CPU_PCT_CEILING]
            ax_cpu.plot([s[0] for s in cs], [s[1] for s in cs], alpha=0.7, linewidth=1,
                        label=f"rep {i}")
        ax_gpu.set_ylim(-5, 105)
        ax_gpu.set_title(f"N={n_conc}: whole-GPU utilization (%)")
        ax_gpu.set_xlabel("wall time (s)")
        ax_gpu.legend(fontsize=7)
        ax_gpu.grid(alpha=0.3)

        ax_cpu.axhline(PHYSICAL_CPU_PCT_CEILING, color="red", linestyle="--", linewidth=1,
                       alpha=0.6, label="28-thread ceiling (2800%)")
        ax_cpu.set_title(f"N={n_conc}: aggregate CPU% (sum over all worker process trees)"
                         + (" [old sampler, see caveat]" if n_conc in (1, 2) else ""))
        ax_cpu.set_xlabel("wall time (s)")
        ax_cpu.legend(fontsize=7)
        ax_cpu.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gpu_cpu_util_grid.png", dpi=150)
    print(f"[Saved] {RESULTS_DIR / 'gpu_cpu_util_grid.png'}")


def main():
    records = load()
    levels = group_by_level(records)
    rows = summarize(levels)
    write_csv(rows)
    plot_throughput_slowdown(rows)
    plot_util_grid(levels)


if __name__ == "__main__":
    main()
