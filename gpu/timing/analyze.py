"""
analyze.py
读取 time_pipeline.py 产出的 results/timing_raw.json，汇总成:
  - results/timing_summary.csv  (各阶段均值+标准差)
  - results/gpu_cpu_util.png    (5次重复的GPU利用率曲线 + CPU%曲线叠加图)
不改动 time_pipeline.py，独立后处理脚本。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load():
    with open(RESULTS_DIR / "timing_raw.json") as f:
        return json.load(f)


def summarize(records: list) -> dict:
    def stat(vals):
        vals = np.array(vals, dtype=float)
        return float(np.mean(vals)), float(np.std(vals))

    fields = {
        "generate_dataset_wall_s": [r["generate_dataset"]["wall_s"] for r in records],
        "generate_dataset_read_mb_per_s": [r["generate_dataset"]["read_mb_per_s"] for r in records],
        "generate_hull_wall_s": [r["generate_hull"]["wall_s"] for r in records],
        "generate_hull_cpu_busy_frac": [r["generate_hull"]["cpu_busy_frac"] for r in records],
        "train_coldstart_s": [r["train"]["coldstart_s"] for r in records],
        "train_loop_s": [r["train"]["train_loop_s"] for r in records],
        "train_total_s": [r["train"]["total_wall_s"] for r in records],
        "train_gpu_util_mean_pct": [r["train"]["gpu_util_mean"] for r in records],
        "train_gpu_util_max_pct": [r["train"]["gpu_util_max"] for r in records],
        "postprocess_export_s": [r["postprocess"]["export_splat_s"] for r in records],
        "postprocess_load_ply_s": [r["postprocess"]["load_ply_s"] for r in records],
        "postprocess_dbscan_s": [r["postprocess"]["dbscan_clean_s"] for r in records],
        "postprocess_total_s": [r["postprocess"]["total_s"] for r in records],
    }

    summary = {}
    for k, vals in fields.items():
        mean, std = stat(vals)
        summary[k] = {"mean": mean, "std": std, "n": len(vals), "values": vals}

    # end-to-end wall clock per repeat (all 5 stages)
    e2e = [
        r["generate_dataset"]["wall_s"] + r["generate_hull"]["wall_s"] +
        r["train"]["total_wall_s"] + r["postprocess"]["total_s"]
        for r in records
    ]
    mean, std = stat(e2e)
    summary["end_to_end_wall_s"] = {"mean": mean, "std": std, "n": len(e2e), "values": e2e}

    return summary


def write_csv(summary: dict):
    lines = ["stage,mean_s,std_s,share_of_e2e_pct"]
    e2e_mean = summary["end_to_end_wall_s"]["mean"]
    stage_keys = [
        ("generate_dataset_wall_s", "1_generate_dataset"),
        ("generate_hull_wall_s", "2_generate_hull"),
        ("train_coldstart_s", "3_train_coldstart"),
        ("train_loop_s", "4_train_loop"),
        ("postprocess_total_s", "5_postprocess"),
    ]
    for key, label in stage_keys:
        m, s = summary[key]["mean"], summary[key]["std"]
        lines.append(f"{label},{m:.3f},{s:.3f},{100*m/e2e_mean:.1f}")
    lines.append(f"end_to_end,{e2e_mean:.3f},{summary['end_to_end_wall_s']['std']:.3f},100.0")
    (RESULTS_DIR / "timing_summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_util(records: list):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    ax = axes[0]
    for i, r in enumerate(records):
        samples = r["train"]["gpu_samples"]
        ts = [s[0] for s in samples]
        us = [s[1] for s in samples]
        ax.plot(ts, us, alpha=0.7, linewidth=1, label=f"rep {i}")
        coldstart = r["train"]["coldstart_s"]
        if coldstart == coldstart:  # not nan
            ax.axvline(coldstart, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("GPU utilization during ns-train subprocess (5 repeats overlaid)\n"
                 "dashed lines = detected cold-start -> train-loop boundary per repeat")
    ax.set_xlabel("time since subprocess start (s)")
    ax.set_ylabel("GPU util (%)")
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 100)

    ax2 = axes[1]
    for i, r in enumerate(records):
        samples = r["train"]["cpu_samples"]
        ts = [s[0] for s in samples]
        cs = [s[1] for s in samples]
        ax2.plot(ts, cs, alpha=0.7, linewidth=1, label=f"rep {i}")
    ax2.set_title("CPU utilization (sum over ns-train process + children, %-of-1-core) during ns-train")
    ax2.set_xlabel("time since subprocess start (s)")
    ax2.set_ylabel("CPU % (can exceed 100 with multiple threads)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gpu_cpu_util.png", dpi=150)
    print(f"[Saved] {RESULTS_DIR / 'gpu_cpu_util.png'}")


def main():
    records = load()
    summary = summarize(records)
    with open(RESULTS_DIR / "timing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(summary)
    plot_util(records)


if __name__ == "__main__":
    main()
