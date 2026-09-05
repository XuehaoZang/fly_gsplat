"""
aggregate_round1.py

Round 1 / Round 1.5（sweep_hyper_params.md）结果聚合脚本。读 gpu/schedule/schedule.py
跑出来的 outputs/<name>/_progress/*.jsonl，按 param_set 汇总 n_gaussians/scale_ratio/
opacity/extent_overshoot 等指标的mean/median，两个视频(ctrl_119_004/010)的同名param_set
合并成一行(dev子集本来就是"每视频各100帧覆盖>=1个完整拍打周期"，两视频合并才是本轮真正
要看的数字)。baseline不重跑——从已有的480帧全量sweep(outputs/ctrl_3cam_test/ctrl_119_*)
的progress jsonl里按dev帧范围过滤出来，成本为0。

不算dbscan_floater_frac(需要先跑T1+T2，是Round2/3对少数候选组再补的可选步骤，见
sweep_hyper_params.md "评估指标"一节)，这里只用run_task()已经免费产出的4个指标。

用法:
    python -m gpu.schedule.analysis.aggregate_round1
    (或 python gpu/schedule/analysis/aggregate_round1.py，脚本自己处理sys.path)
"""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

OUTPUTS = REPO / "outputs"
BASELINE_SWEEPS = {
    "004": (OUTPUTS / "ctrl_3cam_test" / "ctrl_119_004", "ratio3_sh0_dense", (730, 830)),
    "010": (OUTPUTS / "ctrl_3cam_test" / "ctrl_119_010", "ratio3_sh0_dense", (373, 473)),
}

METRICS = ["n_gaussians", "scale_ratio_median", "scale_ratio_p95", "scale_ratio_frac_over_10",
           "opacity_median", "low_opacity_frac", "extent_overshoot", "wall_s"]


def load_progress(out_base_dir: Path) -> list[dict]:
    records = []
    progress_dir = out_base_dir / "_progress"
    if not progress_dir.exists():
        return records
    for jsonl_path in sorted(progress_dir.glob("*.jsonl")):
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    out = {"n_ok": len(ok), "n_failed": len(records) - len(ok)}
    for metric in METRICS:
        vals = [r[metric] for r in ok if metric in r and r[metric] is not None]
        if vals:
            out[f"{metric}_mean"] = statistics.mean(vals)
            out[f"{metric}_median"] = statistics.median(vals)
    return out


def collect_baseline() -> dict:
    all_records = []
    for mov, (out_dir, param_set, (start, end)) in BASELINE_SWEEPS.items():
        records = load_progress(out_dir)
        filtered = [r for r in records if r["param_set"] == param_set and start <= r["frame"] < end]
        all_records += filtered
    return summarize(all_records)


def collect_round1_core() -> dict[str, dict]:
    """core config的param_set在两个视频config里同名，合并两视频后按param_set分组。"""
    by_param_set: dict[str, list[dict]] = {}
    for mov in ("004", "010"):
        out_dir = OUTPUTS / "round1" / f"ctrl_119_{mov}_core"
        for r in load_progress(out_dir):
            by_param_set.setdefault(r["param_set"], []).append(r)
    return {ps: summarize(recs) for ps, recs in by_param_set.items()}


def collect_round1_iters() -> dict[str, dict]:
    by_iters: dict[str, list[dict]] = {}
    for mov in ("004", "010"):
        for iters in (1000, 1500, 2000, 3000):
            if iters == 2000:
                continue  # 2000 iters是baseline，已经在collect_baseline里
            out_dir = OUTPUTS / "round1" / f"ctrl_119_{mov}_iters{iters}"
            recs = load_progress(out_dir)
            if recs:
                by_iters.setdefault(f"P5_iters{iters}", []).extend(recs)
    return {name: summarize(recs) for name, recs in by_iters.items()}


def collect_round1_5() -> dict[str, dict]:
    variants = ["p7_thresh20", "p7_thresh50", "p8_leg_erosion", "p9_hull30k", "p9_hull100k"]
    by_variant: dict[str, list[dict]] = {}
    for mov in ("004", "010"):
        for variant in variants:
            out_dir = OUTPUTS / "round1_5" / f"ctrl_119_{mov}_{variant}"
            recs = load_progress(out_dir)
            if recs:
                by_variant.setdefault(variant, []).extend(recs)
    return {name: summarize(recs) for name, recs in by_variant.items()}


def print_table(rows: dict[str, dict]) -> str:
    header = ["group", "n_ok", "n_gaussians", "extent_overshoot", "scale_ratio_median",
              "opacity_median", "low_opacity_frac"]
    lines = [" | ".join(header), " | ".join(["---"] * len(header))]
    for name, s in sorted(rows.items(), key=lambda kv: kv[1].get("extent_overshoot_mean", 999)):
        lines.append(" | ".join([
            name,
            str(s.get("n_ok", 0)),
            f"{s.get('n_gaussians_mean', float('nan')):.1f}",
            f"{s.get('extent_overshoot_mean', float('nan')):.3f}",
            f"{s.get('scale_ratio_median_mean', float('nan')):.3f}",
            f"{s.get('opacity_median_mean', float('nan')):.3f}",
            f"{s.get('low_opacity_frac_mean', float('nan')):.3f}",
        ]))
    return "\n".join(lines)


def main():
    baseline = collect_baseline()
    core = collect_round1_core()
    iters = collect_round1_iters()
    r1_5 = collect_round1_5()

    all_rows = {"BASELINE_ratio3_sh0_dense": baseline, **core, **iters, **r1_5}

    out = {
        "baseline": baseline,
        "round1_core": core,
        "round1_iters": iters,
        "round1_5": r1_5,
    }
    out_path = OUTPUTS / "round1_summary.json"
    out_path.write_text(json.dumps(out, indent=2))

    table = print_table(all_rows)
    print(table)
    md_path = OUTPUTS / "round1_summary.md"
    md_path.write_text("# Round 1 / 1.5 结果汇总\n\n(按extent_overshoot均值升序排列, baseline取自已有480帧全量sweep的dev帧子集, 免重跑)\n\n" + table + "\n")
    print(f"\n[saved] {out_path.relative_to(REPO)}")
    print(f"[saved] {md_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
