"""
aggregate_round2.py

4cam round2（round2_4cam_sweep_plan.md）结果聚合脚本。跟3cam的aggregate_round1.py/
aggregate_round2.py同一套训练侧指标口径(n_gaussians/scale_ratio/opacity/extent_overshoot/
wall_s，从各config的_progress/*.jsonl现算mean/median)，两个视频(ctrl_009_004/016)同名
param_set合并成一行。

跟3cam版本的关键差异：输出**不再单独按extent_overshoot排序当作唯一排名依据**。
extent_overshoot对"点少"天然友好(点越少越不容易有远处离群点把bbox撑大)，单独排序会
系统性偏向"更稀疏但漂移小"的组合、掩盖掉"更稠密且漂移可接受"的更优选项——这正是v1版
round2_4cam_sweep_plan.md犯过的方法论错误(见文档"v2相对上一版的核心修正")。这里改成
按(n_gaussians, extent_overshoot)联合帕累托支配关系标注每一行，只有帕累托前沿上的组
(没有别的组同时"更密+更不漂移或相当")才值得作为候选，其余的即使overshoot数字好看也要
标"dominated"。

四个桶：d0(max_iters=1000主网格)、iters750(同一组param_sets, max_iters=750)、
p2(去腿诊断)、opt(可选单点)。帕累托前沿在四个桶合并的全体行里统一计算——最终目的是回答
"这些组里到底该选哪几个"，不是"每个桶内部谁最好"。

用法:
    python -m gpu.schedule.analysis.aggregate_round2
"""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

OUTPUTS = REPO / "outputs"
VIDEOS = ("ctrl_009_004", "ctrl_009_016")
BUCKETS = ("d0", "iters750", "p2", "opt")

METRICS = ["n_gaussians", "scale_ratio_median", "scale_ratio_p95", "scale_ratio_frac_over_10",
           "opacity_median", "low_opacity_frac", "extent_overshoot", "bbox_extent_max", "wall_s"]


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


def summarize_training(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    out = {"n_ok": len(ok), "n_failed": len(records) - len(ok)}
    for metric in METRICS:
        vals = [r[metric] for r in ok if metric in r and r[metric] is not None]
        if vals:
            out[f"{metric}_mean"] = statistics.mean(vals)
            out[f"{metric}_median"] = statistics.median(vals)
    return out


def collect_bucket(bucket_suffix: str) -> dict[str, dict]:
    by_param_set: dict[str, list[dict]] = {}
    for video in VIDEOS:
        sweep_name = f"round2/{video}_{bucket_suffix}"
        out_dir = OUTPUTS / sweep_name
        for r in load_progress(out_dir):
            by_param_set.setdefault(r["param_set"], []).append(r)
    return {ps: summarize_training(recs) for ps, recs in by_param_set.items()}


def pareto_front(rows: dict[str, dict]) -> set[str]:
    """行需要同时有n_gaussians_mean/extent_overshoot_mean才参与比较(训练全失败的组
    两者都缺，直接不算在前沿判定里，但仍然会在表里显示n_ok=0)。"""
    comparable = {name: row for name, row in rows.items()
                  if "n_gaussians_mean" in row and "extent_overshoot_mean" in row}
    front = set()
    for name, row in comparable.items():
        n_g, ov = row["n_gaussians_mean"], row["extent_overshoot_mean"]
        dominated = any(
            other["n_gaussians_mean"] >= n_g and other["extent_overshoot_mean"] <= ov
            and (other["n_gaussians_mean"] > n_g or other["extent_overshoot_mean"] < ov)
            for other_name, other in comparable.items() if other_name != name
        )
        if not dominated:
            front.add(name)
    return front


def print_table(title: str, rows: dict[str, dict], front: set[str]) -> None:
    print(f"\n=== {title} ===")
    header = ("group", "n_ok", "n_gauss", "overshoot", "scale_ratio", "opacity",
              "low_op_frac", "pareto")
    print(" | ".join(header))
    for ps, row in sorted(rows.items(), key=lambda kv: kv[1].get("extent_overshoot_mean", 999)):
        flag = "front" if ps in front else "dominated"
        print(f"{ps} | {row.get('n_ok', 0)} | "
              f"{row.get('n_gaussians_mean', float('nan')):.1f} | "
              f"{row.get('extent_overshoot_mean', float('nan')):.3f} | "
              f"{row.get('scale_ratio_median_mean', float('nan')):.3f} | "
              f"{row.get('opacity_median_mean', float('nan')):.3f} | "
              f"{row.get('low_opacity_frac_mean', float('nan')):.3f} | {flag}")


def main() -> None:
    all_results = {bucket: collect_bucket(bucket) for bucket in BUCKETS}

    merged: dict[str, dict] = {}
    for bucket, rows in all_results.items():
        for ps, row in rows.items():
            merged[f"{bucket}/{ps}"] = row
    front = pareto_front(merged)

    for bucket in BUCKETS:
        bucket_front = {ps for key in front if key.startswith(f"{bucket}/")
                         for ps in [key.split("/", 1)[1]]}
        print_table(f"Round2 {bucket} bucket", all_results[bucket], bucket_front)

    print("\n=== Joint Pareto front across ALL buckets (n_gaussians vs extent_overshoot) ===")
    print("(不要只看extent_overshoot——下面这些组是唯一没有被任何组同时'更密+更不漂移'支配的)")
    front_rows = sorted(
        ((key, merged[key]) for key in front),
        key=lambda kv: kv[1]["n_gaussians_mean"]
    )
    for key, row in front_rows:
        print(f"  {key:35s} n_gaussians={row['n_gaussians_mean']:.1f}  "
              f"extent_overshoot={row['extent_overshoot_mean']:.3f}")

    out_path = OUTPUTS / "round2_4cam_summary.json"
    out_path.write_text(json.dumps(
        {"buckets": all_results, "pareto_front": sorted(front)}, indent=2))
    print(f"\n[aggregate_round2] wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
