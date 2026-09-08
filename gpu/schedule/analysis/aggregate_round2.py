"""
aggregate_round2.py

Round 2结果聚合脚本，跟aggregate_round1.py同一套指标口径(n_gaussians/scale_ratio/
opacity/extent_overshoot/wall_s，从各config的_progress/*.jsonl现算mean/median)，
但额外读run_round2_kinematics.py跑完后落盘的kinematics_<group>.csv，附加
"kinematics status=ok的帧占比"这一列——round2的核心诉求之一是"每个pipeline都要过一遍
kinematics"，光有训练侧指标看不出T3/T4实际跑通率，两块拼在一起才是完整的结果表。

三个config桶(d1: max_iters=1000, iters750, d2: max_iters=2000)分别聚合，同名
param_set(比如两个视频各自的"D1_baseline")按名字合并成一行。

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
CONFIG_DIR = REPO / "gpu" / "schedule" / "configs" / "round2"

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


def summarize_kinematics(sweep_name: str, group: str) -> dict:
    """从run_round2_kinematics.py的产出kinematics_<group>.csv里读status列，算
    ok帧占比。跑到这一步之前的组(kinematics还没跑/跑失败)返回n_kinematics_ok=None，
    调用方据此在表里显示"pending"而不是0，避免跟"kinematics跑完但全部失败"混淆。"""
    import pandas as pd
    csv_path = OUTPUTS / sweep_name / "kinematics" / f"kinematics_{group}.csv"
    if not csv_path.exists():
        return {"n_kinematics_ok": None, "n_kinematics_total": None}
    df = pd.read_csv(csv_path)
    return {"n_kinematics_ok": int((df["status"] == "ok").sum()), "n_kinematics_total": len(df)}


def collect_bucket(bucket_suffix: str) -> dict[str, dict]:
    """bucket_suffix in {'d1', 'iters750', 'd2'}. 按param_set合并两个视频。"""
    by_param_set: dict[str, list[dict]] = {}
    sweep_names_by_ps: dict[str, str] = {}
    for mov in ("ctrl_119_004", "ctrl_119_010"):
        sweep_name = f"round2/{mov}_{bucket_suffix}"
        out_dir = OUTPUTS / sweep_name
        for r in load_progress(out_dir):
            by_param_set.setdefault(r["param_set"], []).append(r)
            sweep_names_by_ps[r["param_set"]] = sweep_name

    result = {}
    for ps, recs in by_param_set.items():
        row = summarize_training(recs)
        row.update(summarize_kinematics(sweep_names_by_ps[ps], ps))
        result[ps] = row
    return result


def print_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    header = ("group", "n_ok", "n_gauss", "overshoot", "scale_ratio", "opacity",
              "low_op_frac", "kinematics_ok/total")
    print(" | ".join(header))
    for ps, row in sorted(rows.items(), key=lambda kv: kv[1].get("extent_overshoot_mean", 999)):
        kin = (f"{row['n_kinematics_ok']}/{row['n_kinematics_total']}"
               if row.get("n_kinematics_ok") is not None else "pending")
        print(f"{ps} | {row.get('n_ok', 0)} | "
              f"{row.get('n_gaussians_mean', float('nan')):.1f} | "
              f"{row.get('extent_overshoot_mean', float('nan')):.3f} | "
              f"{row.get('scale_ratio_median_mean', float('nan')):.3f} | "
              f"{row.get('opacity_median_mean', float('nan')):.3f} | "
              f"{row.get('low_opacity_frac_mean', float('nan')):.3f} | {kin}")


def main() -> None:
    all_results = {}
    for bucket, title in (("d1", "Round2 D1 bucket (max_iters=1000, 25 param_sets)"),
                           ("iters750", "Round2 iters750 bucket"),
                           ("d2", "Round2 D2 bucket (max_iters=2000)")):
        rows = collect_bucket(bucket)
        all_results[bucket] = rows
        print_table(title, rows)

    out_path = OUTPUTS / "round2_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[aggregate_round2] wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
