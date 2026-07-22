"""
gpu/disk/audit.py
磁盘诊断脚本：WSL2根文件系统(outputs/、data/所在)剩余空间 + 该文件系统类型(原生ext4 还是
DrvFS/9p挂进来的Windows盘) + Windows C盘剩余空间(经/mnt/c) + 对已完成的splatfacto-checkpoint
产出目录抽样统计各子目录/文件大小 + 基于抽样均值estimate删checkpoint/删tensorboard事件文件
能省多少空间、以及跑到更多帧数还需要多少空间。

用法: python -m gpu.disk.audit [--sample-n 10] [--seed 42] [--targets 1000 2000 5000]
输出: 打印到stdout(含每个抽样目录的完整find+ls+du明细) + 写入精简版 gpu/disk/DISK.md
(DISK.md只保留汇总结论，不含逐目录明细——明细看stdout/终端log)

!!! 重要约束 !!! debug_checkpoints/ 不能被当成"可删/可省空间"的对象统计，
后续任何清理脚本(包括任务3)也不能删除它。原因：debug_checkpoints/stats/step_*_stats.json
是 n_gaussians / scale_ratio.median|p95|... / opacity.* 等指标的唯一数据源，
gpu/schedule/common.py 的 is_task_done() 用它判断任务是否完成，run_task() 用它的
最终step读取scale_ratio.median写入结果。删掉整个debug_checkpoints/会直接打断
指标流水线(is_task_done会误判为未完成、已跑完的结果也拿不到指标了)。
本脚本只统计它的大小用于了解占比，绝不计入"能省的空间"。
"""

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUTPUTS = REPO / "outputs"
DATA = REPO / "data"
DISK_MD = Path(__file__).resolve().parent / "DISK.md"

# 见模块docstring: 这个目录只统计大小，不参与"能省空间"的estimate，也不能被清理脚本删除。
NON_DELETABLE_CATEGORY = "debug_checkpoints"


def sh(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def human_size(nbytes: float) -> str:
    for unit in ["B", "K", "M", "G", "T"]:
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}P"


def du_bytes(path: Path) -> int:
    """实际占用的磁盘空间(分配的block数)，不是apparent size。二者差异对debug_checkpoints/这种
    "很多小文件"的目录很大(实测一个debug_checkpoints/: du -sb=1.8K的apparent size，
    但du -sh/du -sk算出的block占用是20K)，任务问的是磁盘空间能省多少，必须用block占用口径。"""
    out = sh(["du", "-sk", str(path)])
    if not out:
        return 0
    return int(out.split()[0]) * 1024


# -------------------------------------------------------------- 1. WSL2磁盘 --

def report_wsl_disk(lines: list) -> str:
    out = sh(["df", "-h", str(OUTPUTS)])
    print(out)
    data_lines = out.splitlines()
    summary = data_lines[1] if len(data_lines) > 1 else out
    lines.append(f"- WSL2 (outputs/、data/ 所在文件系统): `{summary}`")
    return out


# -------------------------------------------------- 1b. 文件系统类型(ext4? DrvFS/9p?) --

def report_filesystem_type(lines: list) -> None:
    """确认outputs/、data/是不是原生ext4(而不是经/mnt/c、/mnt/x这种DrvFS/9p挂进来的Windows盘)。"""
    results = {}
    for label, path in [("outputs/", OUTPUTS), ("data/", DATA)]:
        target = path if path.exists() else path.parent
        proc = subprocess.run(["findmnt", "-no", "SOURCE,FSTYPE", "-T", str(target)],
                               capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            results[label] = ("?", "?")
            continue
        source, fstype = proc.stdout.strip().split(None, 1)
        results[label] = (source, fstype)
        print(f"findmnt -T {target} -> source={source} fstype={fstype}")

    fstypes = {v[1] for v in results.values()}
    if fstypes == {"ext4"}:
        verdict = "是原生ext4 (不是DrvFS/9p挂进来的Windows盘)。"
    elif "?" in fstypes:
        verdict = f"无法完全确认 (findmnt失败)，原始结果: {results}"
    else:
        verdict = f"不是原生ext4，实际类型: {results}"
    line = f"- 文件系统类型: **{verdict}** (" + "; ".join(f"{k}→{v[0]}({v[1]})" for k, v in results.items()) + ")"
    print(line)
    lines.append(line)


# ---------------------------------------------------------- 2. Windows C盘 --

def report_windows_c_disk(lines: list) -> None:
    mnt_c = Path("/mnt/c")
    if not mnt_c.exists():
        msg = "- Windows C盘: 错误 — /mnt/c 不存在，WSL2里访问不到，无法报告(不做假设)。"
        lines.append(msg)
        print(msg)
        return
    proc = subprocess.run(["df", "-h", "/mnt/c"], capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        msg = (f"- Windows C盘: 错误 — /mnt/c 存在但 `df -h /mnt/c` 失败 "
               f"(returncode={proc.returncode}, stderr={proc.stderr.strip()!r})，无法报告(不做假设)。")
        lines.append(msg)
        print(msg)
        return
    print(proc.stdout.strip())
    out_lines = proc.stdout.strip().splitlines()
    summary = out_lines[1] if len(out_lines) > 1 else proc.stdout.strip()
    lines.append(f"- Windows C盘 (经 /mnt/c): `{summary}`")


# ------------------------------------------------ 3. 抽样已完成的checkpoint目录 --

def find_completed_checkpoint_dirs() -> list:
    """判定标准和gpu/schedule/common.py::is_task_done()一致：
    splat.ply存在 + debug_checkpoints/stats/下有能parse的stats.json + scale_ratio.median非空。"""
    done = []
    for ckpt_root in OUTPUTS.rglob("splatfacto-checkpoint"):
        if not ckpt_root.is_dir():
            continue
        for ts_dir in ckpt_root.iterdir():
            if not ts_dir.is_dir():
                continue
            if not (ts_dir / "splat.ply").exists():
                continue
            stats_files = sorted((ts_dir / "debug_checkpoints" / "stats").glob("step_*_stats.json"))
            if not stats_files:
                continue
            try:
                stats = json.loads(stats_files[-1].read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if stats.get("scale_ratio", {}).get("median") is None:
                continue
            done.append(ts_dir)
    return done


def categorize(name: str) -> str:
    if name == "nerfstudio_models":
        return "nerfstudio_models"
    if name == "debug_checkpoints":
        return NON_DELETABLE_CATEGORY
    if name.startswith("events.out.tfevents"):
        return "tensorboard_events"
    if name == "splat.ply":
        return "splat.ply"
    if name == "dataparser_transforms.json":
        return "dataparser_transforms.json"
    if name == "config.yml":
        return "config.yml"
    return "other"  # 抽样中实际见过的例子: gaussian_features_f0032.csv 之类的额外产出


def audit_sample(sample_dirs: list):
    """对每个抽样目录: 先find+ls完整打印这一层的目录结构(不假设子目录名字)，
    再对每个一级子项跑du统计大小，按categorize()分类累计。逐目录明细只打印到stdout，
    不写入DISK.md(DISK.md只保留汇总结论)。"""
    sizes_by_category = defaultdict(list)   # category -> [bytes, ...]
    per_dir_total = []                      # (dir, total_bytes) 用于估算整个run的平均总大小

    for d in sample_dirs:
        rel = d.relative_to(REPO)
        print(f"\n=== {rel} ===")
        print("[find -maxdepth 1]")
        print(sh(["find", str(d), "-maxdepth", "1"]))
        print("[ls -la]")
        print(sh(["ls", "-la", str(d)]))

        entries = sorted(p for p in d.iterdir())
        dir_total = 0
        for entry in entries:
            b = du_bytes(entry)
            cat = categorize(entry.name)
            sizes_by_category[cat].append(b)
            dir_total += b
            print(f"  du -sh {entry.name} -> {human_size(b)}  [{cat}]")
        per_dir_total.append((d, dir_total))

    return sizes_by_category, per_dir_total


# --------------------------------------------------------- 4/5. estimate + summary --

def build_estimates(sizes_by_category: dict, per_dir_total: list, targets: list, lines: list):
    avg_by_category = {cat: sum(vals) / len(vals) for cat, vals in sizes_by_category.items()}

    tbl = ["| 分类 | 平均大小 | 备注 |", "|---|---|---|"]
    for cat, avg in sorted(avg_by_category.items()):
        note = "**不可删/不计入可省空间**" if cat == NON_DELETABLE_CATEGORY else ""
        tbl.append(f"| {cat} | {human_size(avg)} | {note} |")
    print("\n".join(tbl))
    lines.append("\n## 每run分类别平均大小 (抽样均值)")
    lines.extend(tbl)

    avg_total_per_run = sum(t for _, t in per_dir_total) / len(per_dir_total) if per_dir_total else 0
    total_runs = len(list(OUTPUTS.rglob("splat.ply")))  # 等价于 find outputs -name splat.ply | wc -l
    avg_nerfstudio_models = avg_by_category.get("nerfstudio_models", 0)
    avg_tfevents = avg_by_category.get("tensorboard_events", 0)
    checkpoint_savings = total_runs * avg_nerfstudio_models
    tfevents_savings = total_runs * avg_tfevents
    current_total = avg_total_per_run * total_runs

    lines.append("\n## 空间预估")
    lines.append(f"- 当前总run数: **{total_runs}** ，当前总占用: 约 **{human_size(current_total)}**")
    lines.append(f"- 删除所有run的 `nerfstudio_models/`(训练ckpt) 能省: 约 **{human_size(checkpoint_savings)}**"
                 f" ({total_runs} × {human_size(avg_nerfstudio_models)})")
    lines.append(f"- 不再产生tensorboard事件文件能省: 约 **{human_size(tfevents_savings)}**"
                 f" ({total_runs} × {human_size(avg_tfevents)})")
    lines.append(f"- `debug_checkpoints/` 不计入以上可省空间也不能删 (唯一指标数据源，见脚本docstring)，"
                 f"平均 {human_size(avg_by_category.get(NON_DELETABLE_CATEGORY, 0))}/run")

    lines.append("\n### 跑到更多帧数预计还需要的总空间 (假设每帧1个run；多个param_set并行跑同一帧要再乘以param_set数)")
    tbl2 = [f"| 目标run数 | 预计总占用 | 相对当前({total_runs}个run)的增量 |", "|---|---|---|"]
    for n in targets:
        est_total = avg_total_per_run * n
        tbl2.append(f"| {n} | {human_size(est_total)} | {human_size(est_total - current_total)} |")
    lines.extend(tbl2)
    print("\n".join(tbl2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", type=int, nargs="+", default=[1000, 2000, 5000])
    args = ap.parse_args()

    lines = ["# 磁盘诊断报告 (gpu/disk/audit.py)\n", "## 磁盘空间"]

    report_wsl_disk(lines)
    report_filesystem_type(lines)
    report_windows_c_disk(lines)

    print("\n扫描已完成的splatfacto-checkpoint目录...")
    completed = find_completed_checkpoint_dirs()
    n_sample = min(args.sample_n, len(completed))
    print(f"共找到 {len(completed)} 个已完成目录，抽样 {n_sample} 个 (明细见上方stdout，不写入DISK.md)")
    if not completed:
        lines.append("\n没有找到任何已完成的splatfacto-checkpoint目录，跳过抽样统计。")
        print("没有找到任何已完成的splatfacto-checkpoint目录，跳过抽样统计。")
    else:
        sample = random.Random(args.seed).sample(completed, n_sample)
        sizes_by_category, per_dir_total = audit_sample(sample)
        build_estimates(sizes_by_category, per_dir_total, args.targets, lines)

    text = "\n".join(lines) + "\n"
    DISK_MD.write_text(text)
    print(f"\n已写入 {DISK_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
