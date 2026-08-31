"""
select_frame_window.py

从EasyWand sparse pixel .mat文件里，为一个多相机session下的每个视频挑选一段
"4相机同时有真实检测点"的可用信号窗口，排除拍摄边缘(果蝇刚入画/快出画的过渡帧，
物理上不可靠、不该拿去做3DGS重建)，最终给每个视频选定一段固定长度的训练帧窗口。

背景: 2026-08-20的首轮29视频sweep把"sparse mat文件的总帧槽位数"(每个视频都是2401)
误当成了"有效追踪信号长度"，直接取总数的中间200帧，结果19/29视频崩在Phase A——因为
那些视频的真实追踪数据(DLTdv/EasyWand自动跟踪)远没有覆盖到总帧数的中段，3个相机在
很早就已经完全丢失目标(sparse mat对应帧变成退化的1维空数组，不是正常的(N,3) indIm)。
本模块就是补上"筛选阶段"，2026-08-24与用户对齐算法后实现:

  1. 每个相机分别找自己的原始有效帧(indIm是2维数组、非退化空数组的帧)，按连续性
     切成若干段(run)；每段各掐头去尾MARGIN_FRAMES帧(入画/出画过渡期，不可靠)；
     trim后长度<=2*MARGIN_FRAMES的段整段作废(掐完不剩东西)。
  2. 4个相机(各自trim完的)有效帧集合取交集。
  3. 交集如果断成多段(某相机中途掉线又恢复)，取最长的一段——不再对这一段二次trim
     (跟第1步的逐相机trim是两回事，这段边界如果不是原始录制的入画/出画边缘，用户
     确认不需要再trim一轮)。
  4. 这段长度必须 > MIN_SIGNAL_FRAMES，否则整个视频判定"没有可用信号"，筛掉，不
     进入训练。
  5. 通过的视频，从这段里居中取TRAIN_FRAMES帧，作为该视频最终的GS训练窗口。

用法:
    python select_frame_window.py \\
        --sparse-root "X:\\antenna\\control\\009_25052026\\Sparse" \\
        --out-csv gpu/schedule/configs/ctrl_009_valid480/frame_selection.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py

MARGIN_FRAMES = 160
"""~10ms @ 16000fps: 果蝇入画/出画过渡期，即使有检测点也不可靠，见docstring。"""
MIN_SIGNAL_FRAMES = 480
"""30ms @ 16000fps: 4相机交集最长段低于这个长度，视频整体判定无可用信号。"""
TRAIN_FRAMES = 480
"""每个通过筛选的视频，最终固定训练这么多帧(2026-08-24用户确认: 全部视频统一
训练相同长度，而不是各自用完整的可用段——不然训练量能差4倍)。"""
FPS = 16000.0
"""拍摄帧率，user-confirmed(同postprocessing/kinematics/diagnostics.py::FPS)。"""


@dataclass
class WindowResult:
    mov: str
    n_cams: int
    raw_intersection_range: tuple[int, int] | None
    raw_intersection_len: int
    trimmed_segment_range: tuple[int, int] | None
    trimmed_segment_len: int
    passed: bool
    train_range: tuple[int, int] | None       # (start, end) end-exclusive, 同schedule.py config的frames字段


def _raw_valid_frames(sparse_file: Path) -> list[int]:
    """一个相机sparse mat文件里，indIm是正常2维数组(非退化空数组)的帧号，升序。"""
    with h5py.File(sparse_file, "r") as sp:
        refs = sp["/frames/indIm"][0]
        return [fi for fi in range(len(refs)) if sp[refs[fi]].ndim == 2]


def _contiguous_runs(sorted_frames: list[int]) -> list[tuple[int, int]]:
    if not sorted_frames:
        return []
    runs = []
    lo = prev = sorted_frames[0]
    for f in sorted_frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append((lo, prev))
            lo = prev = f
    runs.append((lo, prev))
    return runs


def _trim_runs(runs: list[tuple[int, int]], margin: int) -> set[int]:
    out: set[int] = set()
    for lo, hi in runs:
        if hi - lo + 1 > 2 * margin:
            out.update(range(lo + margin, hi - margin + 1))
    return out


def _longest_run(runs: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not runs:
        return None
    return max(runs, key=lambda lohi: lohi[1] - lohi[0] + 1)


def select_window(sparse_dir: Path, mov: str,
                   margin: int = MARGIN_FRAMES, min_len: int = MIN_SIGNAL_FRAMES,
                   train_len: int = TRAIN_FRAMES) -> WindowResult:
    cam_files = sorted(sparse_dir.glob("Camera*_sparse.mat"))
    per_cam_raw = [_raw_valid_frames(c) for c in cam_files]

    # 未trim的4相机交集(仅供CSV参考对比，不用于选窗)
    raw_sets = [set(g) for g in per_cam_raw]
    raw_inter = sorted(set.intersection(*raw_sets)) if raw_sets else []
    raw_run = _longest_run(_contiguous_runs(raw_inter))
    raw_len = (raw_run[1] - raw_run[0] + 1) if raw_run else 0

    # 逐相机、逐run trim MARGIN帧，再取4相机交集
    trimmed_sets = [_trim_runs(_contiguous_runs(g), margin) for g in per_cam_raw]
    trimmed_inter = sorted(set.intersection(*trimmed_sets)) if trimmed_sets else []
    best_run = _longest_run(_contiguous_runs(trimmed_inter))
    trimmed_len = (best_run[1] - best_run[0] + 1) if best_run else 0

    passed = trimmed_len > min_len
    train_range = None
    if passed:
        lo, hi = best_run
        start = lo + (trimmed_len - train_len) // 2
        train_range = (start, start + train_len)

    return WindowResult(
        mov=mov, n_cams=len(cam_files),
        raw_intersection_range=raw_run, raw_intersection_len=raw_len,
        trimmed_segment_range=best_run, trimmed_segment_len=trimmed_len,
        passed=passed, train_range=train_range,
    )


def scan_session(sparse_root: Path, margin: int = MARGIN_FRAMES,
                  min_len: int = MIN_SIGNAL_FRAMES, train_len: int = TRAIN_FRAMES) -> list[WindowResult]:
    video_dirs = sorted(p for p in sparse_root.iterdir()
                         if p.is_dir() and p.name.startswith("Expr_") and "mov_" in p.name)
    results = []
    for vdir in video_dirs:
        mov = vdir.name.split("mov_")[-1]
        print(f"[scan] {vdir.name} ...")
        results.append(select_window(vdir, mov, margin, min_len, train_len))
    return results


def write_csv(results: list[WindowResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mov", "n_cams",
                    "raw_intersection_start", "raw_intersection_end", "raw_intersection_len",
                    "trimmed_segment_start", "trimmed_segment_end", "trimmed_segment_len",
                    "selected", "train_start", "train_end",
                    "train_start_ms", "train_end_ms"])
        for r in results:
            raw_lo, raw_hi = r.raw_intersection_range if r.raw_intersection_range else (None, None)
            trim_lo, trim_hi = r.trimmed_segment_range if r.trimmed_segment_range else (None, None)
            train_lo, train_hi = r.train_range if r.train_range else (None, None)
            w.writerow([
                r.mov, r.n_cams,
                raw_lo, raw_hi, r.raw_intersection_len,
                trim_lo, trim_hi, r.trimmed_segment_len,
                r.passed, train_lo, train_hi,
                f"{train_lo / FPS * 1000:.3f}" if train_lo is not None else "",
                f"{train_hi / FPS * 1000:.3f}" if train_hi is not None else "",
            ])
    print(f"[csv] -> {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse-root", type=str, required=True,
                     help=r'session的Sparse根目录，如 "X:\antenna\control\009_25052026\Sparse"')
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--margin", type=int, default=MARGIN_FRAMES)
    ap.add_argument("--min-len", type=int, default=MIN_SIGNAL_FRAMES)
    ap.add_argument("--train-len", type=int, default=TRAIN_FRAMES)
    args = ap.parse_args()

    sparse_root = Path(args.sparse_root.replace("X:", "/mnt/x").replace("\\", "/"))
    results = scan_session(sparse_root, args.margin, args.min_len, args.train_len)
    write_csv(results, Path(args.out_csv))

    n_pass = sum(r.passed for r in results)
    print(f"\n{n_pass}/{len(results)} videos passed (trimmed segment > {args.min_len} frames)")


if __name__ == "__main__":
    main()
