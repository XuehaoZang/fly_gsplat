"""
run_round2_kinematics.py

Round 2训练完成后的CPU侧后处理：对gpu/schedule/configs/round2/*.json里的每个
(sweep_name, param_set)跑一遍完整T1-T4 kinematics(复用
postprocessing.batch_calc_kinematics.run_one同一套T1-T4+角度图逻辑，只做一处必要
修正)+ 新增的标注点云视频。

跟batch_calc_kinematics.run_one的唯一区别：run_one固定用cfg["base_name"](config
顶层默认值)当raw_data_dir，不知道某个param_set自己把base_name override成了别的
数据变体(去腿/mask阈值/hull采样数)——这只会让重投影QC图叠加错图(不影响kinematics
csv本身，因为raw_data_dir没传进run_cleaning_and_labeling/pipeline)，但round2恰好
新增了好几个这样的数据变体组，且重投影QC图正是本轮发现leg-erosion问题的手段，不能
让它对新变体也悄悄失真——这里显式按每个param_set自己实际的base_name解析raw_data_dir。

half_window显式传18(8000fps的3相机数据集修正值，不能用calc_kinematics默认的
16000fps锁定值36，见density.py docstring / sweep_hyper_params.md Round2记录)。

用法:
    python -m gpu.schedule.analysis.run_round2_kinematics                      # 全部跑
    python -m gpu.schedule.analysis.run_round2_kinematics --workers 6          # 并发度
    python -m gpu.schedule.analysis.run_round2_kinematics --only ctrl_119_004_d1  # 只跑一个config
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "gpu" / "schedule" / "configs" / "round2"
HALF_WINDOW_8KFPS = 18


def _param_set_base_name(value, default_base_name: str) -> str:
    return value["base_name"] if isinstance(value, dict) and "base_name" in value else default_base_name


def enumerate_groups(only: str | None = None) -> list[dict]:
    """扫描configs/round2/*.json，展开成[{sweep_name, group, raw_data_dir}, ...]。"""
    groups = []
    for cfg_path in sorted(CONFIG_DIR.glob("*.json")):
        if cfg_path.name.startswith("_"):
            continue  # _smoketest.json etc -- throwaway configs, never part of the real sweep
        if only is not None and cfg_path.stem != only:
            continue
        cfg = json.loads(cfg_path.read_text())
        for group, value in cfg["param_sets"].items():
            groups.append({
                "sweep_name": cfg["name"],
                "group": group,
                "raw_data_dir": _param_set_base_name(value, cfg["base_name"]),
            })
    return groups


def run_group(sweep_name: str, group: str, raw_data_dir_rel: str,
              half_window: int = HALF_WINDOW_8KFPS) -> dict:
    """单个(sweep_name, group)的完整T1-T4+角度图+重投影+标注视频。任何一步异常都
    捕获返回失败记录，不中断调用方(可能是多进程池里跑着别的group)。"""
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    from postprocessing import calc_kinematics as ck
    from postprocessing.kinematics import pipeline
    from postprocessing.kinematics.diagnostics import plot_body_angles
    from postprocessing.viz import reprojection_viewer
    from gpu.schedule.analysis.render_labeled_video import render_labeled_video

    dataset_root = REPO_ROOT / "outputs" / sweep_name / group
    raw_data_dir = REPO_ROOT / raw_data_dir_rel
    out_dir = dataset_root.parent / "kinematics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"kinematics_{dataset_root.name}.csv"

    if not dataset_root.exists():
        return {"sweep_name": sweep_name, "group": group, "status": "skipped:no_dataset_root"}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        if any(dataset_root.glob(ck.LABELED_FRAME_GLOB)):
            pass  # T3已跑过，只补T4
        else:
            start, end = ck.discover_frame_range(dataset_root)
            ck.run_cleaning_and_labeling(dataset_root, start, end, half_window=half_window)

        if not any(dataset_root.glob(ck.LABELED_FRAME_GLOB)):
            # T3全帧失败(比如帧数不够撑起motion累加窗口)时，pipeline.
            # run_dataset_with_sequence_correction会在build_sequence.py里对空序列抛
            # IndexError(核心pipeline代码没预期"一帧都没有"这种输入)，这里提前短路，
            # 给出可读的status而不是让一个不相关的下标越界异常冒泡。
            return {"sweep_name": sweep_name, "group": group, "status": "no_labeled_frames"}

        config = pipeline.PipelineConfig(output_dir=out_dir, write_debug=True,
                                          frame_glob=ck.LABELED_FRAME_GLOB)
        df = pipeline.run_dataset_with_eta_unwrap(dataset_root, config)

    ok = df[df["status"] == "ok"].reset_index(drop=True)
    result = {"sweep_name": sweep_name, "group": group, "status": "ok",
              "n_frames": len(df), "n_ok": len(ok)}

    if not ok.empty:
        # out_dir(outputs/<sweep_name>/kinematics/) is SHARED by every param_set/group
        # under the same sweep_name (a round2 "bucket" config has up to 25 of them) --
        # every filename written here must be namespaced by group, or concurrent groups
        # racing in the multiprocessing pool silently clobber each other's plots down
        # to whichever one happened to finish last (found this the hard way: only 6/56
        # body_angles.png survived after the first real run, one per sweep_name).
        plot_body_angles(ok, out_dir / f"{group}_body_angles.png")
        ck.plot_wing_angles(ok, out_dir / f"{group}_wing_angles.png")

        try:
            reproj_dir = out_dir / "reprojection" / group
            frames = ck.pick_reprojection_frames(ok["frame_id"].tolist(), ck.N_FRAMES)
            reprojection_viewer.run_batch(frames, dataset_root, raw_data_dir, reproj_dir)
        except Exception as e:  # noqa: BLE001
            result["reprojection_error"] = f"{type(e).__name__}: {e}"

        video_path = out_dir / f"{group}_labeled_video.mp4"
        render_labeled_video(dataset_root, video_path, label=f"{sweep_name}/{group}")
    else:
        result["status"] = "no_ok_frames"

    return result


def _run_group_safe(task: dict) -> dict:
    try:
        return run_group(task["sweep_name"], task["group"], task["raw_data_dir"])
    except Exception as e:  # noqa: BLE001
        return {"sweep_name": task["sweep_name"], "group": task["group"], "status": "failed",
                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                     help="并发进程数(CPU侧，跟GPU sweep的12个worker共存时建议<=6，"
                          "训练全部跑完后单独跑可以调大)")
    ap.add_argument("--only", type=str, default=None,
                     help="只处理某一个config(stem名，如ctrl_119_004_d1)，不传则全部")
    args = ap.parse_args()

    tasks = enumerate_groups(only=args.only)
    print(f"[run_round2_kinematics] {len(tasks)} group(s) queued, workers={args.workers}")

    ok, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_group_safe, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            res = fut.result()
            tag = f"{res['sweep_name']}/{res['group']}"
            if res["status"] in ("ok", "no_ok_frames", "skipped:no_dataset_root", "no_labeled_frames"):
                ok.append(tag)
                print(f"[{tag}] {res['status']} "
                      f"({res.get('n_ok', '?')}/{res.get('n_frames', '?')} ok frames)")
            else:
                failed.append(tag)
                print(f"[{tag}] FAILED: {res.get('error')}")

    print(f"\ndone: {len(ok)} ok/skipped, {len(failed)} failed")
    if failed:
        print(f"failed groups: {failed}")


if __name__ == "__main__":
    main()
