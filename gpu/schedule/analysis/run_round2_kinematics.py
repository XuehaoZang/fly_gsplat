"""
run_round2_kinematics.py

Round 2训练完成后的CPU侧后处理：对gpu/schedule/configs/round2/*.json里的每个
(sweep_name, param_set)跑一遍完整T1-T4 kinematics + 新增的标注点云视频，纯目视QC用
(数值指标好看不代表点云/角度曲线没问题)。

本脚本现在同时喂3cam(ctrl_119_*)和4cam(ctrl_009_*)两套round2 config，两者拍摄fps不同
(3cam=8000fps，4cam=16000fps)，T3的motion累加窗口half_window必须按fps折算(见
postprocessing.labeling.motion.density.HALF_WINDOW的docstring)——4cam用默认值36即可，
3cam必须显式传18，否则窗口覆盖的物理时长变成2倍，跨帧累加进更多个振翅周期，body/wing
判据被稀释混淆(2026-09-10按用户报告排查confirmed: main分支上这条half_window链路曾经
被静默去掉过，见_half_window_for()下方注释)。按raw_data_dir里是否含"ctrl_119"识别
数据集来源，不依赖sweep_name本身的命名，因为base_name override(去腿等数据变体)也一样
落在ctrl_119_3cam_*前缀下。

跟postprocessing.batch_calc_kinematics.run_one同一套T1-T4+角度图逻辑，唯一区别：
run_one固定用cfg["base_name"](config顶层默认值)当raw_data_dir，不知道某个param_set
自己把base_name override成了别的数据变体(比如round2_4cam的p2去腿diagnostics)——这只
会让重投影QC图叠加错图(不影响kinematics csv本身)，但重投影QC图正是核查去腿是否啃到
身体的手段，不能让它对变体组也悄悄失真——这里显式按每个param_set自己实际的base_name
解析raw_data_dir。

用法:
    python -m gpu.schedule.analysis.run_round2_kinematics                        # 全部跑(含3cam+4cam)
    python -m gpu.schedule.analysis.run_round2_kinematics --workers 10           # 并发度
    python -m gpu.schedule.analysis.run_round2_kinematics --only ctrl_009_004_d0 # 只跑一个config
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


def _param_set_base_name(value, default_base_name: str) -> str:
    return value["base_name"] if isinstance(value, dict) and "base_name" in value else default_base_name


def _half_window_for(raw_data_dir_rel: str) -> int:
    """3cam(ctrl_119, 8000fps)数据集的所有base_name变体都落在"ctrl_119_3cam"或
    "ctrl_119_3cam_<variant>"前缀下(见generate_round2_configs.py/
    prepare_round2_leg_erosion.py)，据此识别，不用sweep_name(round2/ctrl_119_004_d1
    这类命名本身也含"119"，但更脆弱，直接认数据路径更可靠)。4cam(ctrl_009,16000fps)
    落在别的default值(motion_density.HALF_WINDOW=36)。"""
    from postprocessing.labeling.motion import density as motion_density
    return 18 if "ctrl_119_3cam" in raw_data_dir_rel else motion_density.HALF_WINDOW


def enumerate_groups(only: str | None = None, prefix: str | None = None) -> list[dict]:
    """扫描configs/round2/*.json，展开成[{sweep_name, group, raw_data_dir}, ...]。"""
    groups = []
    for cfg_path in sorted(CONFIG_DIR.glob("*.json")):
        if cfg_path.name.startswith("_"):
            continue  # _smoketest.json etc -- throwaway configs, never part of the real sweep
        if only is not None and cfg_path.stem != only:
            continue
        if prefix is not None and not cfg_path.stem.startswith(prefix):
            continue
        cfg = json.loads(cfg_path.read_text())
        for group, value in cfg["param_sets"].items():
            groups.append({
                "sweep_name": cfg["name"],
                "group": group,
                "raw_data_dir": _param_set_base_name(value, cfg["base_name"]),
            })
    return groups


def run_group(sweep_name: str, group: str, raw_data_dir_rel: str) -> dict:
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
            half_window = _half_window_for(raw_data_dir_rel)
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
        # under the same sweep_name (a round2 "bucket" config has several of them) --
        # every filename written here must be namespaced by group, or concurrent groups
        # racing in the multiprocessing pool silently clobber each other's plots down
        # to whichever one happened to finish last.
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
                     help="并发进程数(CPU侧任务，本机nproc=28)")
    ap.add_argument("--only", type=str, default=None,
                     help="只处理某一个config(stem名，如ctrl_009_004_d0)，不传则处理"
                          "configs/round2/下全部config(含3cam+4cam)")
    ap.add_argument("--prefix", type=str, default=None,
                     help="只处理stem以此前缀开头的config，如ctrl_009_ 覆盖4cam全部8个"
                          "config但排除ctrl_119_的3cam config；跟--only互斥，--only优先")
    args = ap.parse_args()

    tasks = enumerate_groups(only=args.only, prefix=args.prefix)
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
