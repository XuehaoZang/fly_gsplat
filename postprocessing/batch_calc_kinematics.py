"""
批量/非交互版 calc_kinematics —— 对多个数据集根目录依次跑T1-T4+画图，不拉起
calc_kinematics.py末尾那个阻塞的交互式viser viewer。逐帧处理逻辑直接复用
calc_kinematics.py里的函数，只是把"单数据集+交互viewer"换成"多数据集循环+
无viewer"，且raw_data_dir按每个数据集自己的sweep config(base_name字段)动态
解析，而不是calc_kinematics.py里写死的RAW_DATA_DIR(=data/ctrl_009_002，只对
那一个数据集成立)。

用法:
    # 单个sweep(按sweep_name在gpu/schedule/configs/**/下找同名json，从里面的
    # base_name字段解析raw_data_dir；dataset_root固定是
    # outputs/<sweep_name>/<把param_sets展开出的那个group目录>)
    python -m postprocessing.batch_calc_kinematics --sweep-name ctrl_009_013_ratio3_sh0_dense_mid200 --group ratio3_sh0_dense

    # 批量: 对gpu/schedule/configs/ctrl_009_mid200/下所有config跑一遍
    python -m postprocessing.batch_calc_kinematics --configs-glob "gpu/schedule/configs/ctrl_009_mid200/*.json" --group ratio3_sh0_dense
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from postprocessing import calc_kinematics as ck  # noqa: E402
from postprocessing.kinematics import pipeline  # noqa: E402
from postprocessing.kinematics.diagnostics import plot_body_angles  # noqa: E402
from postprocessing.labeling.motion import density as motion_density  # noqa: E402
from postprocessing.viz import reprojection_viewer  # noqa: E402


def find_config(sweep_name: str) -> dict:
    matches = list((REPO_ROOT / "gpu" / "schedule" / "configs").rglob(f"{sweep_name}.json"))
    if not matches:
        raise FileNotFoundError(f"no config named {sweep_name}.json under gpu/schedule/configs/")
    if len(matches) > 1:
        raise ValueError(f"ambiguous config name {sweep_name}.json: {matches}")
    return json.loads(matches[0].read_text())


def run_one(sweep_name: str, group: str, half_window: int = motion_density.HALF_WINDOW) -> None:
    """跟calc_kinematics.main()同一套T1-T4+画图逻辑，唯一区别: raw_data_dir从
    该sweep自己的config.base_name动态解析，且跑完不拉起交互式viewer。

    half_window透传给T3(见calc_kinematics.run_cleaning_and_labeling的docstring)，
    默认按16000fps锁定=36，别的拍摄fps数据集需要按比例换算显式传入。"""
    cfg = find_config(sweep_name)
    dataset_root = REPO_ROOT / "outputs" / sweep_name / group
    raw_data_dir = REPO_ROOT / cfg["base_name"]
    out_dir = dataset_root.parent / "kinematics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"kinematics_{dataset_root.name}.csv"

    print(f"\n{'=' * 60}\n{sweep_name}  (raw_data_dir={raw_data_dir})\n{'=' * 60}")

    if csv_path.exists():
        print(f"== 已有kinematics结果 {csv_path}，跳过T1-T4，只补画图 ==")
        df = pd.read_csv(csv_path)
    else:
        if any(dataset_root.glob(ck.LABELED_FRAME_GLOB)):
            print("== 已有T3标注(_labeled.csv)，跳过T1-T3，只跑T4 ==")
        else:
            print("== 未检测到T3标注，从raw splat.ply开始跑T1-T3 ==")
            start, end = ck.discover_frame_range(dataset_root)
            ck.run_cleaning_and_labeling(dataset_root, start, end, half_window=half_window)

        print(f"\n== T4 kinematics pipeline: {dataset_root} ==")
        config = pipeline.PipelineConfig(output_dir=out_dir, write_debug=True,
                                          frame_glob=ck.LABELED_FRAME_GLOB)
        # 用 run_dataset_with_eta_unwrap(不是 run_dataset_with_sequence_correction)
        # 跟 calc_kinematics.py 的T4入口保持一致,写CSV前对eta_L/eta_R做整段unwrap
        # (见 pipeline.run_dataset_with_eta_unwrap / eta_unwrap.py 的模块级说明)。
        df = pipeline.run_dataset_with_eta_unwrap(dataset_root, config)
        print(f"  -> {csv_path}  ({len(df)} frame(s))")

    ok = df[df["status"] == "ok"].reset_index(drop=True)
    bad = df[df["status"] != "ok"]
    print(f"  status=ok: {len(ok)}/{len(df)}")
    if len(bad):
        print(f"  非ok帧: {list(zip(bad['frame_id'].tolist(), bad['status'].tolist()))}")

    if not ok.empty:
        plot_body_angles(ok, out_dir / "body_angles.png")
        ck.plot_wing_angles(ok, out_dir / "wing_angles.png")

        reproj_dir = out_dir / "reprojection"
        frames = ck.pick_reprojection_frames(ok["frame_id"].tolist(), ck.N_FRAMES)
        reprojection_viewer.run_batch(frames, dataset_root, raw_data_dir, reproj_dir)
        print(f"  -> {out_dir}")
    else:
        print("  [WARN] 没有status=ok的帧，跳过画图/重投影。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-name", type=str, help="单个sweep name(对应outputs/<name>/)")
    ap.add_argument("--configs-glob", type=str,
                     help='批量: glob匹配一批config json，逐个取"name"字段跑，例如'
                          ' "gpu/schedule/configs/ctrl_009_mid200/*.json"')
    ap.add_argument("--group", type=str, required=True,
                     help="dataset_root = outputs/<sweep_name>/<group>，通常是param_sets里的那个key")
    ap.add_argument("--half-window", type=int, default=motion_density.HALF_WINDOW,
                     help="T3 motion累加窗口半宽(帧)，只在从raw splat.ply现跑T1-T3时生效。"
                          "默认按16000fps锁定=36，别的拍摄fps数据集需要按比例换算显式传入，"
                          "例如8000fps的3相机数据集应传18，见density.HALF_WINDOW的docstring。")
    args = ap.parse_args()

    if bool(args.sweep_name) == bool(args.configs_glob):
        ap.error("必须且只能指定 --sweep-name 或 --configs-glob 其中一个")

    if args.sweep_name:
        sweep_names = [args.sweep_name]
    else:
        cfg_paths = sorted(glob.glob(args.configs_glob))
        if not cfg_paths:
            ap.error(f"--configs-glob 没匹配到任何文件: {args.configs_glob}")
        sweep_names = [json.loads(Path(p).read_text())["name"] for p in cfg_paths]

    failures = []
    for name in sweep_names:
        try:
            run_one(name, args.group, half_window=args.half_window)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failures.append((name, str(e)))

    print(f"\n{'=' * 60}\ndone: {len(sweep_names) - len(failures)}/{len(sweep_names)} ok")
    if failures:
        print(f"failed: {failures}")


if __name__ == "__main__":
    main()
