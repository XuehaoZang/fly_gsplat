"""
round1_diag_reproj.py

对Round1/Round1.5的每一组(param_set/variant)，在同一帧(默认f0730, ctrl_119_004)上
把训练出的splat.ply重投影叠加到原始相机图上，输出到 outputs/round1/diag/，方便纯目视
对比各组的点云形态(空洞/漂移/去腿是否伤到翅膀等)，而不是只看extent_overshoot等汇总数字。

复用 postprocessing/viz/reprojection_viewer.py::plot_reprojection_overlay(只需要x/y/z列，
不需要跑完整的T1(gaussian_features)/T2/T3流程)，点云坐标用
gpu/schedule/common.py同款的unrescale(ply里的xyz是dataparser归一化坐标，要用each
group自己的dataparser_transforms.json换回物理坐标，跟reprojection用的transforms.json
相机标定在同一套尺度下)。

不跑T1-T4，只需要gpu/schedule/schedule.py训练完直接产出的splat.ply+
dataparser_transforms.json，Round1/1.5跑完就有，不需要额外训练/后处理。

用法:
    python gpu/schedule/analysis/round1_diag_reproj.py
    python gpu/schedule/analysis/round1_diag_reproj.py --frame f0373 --mov 010
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gpu" / "schedule"))

import common  # noqa: E402
from utils.ply import load_ply_with_attrs, unrescale  # noqa: E402
from postprocessing.viz.reprojection_viewer import plot_reprojection_overlay  # noqa: E402

OUT_DIR = REPO / "outputs" / "round1" / "diag"


def groups_for(mov: str) -> dict[str, tuple[str, str, str]]:
    """group_name -> (sweep_name, param_set, base_name)"""
    base = f"data/ctrl_119_3cam/{mov}"
    core_sweep = f"round1/ctrl_119_{mov}_core"
    g = {
        "BASELINE": (f"ctrl_3cam_test/ctrl_119_{mov}", "ratio3_sh0_dense", base),
        "P1a_grad_conservative": (core_sweep, "P1a_grad_conservative", base),
        "P1b_grad_aggressive": (core_sweep, "P1b_grad_aggressive", base),
        "P1c_refine_slow": (core_sweep, "P1c_refine_slow", base),
        "P2_freeze_early": (core_sweep, "P2_freeze_early", base),
        "P3_camera_optimizer": (core_sweep, "P3_camera_optimizer", base),
        "P4a_cull_strict": (core_sweep, "P4a_cull_strict", base),
        "P4b_cull_stricter": (core_sweep, "P4b_cull_stricter", base),
        "P6_sh1": (core_sweep, "P6_sh1", base),
        "P5_iters1000": (f"round1/ctrl_119_{mov}_iters1000", "P5_baseline", base),
        "P5_iters1500": (f"round1/ctrl_119_{mov}_iters1500", "P5_baseline", base),
        "P5_iters3000": (f"round1/ctrl_119_{mov}_iters3000", "P5_baseline", base),
    }
    for variant in ("p7_thresh20", "p7_thresh50", "p8_leg_erosion", "p9_hull30k", "p9_hull100k"):
        g[variant] = (f"round1_5/ctrl_119_{mov}_{variant}", "P_baseline",
                      f"data/ctrl_119_3cam_{variant}/{mov}")
    return g


def load_physical_xyz(sweep_name: str, param_set: str, frame_idx: int) -> "np.ndarray":
    exp_name = common.exp_name_for(sweep_name, param_set, frame_idx)
    splat_dir = common.find_splat_dir(exp_name, "splatfacto")
    if splat_dir is None:
        raise FileNotFoundError(f"no splatfacto output dir for {exp_name}")

    ply_path = splat_dir / "splat.ply"
    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    import numpy as np
    R = np.array(dp["transform"])[:3, :3]
    t = np.array(dp["transform"])[:3, 3]
    scale = float(dp["scale"])

    attrs = load_ply_with_attrs(ply_path)
    return unrescale(attrs["xyz"], R, t, scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=str, default="f0730", help="目标帧，如 f0730")
    ap.add_argument("--mov", type=str, default="004", choices=["004", "010"])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    frame = args.frame
    frame_idx = int(frame.lstrip("f"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = groups_for(args.mov)
    ok, failed = [], []
    for group_name, (sweep_name, param_set, base_name) in groups.items():
        try:
            xyz = load_physical_xyz(sweep_name, param_set, frame_idx)
            df = pd.DataFrame(xyz, columns=["x", "y", "z"])
            out_path = out_dir / f"reproj_{frame}_{group_name}.png"
            plot_reprojection_overlay(frame, df, out_path,
                                       raw_data_dir=REPO / base_name,
                                       title_suffix=f"  [{group_name}]  n={len(df)}")
            print(f"[{group_name}] n_points={len(df)} -> {out_path.relative_to(REPO)}")
            ok.append(group_name)
        except Exception as e:
            print(f"[{group_name}] FAILED: {type(e).__name__}: {e}")
            failed.append(group_name)

    print(f"\n{len(ok)} ok, {len(failed)} failed" + (f": {failed}" if failed else ""))


if __name__ == "__main__":
    main()
