"""
batch_reconstruct.py
对 ctrl_009_002 帧 10~110 逐帧独立跑：
  generate_dataset -> generate_hull -> ns-train (子进程) -> 数值验证(clean+stats，无viser)
每帧输出到 data/ctrl_009_002_f{N}/ 和 outputs/ctrl_009_002_f{N}/
结果汇总到 batch_results.json
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

from generate_dataset import generate_dataset
from generate_hull import generate_hull
from utils.camera import CameraConfig
from utils.ply import export_splat, load_ply, unrescale

FRAME_RANGE = range(0,640)
SPARSE_DIR  = r"X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002"
BASE_NAME   = "ctrl_009_002"
MAX_ITERS   = 10000
CHECKPOINT_EVERY = 1000

def load_cameras(json_path: Path) -> list:
    with open(json_path) as f:
        frames = json.load(f)["frames"]
    cams = []
    for i, fr in enumerate(frames):
        cam = CameraConfig.from_opengl(fr)
        cam.cam_idx = i + 1
        cams.append(cam)
    return cams


def process_frame(frame_idx: int) -> dict:
    """单帧完整流程，不抛异常中断整体批处理，失败原样记录到结果里。"""
    base_dir =  f"./data/{BASE_NAME}"
    data_dir  = Path(f"./data/{BASE_NAME}_f{frame_idx}")
    exp_name  = f"{BASE_NAME}/f{frame_idx:04d}"

    try:
        # Step 1: 生成数据集（白底/无裁剪/无mask，和 ctrl_009_002 主实验保持一致）
        generate_dataset(str(data_dir), SPARSE_DIR, target_frame=frame_idx,
                          if_crop=False, white_bg=True, if_mask=False, calib_dir=base_dir)

        # Step 2: 生成 hull 初始化（逐帧独立现算，不依赖前一帧）
        generate_hull(str(data_dir), if_viser=False)

        # Step 3: 训练
        subprocess.run([
            "ns-train", "splatfacto-checkpoint",
            "--data", str(data_dir),
            "--vis", "tensorboard",
            "--max-num-iterations", str(MAX_ITERS),
            "--pipeline.model.background-color", "white",
            "--pipeline.model.checkpoint-every", str(CHECKPOINT_EVERY),
            "--experiment-name", exp_name,
            "nerfstudio-data", "--eval-mode", "all"
        ], check=True)

        # 找到刚才训练出的最新 timestamp 目录（method 名对应改成 splatfacto-checkpoint）
        splat_dir = sorted((Path("outputs") / exp_name / "splatfacto-checkpoint").iterdir())[-1]

        # Step 4: 数值验证（clean + stats，不开 viser）
        export_splat(splat_dir)
        hull_pts = load_ply(data_dir / "init_points.ply")

        with open(splat_dir / "dataparser_transforms.json") as f:
            dp = json.load(f)
        R_ns, t_ns, scale = np.array(dp["transform"])[:3,:3], np.array(dp["transform"])[:3,3], float(dp["scale"])

        splat_pts = load_ply(splat_dir / "splat.ply")
        splat_pts_physical = unrescale(splat_pts, R_ns, t_ns, scale)

        return {
            "frame": frame_idx, "status": "ok",
            "n_hull": len(hull_pts), "n_splat_raw": len(splat_pts),
            "hull_extent": (hull_pts.max(0) - hull_pts.min(0)).tolist(),
            "splat_extent": (splat_pts_physical.max(0) - splat_pts_physical.min(0)).tolist(),
            "splat_dir": str(splat_dir),   # 后面批量跑 debug_checkpoints 要用到
        }


    except Exception as e:
        return {"frame": frame_idx, "status": "failed", "error": str(e)}

def main():
    results = []
    for f in FRAME_RANGE:
        print(f"\n{'='*20} Frame {f} {'='*20}")
        results.append(process_frame(f))

    out_path = Path(f"./outputs/batch_results_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"\nDone: {n_ok}/{len(results)} succeeded. Saved -> {out_path}")

    # 全部训练完之后，统一批量跑 checkpoint 分析（收敛曲线 + 逐step重投影图）
    from debug.debug_checkpoints import debug_checkpoints
    for r in results:
        if r["status"] != "ok":
            continue
        f = r["frame"]
        data_dir = f"./data/{BASE_NAME}_f{f}"
        splat_dir = r["splat_dir"]
        try:
            debug_checkpoints(
                data_dir=data_dir,
                splat_dir=splat_dir,
                checkpoint_dir=str(Path(splat_dir) / "debug_checkpoints"),
            )
        except Exception as e:
            print(f"[analyze frame {f}] failed: {e}")


if __name__ == "__main__":
    main()