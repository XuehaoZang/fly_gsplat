"""第 1 步 dataset (b)：easyWand 标定 -> 每帧 transforms.json（nerfstudio 格式）。

    python -m dataset.make_transforms --calib <calibration_easyWandData.mat> --out data/ctrl_009/003

对 <out> 下每个含 images/ 的 f#### 目录写 transforms.json：camera_model="OPENCV"，ply_file_path="init_points.ply"，
frames[] 每台相机一项（file_path, fl_x, fl_y, cx, cy, w, h, transform_matrix = OpenGL c2w）。
各帧的相机都相同，只有图片文件名 images/P{f}CAM{c}.png 不同。坐标约定见 utils/camera.py。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import scipy.io as sio

from utils.camera import CameraConfig


def load_easywand(path: Path):
    return sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)["easyWandData"]


def transforms_for_frame(cameras: list[CameraConfig], frame: int) -> dict:
    return {
        "ply_file_path": "init_points.ply",
        "camera_model": "OPENCV",
        "frames": [{
            "file_path": f"images/P{frame}CAM{cam.cam_idx}.png",
            "fl_x": cam.fx, "fl_y": cam.fy, "cx": cam.cx, "cy": cam.cy, "w": cam.w, "h": cam.h,
            "transform_matrix": cam.transform_opengl.tolist(),
        } for cam in cameras],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ew = load_easywand(args.calib)
    cameras = [CameraConfig.easywand_dlt(ew, i) for i in range(int(ew.nCams))]
    frame_dirs = sorted(d for d in args.out.glob("f*") if re.fullmatch(r"f\d+", d.name) and (d / "images").is_dir())
    if not frame_dirs:
        raise SystemExit(f"no f####/images under {args.out} (run dataset.sparse_to_images first)")
    for d in frame_dirs:
        with open(d / "transforms.json", "w") as f:
            json.dump(transforms_for_frame(cameras, int(d.name[1:])), f, indent=4)
    print(f"[make_transforms] {len(cameras)} cameras -> {len(frame_dirs)} transforms.json under {args.out}")


if __name__ == "__main__":
    main()
