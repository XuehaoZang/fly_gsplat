"""
debug_splat_ply.py
Compare hull init_points.ply vs trained splat.ply in Viser.
Prints coordinate-space diagnostics to help identify misalignment.

Edit __main__: set data_dir and splat_dir.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path

from utils.camera import CameraConfig
from utils.viz    import start_viser, add_camera_axes, add_point_cloud, stop_viser


def load_ply(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


def print_stats(label: str, pts: np.ndarray) -> None:
    if len(pts) == 0:
        print(f"[{label}] EMPTY")
        return
    print(f"[{label}]")
    print(f"  n       = {len(pts)}")
    print(f"  center  = {pts.mean(axis=0)}")
    print(f"  min     = {pts.min(axis=0)}")
    print(f"  max     = {pts.max(axis=0)}")
    print(f"  extent  = {pts.max(axis=0) - pts.min(axis=0)}")


def load_cameras(json_path: Path) -> list:
    with open(json_path) as f:
        frames = json.load(f)["frames"]
    cameras = []
    for i, fr in enumerate(frames):
        cam = CameraConfig.from_opengl(fr)
        cam.cam_idx = i + 1
        cameras.append(cam)
    return cameras


def main(data_dir: Path, splat_dir: Path) -> None:
    hull_path  = data_dir  / "init_points.ply"
    splat_path = splat_dir / "splat.ply"
    json_path  = data_dir  / "transforms.json"
    transform_path = splat_dir / "dataparser_transforms.json"

    # ---------------------------------------------------------------- stats --
    print("=" * 60)

    hull_pts = load_ply(hull_path) if hull_path.exists() else np.empty((0, 3))
    # print_stats("hull  (physical coords)", hull_pts)
    # print()

    # 加载 nerfstudio 的 dataparser 变换
    with open(transform_path) as f:
        dp = json.load(f)
    R_ns  = np.array(dp["transform"])[:3, :3]
    t_ns  = np.array(dp["transform"])[:3,  3]
    scale = float(dp["scale"])

    # 把 hull 变换到 rescaled 坐标系
    hull_rescaled = (scale * (R_ns @ hull_pts.T + t_ns[:, None])).T
    print_stats("hull (rescaled, should match splat)", hull_rescaled)
    print()

    splat_pts = load_ply(splat_path) if splat_path.exists() else np.empty((0, 3))
    print_stats("splat (rescaled coords)", splat_pts)
    print()

    cameras = load_cameras(json_path)
    cam_positions = np.array([c.X0 for c in cameras])
    print_stats("cameras (transforms.json X0)", cam_positions)
    for c in cameras:
        print(f"  Cam{c.cam_idx} X0 = {c.X0}")
    print("=" * 60)

    # --------------------------------------------------------- Viser viewer --
    server = start_viser()
    # add_camera_axes(server, cameras)

    offset = hull_rescaled.mean(axis=0)

    if len(hull_pts):
        # add_point_cloud(server, hull_pts,
        #                 np.tile(np.uint8([50, 200, 50]),  (len(hull_pts),  1)),
        #                 name="/hull",  point_size=0.0002)

        add_point_cloud(server, hull_rescaled - offset,
                        np.tile(np.uint8([50, 200, 50]), (len(hull_rescaled), 1)),
                        name="/hull_rescaled", point_size=0.0002)

    # if len(splat_pts):
    #     add_point_cloud(server, splat_pts - offset,
    #                     np.tile(np.uint8([200, 50, 200]), (len(splat_pts), 1)),
    #                     name="/splat", point_size=0.0002)

    stop_viser(server)


if __name__ == "__main__":
    data_dir  = Path("./data/ctrl_009_002")
    splat_dir = Path("./outputs/ctrl_009_002/splatfacto/2026-06-11_161701")

    main(data_dir, splat_dir)