"""
validate_dataset.py
Verify a generated dataset (transforms.json + images) by visualising
camera geometry and mask frustum beams in Viser.

Input : data_dir/transforms.json  +  data_dir/images/
Output: data_dir/debug/ (mask + centroid images)  +  Viser at localhost:8080
"""

import json
import sys
import numpy as np
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.camera import CameraConfig
from utils.calib  import backproj, triangulate, mask_centroid
from utils.image  import binarize_mask, dilate_mask
from utils.viz    import (cam_colors, start_viser, add_camera_axes,
                           add_point_cloud, stop_viser, build_mask_frustum)


def main():
    data_dir  = Path("./data/ctrl_009_002")
    json_path = data_dir / "transforms.json"
    debug_dir = data_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    with open(json_path) as f:
        frames = json.load(f)["frames"]
    print(f"Loaded {len(frames)} frames from {json_path}")

    cameras, masks, centroids = [], [], []

    for idx, frame in enumerate(frames):
        img_path = data_dir / frame["file_path"]
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"[Warning] Cam {idx+1}: image not found at {img_path}")
            continue

        # build CameraConfig from OpenGL transform_matrix in transforms.json
        cam = CameraConfig.from_opengl(frame)
        cam.cam_idx = idx + 1

        # masks
        binary_mask  = binarize_mask(im, threshold=1)
        dilated_mask = dilate_mask(binary_mask, kernel_size=3, iterations=2)
        cv2.imwrite(str(debug_dir / f"mask_cam_{idx+1:02d}.png"), dilated_mask)

        # centroid; fall back to principal point if mask is empty
        u, v = mask_centroid(binary_mask)
        if np.isnan(u) or np.isnan(v):
            u, v = cam.cx, cam.cy
            print(f"[Warning] Cam {idx+1}: empty mask, using principal point")

        # save centroid visualisation (red cross = centroid, blue dot = principal point)
        vis = im.copy()
        cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(vis, (int(cam.cx), int(cam.cy)), 10, (255, 0, 0), -1)
        cv2.imwrite(str(debug_dir / f"centroid_cam_{idx+1:02d}.png"), vis)

        cameras.append(cam)
        masks.append(dilated_mask)
        centroids.append((u, v))
        print(f"Cam {idx+1}: active={np.count_nonzero(dilated_mask)}px  centroid=({u:.0f},{v:.0f})")

    print(f"\n{len(cameras)} cameras ready")

    # triangulate shared target from mask centroids
    rays = [(cam.X0, backproj(cam.K, cam.R_w2c, u, v))
            for cam, (u, v) in zip(cameras, centroids)]
    target, res = triangulate(rays)
    print(f"Target: {target}  residual={res*1000:.3f} mm")

    # angle between camera forward axis (-Z in OpenGL) and target direction
    for cam in cameras:
        fwd   = -cam.transform_opengl[:3, 2]
        to_tgt = (target - cam.X0)
        to_tgt /= np.linalg.norm(to_tgt)
        angle  = np.degrees(np.arccos(np.clip(np.dot(fwd / np.linalg.norm(fwd), to_tgt), -1, 1)))
        print(f"  Cam {cam.cam_idx}: angle to target = {angle:.1f}°")

    # launch Viser
    server = start_viser()
    add_camera_axes(server, cameras)

    for cam, mask in zip(cameras, masks):
        color = cam_colors.get(cam.cam_idx, [255, 255, 255])
        pts, cols = build_mask_frustum(cam, mask, target, color, depth_steps=800, pixel_step=2)
        add_point_cloud(server, pts, cols, name=f"/beams/cam{cam.cam_idx}")
        print(f"  Cam {cam.cam_idx}: {len(pts)} beam points")

    stop_viser(server)


if __name__ == "__main__":
    main()