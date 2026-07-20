"""
冒烟测试：把splat.ply的3D点(还原到物理坐标后)投影到4个相机的label_map上查标签，
验证投影坐标系是否对齐。这一步只做查询，不做多相机投票融合。

相机用 CameraConfig.easywand_dlt 直接从 calibration_easyWandData.mat 构造
（不走 transforms.json / from_opengl，避免crop等因素导致坐标系不一致）。

用法示例:
    python -m debug.smoke_test_reproject \
        --calib-mat data/ctrl_009_002/calibration_easyWandData.mat \
        --splat-dir "outputs/ctrl_009_002/f0100/splatfacto-checkpoint/2026-07-04_022821" \
        --sparse-dir "X:\\antenna\\control\\009_25052026\\Sparse\\Expr_009_mov_002" \
        --frame-idx 100 \
        --out-dir outputs/reproject_debug
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

from utils.camera import CameraConfig
from utils.ply import load_ply_with_attrs, unrescale
from utils.reproject import load_dataparser_transform, lookup_labels, project_points
from utils.seg2d import segment_body_wing

# BGR (cv2约定)
LABEL_COLORS_BGR = {
    -1: (255, 0, 255),  # 深度<=0 或出界 -> 品红，突出显示异常点
    0: (40, 40, 40),    # 背景 -> 深灰(区别于纯黑的图像边界)
    1: (0, 255, 0),      # body -> 绿
    2: (0, 0, 255),      # wing -> 红
    3: (0, 255, 255),    # 未分类 -> 黄
}


def label_map_to_color(label_map: np.ndarray) -> np.ndarray:
    color = np.full((*label_map.shape, 3), LABEL_COLORS_BGR[0], dtype=np.uint8)
    for label_val, c in LABEL_COLORS_BGR.items():
        if label_val in (-1, 0):
            continue
        color[label_map == label_val] = c
    return color


def draw_points(base: np.ndarray, uv: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = base.copy()
    H, W = base.shape[:2]
    margin = 50
    for (u, v), lb in zip(uv, labels):
        if not (-margin <= u < W + margin and -margin <= v < H + margin):
            continue
        color = LABEL_COLORS_BGR.get(int(lb), (255, 255, 255))
        center = (int(round(u)), int(round(v)))
        cv2.circle(out, center, 4, (0, 0, 0), -1)   # 黑色描边，跟底图区分
        cv2.circle(out, center, 2, color, -1)       # 按查到的label上色
    return out


def print_histogram(cam_idx: int, labels: np.ndarray) -> None:
    n = len(labels)
    parts = []
    for v in (-1, 0, 1, 2, 3):
        c = int((labels == v).sum())
        parts.append(f"{v}={c}({100 * c / n:.1f}%)")
    print(f"cam{cam_idx}: n={n} " + " ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-mat", type=str, required=True,
                        help="calibration_easyWandData.mat 路径")
    parser.add_argument("--splat-dir", type=str, required=True,
                        help="包含 splat.ply + dataparser_transforms.json 的目录")
    parser.add_argument("--sparse-dir", type=str, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--scatter-cam", type=int, default=1, help="出叠加验证图用哪个相机(1-based)")
    parser.add_argument("--out-dir", type=str, default="outputs/reproject_debug")
    args = parser.parse_args()

    splat_dir = Path(args.splat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载splat点云，转回物理坐标 (transforms.json空间)
    ply_data = load_ply_with_attrs(splat_dir / "splat.ply")
    R_ns, t_ns, scale = load_dataparser_transform(splat_dir / "dataparser_transforms.json")
    xyz_physical = unrescale(ply_data["xyz"], R_ns, t_ns, scale)
    print(f"[ply] n_points={len(xyz_physical)}")

    # 2. 4相机 CameraConfig via easywand_dlt
    mat = sio.loadmat(args.calib_mat, struct_as_record=False, squeeze_me=True)
    ew_data = mat["easyWandData"]
    n_cams = int(ew_data.nCams)

    sparse_dir = Path(args.sparse_dir.replace("X:", "/mnt/x").replace("\\", "/"))
    sparse_files = sorted(sparse_dir.glob("Camera*_sparse.mat"))
    if len(sparse_files) != n_cams:
        print(f"Error: found {len(sparse_files)} sparse files under {sparse_dir}, "
              f"expected n_cams={n_cams}")
        return

    label_maps = {}
    uv_by_cam = {}
    labels_by_cam = {}
    for i in range(n_cams):
        cam_idx = i + 1
        cam = CameraConfig.easywand_dlt(ew_data, i)
        label_map, _meta = segment_body_wing(sparse_files[i], cam_idx, args.frame_idx)
        label_maps[cam_idx] = label_map

        uv, depth = project_points(xyz_physical, cam)
        labels = lookup_labels(uv, depth, label_map)
        uv_by_cam[cam_idx] = uv
        labels_by_cam[cam_idx] = labels

        print_histogram(cam_idx, labels)

    # 3. 挑一个相机画叠加验证图：label_map为底图，投影散点按查到的label上色
    cam_idx = args.scatter_cam
    base = label_map_to_color(label_maps[cam_idx])
    overlay = draw_points(base, uv_by_cam[cam_idx], labels_by_cam[cam_idx])
    out_path = out_dir / f"reproject_debug_cam{cam_idx}_frame{args.frame_idx}.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
