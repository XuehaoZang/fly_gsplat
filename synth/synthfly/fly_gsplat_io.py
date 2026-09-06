"""Writers for the fly_gsplat dataset layout.

fly_gsplat (github.com/XuehaoZang/fly_gsplat) trains one splatfacto model per
frame from

    <dataset>/f%04d/images/P<frame>CAM<k>.png   1280 x 800, 8-bit single-channel, background exactly 255
    <dataset>/f%04d/transforms.json             nerfstudio format, camera_model OPENCV, c2w in OpenGL axes, metres
    <dataset>/f%04d/init_points.ply             Open3D binary PLY, double xyz (metres) + uchar rgb, initial Gaussians

The frame number in the image name is unpadded (f0200/images/P200CAM1.png).
The camera entries of transforms.json are identical in every frame folder
except `file_path`. Nothing else (masks, crops, distortion) is written.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import PART_NAMES

# Open3D-compatible point colours for the part-labelled ground truth
PART_RGB = np.array([
    [255, 255, 255], [60, 60, 60], [220, 40, 40], [240, 160, 40], [120, 80, 20], [40, 120, 240],
    [40, 200, 240], [150, 40, 200], [200, 120, 240], [40, 180, 60], [120, 220, 80],
], dtype=np.uint8)
HULL_GRAY = 153  # colour fly_gsplat gives its visual-hull points (0.6 * 255)


def write_open3d_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY with double xyz and uchar rgb, header as Open3D writes it."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if len(xyz) != len(rgb):
        raise ValueError("xyz and rgb lengths differ")
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", "comment Created by Open3D", f"element vertex {len(xyz)}",
        "property double x", "property double y", "property double z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header",
    ]) + "\n"
    rec = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rows = np.empty(len(xyz), dtype=rec)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["r"], rows["g"], rows["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def read_open3d_ply(path: Path):
    """Read a PLY written by write_open3d_ply or Open3D (double or float xyz, optional uchar rgb)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    props, count, fmt = [], 0, ""
    for line in header:
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            count = int(parts[2])
        elif parts[0] == "property" and parts[1] != "list":
            props.append((parts[1], parts[2]))
    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: unsupported PLY format {fmt}")
    typemap = {"double": "<f8", "float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    rec = np.dtype([(name, typemap[t]) for t, name in props])
    rows = np.frombuffer(data[end:end + rec.itemsize * count], dtype=rec)
    xyz = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)
    rgb = None
    if all(n in rows.dtype.names for n in ("red", "green", "blue")):
        rgb = np.stack([rows["red"], rows["green"], rows["blue"]], axis=1)
    return xyz, rgb


def write_transforms(path: Path, calibrations, frame_number: int, units_scale: float = 0.01) -> dict:
    payload = {
        "ply_file_path": "init_points.ply",
        "camera_model": "OPENCV",
        "frames": [cal.frame_dict(f"images/P{frame_number}CAM{k + 1}.png", units_scale)
                   for k, cal in enumerate(calibrations)],
    }
    Path(path).write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return payload


def write_part_palette(dataset_dir: Path) -> Path:
    """Colour -> part table for gt_points.ply, written once per dataset folder."""
    path = Path(dataset_dir) / "part_palette.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"gt_points_ply": "vertex colour encodes the body part; gt.h5 surface/part_id has the same ids",
                   "parts": [{"id": i, "name": PART_NAMES[i], "rgb": PART_RGB[i].tolist()} for i in range(len(PART_NAMES))],
                   "init_points_ply": f"visible ground-truth samples, grey {HULL_GRAY}, stand-in for the visual hull"}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_frame(dataset_dir: Path, frame_number: int, grays: list[np.ndarray], calibrations,
                xyz_cm: np.ndarray, part_id: np.ndarray, gray_pts: np.ndarray, visible_any: np.ndarray,
                init_points_max: int = 2000, units_scale: float = 0.01, rng=None, write_csv: bool = False) -> Path:
    """Write one fly_gsplat frame folder from rendered images and ground-truth samples.

    grays: one uint8 (H, W) image per camera, already post-processed (background 255).
    xyz_cm: (P, 3) sample positions in world cm; part_id, gray_pts, visible_any: (P,).
    init_points.ply gets up to `init_points_max` visible samples in metres, grey 153,
    standing in for the visual hull; gt_points.ply keeps every sample coloured by
    part (palette in part_palette.json); gt_points.csv with explicit labels only
    when `write_csv` (large: about 300 kB per frame)."""
    import imageio.v2 as imageio

    frame_dir = Path(dataset_dir) / f"f{frame_number:04d}"
    (frame_dir / "images").mkdir(parents=True, exist_ok=True)
    write_part_palette(dataset_dir)
    for k, gray in enumerate(grays):
        imageio.imwrite(str(frame_dir / "images" / f"P{frame_number}CAM{k + 1}.png"), np.ascontiguousarray(gray))
    write_transforms(frame_dir / "transforms.json", calibrations, frame_number, units_scale)

    xyz_m = np.asarray(xyz_cm, dtype=np.float64) * units_scale
    vis_idx = np.nonzero(visible_any)[0]
    if vis_idx.size == 0:
        vis_idx = np.arange(len(xyz_m))
    if init_points_max > 0 and vis_idx.size > init_points_max:
        rng = rng or np.random.default_rng(frame_number)
        vis_idx = np.sort(rng.choice(vis_idx, size=init_points_max, replace=False))
    write_open3d_ply(frame_dir / "init_points.ply", xyz_m[vis_idx], np.full((vis_idx.size, 3), HULL_GRAY, np.uint8))
    write_open3d_ply(frame_dir / "gt_points.ply", xyz_m, PART_RGB[np.asarray(part_id, dtype=np.int64)])
    if write_csv:
        with open(frame_dir / "gt_points.csv", "w", encoding="utf-8") as f:
            f.write("x_m,y_m,z_m,part_id,part,gray,visible\n")
            for p, pid, g, v in zip(xyz_m, part_id, gray_pts, visible_any):
                f.write(f"{p[0]:.7f},{p[1]:.7f},{p[2]:.7f},{int(pid)},{PART_NAMES[int(pid)]},{int(g)},{int(bool(v))}\n")
    return frame_dir


def copy_calibration_files(source_transforms: Path, dataset_dir: Path) -> list[str]:
    """Copy calibration .mat files sitting next to the source frames (for fly_gsplat's debug scripts)."""
    import shutil

    src_root = Path(source_transforms).parent
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for root in (src_root, src_root.parent):
        for name in ("camera_KRX0.mat", "calibration_easyWandData.mat"):
            f = root / name
            if f.is_file() and not (dataset_dir / name).exists():
                shutil.copy2(f, dataset_dir / name)
                copied.append(name)
        if copied:
            break
    return copied
