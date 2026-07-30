from pathlib import Path
import numpy as np
import h5py
from typing import Dict, Any
from utils.camera import CameraConfig

def reconstruct_frame_image(sparse_file: Path, target_frame: int, w: int, h: int,
                             white_bg: bool = False) -> np.ndarray:
    """
    Reconstruct a single grayscale frame from a Camera*_sparse.mat file.
    sparse_file : path to Camera<i>_sparse.mat (indexed sparse frame data)
    target_frame: 0-based frame index into /frames/indIm
    w, h        : full image dimensions
    white_bg    : True -> background 255, False -> background 0
    """
    with h5py.File(sparse_file, 'r') as sp:
        refs = sp['/frames/indIm'][0]
        indIm = sp[refs[target_frame]][:]
        if indIm.shape[0] == 3:
            indIm = indIm.T

        frame_size = (h, w)
        im = np.full(frame_size, 255, dtype=np.uint8) if white_bg else np.zeros(frame_size, dtype=np.uint8)

        if indIm.size > 0:
            rows = indIm[:, 0].astype(int) - 1
            cols = indIm[:, 1].astype(int) - 1
            vals = indIm[:, 2].astype(float)

            valid = (rows >= 0) & (rows < frame_size[0]) & (cols >= 0) & (cols < frame_size[1])
            im[rows[valid], cols[valid]] = vals[valid].astype(np.uint8)
    return im


def count_sparse_frames(sparse_file: Path) -> int:
    """Number of frames available in a Camera*_sparse.mat file."""
    with h5py.File(sparse_file, 'r') as sp:
        return len(sp['/frames/indIm'][0])


def generate_frame_dict(img_name: str, mask_name: str, cam: "CameraConfig") -> Dict[str, Any]:

    """
    Format the camera parameters into OpenGL (Nerfstudio) frame dictionary.
    """
    fl_x = cam.fx
    fl_y = cam.fy
    cx   = cam.cx
    cy   = cam.cy
    transform_matrix = cam.transform_opengl

    frame = {
        "file_path": f"images/{img_name}",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": cam.w,
        "h": cam.h,
        "transform_matrix": transform_matrix.tolist()
    }
    if mask_name is not None:
        frame["mask_path"] = f"masks/{mask_name}"
    return frame
