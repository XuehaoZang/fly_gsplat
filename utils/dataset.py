import numpy as np
from typing import Dict, Any
from utils.camera import CameraConfig

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
