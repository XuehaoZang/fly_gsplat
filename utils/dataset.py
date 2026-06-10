import numpy as np
from typing import Dict, Any, Tuple

def generate_frame_dict(img_name: str, w: int, h: int, K: np.ndarray, R: np.ndarray, X0: np.ndarray) -> Dict[str, Any]:
    """
    Format the camera parameters into a Nerfstudio-compatible frame dictionary.
    """
    fl_x = K[0,0]
    fl_y = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = X0.flatten()

    flip = np.array([
            [1,  0,  0, 0],
            [0,  -1,  0, 0],
            [0,  0,  -1, 0],
            [0,  0,  0, 1]
        ])
    # transform_matrix = flip_gravity @ M @ flip_gravity
    transform_matrix =  M

    return {
        "file_path": f"images/{img_name}",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "transform_matrix": transform_matrix.tolist()
    }
