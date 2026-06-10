import time
from typing import List, TYPE_CHECKING
import numpy as np
import viser
from scipy.spatial.transform import Rotation
if TYPE_CHECKING:
    from utils.camera import CameraConfig

cam_colors = {
        1: [255, 30, 30],
        2: [30, 255, 30],
        3: [50, 150, 255],
        4: [255, 255, 30]
    }

def start_viser(port: int = 8080) -> viser.ViserServer:
    """Start Viser server and add world origin axes. Returns server handle."""
    server = viser.ViserServer(port=port)
    server.scene.add_frame("/World", axes_length=0.002, axes_radius=0.0002)
    print(f"Viser running at http://localhost:{port}")
    return server


def add_camera_axes(server: viser.ViserServer, cameras: List["CameraConfig"]) -> None:
    """Add a coordinate frame and label for each camera (OpenGL convention)."""
    for cam in cameras:
        M        = cam.transform_opengl
        pos      = M[:3, 3]
        quat_xyzw = Rotation.from_matrix(M[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        server.scene.add_frame(
            f"/World/Cam_{cam.cam_idx}",
            position=pos, wxyz=quat_wxyz,
            axes_length=0.05, axes_radius=0.001
        )
        server.scene.add_label(
            f"/World/Cam_{cam.cam_idx}_label",
            text=f"Cam {cam.cam_idx}", position=pos
        )


def add_point_cloud(server: viser.ViserServer, points: np.ndarray, colors: np.ndarray,
                    name: str, point_size: float = 0.0001) -> None:
    """Add a point cloud layer. name uses path format e.g. '/beams/cam1'.
    Viser groups layers by prefix in the sidebar for individual toggle."""
    if len(points) == 0:
        return
    server.scene.add_point_cloud(name=name, points=points,
                                  colors=colors, point_size=point_size)


def stop_viser(server: viser.ViserServer) -> None:
    """Block until user clicks Continue in the browser, then resume."""
    paused      = True
    continue_btn = server.gui.add_button("Continue", color="green")

    @continue_btn.on_click
    def _(_):
        nonlocal paused
        paused = False

    print("Paused — click Continue in browser to proceed.")
    try:
        while paused:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print("Viser closed, continuing...")

def build_mask_frustum(cam: "CameraConfig", mask: np.ndarray, target_center: np.ndarray,
                       color, depth_steps: int = 50, pixel_step: int = 2):
    """Build a filled beam point cloud from camera centre to target along mask pixels.

    cam          : CameraConfig (OpenGL convention used internally)
    mask         : binary mask (H x W uint8), non-zero pixels define the beam shape
    target_center: (3,) world-frame point, determines beam length
    color        : RGB tuple/list for point colours
    depth_steps  : number of depth slices along each ray
    pixel_step   : mask downsampling stride (2 = 1/4 of pixels)
    Returns (points (N,3), colors (N,3))."""

    cam_pos = cam.X0
    R       = cam.transform_opengl[:3, :3]   # OpenGL R_c2w, matches ray convention below

    # downsampled mask pixel coordinates
    v, u = np.where(mask[::pixel_step, ::pixel_step] > 0)
    if len(u) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    u = u * pixel_step
    v = v * pixel_step

    # OpenGL ray directions: X right, Y up (-v), Z backward (-1)
    dirs_local = np.column_stack([
        (u - cam.cx) / cam.fx,
        -(v - cam.cy) / cam.fy,
        np.full_like(u, -1.0)
    ])
    dirs_local = dirs_local / np.linalg.norm(dirs_local, axis=1, keepdims=True)
    dirs_world = (R @ dirs_local.T).T   # (N, 3)

    # sample points along each ray up to target distance
    max_depth = np.linalg.norm(target_center - cam_pos)
    t      = np.linspace(0.0001, max_depth * 1.1, depth_steps)
    points = cam_pos[None, None, :] + t[:, None, None] * dirs_world[None, :, :]
    points = points.reshape(-1, 3)

    colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    return points, colors