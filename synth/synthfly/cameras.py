"""Camera rigs and calibration export.

Two kinds of rig:

- preset rigs built from positions and look-at targets (`rig_cameras`): the
  Cornell three-camera arrangement and the Beatus four-camera one;
- calibrated rigs loaded from a fly_gsplat `transforms.json`
  (`load_fly_gsplat_rig`): the real four-camera lab setup, reproduced exactly
  (intrinsics with off-centre principal points, extrinsics, image size).

Conventions:
- world units are cm (flybody CGS); fly_gsplat's world is the EasyWand
  calibration frame in metres, so its camera centres are multiplied by 100;
- a MuJoCo camera looks along its local -z axis with +y up, +x right, which is
  the OpenGL camera frame that a nerfstudio `transform_matrix` stores;
- exported extrinsics are OpenCV world-to-camera (x right, y down, z forward):
  R_cv = diag(1, -1, -1) @ R_gl^T, t_cv = -R_cv @ p, X_cv = R_cv @ X_w + t_cv;
- pixel coordinates (u, v) are continuous, u to the right, v down, the image
  spanning [0, W] x [0, H]; pixel index = floor(u), floor(v).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mathutil import mat_to_quat

FLIP_YZ = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraSpec:
    """A preset camera: position, look-at target and image-up hint (all world cm)."""
    name: str
    pos: tuple[float, float, float]
    up: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CalibratedCamera:
    """A camera copied from a calibration: full pinhole K, pose, image size.

    `rot_gl` has the camera x (right), y (image up) and z (backward) axes as
    columns in world coordinates, i.e. the rotation block of a nerfstudio
    camera-to-world matrix and the MuJoCo camera frame. `source` keeps the
    original transforms.json frame entry so it can be written back verbatim."""
    name: str
    pos_cm: np.ndarray
    rot_gl: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: dict = field(default_factory=dict)

    @property
    def R_w2c(self) -> np.ndarray:
        return (self.rot_gl @ FLIP_YZ).T


def rig_cameras(rig: str, distance_cm: float, center=(0.0, 0.0, 0.0)) -> list[CameraSpec]:
    """Preset rigs (distance D from the arena centre c).

    cornell3: one camera looking straight down from c + (0, 0, D) with image-up
    = +x, two horizontal at c + (D, 0, 0) and c + (0, D, 0) with image-up = +z
    (Ristroph 2009, Beatus 2015, Whitehead 2022).
    beatus4: three cameras at azimuths 0/120/240 deg, 36 deg below the
    horizontal plane looking up, plus one looking up from c - (0, 0, D)
    (Ben-Dov and Beatus 2022; Maya et al. 2023)."""
    c = np.asarray(center, dtype=np.float64)
    D = float(distance_cm)
    tgt = tuple(float(v) for v in c)
    if rig == "cornell3":
        return [
            CameraSpec("cam_top", tuple(c + [0, 0, D]), (1.0, 0.0, 0.0), tgt),
            CameraSpec("cam_side", tuple(c + [D, 0, 0]), (0.0, 0.0, 1.0), tgt),
            CameraSpec("cam_front", tuple(c + [0, D, 0]), (0.0, 0.0, 1.0), tgt),
        ]
    if rig == "beatus4":
        cams = []
        elev = math.radians(36.0)
        for k, az_deg in enumerate((0.0, 120.0, 240.0)):
            az = math.radians(az_deg)
            pos = c + D * np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), -math.sin(elev)])
            cams.append(CameraSpec(f"cam_{k}", tuple(pos), (0.0, 0.0, 1.0), tgt))
        cams.append(CameraSpec("cam_bottom", tuple(c + [0, 0, -D]), (1.0, 0.0, 0.0), tgt))
        return cams
    raise ValueError(f"unknown preset rig {rig!r}; choose cornell3 or beatus4 (the calibrated rig comes from "
                     "load_fly_gsplat_rig)")


def find_transforms_json(path: Path) -> Path:
    """Accept a transforms.json, a frame folder, or a dataset root with f*/ folders."""
    path = Path(path)
    if path.is_file():
        return path
    direct = path / "transforms.json"
    if direct.is_file():
        return direct
    frames = sorted(p for p in path.glob("f*/transforms.json"))
    if frames:
        return frames[0]
    raise FileNotFoundError(f"no transforms.json under {path}")


def load_fly_gsplat_rig(path: Path, units_scale: float = 100.0) -> list[CalibratedCamera]:
    """Load the cameras of a fly_gsplat transforms.json (nerfstudio format, OpenGL c2w, metres).

    Cameras are returned in file order (CAM1..CAMn) as `cam1`, `cam2`, ...;
    positions are multiplied by `units_scale` (metres -> cm)."""
    tj = find_transforms_json(path)
    payload = json.loads(tj.read_text(encoding="utf-8"))
    cams = []
    for k, fr in enumerate(payload["frames"]):
        M = np.asarray(fr["transform_matrix"], dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(f"{tj}: transform_matrix of frame {k} is not 4x4")
        rot = M[:3, :3]
        if abs(np.linalg.det(rot) - 1.0) > 1e-6 or np.abs(rot @ rot.T - np.eye(3)).max() > 1e-6:
            raise ValueError(f"{tj}: rotation of frame {k} is not orthonormal")
        cams.append(CalibratedCamera(
            name=f"cam{k + 1}", pos_cm=M[:3, 3] * units_scale, rot_gl=rot.copy(),
            fx=float(fr["fl_x"]), fy=float(fr["fl_y"]), cx=float(fr["cx"]), cy=float(fr["cy"]),
            width=int(fr["w"]), height=int(fr["h"]), source=dict(fr)))
    if not cams:
        raise ValueError(f"{tj} has no frames")
    return cams


def lookat_rotation(pos, target, up) -> np.ndarray:
    """Rotation matrix whose columns are the camera x, y, z axes in world coordinates.

    The camera looks along -z (MuJoCo convention), so z = -(target - pos)."""
    pos = np.asarray(pos, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - pos
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(np.asarray(up, dtype=np.float64), z)
    n = np.linalg.norm(x)
    if n < 1e-9:
        raise ValueError("camera up vector is parallel to the viewing direction")
    x /= n
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def lookat_quat(pos, target, up) -> np.ndarray:
    """wxyz quaternion for MjsCamera.quat."""
    return mat_to_quat(lookat_rotation(pos, target, up))


def fovy_for_side(side_cm: float, distance_cm: float, margin: float = 1.1) -> float:
    """Vertical field of view (deg) so that `margin * side_cm` fits at the centre plane."""
    return 2.0 * math.degrees(math.atan(0.5 * margin * side_cm / distance_cm))


@dataclass
class Calibration:
    """One camera as the renderer realised it, with everything a consumer needs."""
    name: str
    width: int
    height: int
    projection: str  # "pinhole" | "orthographic"
    fx: float  # pinhole: px; orthographic: px per cm
    fy: float
    cx: float
    cy: float
    pos_world: np.ndarray  # (3,) camera centre in world (cm)
    rot_mj: np.ndarray  # (3, 3) columns = camera axes in world (MuJoCo / OpenGL convention)
    fovy: float = 0.0  # MuJoCo vertical fov (deg) or orthographic extent (cm), informational
    um_per_px: float = 0.0
    source: dict = field(default_factory=dict)

    @classmethod
    def from_fovy(cls, name, width, height, projection, fovy, pos_world, rot_mj, center) -> "Calibration":
        pos_world = np.asarray(pos_world, dtype=np.float64)
        if projection == "pinhole":
            f = height / (2.0 * math.tan(math.radians(fovy) / 2.0))
            dist = float(np.linalg.norm(np.asarray(center, dtype=np.float64) - pos_world))
            um = 1e4 * dist * 2.0 * math.tan(math.radians(fovy) / 2.0) / height
        elif projection == "orthographic":
            f = height / fovy
            um = 1e4 * fovy / height
        else:
            raise ValueError(f"unknown projection {projection!r}")
        return cls(name, width, height, projection, f, f, width / 2.0, height / 2.0,
                   pos_world, np.asarray(rot_mj, dtype=np.float64), float(fovy), um)

    @classmethod
    def from_calibrated(cls, cam: CalibratedCamera, center) -> "Calibration":
        dist = float(np.linalg.norm(np.asarray(center, dtype=np.float64) - cam.pos_cm))
        fovy = 2.0 * math.degrees(math.atan(cam.height / (2.0 * cam.fy)))
        return cls(cam.name, cam.width, cam.height, "pinhole", cam.fx, cam.fy, cam.cx, cam.cy,
                   np.asarray(cam.pos_cm, dtype=np.float64), np.asarray(cam.rot_gl, dtype=np.float64),
                   fovy, 1e4 * dist / cam.fy, dict(cam.source))

    # --- derived quantities -------------------------------------------------
    @property
    def quat_wxyz(self) -> np.ndarray:
        return mat_to_quat(self.rot_mj)

    @property
    def R_cv(self) -> np.ndarray:
        return FLIP_YZ @ self.rot_mj.T

    @property
    def t_cv(self) -> np.ndarray:
        return -self.R_cv @ self.pos_world

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def P(self) -> np.ndarray:
        """3x4 projection: pinhole P = K [R | t]; orthographic P maps to (u, v, 1) affinely."""
        Rt = np.hstack([self.R_cv, self.t_cv[:, None]])
        if self.projection == "pinhole":
            return self.K @ Rt
        K = self.K
        P = np.zeros((3, 4))
        P[0] = K[0, 0] * Rt[0]
        P[1] = K[1, 1] * Rt[1]
        P[0, 3] += K[0, 2]
        P[1, 3] += K[1, 2]
        P[2, 3] = 1.0
        return P

    def project(self, X: np.ndarray):
        """World points (N, 3) -> pixel (N, 2) and depth along the optical axis (N,)."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        Xc = X @ self.R_cv.T + self.t_cv
        depth = Xc[:, 2]
        if self.projection == "pinhole":
            z = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
            u = self.fx * Xc[:, 0] / z + self.cx
            v = self.fy * Xc[:, 1] / z + self.cy
        else:
            u = self.fx * Xc[:, 0] + self.cx
            v = self.fy * Xc[:, 1] + self.cy
        return np.stack([u, v], axis=1), depth

    def c2w_gl(self, units_scale: float = 1.0) -> np.ndarray:
        """4x4 camera-to-world in the OpenGL convention (what nerfstudio stores)."""
        M = np.eye(4)
        M[:3, :3] = self.rot_mj
        M[:3, 3] = self.pos_world * units_scale
        return M

    def frame_dict(self, file_path: str, units_scale: float = 0.01) -> dict:
        """A nerfstudio transforms.json frame entry.

        A camera loaded from a fly_gsplat calibration returns its source entry
        verbatim (only `file_path` replaced); a preset camera builds one with
        its position scaled by `units_scale` (cm -> m)."""
        if self.source:
            d = dict(self.source)
            d["file_path"] = file_path
            return d
        return {
            "file_path": file_path,
            "fl_x": float(self.fx), "fl_y": float(self.fy), "cx": float(self.cx), "cy": float(self.cy),
            "w": int(self.width), "h": int(self.height),
            "transform_matrix": self.c2w_gl(units_scale).tolist(),
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "projection": self.projection,
            "fovy": float(self.fovy),
            "fovy_unit": "deg" if self.projection == "pinhole" else "cm",
            "K": self.K.tolist(),
            "R_cv": self.R_cv.tolist(),
            "t_cv": self.t_cv.tolist(),
            "P": self.P.tolist(),
            "convention": "opencv_world_to_camera",
            "mujoco": {"pos": self.pos_world.tolist(), "quat_wxyz": self.quat_wxyz.tolist(),
                       "fovy": float(self.fovy), "projection": self.projection},
            "um_per_px_at_center": float(self.um_per_px),
            "units": "cm",
        }


def calibration_from_model(model, data, cam_name: str, width: int, height: int, center) -> Calibration:
    """Read a compiled MuJoCo camera (after mj_forward) into a Calibration (fovy-defined cameras)."""
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise KeyError(f"camera {cam_name!r} not in model")
    pos = np.array(data.cam_xpos[cid], dtype=np.float64)
    rot = np.array(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
    ortho = int(model.cam_projection[cid]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
    projection = "orthographic" if ortho else "pinhole"
    return Calibration.from_fovy(cam_name, width, height, projection, float(model.cam_fovy[cid]), pos, rot, center)


def write_calibration(path, calibrations: list[Calibration], extra: dict | None = None) -> None:
    payload = {"cameras": [c.to_dict() for c in calibrations]}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
