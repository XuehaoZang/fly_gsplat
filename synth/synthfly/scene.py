"""CPU MuJoCo scene for kinematic replay: fixed cameras, backlit-silhouette look.

The model is the self-contained `model/fruitfly_synth.xml` (the flybody fruit
fly with the fly_mimic flight patches baked in: 45 degree stroke plane, legs
frozen in the retracted flight pose), so recorded qpos vectors can be written
straight into MjData. Nothing here steps physics.

Cameras come either from a preset rig (position, look-at, vertical fov) or
from a calibration (`CalibratedCamera`: full K with an off-centre principal
point, pose and image size), in which case MuJoCo's intrinsic camera
parameters are used. MuJoCo's `principal_pixel` is an offset from the image
centre with both axes pointing the opposite way to the OpenCV (cx, cy), so it
is set to (W/2 - cx, H/2 - cy); this was verified by rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import LEG_PATTERNS, MODEL_XML, PART_IDS, PART_NAMES, part_of_body
from .cameras import (CalibratedCamera, Calibration, CameraSpec, calibration_from_model, fovy_for_side,
                      lookat_quat)
from .mathutil import mat_to_quat

SENSOR_PX = 1e-5  # sensor length per pixel handed to MuJoCo (any value works; intrinsics scale with it)


@dataclass
class SceneConfig:
    width: int = 512
    height: int = 512
    cameras: list[CameraSpec] = field(default_factory=list)  # preset rig
    calibrated: list[CalibratedCamera] = field(default_factory=list)  # takes precedence over `cameras`
    projection: str = "pinhole"  # pinhole | orthographic (preset rigs only)
    arena_side_cm: float = 3.5
    arena_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_margin: float = 1.1
    body_gray: float = 0.0  # albedo of the body against the backlight (0 = black, 0.16 ~ real recordings)
    membrane_alpha: float = 0.6
    offsamples: int = 4
    show_legs: bool = True


class FlyScene:
    """Compiled fly model with fixed cameras and the silhouette materials."""

    VISUAL_GROUP = 1
    HIDDEN_GROUP = 2

    def __init__(self, cfg: SceneConfig) -> None:
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg
        if not MODEL_XML.is_file():
            raise FileNotFoundError(f"fly model not found: {MODEL_XML}")
        spec = mujoco.MjSpec.from_file(str(MODEL_XML))

        if cfg.calibrated:
            sizes = {(c.width, c.height) for c in cfg.calibrated}
            if len(sizes) != 1:
                raise ValueError(f"calibrated cameras have mixed image sizes {sizes}")
            cfg.width, cfg.height = cfg.calibrated[0].width, cfg.calibrated[0].height
            for cam in cfg.calibrated:
                W, H = cam.width, cam.height
                spec.worldbody.add_camera(
                    name=cam.name, pos=[float(v) for v in cam.pos_cm], quat=[float(v) for v in mat_to_quat(cam.rot_gl)],
                    resolution=[W, H], focal_pixel=[cam.fx, cam.fy],
                    principal_pixel=[W / 2.0 - cam.cx, H / 2.0 - cam.cy],
                    sensor_size=[W * SENSOR_PX, H * SENSOR_PX])
            self.camera_names = [c.name for c in cfg.calibrated]
        elif cfg.cameras:
            for cam in cfg.cameras:
                q = lookat_quat(cam.pos, cam.target, cam.up)
                if cfg.projection == "pinhole":
                    dist = float(np.linalg.norm(np.asarray(cam.pos) - np.asarray(cam.target)))
                    fovy = fovy_for_side(cfg.arena_side_cm, dist, cfg.fov_margin)
                elif cfg.projection == "orthographic":
                    fovy = cfg.fov_margin * cfg.arena_side_cm
                else:
                    raise ValueError(f"unknown projection {cfg.projection!r}")
                c = spec.worldbody.add_camera(name=cam.name, pos=list(cam.pos), quat=list(q), fovy=float(fovy))
                if cfg.projection == "orthographic":
                    c.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
            self.camera_names = [c.name for c in cfg.cameras]
        else:
            raise ValueError("SceneConfig needs cameras or calibrated cameras")

        # backlit look: white sky, self-lit grey/black materials, translucent membrane, no lights
        spec.add_texture(name="synth_white_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                         rgb1=[1.0, 1.0, 1.0], rgb2=[1.0, 1.0, 1.0], width=32, height=32)
        g = float(np.clip(cfg.body_gray, 0.0, 1.0))
        for m in spec.materials:
            alpha = cfg.membrane_alpha if m.name == "membrane" else 1.0
            m.rgba = [g, g, g, float(alpha)]
            m.emission = 1.0  # colour = rgba, independent of lighting
            m.specular = 0.0
            m.shininess = 0.0
            m.reflectance = 0.0
        spec.visual.headlight.active = 0
        spec.visual.quality.offsamples = int(cfg.offsamples)
        spec.visual.quality.shadowsize = 0
        gl = spec.visual.global_
        gl.offwidth = max(int(gl.offwidth), cfg.width)
        gl.offheight = max(int(gl.offheight), cfg.height)

        model = spec.compile()
        model.light_active[:] = 0
        cam_pos = [c.pos_cm for c in cfg.calibrated] if cfg.calibrated else [np.asarray(c.pos) for c in cfg.cameras]
        far_cm = max(np.linalg.norm(np.asarray(p) - np.asarray(cfg.arena_center)) for p in cam_pos) + 2 * cfg.arena_side_cm
        model.vis.map.zfar = float(far_cm / model.stat.extent) * 1.5
        model.vis.map.znear = float(0.05 / model.stat.extent)
        self.model = model
        self.data = mujoco.MjData(model)

        # part labels per geom, mesh geoms per part
        self.body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
        self.geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_) or "" for g_ in range(model.ngeom)]
        self.body_part = np.zeros(model.nbody, dtype=np.int64)
        for b, name in enumerate(self.body_names):
            self.body_part[b] = 0 if b == 0 else PART_IDS[part_of_body(name)]
        self.geom_part = self.body_part[model.geom_bodyid]
        is_mesh = model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
        self.visual_geoms = np.nonzero(is_mesh & (model.geom_group == self.VISUAL_GROUP))[0]
        leg = np.array([any(p in n for p in LEG_PATTERNS) for n in self.geom_names])
        if not cfg.show_legs:
            model.geom_group[leg] = self.HIDDEN_GROUP
            self.visual_geoms = self.visual_geoms[~leg[self.visual_geoms]]
        self.membrane_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "membrane")
        lut = np.zeros(model.ngeom + 1, dtype=np.uint8)
        lut[1:] = self.geom_part
        self._part_lut = lut

        self.renderer = mujoco.Renderer(model, cfg.height, cfg.width)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
            self.renderer.scene.flags[flag] = 0
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        self.scene_option.geomgroup[self.VISUAL_GROUP] = 1
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in self.camera_names}

        mujoco.mj_forward(model, self.data)
        if cfg.calibrated:
            self.calibrations = [Calibration.from_calibrated(c, cfg.arena_center) for c in cfg.calibrated]
            for cam, cal in zip(cfg.calibrated, self.calibrations):
                cid = self.camera_ids[cam.name]
                dp = np.abs(np.asarray(self.data.cam_xpos[cid]) - cal.pos_world).max()
                dr = np.abs(np.asarray(self.data.cam_xmat[cid]).reshape(3, 3) - cal.rot_mj).max()
                if dp > 1e-6 or dr > 1e-6:
                    raise RuntimeError(f"camera {cam.name}: compiled pose differs from calibration (dp {dp}, dR {dr})")
        else:
            self.calibrations = [calibration_from_model(model, self.data, n, cfg.width, cfg.height, cfg.arena_center)
                                 for n in self.camera_names]

    # --- state -----------------------------------------------------------------
    def set_state(self, qpos: np.ndarray, world_offset=(0.0, 0.0, 0.0)) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos has shape {qpos.shape}, model expects ({self.model.nq},)")
        self.data.qpos[:] = qpos
        self.data.qpos[0:3] += np.asarray(world_offset, dtype=np.float64)
        self.mujoco.mj_forward(self.model, self.data)

    # --- passes ----------------------------------------------------------------
    def _update(self, cam_name: str) -> None:
        self._camera.fixedcamid = self.camera_ids[cam_name]
        self.renderer.update_scene(self.data, camera=self._camera, scene_option=self.scene_option)

    def render_gray(self, cam_name: str) -> np.ndarray:
        """Backlit silhouette, uint8 (H, W): white background, dark body, grey membrane."""
        self._update(cam_name)
        rgb = self.renderer.render()
        return rgb.mean(axis=2).round().astype(np.uint8)

    def render_parts(self, cam_name: str) -> np.ndarray:
        """Per-pixel part id, uint8 (H, W), 0 = background (see PART_NAMES)."""
        r = self.renderer
        r.enable_segmentation_rendering()
        try:
            self._update(cam_name)
            seg = r.render()
        finally:
            r.disable_segmentation_rendering()
        geom_id = seg[:, :, 0].astype(np.int64)
        return self._part_lut[geom_id + 1]

    def render_depth(self, cam_name: str) -> np.ndarray:
        """Depth along the optical axis in cm, float32 (H, W); membrane made opaque for the pass."""
        r = self.renderer
        alpha = float(self.model.mat_rgba[self.membrane_mat, 3])
        self.model.mat_rgba[self.membrane_mat, 3] = 1.0
        r.enable_depth_rendering()
        try:
            self._update(cam_name)
            depth = r.render().astype(np.float32)
        finally:
            r.disable_depth_rendering()
            self.model.mat_rgba[self.membrane_mat, 3] = alpha
        return depth

    # --- geometry --------------------------------------------------------------
    def geom_transforms(self):
        """(ngeom, 3) positions and (ngeom, 3, 3) rotations of every geom in world coordinates."""
        return np.array(self.data.geom_xpos), np.array(self.data.geom_xmat).reshape(-1, 3, 3)

    def body_transforms(self):
        return np.array(self.data.xpos), np.array(self.data.xquat)

    def part_table(self) -> dict:
        return {"names": list(PART_NAMES), "body_part": self.body_part.tolist(), "body_names": self.body_names}

    def close(self) -> None:
        self.renderer.close()
