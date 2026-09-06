"""Kinematic replay of a recorded rollout: frames, part masks, surface samples, ground truth.

One call of `render_clip` turns a rollout record into

    <clip_dir>/gt.h5              everything, see README.md
    <clip_dir>/calib.json         camera calibrations (also inside gt.h5)
    <clip_dir>/clip.json          scalar metadata
    <clip_dir>/<camera>.mp4       grayscale preview video (optional)
    <clip_dir>/frames/<camera>/   PNG frames (optional)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import (ABDOMEN_JOINTS, COM_OFFSET_CM, CONTROL_DT, HALTERE_JOINTS, HEAD_JOINTS, PART_NAMES,
               WING_ANGLE_NAMES, WING_JOINTS, wing_joint_to_angles_deg)
from .cameras import write_calibration
from .mathutil import apply_photometry, interp_quaternions, interp_series, quats_to_mats
from .surface import SurfaceSamples, visibility, world_normals, world_points

COM_OFFSET = np.asarray(COM_OFFSET_CM, dtype=np.float64)  # cm, thorax frame
SCHEMA = "synthfly/1"


@dataclass
class RenderConfig:
    fps: int = 5000  # 5000 = native; anything else resamples the recorded states
    max_frames: int = 0  # 0 = all
    recenter: str = "bbox"  # bbox | fit | none (fit: maximise the longest run of frames every camera sees whole)
    arena_side_cm: float = 3.5
    arena_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    blur_sigma_px: float = 0.0
    noise_sigma: float = 0.0
    visibility_tol_cm: float = 0.02
    write_video: bool = True
    video_fps: int = 25
    write_png: bool = False
    write_depth: bool = False
    store_uv: bool = False  # per-camera pixel coordinates of every sample; recomputable from xyz + calib
    store_gray_images: bool = True  # images/gray in gt.h5 (the fly_gsplat PNGs carry the same pixels)
    store_part_images: bool = True  # images/part in gt.h5
    white_clip: int = 0  # > 0: grey levels at or above this become exactly 255 (clean backlight like the real data)
    fly_gsplat_dir: Path | None = None  # also write the clip in fly_gsplat's per-frame layout here
    fly_gsplat_frames: str = "longest_run"  # longest_run: only the longest run of fully visible frames, renumbered | all
    visibility_margin_px: float = 4.0  # a frame is fully visible if every sample is this far inside every image
    fit_search_cm: float = 2.0  # half-width of the offset search box for recenter = "fit"
    fit_step_cm: float = 0.25
    init_points_max: int = 2000  # visible ground-truth samples written as init_points.ply per frame
    gsplat_csv: bool = False  # also write gt_points.csv per frame (labels as text; large)
    units_scale_out: float = 0.01  # world cm -> fly_gsplat metres
    photometry_seed: int = 0
    extra_meta: dict = field(default_factory=dict)


def _joint_index(names: list[str], wanted) -> np.ndarray:
    return np.array([names.index(w) for w in wanted], dtype=np.int64)


def _relative_or_absolute(path, base: Path):
    """Path as a string relative to `base` when possible (keeps metadata free of machine paths)."""
    if path is None:
        return None
    import os

    try:
        return os.path.relpath(Path(path).resolve(), Path(base).resolve()).replace("\\", "/")
    except ValueError:  # different drives on Windows
        return str(path)


def resample_states(rec: dict, fps: int):
    """Resample qpos/qvel and scalar tracks onto a new frame rate (hinges linear, root quaternion slerp)."""
    nv = int(rec["n_valid_frames"])
    t_src = np.arange(nv) * float(rec["control_dt"])
    if fps == int(round(1.0 / float(rec["control_dt"]))):
        t_dst = t_src
        interpolated = np.zeros(nv, dtype=bool)
    else:
        t_dst = np.arange(0.0, t_src[-1] + 1e-12, 1.0 / fps)
        interpolated = ~np.isclose(t_dst[:, None], t_src[None, :], atol=1e-9).any(axis=1)
    qpos = rec["qpos"][:nv]
    qvel = rec["qvel"][:nv]
    root = int(rec["root_qposadr"])
    q_lin = interp_series(t_src, np.delete(qpos, np.s_[root + 3:root + 7], axis=1), t_dst)
    quat = interp_quaternions(t_src, qpos[:, root + 3:root + 7], t_dst)
    qpos_new = np.concatenate([q_lin[:, :root + 3], quat, q_lin[:, root + 3:]], axis=1)
    qvel_new = interp_series(t_src, qvel, t_dst)
    src_index = np.clip(np.round(t_dst / float(rec["control_dt"])).astype(np.int64), 0, nv - 1)
    tracks = {}
    for key in ("carrier_phase", "ctrl_freq_hz"):
        v = rec.get(key)
        if v is None:
            continue
        v = np.asarray(v[:nv], dtype=np.float64)
        if key == "carrier_phase":
            v = np.unwrap(v)
        tracks[key] = np.interp(t_dst, t_src, v)
    return t_dst, qpos_new, qvel_new, src_index, interpolated, tracks


def _bbox_offset(cfg: RenderConfig, qpos: np.ndarray, root: int) -> np.ndarray:
    R = quats_to_mats(qpos[:, root + 3:root + 7])
    com = qpos[:, root:root + 3] + np.einsum("fij,j->fi", R, COM_OFFSET)
    centre = 0.5 * (com.min(axis=0) + com.max(axis=0))
    return np.asarray(cfg.arena_center, dtype=np.float64) - centre


def frame_extents(scene, samples: SurfaceSamples, qpos: np.ndarray) -> np.ndarray:
    """(F, 2, 3) min and max corner of the surface samples of every frame, raw world frame."""
    out = np.empty((len(qpos), 2, 3))
    for f in range(len(qpos)):
        scene.set_state(qpos[f])
        gx, gR = scene.geom_transforms()
        pts = world_points(samples, gx, gR)
        out[f, 0] = pts.min(axis=0)
        out[f, 1] = pts.max(axis=0)
    return out


def _corners(extents: np.ndarray) -> np.ndarray:
    """(F, 8, 3) bounding-box corners."""
    lo, hi = extents[:, 0], extents[:, 1]
    corners = np.empty((len(extents), 8, 3))
    for i in range(8):
        pick = np.array([(i >> a) & 1 for a in range(3)], dtype=bool)
        corners[:, i] = np.where(pick, hi, lo)
    return corners


def fully_visible_mask(calibs, extents: np.ndarray, offset: np.ndarray, margin_px: float) -> np.ndarray:
    """(F,) True when the whole bounding box of the fly is inside every camera image."""
    corners = _corners(extents) + np.asarray(offset, dtype=np.float64)
    flat = corners.reshape(-1, 3)
    ok = np.ones(len(extents), dtype=bool)
    for cal in calibs:
        uv, depth = cal.project(flat)
        inside = ((depth > 0) & (uv[:, 0] >= margin_px) & (uv[:, 0] <= cal.width - margin_px)
                  & (uv[:, 1] >= margin_px) & (uv[:, 1] <= cal.height - margin_px))
        ok &= inside.reshape(len(extents), 8).all(axis=1)
    return ok


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of True values (length 0 if none)."""
    best_start, best_len, start = 0, 0, None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    return best_start, best_len


def fit_offset(cfg: RenderConfig, calibs, extents: np.ndarray) -> np.ndarray:
    """Offset (added to the raw world frame) maximising the longest fully visible run.

    Coarse grid around the bbox-centred offset, evaluated on a subsample of
    frames. Ties are broken towards the bbox-centred offset, so a clip that
    fits everywhere stays centred; if no offset makes any frame fully
    visible the bbox-centred offset is returned."""
    base = np.asarray(cfg.arena_center, dtype=np.float64) - 0.5 * (extents[:, 0].min(axis=0) + extents[:, 1].max(axis=0))
    steps = np.arange(-cfg.fit_search_cm, cfg.fit_search_cm + 1e-9, cfg.fit_step_cm)
    stride = max(1, len(extents) // 400)
    sub = extents[::stride]

    def score(off):
        mask = fully_visible_mask(calibs, sub, off, cfg.visibility_margin_px)
        return (longest_run(mask)[1], int(mask.sum()))

    best, best_key, best_dist = base, score(base), 0.0
    for dx in steps:
        for dy in steps:
            for dz in steps:
                delta = np.array([dx, dy, dz])
                dist = float(np.linalg.norm(delta))
                key = score(base + delta)
                if key > best_key or (key == best_key and dist < best_dist):
                    best, best_key, best_dist = base + delta, key, dist
    if best_key[0] == 0:
        return base
    return best


def _recenter_offset(cfg: RenderConfig, qpos: np.ndarray, root: int, scene=None, samples=None, calibs=None) -> np.ndarray:
    if cfg.recenter == "none":
        return np.zeros(3)
    if cfg.recenter == "bbox":
        return _bbox_offset(cfg, qpos, root)
    if cfg.recenter == "fit":
        extents = frame_extents(scene, samples, qpos)
        return fit_offset(cfg, calibs, extents)
    raise ValueError(f"unknown recenter mode {cfg.recenter!r}")


def render_clip(rec: dict, scene, samples: SurfaceSamples, cfg: RenderConfig, clip_dir: Path, log=print) -> Path:
    import h5py

    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names = list(rec["joint_names"])
    root = int(rec["root_qposadr"])
    qposadr = np.asarray(rec["jnt_qposadr"])
    t_s, qpos, qvel, src_index, interpolated, tracks = resample_states(rec, cfg.fps)
    if cfg.max_frames > 0:
        keep = slice(0, cfg.max_frames)
        t_s, qpos, qvel, src_index, interpolated = t_s[keep], qpos[keep], qvel[keep], src_index[keep], interpolated[keep]
        tracks = {k: v[keep] for k, v in tracks.items()}
    F = len(t_s)
    cams = scene.camera_names
    calibs = scene.calibrations
    offset = _recenter_offset(cfg, qpos, root, scene, samples, calibs)
    qpos_world = qpos.copy()
    qpos_world[:, root:root + 3] += offset
    # which frames every camera sees whole (geometric pre-pass, no occlusion)
    extents = frame_extents(scene, samples, qpos_world)
    fully_visible = fully_visible_mask(calibs, extents, np.zeros(3), cfg.visibility_margin_px)
    run_start, run_len = longest_run(fully_visible)
    if cfg.fly_gsplat_dir is not None:
        if cfg.fly_gsplat_frames == "longest_run":
            gsplat_frames = {f: f - run_start for f in range(run_start, run_start + run_len)}
        elif cfg.fly_gsplat_frames == "all":
            gsplat_frames = {f: f for f in range(F)}
        else:
            raise ValueError(f"unknown fly_gsplat_frames mode {cfg.fly_gsplat_frames!r}")
        log(f"  fully visible in every camera: {int(fully_visible.sum())}/{F} frames, "
            f"longest run {run_len} from frame {run_start}; writing {len(gsplat_frames)} fly_gsplat frames")
    else:
        gsplat_frames = {}

    C, H, W = len(cams), scene.cfg.height, scene.cfg.width
    P = samples.count
    half = 0.5 * cfg.arena_side_cm
    centre = np.asarray(cfg.arena_center, dtype=np.float64)
    rng = np.random.default_rng(cfg.photometry_seed)

    # derived joint groups
    wing_q = qpos_world[:, qposadr[_joint_index(names, WING_JOINTS)]]
    head_q = qpos_world[:, qposadr[_joint_index(names, HEAD_JOINTS)]]
    abd_q = qpos_world[:, qposadr[_joint_index(names, ABDOMEN_JOINTS)]]
    hal_q = qpos_world[:, qposadr[_joint_index(names, HALTERE_JOINTS)]]
    hinge = np.asarray(rec["jnt_width"]) == 1
    hinge_names = [n for n, h in zip(names, hinge) if h]
    hinge_q = qpos_world[:, qposadr[hinge]]
    R_root = quats_to_mats(qpos_world[:, root + 3:root + 7])
    com = qpos_world[:, root:root + 3] + np.einsum("fij,j->fi", R_root, COM_OFFSET)

    video_writers = {}
    if cfg.write_video:
        import imageio.v2 as imageio

        for cam in cams:
            video_writers[cam] = imageio.get_writer(str(clip_dir / f"{cam}.mp4"), fps=cfg.video_fps,
                                                    codec="libx264", macro_block_size=None, quality=8)
    if cfg.write_png:
        for cam in cams:
            (clip_dir / "frames" / cam).mkdir(parents=True, exist_ok=True)

    with h5py.File(clip_dir / "gt.h5", "w") as h5:
        h5.attrs["schema"] = SCHEMA
        h5.attrs["units"] = "cm, s, rad; wing_angles_deg in degrees"
        h5.attrs["trial_id"] = str(rec["trial_id"])
        h5.attrs["body_trial"] = str(rec.get("body_trial", rec["trial_id"]))
        h5.attrs["template_trial"] = str(rec.get("template_trial", rec["trial_id"]))
        h5.attrs["split"] = str(rec["split"])
        h5.attrs["checkpoint"] = str(rec["checkpoint"])
        h5.attrs["checkpoint_sha256"] = str(rec["checkpoint_sha256"])
        h5.attrs["arm"] = str(rec["arm"])
        h5.attrs["rollout_seed"] = int(rec["seed"])
        h5.attrs["control_dt_s"] = float(rec["control_dt"])
        h5.attrs["fps"] = int(cfg.fps)
        h5.attrs["n_frames"] = F
        h5.attrs["world_offset_cm"] = offset
        h5.attrs["arena_side_cm"] = float(cfg.arena_side_cm)
        h5.attrs["arena_center_cm"] = centre
        h5.attrs["base_freq_hz"] = float(rec.get("base_freq_hz", float("nan")))
        h5.attrs["ended_by"] = str(rec["ended_by"])
        h5.attrs["camera_names"] = np.array(cams, dtype="S")
        for k, v in cfg.extra_meta.items():
            h5.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v

        g = h5.create_group("cameras")
        for cal in calibs:
            gc = g.create_group(cal.name)
            for k, v in cal.to_dict().items():
                if k == "mujoco":
                    for kk, vv in v.items():
                        gc.attrs[f"mujoco_{kk}"] = vv
                elif isinstance(v, list):
                    gc.create_dataset(k, data=np.asarray(v, dtype=np.float64))
                else:
                    gc.attrs[k] = v

        gj = h5.create_group("joints")
        gj.create_dataset("names", data=np.array(names, dtype="S"))
        gj.create_dataset("qposadr", data=qposadr)
        gj.create_dataset("dofadr", data=np.asarray(rec["jnt_dofadr"]))
        gj.create_dataset("hinge_names", data=np.array(hinge_names, dtype="S"))
        gj.create_dataset("wing_joint_names", data=np.array(WING_JOINTS, dtype="S"))
        gj.create_dataset("wing_angle_names", data=np.array(WING_ANGLE_NAMES, dtype="S"))
        gj.attrs["wing_mapping"] = "stroke = deg(yaw) + 90; deviation = -deg(roll); rotation = deg(pitch) + 45"
        gj.attrs["root_qposadr"] = root
        gj.attrs["com_offset_thorax_cm"] = COM_OFFSET

        gp = h5.create_group("parts")
        gp.create_dataset("names", data=np.array(PART_NAMES, dtype="S"))
        gp.create_dataset("body_names", data=np.array(scene.body_names, dtype="S"))
        gp.create_dataset("body_part", data=scene.body_part)

        gf = h5.create_group("frames")
        gf.create_dataset("t_s", data=t_s)
        gf.create_dataset("frame_index", data=np.arange(F, dtype=np.int64))
        gf.create_dataset("source_step", data=src_index)
        gf.create_dataset("interpolated", data=interpolated)
        gf.create_dataset("fully_visible", data=fully_visible)
        gf.create_dataset("qpos", data=qpos_world)
        gf.create_dataset("qvel", data=qvel)
        gf.create_dataset("root_pos_cm", data=qpos_world[:, root:root + 3])
        gf.create_dataset("root_quat_wxyz", data=qpos_world[:, root + 3:root + 7])
        gf.create_dataset("com_pos_cm", data=com)
        gf.create_dataset("root_lin_vel_cm_s", data=qvel[:, 0:3])
        gf.create_dataset("root_ang_vel_body_rad_s", data=qvel[:, 3:6])
        gf.create_dataset("hinge_rad", data=hinge_q)
        gf.create_dataset("wing_joint_rad", data=wing_q)
        gf.create_dataset("wing_angles_deg", data=wing_joint_to_angles_deg(wing_q))
        gf.create_dataset("head_rad", data=head_q)
        gf.create_dataset("abdomen_rad", data=abd_q)
        gf.create_dataset("haltere_rad", data=hal_q)
        for key, v in tracks.items():
            gf.create_dataset(key, data=v)
        # reference tracks are interpolated onto the frame times like the state (positions
        # linearly, quaternions by slerp); the action is a per-control-step quantity and is
        # taken from the nearest control step, NaN where no action was applied
        for key in ("ref_root", "ref_com", "ref_quat", "ref_wing_qpos"):
            v = rec.get(key)
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float64)
            t_ref = np.arange(len(v)) * float(rec["control_dt"])
            t_q = np.clip(t_s, 0.0, t_ref[-1])
            if key == "ref_quat":
                out = interp_quaternions(t_ref, v, t_q)
            elif key == "ref_root":
                out = np.concatenate([interp_series(t_ref, v[:, :3], t_q), interp_quaternions(t_ref, v[:, 3:7], t_q)], axis=1)
            else:
                out = interp_series(t_ref, v, t_q)
            if key in ("ref_root", "ref_com"):
                out[:, :3] += offset
            gf.create_dataset(key, data=out)
        act = rec.get("action")
        if act is not None and len(act):
            act = np.asarray(act, dtype=np.float64)
            out = np.full((F, act.shape[1]), np.nan)
            has = src_index < len(act)
            out[has] = act[src_index[has]]
            d_act = gf.create_dataset("action", data=out)
            d_act.attrs["convention"] = ("action[k] is computed from frame k and produces frame k+1; ctrl_freq_hz, "
                                         "carrier_phase and wing_command at frame k follow from action[k-1]; "
                                         "nearest control step at non-native fps; NaN where no action was applied")
        d_body_pos = gf.create_dataset("body_xpos_cm", (F, scene.model.nbody, 3), dtype=np.float32)
        d_body_quat = gf.create_dataset("body_xquat_wxyz", (F, scene.model.nbody, 4), dtype=np.float32)
        d_in_volume = gf.create_dataset("in_volume", (F,), dtype=bool)

        gs = h5.create_group("surface")
        gs.create_dataset("local_cm", data=samples.local.astype(np.float32))
        gs.create_dataset("geom_id", data=samples.geom_id)
        gs.create_dataset("part_id", data=samples.part_id)
        gs.create_dataset("material_gray", data=samples.material_gray)
        d_xyz = gs.create_dataset("xyz_cm", (F, P, 3), dtype=np.float32, chunks=(1, P, 3), compression="gzip", compression_opts=4)
        d_nrm = gs.create_dataset("normal", (F, P, 3), dtype=np.float16, chunks=(1, P, 3), compression="gzip", compression_opts=4)
        d_gray = gs.create_dataset("gray", (F, P), dtype=np.uint8, chunks=(1, P), compression="gzip", compression_opts=4)
        d_vis = gs.create_dataset("visible", (F, P, C), dtype=bool, chunks=(1, P, C), compression="gzip", compression_opts=4)
        d_uv = None
        if cfg.store_uv:
            d_uv = gs.create_dataset("uv_px", (F, P, C, 2), dtype=np.float32, chunks=(1, P, C, 2), compression="gzip", compression_opts=4)

        gi = h5.create_group("images")
        gi.attrs["layout"] = "(frame, camera, row, col); gray 0..255 with 255 = backlight; part ids per parts/names"
        d_img = d_part = None
        if cfg.store_gray_images:
            d_img = gi.create_dataset("gray", (F, C, H, W), dtype=np.uint8, chunks=(1, 1, H, W), compression="gzip", compression_opts=4)
        if cfg.store_part_images:
            d_part = gi.create_dataset("part", (F, C, H, W), dtype=np.uint8, chunks=(1, 1, H, W), compression="gzip", compression_opts=4)
        d_depth = None
        if cfg.write_depth:
            d_depth = gi.create_dataset("depth_cm", (F, C, H, W), dtype=np.float16, chunks=(1, 1, H, W), compression="gzip", compression_opts=4)

        for f in range(F):
            scene.set_state(qpos_world[f])
            gx, gR = scene.geom_transforms()
            pts = world_points(samples, gx, gR)
            d_xyz[f] = pts.astype(np.float32)
            d_nrm[f] = world_normals(samples, gR).astype(np.float16)
            bx, bq = scene.body_transforms()
            d_body_pos[f] = bx.astype(np.float32)
            d_body_quat[f] = bq.astype(np.float32)
            d_in_volume[f] = bool(np.all(np.abs(pts - centre) <= half))
            gray_sum = np.zeros(P, dtype=np.float64)
            gray_n = np.zeros(P, dtype=np.int64)
            vis_all = np.zeros((P, C), dtype=bool)
            uv_all = np.zeros((P, C, 2), dtype=np.float32)
            grays = []
            for ci, cam in enumerate(cams):
                gray = apply_photometry(scene.render_gray(cam), cfg.blur_sigma_px, cfg.noise_sigma, rng)
                if cfg.white_clip > 0:
                    gray = np.where(gray >= cfg.white_clip, 255, gray).astype(np.uint8)
                grays.append(gray)
                part = scene.render_parts(cam)
                depth = scene.render_depth(cam)
                if d_img is not None:
                    d_img[f, ci] = gray
                if d_part is not None:
                    d_part[f, ci] = part
                if d_depth is not None:
                    d_depth[f, ci] = depth.astype(np.float16)
                vis, uv, rc = visibility(pts, calibs[ci], depth, cfg.visibility_tol_cm)
                vis_all[:, ci] = vis
                uv_all[:, ci] = uv.astype(np.float32)
                if vis.any():
                    gray_sum[vis] += gray[rc[vis, 0], rc[vis, 1]]
                    gray_n[vis] += 1
                if cam in video_writers:
                    video_writers[cam].append_data(np.repeat(gray[:, :, None], 3, axis=2))
                if cfg.write_png:
                    import imageio.v2 as imageio

                    imageio.imwrite(str(clip_dir / "frames" / cam / f"{f:06d}.png"), gray)
            g_out = samples.material_gray.astype(np.float64)
            seen = gray_n > 0
            g_out[seen] = gray_sum[seen] / gray_n[seen]
            gray_pts = np.clip(np.round(g_out), 0, 255).astype(np.uint8)
            d_gray[f] = gray_pts
            d_vis[f] = vis_all
            if d_uv is not None:
                d_uv[f] = uv_all
            if f in gsplat_frames:
                from .fly_gsplat_io import write_frame

                write_frame(cfg.fly_gsplat_dir, gsplat_frames[f], grays, calibs, pts, samples.part_id, gray_pts,
                            vis_all.any(axis=1), cfg.init_points_max, cfg.units_scale_out, rng, cfg.gsplat_csv)
            if (f + 1) % 200 == 0 or f + 1 == F:
                log(f"  rendered {f + 1}/{F} frames ({time.time() - t0:.0f} s)")

        in_vol = np.asarray(d_in_volume[:])
        h5.attrs["frames_in_volume"] = int(in_vol.sum())
        h5.attrs["frames_fully_visible"] = int(fully_visible.sum())
        h5.attrs["fully_visible_longest_run"] = np.array([run_start, run_len])

    for w in video_writers.values():
        w.close()
    if gsplat_frames:
        with open(Path(cfg.fly_gsplat_dir) / "frames.csv", "w", encoding="utf-8") as fcsv:
            fcsv.write("fly_gsplat_frame,source_frame,t_s\n")
            for f, g in sorted(gsplat_frames.items(), key=lambda kv: kv[1]):
                fcsv.write(f"{g},{f},{t_s[f]:.7f}\n")
    write_calibration(clip_dir / "calib.json", calibs, {
        "arena": {"side_cm": cfg.arena_side_cm, "center_cm": list(centre)},
        "world_offset_cm": offset.tolist(),
        "units": "cm",
    })
    meta = {
        "schema": SCHEMA,
        "trial_id": str(rec["trial_id"]),
        "body_trial": str(rec.get("body_trial", rec["trial_id"])),
        "template_trial": str(rec.get("template_trial", rec["trial_id"])),
        "tracking": rec.get("tracking"),
        "split": str(rec["split"]),
        "arm": str(rec["arm"]),
        "checkpoint": str(rec["checkpoint"]),
        "checkpoint_sha256": str(rec["checkpoint_sha256"]),
        "rollout_seed": int(rec["seed"]),
        "fps": int(cfg.fps),
        "n_frames": int(F),
        "frames_in_volume": int(in_vol.sum()),
        "frames_fully_visible": int(fully_visible.sum()),
        "fully_visible_longest_run": [int(run_start), int(run_len)],
        "fly_gsplat_frames_written": len(gsplat_frames),
        "duration_s": float(t_s[-1]) if F else 0.0,
        "reference_length_steps": int(rec["reference_length"]),
        "valid_rollout_frames": int(rec["n_valid_frames"]),
        "ended_by": str(rec["ended_by"]),
        "base_freq_hz": float(rec.get("base_freq_hz", float("nan"))),
        "world_offset_cm": offset.tolist(),
        "cameras": cams,
        "image_size": [W, H],
        "surface_samples": int(P),
        "fly_gsplat_dir": _relative_or_absolute(cfg.fly_gsplat_dir, clip_dir),
        "extra_meta": cfg.extra_meta,
        "render_seconds": round(time.time() - t0, 1),
    }
    (clip_dir / "clip.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return clip_dir
