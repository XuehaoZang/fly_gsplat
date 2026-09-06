"""Render one recorded fly rollout through the four-camera calibration into a fly_gsplat dataset.

    python render_synth.py --rollout data/rollouts/<clip>.npz --out out/<clip>
        [--calibration ../data/ctrl_009_002/f0200/transforms.json] [--fps 16000] [--max-frames N]
        [--config synth_config.json] [--no-video] [--no-h5-images] [--verify-frames 20]

Writes
    <out>/dataset/f%04d/{images/P<n>CAM<k>.png, transforms.json, init_points.ply, gt_points.ply}
    <out>/dataset/frames.csv, part_palette.json, camera_KRX0.mat, calibration_easyWandData.mat
    <out>/cam1.mp4 .. cam4.mp4        grayscale previews (needs imageio-ffmpeg; skipped otherwise)
    <out>/gt.h5, calib.json, clip.json  full ground truth (see README)

Only frames in which every camera sees the whole fly go into dataset/ (the longest such run,
renumbered from 0; frames.csv maps them back to the clip). Dependencies: numpy, mujoco, h5py, imageio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_CALIBRATION = HERE.parent / "data" / "ctrl_009_002" / "f0200" / "transforms.json"


@dataclass
class SynthConfig:
    """Everything that shapes the images; defaults match the real recordings of ctrl_009_002."""
    arena_side_cm: float = 3.5
    arena_center_cm: tuple = (0.0, 0.0, 0.0)
    recenter: str = "fit"  # fit | bbox | none
    visibility_margin_px: float = 4.0
    fit_search_cm: float = 2.0
    fit_step_cm: float = 0.25
    units_scale: float = 100.0  # calibration metres -> model cm
    body_gray: float = 0.16
    membrane_alpha: float = 0.55
    offsamples: int = 4
    blur_sigma_px: float = 1.0
    noise_sigma: float = 0.0
    white_clip: int = 250
    show_legs: bool = True
    surface_counts: dict = field(default_factory=lambda: {
        "thorax": 800, "head": 600, "antenna": 100, "abdomen": 800, "wing_left": 1000, "wing_right": 1000,
        "haltere_left": 100, "haltere_right": 100, "leg_left": 300, "leg_right": 300})
    visibility_tol_cm: float = 0.02
    init_points_max: int = 2000
    video_fps: int = 25
    fly_gsplat_frames: str = "longest_run"  # longest_run | all
    store_depth: bool = False
    store_uv: bool = False

    @classmethod
    def load(cls, path: Path | None) -> "SynthConfig":
        cfg = cls()
        if path is None:
            return cfg
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name: f for f in fields(cls)}
        unknown = [k for k in data if k not in known]
        if unknown:
            raise SystemExit(f"unknown config keys {unknown}; known: {sorted(known)}")
        for k, v in data.items():
            current = getattr(cfg, k)
            if k == "arena_center_cm":
                v = tuple(float(x) for x in v)
            elif isinstance(current, bool):
                v = bool(v)
            elif isinstance(current, int):
                v = int(v)
            elif isinstance(current, float):
                v = float(v)
            elif isinstance(current, dict):
                v = {str(kk): int(vv) for kk, vv in dict(v).items()}
            elif isinstance(current, str):
                v = str(v)
            setattr(cfg, k, v)
        return cfg


def _portable(path: Path) -> str:
    """A path relative to the fly_gsplat repository root when inside it, else its name."""
    import os

    repo = HERE.parent
    try:
        return os.path.relpath(Path(path).resolve(), repo).replace("\\", "/")
    except ValueError:
        return Path(path).name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rollout", required=True, help="gs_recon rollout .npz (recorded simulator states)")
    p.add_argument("--out", required=True, help="output folder for this clip")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                   help="fly_gsplat transforms.json (or a frame folder / dataset root containing one)")
    p.add_argument("--config", default=None, help="JSON overriding SynthConfig fields")
    p.add_argument("--fps", type=int, default=16000, help="frame rate of the output; 5000 = native, else interpolated")
    p.add_argument("--max-frames", type=int, default=0, help="cap the number of rendered frames (0 = all)")
    p.add_argument("--seed", type=int, default=0, help="seed of the surface sampling and the noise")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-h5-images", action="store_true", help="do not store image stacks in gt.h5")
    p.add_argument("--no-dataset", action="store_true", help="skip the fly_gsplat frame folders")
    p.add_argument("--gt-csv", action="store_true", help="also write gt_points.csv per frame (about 300 kB each)")
    p.add_argument("--verify-frames", type=int, default=20, help="frames to check after rendering (0 = skip)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import numpy as np  # noqa: F401  (import errors surface here, before any work)

    from synthfly.cameras import find_transforms_json, load_fly_gsplat_rig
    from synthfly.fly_gsplat_io import copy_calibration_files
    from synthfly.render import RenderConfig, render_clip
    from synthfly.rollout_io import load_rollout, rollout_summary
    from synthfly.scene import FlyScene, SceneConfig
    from synthfly.surface import sample_surface

    t0 = time.time()
    cfg = SynthConfig.load(Path(args.config) if args.config else None)
    rollout_path = Path(args.rollout)
    if not rollout_path.is_file():
        raise SystemExit(f"rollout not found: {rollout_path}")
    calibration = find_transforms_json(Path(args.calibration))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = load_rollout(rollout_path)
    # metadata must not carry another machine's paths: keep file names, the hash identifies the policy
    rec["checkpoint"] = Path(str(rec["checkpoint"])).name
    summary = rollout_summary(rec)
    print("rollout:", json.dumps(summary))
    cams = load_fly_gsplat_rig(calibration, cfg.units_scale)
    scene = FlyScene(SceneConfig(
        calibrated=cams, arena_side_cm=cfg.arena_side_cm, arena_center=tuple(cfg.arena_center_cm),
        body_gray=cfg.body_gray, membrane_alpha=cfg.membrane_alpha, offsamples=cfg.offsamples,
        show_legs=cfg.show_legs))
    samples = sample_surface(scene, cfg.surface_counts, seed=args.seed)
    print(f"scene: {len(scene.camera_names)} cameras {scene.cfg.width}x{scene.cfg.height}, "
          f"{samples.count} surface samples, calibration {calibration}")

    dataset_dir = None if args.no_dataset else out / "dataset"
    write_video = not args.no_video
    if write_video:
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            print("imageio-ffmpeg not installed: skipping the mp4 previews (pip install imageio-ffmpeg)")
            write_video = False
    rcfg = RenderConfig(
        fps=args.fps, max_frames=args.max_frames, recenter=cfg.recenter,
        arena_side_cm=cfg.arena_side_cm, arena_center=tuple(cfg.arena_center_cm),
        blur_sigma_px=cfg.blur_sigma_px, noise_sigma=cfg.noise_sigma, visibility_tol_cm=cfg.visibility_tol_cm,
        write_video=write_video, video_fps=cfg.video_fps, write_png=False, write_depth=cfg.store_depth,
        store_uv=cfg.store_uv, store_gray_images=not args.no_h5_images, store_part_images=not args.no_h5_images,
        white_clip=cfg.white_clip, fly_gsplat_dir=dataset_dir, fly_gsplat_frames=cfg.fly_gsplat_frames,
        visibility_margin_px=cfg.visibility_margin_px, fit_search_cm=cfg.fit_search_cm, fit_step_cm=cfg.fit_step_cm,
        init_points_max=cfg.init_points_max, gsplat_csv=args.gt_csv, units_scale_out=1.0 / cfg.units_scale,
        photometry_seed=args.seed,
        extra_meta={"rig": "fly_gsplat", "calibration": _portable(calibration), "renderer": "synthfly",
                    "rollout_file": rollout_path.name, "config": asdict(cfg)},
    )
    try:
        render_clip(rec, scene, samples, rcfg, out)
    finally:
        scene.close()
    meta = json.loads((out / "clip.json").read_text(encoding="utf-8"))
    if dataset_dir is not None and meta.get("fly_gsplat_frames_written", 0) > 0:
        copy_calibration_files(calibration, dataset_dir)
    print(f"frames: {meta['n_frames']} rendered, {meta['frames_fully_visible']} fully visible in every camera, "
          f"{meta['fly_gsplat_frames_written']} written to dataset/ ({time.time() - t0:.0f} s)")

    if args.verify_frames > 0 and dataset_dir is not None and meta.get("fly_gsplat_frames_written", 0) > 0:
        from synthfly.verify import main as verify_main

        rc = verify_main([str(dataset_dir), "--reference", str(calibration), "--max-frames", str(args.verify_frames)])
        if rc != 0:
            print("verification reported problems (see above)")
            return rc
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
