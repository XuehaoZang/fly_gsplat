"""Acceptance checks for a synthetic dataset in the fly_gsplat layout.

    python verify_synth.py <dataset_dir>
        [--reference <real transforms.json>] [--max-frames N] [--min-hit 0.9]

Per-camera summary line: `hit` is the fraction of ground-truth points that
project onto non-white pixels (must reach --min-hit), `row-flipped` the same
with the image rows mirrored (must stay clearly below `hit`), `dark px` the
number of pixels below 128, `background 255` the fraction of pixels exactly
255. With --reference the camera entries are compared field by field.

Per frame folder it checks that
- images are 1280 x 800 (or the calibration's size), 8-bit, single-channel,
  with the background exactly 255 and a dark fly present in every camera;
- transforms.json has the expected keys and, if a reference is given, camera
  entries identical to the real calibration (only file_path differs);
- the ground-truth points (gt_points.ply, or init_points.ply) projected with
  K [R | t] recovered from transforms.json land on non-white pixels in every
  camera, and do not when the image row is flipped.
Exit status 1 if any frame fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .fly_gsplat_io import read_open3d_ply

FLIP = np.diag([1.0, -1.0, -1.0])
FRAME_KEYS = ("file_path", "fl_x", "fl_y", "cx", "cy", "w", "h", "transform_matrix")


def camera_from_frame(fr: dict):
    M = np.asarray(fr["transform_matrix"], dtype=np.float64)
    R_w2c = (M[:3, :3] @ FLIP).T
    X0 = M[:3, 3]
    return R_w2c, X0, float(fr["fl_x"]), float(fr["fl_y"]), float(fr["cx"]), float(fr["cy"]), int(fr["w"]), int(fr["h"])


def hit_fraction(points_m, fr, img, flip_rows=False):
    R, X0, fx, fy, cx, cy, W, H = camera_from_frame(fr)
    xc = (R @ (points_m - X0).T).T
    z = xc[:, 2]
    ok = z > 1e-6
    u = fx * xc[ok, 0] / z[ok] + cx
    v = fy * xc[ok, 1] / z[ok] + cy
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inside.sum() == 0:
        return 0.0, 0
    ui = np.floor(u[inside]).astype(int)
    vi = np.floor(v[inside]).astype(int)
    if flip_rows:
        vi = H - 1 - vi
    return float((img[vi, ui] < 255).mean()), int(inside.sum())


def check_frame(frame_dir: Path, reference: list[dict] | None, min_hit: float) -> list[str]:
    import imageio.v2 as imageio

    problems = []
    tj = frame_dir / "transforms.json"
    if not tj.is_file():
        return [f"{frame_dir.name}: missing transforms.json"]
    payload = json.loads(tj.read_text(encoding="utf-8"))
    if payload.get("camera_model") != "OPENCV" or payload.get("ply_file_path") != "init_points.ply":
        problems.append(f"{frame_dir.name}: top-level keys differ from fly_gsplat's")
    frames = payload.get("frames", [])
    if reference is not None:
        if len(frames) != len(reference):
            problems.append(f"{frame_dir.name}: {len(frames)} cameras, reference has {len(reference)}")
        for k, (fr, ref) in enumerate(zip(frames, reference)):
            for key in FRAME_KEYS:
                if key == "file_path":
                    continue
                a, b = np.asarray(fr.get(key), dtype=np.float64), np.asarray(ref.get(key), dtype=np.float64)
                if a.shape != b.shape or np.abs(a - b).max() > 1e-9:
                    problems.append(f"{frame_dir.name}: camera {k + 1} field {key} differs from the reference")
    pts_file = frame_dir / "gt_points.ply" if (frame_dir / "gt_points.ply").is_file() else frame_dir / "init_points.ply"
    if not pts_file.is_file():
        problems.append(f"{frame_dir.name}: no gt_points.ply or init_points.ply")
        return problems
    points, _ = read_open3d_ply(pts_file)
    for k, fr in enumerate(frames):
        if tuple(fr.keys()) != FRAME_KEYS and set(fr.keys()) != set(FRAME_KEYS):
            problems.append(f"{frame_dir.name}: camera {k + 1} has keys {sorted(fr.keys())}")
        img_path = frame_dir / fr["file_path"]
        if not img_path.is_file():
            problems.append(f"{frame_dir.name}: missing {fr['file_path']}")
            continue
        img = imageio.imread(img_path)
        if img.ndim != 2 or img.dtype != np.uint8:
            problems.append(f"{frame_dir.name}: {img_path.name} is not an 8-bit single-channel image ({img.shape}, {img.dtype})")
            continue
        if img.shape != (int(fr["h"]), int(fr["w"])):
            problems.append(f"{frame_dir.name}: {img_path.name} is {img.shape}, transforms says {(fr['h'], fr['w'])}")
        white = float((img == 255).mean())
        dark = int((img < 128).sum())
        if white < 0.99:
            problems.append(f"{frame_dir.name}: {img_path.name} background is not clean (only {white:.3%} pixels are 255)")
        if dark < 100:
            problems.append(f"{frame_dir.name}: {img_path.name} has almost no dark pixels ({dark})")
        hit, n = hit_fraction(points, fr, img)
        hit_flipped, _ = hit_fraction(points, fr, img, flip_rows=True)
        if n == 0 or hit < min_hit:
            problems.append(f"{frame_dir.name}: camera {k + 1} projection hit {hit:.3f} (< {min_hit}) on {n} points")
        # a fly centred on the image midline also overlaps its own row-flipped silhouette, so the
        # flipped projection is only wrong if it scores as well as the stored-row convention
        if hit_flipped >= hit - 0.1:
            problems.append(f"{frame_dir.name}: camera {k + 1} flipped-row hit {hit_flipped:.3f} vs {hit:.3f}; "
                            "row convention wrong?")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset_dir")
    p.add_argument("--reference", default=None, help="a real transforms.json to compare camera entries against")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--min-hit", type=float, default=0.9)
    args = p.parse_args(argv)
    root = Path(args.dataset_dir)
    frames = sorted(d for d in root.glob("f*") if d.is_dir())
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    if not frames:
        print(f"no frame folders under {root}")
        return 1
    reference = None
    if args.reference:
        reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))["frames"]
        print(f"comparing camera entries against {args.reference} ({len(reference)} cameras)")
    failures = 0
    for fd in frames:
        problems = check_frame(fd, reference, args.min_hit)
        if problems:
            failures += 1
            for pr in problems:
                print("FAIL", pr)
    # summary of the projection agreement on the first frame
    first = frames[0]
    payload = json.loads((first / "transforms.json").read_text(encoding="utf-8"))
    pts_file = first / "gt_points.ply" if (first / "gt_points.ply").is_file() else first / "init_points.ply"
    if pts_file.is_file():
        import imageio.v2 as imageio

        points, _ = read_open3d_ply(pts_file)
        for k, fr in enumerate(payload["frames"]):
            img = imageio.imread(first / fr["file_path"])
            hit, n = hit_fraction(points, fr, img)
            flipped, _ = hit_fraction(points, fr, img, flip_rows=True)
            print(f"{first.name} camera {k + 1}: {n} points in view, hit {hit:.3f}, row-flipped {flipped:.3f}, "
                  f"dark px {(img < 128).sum()}, background 255: {(img == 255).mean():.4%}")
    if reference is not None and not failures:
        print("camera entries identical to the reference in every checked frame")
    print(f"checked {len(frames)} frame folders, {failures} with problems "
          f"(hit >= {args.min_hit} required; row-flipped must be clearly lower)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
