"""
preprocessing/calib/viz_calib.py

Quick A/B quality check for one or more `calibration_easyWandData.mat` files,
against a shared set of `Camera*_sparse.mat` frames — without running the full
`generate_dataset.py` pipeline.

Only the production decoding path is exercised:
  CameraConfig.easywand_dlt(ew, i)   (RQ decomposition of the DLT coefs)
No dependency on camera_KRX0.mat / the "roni"/"native" comparison methods in
debug/validate_calib.py.

For each easyWandData.mat and each requested frame:
  1. decode (K, R_w2c, X0) per camera via CameraConfig.easywand_dlt
  2. reconstruct a grayscale frame from the sparse mats, get a 2D detection
     centroid per camera (utils.calib.mask_centroid)
  3. backproj + triangulate the centroids -> 3D target + triangulation residual
  4. proj the target back into every camera -> per-camera reprojection error (px)
  5. check_ortho on every R_w2c

Outputs (under --out-dir, one subfolder per ew tag):
  outputs/{tag}/frame_{f:04d}_reproj.png   4-up overlay: x = detected centroid,
                                            o = reprojected target
  outputs/comparison.json / comparison.csv  flat per ew x frame x camera table
  terminal summary table + a Viser rig view per ew file (unless --no-viser)

Example:
  python -m preprocessing.calib.viz_calib \
      --ew /mnt/x/.../bad_calib/calibration_easyWandData.mat \
           /mnt/x/.../good_calib/calibration_easyWandData.mat \
      --sparse-dir /mnt/x/antenna/control/009_25052026/Sparse/Expr_009_mov_002 \
      --frames 0 10 50 100 200
  python -m preprocessing.calib.viz_calib \
      --ew /mnt/x/antenna/control/009_25052026/calibration/calibration_easyWandData.mat \
           /mnt/x/antenna/control/005_15052025/calibration/calibration_easyWandData.mat \
      --sparse-dir /mnt/x/antenna/control/005_15052025/Sparse/Expr_005_mov_003 \
      --frames 0 10 50 100 200

"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.camera import CameraConfig
from utils.calib import proj, backproj, triangulate, check_ortho, mask_centroid
from utils.dataset import reconstruct_frame_image, count_sparse_frames
from utils.viz import start_viser, add_camera_axes, stop_viser


# --------------------------------------------------------------------- setup --
def load_cameras(ew_path: Path) -> list:
    mat = sio.loadmat(str(ew_path), struct_as_record=False, squeeze_me=True)
    ew = mat["easyWandData"]
    n_cams = int(ew.nCams)
    return [CameraConfig.easywand_dlt(ew, i) for i in range(n_cams)]


def make_tags(ew_paths: list, tags_arg) -> list:
    if tags_arg:
        if len(tags_arg) != len(ew_paths):
            raise ValueError("--tags must have the same length as --ew")
        return list(tags_arg)

    candidates = [p.parent.name for p in ew_paths]
    if len(set(candidates)) == len(candidates):
        return candidates

    candidates = [f"{p.parent.name}_{p.stem}" for p in ew_paths]
    if len(set(candidates)) == len(candidates):
        return candidates

    return [f"{c}_{i}" for i, c in enumerate(candidates)]


def auto_frames(sparse_files: list, n_auto: int) -> list:
    total = count_sparse_frames(sparse_files[0])
    n = min(n_auto, total)
    idx = np.linspace(0, total - 1, n).round().astype(int)
    return sorted(set(int(i) for i in idx))


# ------------------------------------------------------------------- 4-up viz --
def plot_frame_overlay(cameras: list, images: list, detected: list,
                        reprojected: list, out_path: Path) -> None:
    n = len(cameras)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, cam in enumerate(cameras):
        ax = axes[i]
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        cu, cv = detected[i]
        ru, rv = reprojected[i]
        ax.plot(cu, cv, marker="x", color="red", markersize=10, mew=2, label="detected")
        ax.plot(ru, rv, marker="o", markerfacecolor="none", markeredgecolor="lime",
                 markersize=14, mew=2, label="reprojected")
        ax.set_title(f"Cam {cam.cam_idx}")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------- run --
def evaluate_one_ew(ew_path: Path, sparse_files: list, frames: list,
                     out_dir: Path, white_bg: bool) -> list:
    """Returns a list of row dicts: one per (frame, camera)."""
    cameras = load_cameras(ew_path)
    n_cams = len(cameras)
    if len(sparse_files) != n_cams:
        print(f"[Warning] {ew_path}: nCams={n_cams} but found {len(sparse_files)} sparse files")

    print(f"\n=== {ew_path} ===  ({n_cams} cams)")
    for cam in cameras:
        det, ortho, ok = check_ortho(cam.R_w2c)
        flag = "OK " if ok else "BAD"
        print(f"  Cam{cam.cam_idx} ortho [{flag}] det={det:+.4f} residual={ortho:.1e}")

    rows = []
    for frame in frames:
        images, detected = [], []
        for cam, sfile in zip(cameras, sparse_files):
            im = reconstruct_frame_image(sfile, frame, cam.w, cam.h, white_bg=white_bg)
            u, v = mask_centroid(im)
            if np.isnan(u) or np.isnan(v):
                u, v = cam.cx, cam.cy
                print(f"[Warning] {ew_path.parent.name} frame {frame} cam {cam.cam_idx}: "
                      f"empty mask, using principal point")
            images.append(im)
            detected.append((u, v))

        rays = [(cam.X0, backproj(cam.K, cam.R_w2c, u, v))
                for cam, (u, v) in zip(cameras, detected)]
        target, tri_res = triangulate(rays)

        reprojected, errs = [], []
        for cam, (u, v) in zip(cameras, detected):
            ru, rv, depth = proj(cam.K, cam.R_w2c, cam.X0, target)
            err = float(np.hypot(ru - u, rv - v))
            reprojected.append((ru, rv))
            errs.append(err)

            det, ortho, ok = check_ortho(cam.R_w2c)
            rows.append({
                "frame": frame, "cam_idx": cam.cam_idx,
                "reproj_err_px": err, "depth_m": float(depth),
                "tri_residual_mm": float(tri_res * 1000),
                "ortho_det": float(det), "ortho_norm": float(ortho), "ortho_ok": bool(ok),
            })

        print(f"  frame {frame:5d}  tri_res={tri_res*1000:6.3f}mm  "
              f"reproj(px)=[{', '.join(f'{e:5.2f}' for e in errs)}]  "
              f"mean={np.mean(errs):5.2f}")

        plot_frame_overlay(cameras, images, detected, reprojected,
                            out_dir / f"frame_{frame:04d}_reproj.png")

    errs_all = [r["reproj_err_px"] for r in rows]
    print(f"  -> overall reproj px: mean={np.mean(errs_all):.2f} std={np.std(errs_all):.2f}")
    return rows, cameras


def write_results(all_rows: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    csv_path = out_dir / "comparison.csv"

    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    fieldnames = ["ew_tag", "ew_path", "frame", "cam_idx", "reproj_err_px", "depth_m",
                  "tri_residual_mm", "ortho_det", "ortho_norm", "ortho_ok"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {csv_path}")


def print_summary(all_rows: list) -> None:
    tags = sorted(set(r["ew_tag"] for r in all_rows))
    print("\n=== Summary (reprojection error, px) ===")
    for tag in tags:
        errs = [r["reproj_err_px"] for r in all_rows if r["ew_tag"] == tag]
        tri = [r["tri_residual_mm"] for r in all_rows if r["ew_tag"] == tag]
        print(f"  {tag:20s}  reproj mean={np.mean(errs):6.3f} std={np.std(errs):6.3f} px   "
              f"tri_res mean={np.mean(tri):6.3f} std={np.std(tri):6.3f} mm")


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ew", nargs="+", required=True, type=Path,
                     help="One or more calibration_easyWandData.mat paths")
    ap.add_argument("--sparse-dir", required=True, type=Path,
                     help="Directory containing Camera*_sparse.mat for this video")
    ap.add_argument("--frames", nargs="+", type=int, default=None,
                     help="Frame indices to evaluate (default: auto-sample --n-auto-frames)")
    ap.add_argument("--n-auto-frames", type=int, default=8,
                     help="Number of evenly-spaced frames to sample if --frames is not given")
    ap.add_argument("--tags", nargs="+", default=None,
                     help="Labels for each --ew entry (default: derived from parent dir name)")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parent / "outputs",
                     help="Output root (default: preprocessing/calib/outputs)")
    ap.add_argument("--white-bg", action="store_true",
                     help="Reconstruct frames with white background (default: black, "
                          "required for mask_centroid's gray>0 convention)")
    ap.add_argument("--no-viser", action="store_true", help="Skip the Viser rig view")
    ap.add_argument("--viser-combine", action="store_true",
                     help="Overlay all ew rigs in one Viser scene instead of popping up one at a time")
    args = ap.parse_args()

    sparse_files = sorted(args.sparse_dir.glob("Camera*_sparse.mat"))
    if not sparse_files:
        raise SystemExit(f"No Camera*_sparse.mat found in {args.sparse_dir}")

    frames = args.frames if args.frames else auto_frames(sparse_files, args.n_auto_frames)
    print(f"Evaluating {len(frames)} frames: {frames}")

    tags = make_tags(args.ew, args.tags)

    all_rows = []
    ew_cameras = {}
    for ew_path, tag in zip(args.ew, tags):
        out_dir = args.out_dir / tag
        rows, cameras = evaluate_one_ew(ew_path, sparse_files, frames, out_dir, args.white_bg)
        for r in rows:
            r["ew_tag"] = tag
            r["ew_path"] = str(ew_path)
        all_rows.extend(rows)
        ew_cameras[tag] = cameras

    write_results(all_rows, args.out_dir)
    print_summary(all_rows)

    if not args.no_viser:
        if args.viser_combine:
            server = start_viser()
            for tag, cameras in ew_cameras.items():
                tagged = []
                for cam in cameras:
                    tagged_cam = CameraConfig(cam_idx=f"{tag}_{cam.cam_idx}", K=cam.K,
                                               R_w2c=cam.R_w2c, X0=cam.X0, w=cam.w, h=cam.h)
                    tagged.append(tagged_cam)
                add_camera_axes(server, tagged)
            stop_viser(server)
        else:
            for tag, cameras in ew_cameras.items():
                print(f"\n[Viser] {tag} rig — click Continue in the browser to move to the next one")
                server = start_viser()
                add_camera_axes(server, cameras)
                stop_viser(server)


if __name__ == "__main__":
    main()
