"""
preprocessing/hull/viz_hull.py

Build a visual hull from one calibration + one frame, and immediately check
whether it reprojects cleanly back into the four source images. Used to A/B
different calibrations by comparing hull quality.

Reuses (no reimplementation):
  generate_hull.generate_hull()            hull sampling / voting / outlier removal
  generate_hull.visual_hull_vote()         per-camera "point lands in mask" check
  generate_dataset.generate_dataset()      transforms.json + images/ generation
  debug.validate_reprojection.verify_reprojection()   4-up reprojection overlay

Input (choose one):
  --data-dir DIR                    existing dataset (transforms.json + images/)
  --ew FILE --sparse-dir DIR --frame N     build a temporary dataset first

Output (under --out-dir, default preprocessing/hull/outputs/{tag}):
  init_points.ply (or --out-name)
  debug/hull_reprojection.png
  terminal summary: seed triangulation residual, survived point count,
                     per-camera fraction of hull points landing inside that
                     camera's own mask

Example:
  python -m preprocessing.hull.viz_hull \
      --ew /mnt/x/.../good_calib/calibration_easyWandData.mat \
      --sparse-dir /mnt/x/antenna/control/009_25052026/Sparse/Expr_009_mov_002 \
      --frame 10 --out-name init_points_goodcalib.ply
 python -m preprocessing.hull.viz_hull \
      --ew /mnt/x/antenna/control/005_15052025/calibration/calibration_easyWandData.mat \
      --sparse-dir /mnt/x/antenna/control/005_15052025/Sparse/Expr_005_mov_003 \
      --frame 10 --out-name init_points_Expr_005_mov_003.ply
"""

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from generate_dataset import generate_dataset
from generate_hull import generate_hull, visual_hull_vote
from debug.validate_reprojection import verify_reprojection


def stage_from_data_dir(data_dir: Path, out_dir: Path) -> None:
    """Copy transforms.json + images/ into out_dir so generate_hull's outputs
    land under preprocessing/hull/outputs/, not inside the original data_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / "transforms.json", out_dir / "transforms.json")
    dst_images = out_dir / "images"
    if dst_images.exists():
        shutil.rmtree(dst_images)
    shutil.copytree(data_dir / "images", dst_images)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Existing dataset dir (transforms.json + images/)")
    ap.add_argument("--ew", type=Path, default=None,
                     help="calibration_easyWandData.mat (used with --sparse-dir/--frame)")
    ap.add_argument("--sparse-dir", type=Path, default=None,
                     help="Directory containing Camera*_sparse.mat")
    ap.add_argument("--frame", type=int, default=None, help="Target frame index")
    ap.add_argument("--tag", default=None, help="Output subfolder name (default: derived)")
    ap.add_argument("--out-name", default="init_points.ply", help="Hull ply filename")
    ap.add_argument("--n-samples", type=int, default=10_000, help="Sphere sample count")
    ap.add_argument("--out-root", type=Path,
                     default=Path(__file__).resolve().parent / "outputs",
                     help="Output root (default: preprocessing/hull/outputs)")
    ap.add_argument("--no-viser", action="store_true", help="Skip the Viser hull view")
    args = ap.parse_args()

    if args.data_dir is not None:
        tag = args.tag or args.data_dir.name
        out_dir = args.out_root / tag
        stage_from_data_dir(args.data_dir, out_dir)
    else:
        if args.ew is None or args.sparse_dir is None or args.frame is None:
            raise SystemExit("Provide either --data-dir, or --ew + --sparse-dir + --frame")
        tag = args.tag or args.ew.parent.name
        out_dir = args.out_root / f"{tag}_f{args.frame:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # white_bg=True matches generate_hull's binarize_mask(dark_bg=False)
        # convention, same as production dataset generation (see schedule.py etc.)
        generate_dataset(str(out_dir), str(args.sparse_dir), target_frame=args.frame,
                          if_crop=False, white_bg=True, if_mask=False,
                          calib_dir=str(args.ew.parent))

    stats = generate_hull(str(out_dir), if_viser=not args.no_viser,
                           n_samples=args.n_samples, out_name=args.out_name)

    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    verify_reprojection(data_dir=str(out_dir), ply_path=str(out_dir / args.out_name),
                         save_path=debug_dir / "hull_reprojection.png")

    print(f"\n=== Hull summary ({tag}) ===")
    print(f"seed              : {stats['seed']}")
    print(f"tri residual       : {stats['residual_m'] * 1000:.3f} mm")
    print(f"survived points    : {stats['n_survived']}")

    if stats["n_survived"] > 0:
        print("per-camera fraction of hull points landing inside its own mask:")
        for cam, mask in zip(stats["cameras"], stats["masks"]):
            inside = visual_hull_vote(stats["points"], [cam], [mask])
            frac = float(inside.mean())
            print(f"  Cam {cam.cam_idx}: {frac * 100:5.1f}%  ({inside.sum()}/{len(inside)})")

    print(f"\n[Saved] {out_dir / args.out_name}")
    print(f"[Saved] {debug_dir / 'hull_reprojection.png'}")


if __name__ == "__main__":
    main()
