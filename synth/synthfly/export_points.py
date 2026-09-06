"""Export the part-labelled surface samples of a synthetic clip as a point PLY.

    python -m synthfly.export_points <clip_dir>/gt.h5 --out points.ply [--frames 0 10 20 | --all]
        [--visible-only] [--stride 5]

Each vertex carries x y z (cm, the clip's world frame, i.e. the calibration
frame; divide by 100 for the metres of the fly_gsplat dataset files), t (s),
part (id per parts/names in the HDF5), gray (0..255 rendered intensity),
nx ny nz (surface normal) and frame (index, must be below n_frames). This is the "one primitive = position + colour +
time + part" ground truth for training a dynamic Gaussian-splatting model; it
is not the INRIA Gaussian PLY of the gs_recon data contract.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

PROPERTIES = [("float", "x"), ("float", "y"), ("float", "z"), ("float", "t"), ("uchar", "part"),
              ("uchar", "gray"), ("float", "nx"), ("float", "ny"), ("float", "nz"), ("int", "frame")]
_FMT = {"float": "f", "uchar": "B", "int": "i"}


def write_points_ply(path: Path, rows: np.ndarray) -> None:
    """rows: structured array with the PROPERTIES fields."""
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header += [f"property {t} {n}" for t, n in PROPERTIES]
    header.append("end_header")
    fmt = "<" + "".join(_FMT[t] for t, _ in PROPERTIES)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for r in rows:
            f.write(struct.pack(fmt, *[r[n].item() if hasattr(r[n], "item") else r[n] for _, n in PROPERTIES]))


def collect(h5_path: Path, frames: list[int], visible_only: bool, stride: int) -> np.ndarray:
    import h5py

    dtype = np.dtype([(n, {"float": np.float32, "uchar": np.uint8, "int": np.int32}[t]) for t, n in PROPERTIES])
    chunks = []
    with h5py.File(h5_path, "r") as h5:
        t_s = h5["frames/t_s"][:]
        part = h5["surface/part_id"][:]
        for f in frames:
            xyz = h5["surface/xyz_cm"][f]
            nrm = h5["surface/normal"][f].astype(np.float32)
            gray = h5["surface/gray"][f]
            keep = np.ones(len(xyz), dtype=bool)
            if visible_only:
                keep &= h5["surface/visible"][f].any(axis=1)
            idx = np.nonzero(keep)[0][::max(1, stride)]
            rows = np.empty(len(idx), dtype=dtype)
            rows["x"], rows["y"], rows["z"] = xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]
            rows["t"] = np.float32(t_s[f])
            rows["part"] = part[idx]
            rows["gray"] = gray[idx]
            rows["nx"], rows["ny"], rows["nz"] = nrm[idx, 0], nrm[idx, 1], nrm[idx, 2]
            rows["frame"] = np.int32(f)
            chunks.append(rows)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("gt", help="path to gt.h5")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", nargs="*", type=int, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--visible-only", action="store_true")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th sample")
    args = p.parse_args(argv)
    import h5py

    with h5py.File(args.gt, "r") as h5:
        n = int(h5.attrs["n_frames"])
    frames = list(range(n)) if args.all else (args.frames if args.frames else [0])
    bad = [f for f in frames if f < 0 or f >= n]
    if bad:
        print(f"frame indices {bad} are out of range; this clip has {n} frames (0 .. {n - 1})", file=sys.stderr)
        return 1
    rows = collect(Path(args.gt), frames, args.visible_only, args.stride)
    write_points_ply(Path(args.out), rows)
    print(f"{len(rows)} points from {len(frames)} frame(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
