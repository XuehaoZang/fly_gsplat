"""Export a finished gsplat run into a viz bundle the viewer can load.

    python -m eval.viz.export --run <recon_dir> [<recon_dir> ...] [--out outputs/viz]

Input : <run>/<param_set>/f####/splatfacto/<ts>/gaussian_features_f####{_labeled,_marked,}.csv
        <run>/<param_set>/f####/splatfacto/<ts>/{splat.ply, dataparser_transforms.json}  (orientation)
        <run>/kinematics/angles.csv, else kinematics_<param_set>.csv   (optional; angles + span/normal vectors)
        frame dir of the training input (transforms.json; optional; cameras for the "view from camN" mode)
Output: <out>/<dataset>/gsplat/{meta.json, points.bin, angles.json, cameras.json}

points.bin is float32, row-major, one row per Gaussian, all frames concatenated;
meta.json gives the per-frame row offsets and the column order.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial.transform import Rotation

FIELDS = ["x", "y", "z", "opacity", "if_keep", "label",
          "scale_0", "scale_1", "scale_2", "qx", "qy", "qz", "qw", "R", "G", "B"]
LABELS = ["unlabeled", "body", "wing_L", "wing_R"]
ANGLE_COLS = ["yaw", "pitch", "roll", "phi_L", "theta_L", "eta_L", "phi_R", "theta_R", "eta_R",
              "span_L_x", "span_L_y", "span_L_z", "span_R_x", "span_R_y", "span_R_z",
              "sp_normal_x", "sp_normal_y", "sp_normal_z"]


def dataset_name(run):
    run = os.path.normpath(run)
    base = os.path.basename(run)
    if base == "recon":  # data/synth/<TRIAL>/recon
        return "synth_" + os.path.basename(os.path.dirname(run))
    return base


def frame_csv(frame_dir):
    """Most-processed CSV available for a frame (labeled > marked > raw)."""
    for suffix in ("_labeled", "_marked", ""):
        hits = sorted(glob.glob(os.path.join(frame_dir, "splatfacto", "*",
                                             f"gaussian_features_f*{suffix}.csv")))
        if hits:
            return hits[-1]
    return None


def lab_quats(csv_path, xyz):
    """Per-Gaussian rotation (xyzw) in the lab frame, from splat.ply rot_* and dataparser_transforms.

    nerfstudio: p_ns = s * (T_R p_lab + T_t)  =>  R_lab = T_R^T R_ns.
    The CSV rows are the ply rows in order; checked here via the recomputed centres.
    """
    d = os.path.dirname(csv_path)
    v = PlyData.read(os.path.join(d, "splat.ply"))["vertex"]
    with open(os.path.join(d, "dataparser_transforms.json")) as f:
        t = json.load(f)
    T, s = np.array(t["transform"]), t["scale"]
    p = np.stack([v["x"], v["y"], v["z"]], 1) / s - T[:, 3]
    err = np.abs(p @ T[:, :3] - xyz).max() if len(p) == len(xyz) else np.inf
    if err > 1e-6:
        raise SystemExit(f"{csv_path}: CSV rows do not match splat.ply (max err {err})")
    q_ns = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1)  # wxyz -> xyzw
    return (Rotation.from_matrix(T[:, :3].T) * Rotation.from_quat(q_ns)).as_quat()


def load_frame(path):
    df = pd.read_csv(path)
    xyz = df[["x", "y", "z"]].values
    keep = df["if_keep"].astype(bool) if "if_keep" in df else np.ones(len(df), bool)
    if "part_label" in df:
        label = df["part_label"].map({n: i for i, n in enumerate(LABELS)}).fillna(0)
    else:
        label = np.zeros(len(df))
    scales = df[["scale_phys_0", "scale_phys_1", "scale_phys_2"]].values
    return np.column_stack([xyz, df["opacity"], keep, label, scales, lab_quats(path, xyz),
                            df[["R", "G", "B"]].values]).astype(np.float32)


def frames_root(run):
    """Directory holding the training-input f####/transforms.json for this run, or None."""
    run = os.path.abspath(run)
    if os.path.basename(run) == "recon":  # synth: <trial>/recon next to <trial>/dataset
        return os.path.join(os.path.dirname(run), "dataset")
    meta_path = os.path.join(run, "sweep_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        base = json.load(f).get("base_name", "")
    # base_name is relative to the repo root the sweep ran from: try the ancestors of the run
    d = run
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, base)):
            return os.path.join(d, base)
        d = os.path.dirname(d)
    return None


def export_cameras(run, frames, out):
    root = frames_root(run)
    hits = [os.path.join(root, f"f{f:04d}", "transforms.json") for f in frames] if root else []
    hits = [h for h in hits if os.path.exists(h)]
    if not hits:
        return None
    with open(hits[0]) as f:
        t = json.load(f)
    frame0 = int(os.path.basename(os.path.dirname(hits[0]))[1:])
    cams = []
    for fr in t["frames"]:
        # image path relative to <frames_root>/f####/, with the frame number as "{frame}"
        image = fr["file_path"].replace(f"P{frame0}CAM", "P{frame}CAM")
        if not os.path.exists(os.path.join(os.path.dirname(hits[0]), image.replace("{frame}", str(frame0)))):
            image = None
        cams.append(dict(name=os.path.basename(fr["file_path"]).split("CAM")[-1].split(".")[0],
                         fl_x=fr["fl_x"], fl_y=fr["fl_y"], cx=fr["cx"], cy=fr["cy"], w=fr["w"], h=fr["h"],
                         c2w=fr["transform_matrix"],  # OpenGL c2w, lab frame (metres)
                         image=image))
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(dict(source=os.path.abspath(hits[0]), frames_root=os.path.abspath(root), cameras=cams), f)
    return hits[0]


def export_run(run, param_set, out_root):
    frame_dirs = sorted(glob.glob(os.path.join(run, param_set, "f[0-9][0-9][0-9][0-9]")))
    frames, offsets, blocks, sources = [], [0], [], set()
    for fd in frame_dirs:
        path = frame_csv(fd)
        if path is None:
            continue
        arr = load_frame(path)
        frames.append(int(os.path.basename(fd)[1:]))
        blocks.append(arr)
        offsets.append(offsets[-1] + len(arr))
        sources.add(path.rsplit("_", 1)[-1] if path.endswith(("_labeled.csv", "_marked.csv")) else "raw")
    if not frames:
        raise SystemExit(f"no frames under {run}/{param_set}")

    out = os.path.join(out_root, dataset_name(run), "gsplat")
    os.makedirs(out, exist_ok=True)
    np.concatenate(blocks).tofile(os.path.join(out, "points.bin"))

    angles_csv = os.path.join(run, "kinematics", "angles.csv")  # gsplat-core kinematics.run
    if not os.path.exists(angles_csv):
        angles_csv = os.path.join(run, "kinematics", f"kinematics_{param_set}.csv")  # old pipeline
    has_angles = os.path.exists(angles_csv)
    if has_angles:
        a = pd.read_csv(angles_csv)
        cols = {c: [None if pd.isna(v) else float(v) for v in a[c]] for c in ANGLE_COLS if c in a}
        cols["frame_id"] = a["frame_id"].astype(int).tolist()
        cols["status"] = a["status"].fillna("").astype(str).tolist()
        with open(os.path.join(out, "angles.json"), "w") as f:
            json.dump(cols, f)

    cam_src = export_cameras(run, frames, out)
    meta = dict(method="gsplat", point_type="gaussian", units="m", run=os.path.abspath(run),
                param_set=param_set, frames=frames, offsets=offsets, fields=FIELDS,
                labels=LABELS, csv_stage=sorted(sources), has_angles=has_angles,
                angles_source=os.path.abspath(angles_csv) if has_angles else None,
                has_cameras=cam_src is not None)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"{dataset_name(run)}: {len(frames)} frames, {offsets[-1]} gaussians, "
          f"stage={sorted(sources)}, angles={'yes' if has_angles else 'no'}, "
          f"cameras={'yes' if cam_src else 'no'} -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True, help="recon run dir(s)")
    p.add_argument("--param-set", default="ratio3_sh0_dense")
    p.add_argument("--out", default="outputs/viz")
    args = p.parse_args()
    for run in args.run:
        export_run(run, args.param_set, args.out)


if __name__ == "__main__":
    main()
