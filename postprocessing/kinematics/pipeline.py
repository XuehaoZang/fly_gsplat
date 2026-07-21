"""S5 pipeline: labeled per-point CSVs -> per-frame kinematics table, T4.

Implements calc_kinematics.md §0 (conventions), §1 (input/output contract),
and §6 (T4-adopted definitions) end to end, chaining S2 (`body_frame.py`) ->
S3 (`wing_angles.py`) -> S4a (`chord.py`) per frame. No multi-frame logic
lives here (no unwrap, no smoothing, no stroke-plane bootstrap across
frames) -- S5 is stateless: each frame's output row depends only on that
frame's own CSV, per §2/§4's "single frame" scope.

`estimate_frame` is the single-frame entry point (pure function of one
already-loaded `pd.DataFrame`); `run_dataset` is the I/O layer (globbing,
loading, writing outputs) built on top of it. A frame is never dropped: any
per-stage failure (missing/too-small part, RANSAC or PCA degeneracy) is
caught, leaves the affected output fields NaN, and is recorded in a
`status` string instead of raising -- see `_estimate_frame_impl`.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import body_frame as bf
from . import chord as ch
from . import io_schema
from . import wing_angles as wa

_FRAME_DIR_RE = re.compile(r"^f(\d+)$")
"""Matches the per-frame directory name (`f<NNNN>`) under a dataset root, §1."""

DEBUG_KEYS: tuple[str, ...] = (
    "x_body", "y_body", "z_body", "n_sp",
    "hinge_L", "hinge_R", "body_cm",
    "le_dir_L", "le_dir_R",
    "chord_L", "chord_R",
    "per_bin_chords_L", "per_bin_chords_R",
)
"""Per-frame debug-sidecar keys (task spec): S2 body frame + S3/S4a per-wing
intermediates, for later viz/QA -- not part of the main output CSV schema
(`io_schema.OUTPUT_COLUMNS`)."""


@dataclass
class PipelineConfig:
    """S5 run parameters. Defaults match the values already locked by S2-S4a
    (§0, §2) except `min_points`/`write_debug`/`output_dir`, which are new to
    S5 (per-part point-count guard, debug-sidecar toggle, output location).
    """

    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    stroke_plane_pitch_deg: float = 45.0
    stroke_plane_normal: np.ndarray | tuple[float, float, float] | None = None
    root_mode: str = "root"
    min_points: int = 10
    """Minimum point count for *each* of `body`/`wing_L`/`wing_R` (post
    `apply_if_keep` filtering) before any fitting is attempted; below this,
    the frame fails fast with a `"<part>:too_few_points"` status rather than
    hitting a PCA/RANSAC exception with a less legible message."""
    write_debug: bool = True
    output_dir: str | Path | None = None
    """Where `run_dataset` writes its outputs. `None` (default) writes into
    `dataset_root` itself."""


def _empty_debug() -> dict:
    return {k: None for k in DEBUG_KEYS}


def _all_finite(*values) -> bool:
    return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)


# ---------------------------------------------------------------------------
# Single-frame estimator
# ---------------------------------------------------------------------------


def _estimate_frame_impl(df: pd.DataFrame, frame_id: int, config: PipelineConfig) -> tuple[dict, dict]:
    """Full single-frame pipeline: parts -> S2 body frame -> S3/S4a per wing.

    Returns `(row, debug)`: `row` matches `io_schema.OUTPUT_COLUMNS` plus
    `status`; `debug` has `DEBUG_KEYS`, each `None` for any stage that didn't
    run or failed. Never raises -- every stage is caught and turned into a
    `status` string instead (see module docstring).
    """
    row = io_schema.empty_output_row(frame_id).to_dict()
    debug = _empty_debug()

    try:
        body_xyz = io_schema.get_part(df, "body", apply_if_keep=True)
        wingL_xyz = io_schema.get_part(df, "wing_L", apply_if_keep=True)
        wingR_xyz = io_schema.get_part(df, "wing_R", apply_if_keep=True)
    except Exception as e:  # noqa: BLE001
        row["status"] = f"parts:{e}"
        return row, debug

    for label, xyz in (("body", body_xyz), ("wing_L", wingL_xyz), ("wing_R", wingR_xyz)):
        if xyz.shape[0] < config.min_points:
            row["status"] = f"{label}:too_few_points"
            return row, debug

    try:
        frame = bf.estimate_body_frame(
            body_xyz, wingL_xyz, wingR_xyz,
            up=config.up,
            stroke_plane_pitch_deg=config.stroke_plane_pitch_deg,
            stroke_plane_normal=config.stroke_plane_normal,
            root_mode=config.root_mode,
        )
        if not _all_finite(frame.x_body, frame.y_body, frame.z_body, frame.n_sp,
                            frame.yaw, frame.pitch, frame.roll):
            raise ValueError("non-finite result (degenerate hinge/PCA geometry)")
    except Exception as e:  # noqa: BLE001
        row["status"] = f"body:{e}"
        return row, debug

    row["yaw"], row["pitch"], row["roll"] = frame.yaw, frame.pitch, frame.roll
    row["sp_normal_x"], row["sp_normal_y"], row["sp_normal_z"] = (
        float(frame.n_sp[0]), float(frame.n_sp[1]), float(frame.n_sp[2])
    )
    debug.update(
        x_body=frame.x_body, y_body=frame.y_body, z_body=frame.z_body, n_sp=frame.n_sp,
        hinge_L=frame.hinge_L, hinge_R=frame.hinge_R, body_cm=frame.body_cm,
    )

    wing_statuses = []
    for side, wing_xyz, suffix in (("wing_L", wingL_xyz, "L"), ("wing_R", wingR_xyz, "R")):
        try:
            sweep = wa.stroke_deviation(wing_xyz, frame, side)
            chord_result = ch.estimate_chord(wing_xyz, frame, side, leading_edge=sweep.leading_edge)
            if not _all_finite(sweep.phi, sweep.theta, chord_result.eta, chord_result.chord_conf):
                raise ValueError("non-finite result")
        except Exception as e:  # noqa: BLE001
            wing_statuses.append(f"{side}:{e}")
            continue

        row[f"phi_{suffix}"] = sweep.phi
        row[f"theta_{suffix}"] = sweep.theta
        row[f"eta_{suffix}"] = chord_result.eta
        row[f"chord_conf_{suffix}"] = chord_result.chord_conf
        debug[f"le_dir_{suffix}"] = sweep.leading_edge.le_dir
        debug[f"chord_{suffix}"] = chord_result.chord
        debug[f"per_bin_chords_{suffix}"] = chord_result.per_bin_chords

    row["status"] = "ok" if not wing_statuses else ";".join(wing_statuses)
    return row, debug


def estimate_frame(df: pd.DataFrame, frame_id: int, config: PipelineConfig | None = None) -> dict:
    """One frame's kinematics row (§1 output schema + `status`).

    `df` is an already-loaded per-point table (`io_schema.load_frame`).
    Body/wing points are extracted with `apply_if_keep=True` (§1: T2-rejected
    points are always dropped before any fitting). See module docstring for
    failure handling.
    """
    config = config if config is not None else PipelineConfig()
    row, _debug = _estimate_frame_impl(df, frame_id, config)
    return row


# ---------------------------------------------------------------------------
# Batch / I/O layer
# ---------------------------------------------------------------------------


def _discover_frame_files(dataset_root: Path) -> list[tuple[int, str, Path]]:
    """Find `*_marked.csv` files under `dataset_root/f<NNNN>/splatfacto-checkpoint/<ts>/`.

    Returns `(frame_id, frame_id_str, csv_path)` triples sorted by
    `frame_id`. `frame_id_str` is the zero-padded directory name (e.g.
    `"0090"`) -- kept alongside the parsed int per the task spec, though only
    `frame_id` (int) reaches the output row/CSV.
    """
    out = []
    for csv_path in dataset_root.glob("f*/splatfacto-checkpoint/*/*_marked.csv"):
        rel = csv_path.relative_to(dataset_root)
        m = _FRAME_DIR_RE.match(rel.parts[0])
        if not m:
            continue
        out.append((int(m.group(1)), m.group(1), csv_path))
    out.sort(key=lambda t: t[0])
    return out


def run_dataset(dataset_root: str | Path, config: PipelineConfig | None = None) -> pd.DataFrame:
    """Discover, load, and estimate kinematics for every frame under `dataset_root`.

    Writes `kinematics_<dataset_name>.csv` (`dataset_name` = `dataset_root`'s
    own directory name) into `config.output_dir` (default: `dataset_root`
    itself), plus a pickled debug sidecar
    `kinematics_<dataset_name>_debug.pkl` keyed by `frame_id` (each value a
    `DEBUG_KEYS` dict) unless `config.write_debug` is `False`. Returns the
    same table as a `DataFrame`, sorted by `frame_id`. A frame whose CSV
    can't even be loaded (e.g. missing mandatory column) gets a NaN row with
    `status = "load:<reason>"`, same as any other per-frame failure -- the
    whole dataset is never aborted by one bad frame.
    """
    config = config if config is not None else PipelineConfig()
    dataset_root = Path(dataset_root)

    rows = []
    debug_by_frame = {}
    for frame_id, _frame_id_str, csv_path in _discover_frame_files(dataset_root):
        try:
            df = io_schema.load_frame(csv_path)
        except Exception as e:  # noqa: BLE001
            # io_schema's message is "{csv_path}: reason"; drop the (per-frame,
            # always-distinct) path prefix so identical failures group together
            # in a status breakdown instead of one bucket per frame.
            msg = str(e)
            msg = msg.split(": ", 1)[1] if ": " in msg else msg
            row = io_schema.empty_output_row(frame_id).to_dict()
            row["status"] = f"load:{msg}"
            rows.append(row)
            debug_by_frame[frame_id] = _empty_debug()
            continue

        row, debug = _estimate_frame_impl(df, frame_id, config)
        rows.append(row)
        debug_by_frame[frame_id] = debug

    columns = io_schema.OUTPUT_COLUMNS + ["status"]
    out_df = pd.DataFrame(rows, columns=columns)
    out_df = out_df.sort_values("frame_id").reset_index(drop=True)

    dataset_name = dataset_root.name
    output_dir = Path(config.output_dir) if config.output_dir is not None else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_dir / f"kinematics_{dataset_name}.csv", index=False)

    if config.write_debug:
        with open(output_dir / f"kinematics_{dataset_name}_debug.pkl", "wb") as f:
            pickle.dump(debug_by_frame, f)

    return out_df
