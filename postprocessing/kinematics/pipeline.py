"""S5 pipeline: labeled per-point CSVs -> per-frame kinematics table, T4.

Implements calc_kinematics.md §0 (conventions), §1 (input/output contract),
and §6 (T4-adopted definitions) end to end, chaining S2 (`body_frame.py`) ->
S3 (`wing_angles.py`) -> S4a (`chord.py`) per frame. No multi-frame logic
lives here for wing angles/chord (no unwrap, no smoothing, no stroke-plane
bootstrap across frames) -- per-frame estimation stays a pure function of
that frame's own CSV, per §2/§4's "single frame" scope.

The one exception is `x_body` (§2 step 1's body long-axis sign): a
per-frame-independent PCA guess is not reliable on its own (see
`body_frame.estimate_body_frame`'s docstring), so `PipelineConfig.sequence_x_body`
optionally carries a precomputed `{frame_id: x_body}` table from
`correct_body_axis.sequence_axis.compute_sequence_x_body` (continuity chain +
mandatory anchor verification, see that module) -- `run_dataset_with_
sequence_correction` builds and supplies this automatically; `estimate_frame`/
`run_dataset` fall back to `body_frame.py`'s own single-frame heuristic for
any frame not in the table (e.g. no `sequence_x_body` given at all, or a
frame this round-trip's underlying `identity_flip_stats` loader couldn't
find), never raising or dropping the frame over a missing table entry.

`estimate_frame` is the single-frame entry point (pure function of one
already-loaded `pd.DataFrame` plus, optionally, this frame's own precomputed
`x_body`); `run_dataset` is the I/O layer (globbing, loading, writing
outputs) built on top of it. A frame is never dropped: any per-stage
failure (missing/too-small part, RANSAC or PCA degeneracy) is caught,
leaves the affected output fields NaN, and is recorded in a `status`
string instead of raising -- see `_estimate_frame_impl`.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import body_frame as bf
from . import chord as ch
from . import chord_matlab as chm
from . import io_schema
from . import wing_angles as wa

_FRAME_DIR_RE = re.compile(r"^f(\d+)$")
"""Matches the per-frame directory name (`f<NNNN>`) under a dataset root, §1."""

DEBUG_KEYS: tuple[str, ...] = (
    "x_body", "y_body", "z_body", "n_sp",
    "hinge_L", "hinge_R", "body_cm",
    "le_dir_L", "le_dir_R",
    "span_L", "span_R",
    "chord_L", "chord_R",
    "per_bin_chords_L", "per_bin_chords_R",
)
"""Per-frame debug-sidecar keys (task spec): S2 body frame + S3/S4a per-wing
intermediates, for later viz/QA -- not part of the main output CSV schema
(`io_schema.OUTPUT_COLUMNS`). `span_L`/`span_R` are `wing_angles.estimate_span`'s
wing-PCA span vectors (the S3 revision's phi/theta input); `le_dir_L`/`le_dir_R`
remain the leading-edge vectors `chord.py`'s LE->TE sign uses."""


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
    frame_glob: str = "f*/splatfacto-checkpoint/*/*_marked.csv"
    """Glob (relative to `dataset_root`) for per-frame input CSVs, matched by
    `_discover_frame_files`. Default matches T2's `if_keep`-only output; the
    real T3 labeling step (`postprocessing/labeling/labeling.py`) does not
    label `_marked.csv` in place -- it reads it and writes a sibling
    `*_labeled.csv` (adds `part_label`, `confidence`), leaving `_marked.csv`
    untouched. Point this at `"f*/splatfacto-checkpoint/*/*_labeled.csv"` (or
    any other real layout) to run against actual T3 output."""
    sequence_x_body: dict[int, np.ndarray] | None = None
    """Optional `{frame_id: x_body}` table (unit vectors) overriding
    `body_frame.py`'s own single-frame `x_body` heuristic per frame -- see
    module docstring. `None` (default) keeps every frame on the old
    per-frame heuristic; build one via
    `correct_body_axis.sequence_axis.compute_sequence_x_body`, or just call
    `run_dataset_with_sequence_correction` instead of `run_dataset` to have
    it built and applied automatically. A frame_id absent from the table
    (even when the table itself is non-`None`) falls back to the per-frame
    heuristic for that one frame only."""


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
        x_body = None if config.sequence_x_body is None else config.sequence_x_body.get(frame_id)
        frame = bf.estimate_body_frame(
            body_xyz, wingL_xyz, wingR_xyz,
            up=config.up,
            stroke_plane_pitch_deg=config.stroke_plane_pitch_deg,
            stroke_plane_normal=config.stroke_plane_normal,
            root_mode=config.root_mode,
            x_body=x_body,
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
            if not _all_finite(
                sweep.phi, sweep.theta, chord_result.eta, chord_result.chord_conf, sweep.span_dir
            ):
                raise ValueError("non-finite result")
        except Exception as e:  # noqa: BLE001
            wing_statuses.append(f"{side}:{e}")
            continue

        row[f"phi_{suffix}"] = sweep.phi
        row[f"theta_{suffix}"] = sweep.theta
        row[f"eta_{suffix}"] = chord_result.eta
        row[f"chord_conf_{suffix}"] = chord_result.chord_conf
        row[f"span_{suffix}_x"], row[f"span_{suffix}_y"], row[f"span_{suffix}_z"] = (
            float(sweep.span_dir[0]), float(sweep.span_dir[1]), float(sweep.span_dir[2])
        )
        debug[f"le_dir_{suffix}"] = sweep.leading_edge.le_dir
        debug[f"span_{suffix}"] = sweep.span_dir
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


def _discover_frame_files(dataset_root: Path, frame_glob: str) -> list[tuple[int, str, Path]]:
    """Find per-frame input CSVs under `dataset_root` matching `frame_glob`
    (default `f<NNNN>/splatfacto-checkpoint/<ts>/*_marked.csv`, see
    `PipelineConfig.frame_glob`).

    Returns `(frame_id, frame_id_str, csv_path)` triples sorted by
    `frame_id`. `frame_id_str` is the zero-padded directory name (e.g.
    `"0090"`) -- kept alongside the parsed int per the task spec, though only
    `frame_id` (int) reaches the output row/CSV.
    """
    out = []
    for csv_path in dataset_root.glob(frame_glob):
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
    for frame_id, _frame_id_str, csv_path in _discover_frame_files(dataset_root, config.frame_glob):
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


def run_dataset_with_sequence_correction(
    dataset_root: str | Path, config: PipelineConfig | None = None,
) -> pd.DataFrame:
    """`run_dataset`, but with `config.sequence_x_body` built and applied
    automatically -- the production entry point for a real (non-synthetic,
    non-single-frame-test) dataset; see module docstring and
    `correct_body_axis/sequence_axis.py`.

    Building the table (`correct_body_axis.sequence_axis.compute_sequence_x_body`)
    currently goes through `postprocessing.labeling.motion.diag.
    identity_flip_stats`'s own `_labeled.csv` discovery, a separate code path
    from this function's own `_discover_frame_files`/`io_schema.load_frame` --
    both read the same on-disk `f<NNNN>/.../*_labeled.csv` layout and agree
    on `frame_id`, so the resulting table's keys line up with this function's
    own per-frame loop, but a frame one loader finds and the other doesn't
    (e.g. a stray malformed CSV) simply falls back to the single-frame
    heuristic for that frame rather than erroring -- see `PipelineConfig.
    sequence_x_body`'s own fallback note.

    Imports `correct_body_axis.sequence_axis` lazily (not at this module's
    top level) since that import chain reaches back into
    `postprocessing.calc_kinematics`, which itself imports this module --
    see `sequence_axis.py`'s own docstring for why.
    """
    from .correct_body_axis.sequence_axis import compute_sequence_x_body

    config = config if config is not None else PipelineConfig()
    dataset_root = Path(dataset_root)

    x_body_table, _audit_df = compute_sequence_x_body(dataset_root, up=config.up)
    config = replace(config, sequence_x_body=x_body_table)

    return run_dataset(dataset_root, config)


def run_dataset_with_eta_unwrap(
    dataset_root: str | Path, config: PipelineConfig | None = None,
) -> pd.DataFrame:
    """`run_dataset_with_sequence_correction`, plus a whole-sequence post-pass
    over `eta_L`/`eta_R` (wing pitch) -- the production T4 entry point
    (`calc_kinematics.py` calls this, not `run_dataset_with_sequence_correction`
    directly).

    Per-frame `eta` (`chord.py::estimate_chord`) is an `atan2` angle folded
    to `(-180, 180]`; across a real wingbeat cycle this produces spurious
    ~360 deg jumps at the wrap boundary that no single-frame fix can remove.
    `eta_unwrap.process_eta` (see that module's docstring for the two
    independent failure modes it targets -- the wrap itself, and a real
    +/-180 leading/trailing-edge sign ambiguity in `chord.py` on some
    high-`chord_conf` frames) fixes this as a pure post-pass; it does not
    touch `chord.py`/`wing_angles.py`'s per-frame formula.

    Only `status == "ok"` rows are unwrapped, in `frame_id` order (mirrors
    how the fix was validated); other rows' `eta_L`/`eta_R` are left as-is
    (already NaN). The corrected columns overwrite the CSV
    `run_dataset_with_sequence_correction` already wrote.
    """
    from .eta_unwrap import process_eta

    config = config if config is not None else PipelineConfig()
    dataset_root = Path(dataset_root)

    df = run_dataset_with_sequence_correction(dataset_root, config)

    ok_idx = df.index[df["status"] == "ok"]
    for suffix in ("L", "R"):
        if len(ok_idx) == 0:
            continue
        result = process_eta(
            df.loc[ok_idx, f"eta_{suffix}"].to_numpy(),
            chord_conf=df.loc[ok_idx, f"chord_conf_{suffix}"].to_numpy(),
        )
        df.loc[ok_idx, f"eta_{suffix}"] = result.unwrapped

    dataset_name = dataset_root.name
    output_dir = Path(config.output_dir) if config.output_dir is not None else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"kinematics_{dataset_name}.csv", index=False)

    return df


def run_dataset_with_eta_unwrap_dp(
    dataset_root: str | Path, config: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Same as `run_dataset_with_eta_unwrap`, but the eta post-pass uses
    `eta_unwrap.process_eta_dp` (global two-state DP branch resolution +
    chunked/overlap-stitched unwrap) instead of `process_eta`'s single-pass
    whole-sequence unwrap.

    Trying this as the fix for the unbounded (~800-1000+ deg) drift
    `process_eta` produces on longer (~450+ frame, e.g. `valid480`)
    sequences -- see `eta_unwrap.unwrap_deg_chunked`'s docstring for why a
    single-pass cumulative unwrap turns occasional local branch-resolution
    errors into a permanent staircase on sequences this long. Not yet the
    default production path (`calc_kinematics.py` still calls
    `run_dataset_with_eta_unwrap`); call this directly to opt in.
    """
    from .eta_unwrap import process_eta_dp

    config = config if config is not None else PipelineConfig()
    dataset_root = Path(dataset_root)

    df = run_dataset_with_sequence_correction(dataset_root, config)

    ok_idx = df.index[df["status"] == "ok"]
    for suffix in ("L", "R"):
        if len(ok_idx) == 0:
            continue
        result = process_eta_dp(
            df.loc[ok_idx, f"eta_{suffix}"].to_numpy(),
            chord_conf=df.loc[ok_idx, f"chord_conf_{suffix}"].to_numpy(),
        )
        df.loc[ok_idx, f"eta_{suffix}"] = result.unwrapped

    dataset_name = dataset_root.name
    output_dir = Path(config.output_dir) if config.output_dir is not None else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"kinematics_{dataset_name}.csv", index=False)

    return df


def run_dataset_matlab_chord_eta(
    dataset_root: str | Path, config: PipelineConfig | None = None, use_velocity: bool = False,
) -> pd.DataFrame:
    """Experimental alternative to `run_dataset_with_eta_unwrap`: `eta_L`/
    `eta_R`/`chord_conf_L`/`chord_conf_R` come from `chord_matlab.
    estimate_chord_matlab` (MATLAB `find_chords_quad.m`-style diagonal
    selection, see that module's docstring) instead of `chord.py`'s
    leading-edge-winner-based chord. Everything else (`yaw`/`pitch`/`roll`,
    `phi`/`theta`, `status`, `sequence_x_body`) is computed exactly as
    `run_dataset_with_sequence_correction` does -- this function re-walks the
    same frame files itself (rather than post-processing that function's
    output) only because the new chord needs each frame's raw wing point
    cloud, not just the already-collapsed `eta` column `eta_unwrap.py`'s
    functions post-process.

    No `resolve_180_flip`/`unwrap` pass is applied here -- `chord_matlab`'s
    output is dramatically more stable frame-to-frame (step 14: wrap-crossing
    count ~160-177 -> ~17-19 per side on the real `valid480` dataset this was
    developed against) but not perfectly clean; a caller wanting a fully
    unwrapped column should still run the result through `eta_unwrap.
    process_eta` (or a variant re-tuned for this method's much-lower-noise
    residual failure mode -- step 14 found the *old* `process_eta`, tuned
    against `chord.py`'s noisier output, does not uniformly help this one).

    `use_velocity=False` (default) matches step 14's finding that enabling
    the velocity fallback (once its threshold-scale bug was fixed) did not
    improve on the length-ratio-only result on real data -- see
    `chord_matlab.estimate_chord_matlab`'s own docstring.

    Status: experimental (see `chord_matlab.py` module docstring) -- not
    wired into `calc_kinematics.py`'s default T4 path.
    """
    from .correct_body_axis.sequence_axis import compute_sequence_x_body

    config = config if config is not None else PipelineConfig()
    dataset_root = Path(dataset_root)

    x_body_table, _audit_df = compute_sequence_x_body(dataset_root, up=config.up)
    config = replace(config, sequence_x_body=x_body_table)

    prev_signed_chord = {"wing_L": None, "wing_R": None}
    prev_span_tip = {"wing_L": None, "wing_R": None}
    prev_body_cm = {"wing_L": None, "wing_R": None}

    rows = []
    for frame_id, _frame_id_str, csv_path in _discover_frame_files(dataset_root, config.frame_glob):
        try:
            df_frame = io_schema.load_frame(csv_path)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            msg = msg.split(": ", 1)[1] if ": " in msg else msg
            row = io_schema.empty_output_row(frame_id).to_dict()
            row["status"] = f"load:{msg}"
            rows.append(row)
            continue

        row = io_schema.empty_output_row(frame_id).to_dict()
        try:
            body_xyz = io_schema.get_part(df_frame, "body", apply_if_keep=True)
            wingL_xyz = io_schema.get_part(df_frame, "wing_L", apply_if_keep=True)
            wingR_xyz = io_schema.get_part(df_frame, "wing_R", apply_if_keep=True)
        except Exception as e:  # noqa: BLE001
            row["status"] = f"parts:{e}"
            rows.append(row)
            continue

        too_few = False
        for label, xyz in (("body", body_xyz), ("wing_L", wingL_xyz), ("wing_R", wingR_xyz)):
            if xyz.shape[0] < config.min_points:
                row["status"] = f"{label}:too_few_points"
                rows.append(row)
                too_few = True
                break
        if too_few:
            continue

        try:
            x_body = None if config.sequence_x_body is None else config.sequence_x_body.get(frame_id)
            frame = bf.estimate_body_frame(
                body_xyz, wingL_xyz, wingR_xyz,
                up=config.up, stroke_plane_pitch_deg=config.stroke_plane_pitch_deg,
                stroke_plane_normal=config.stroke_plane_normal, root_mode=config.root_mode, x_body=x_body,
            )
            if not _all_finite(frame.x_body, frame.y_body, frame.z_body, frame.n_sp,
                                frame.yaw, frame.pitch, frame.roll):
                raise ValueError("non-finite result (degenerate hinge/PCA geometry)")
        except Exception as e:  # noqa: BLE001
            row["status"] = f"body:{e}"
            rows.append(row)
            continue

        row["yaw"], row["pitch"], row["roll"] = frame.yaw, frame.pitch, frame.roll
        row["sp_normal_x"], row["sp_normal_y"], row["sp_normal_z"] = (
            float(frame.n_sp[0]), float(frame.n_sp[1]), float(frame.n_sp[2])
        )

        wing_statuses = []
        for side, wing_xyz, suffix in (("wing_L", wingL_xyz, "L"), ("wing_R", wingR_xyz, "R")):
            try:
                sweep = wa.stroke_deviation(wing_xyz, frame, side)
                mc_result = chm.estimate_chord_matlab(
                    wing_xyz, frame, side, sweep.span_dir,
                    prev_signed_chord=prev_signed_chord[side],
                    prev_span_tip=prev_span_tip[side], prev_body_cm=prev_body_cm[side],
                    use_velocity=use_velocity,
                )
                if not _all_finite(sweep.phi, sweep.theta, mc_result.eta, mc_result.chord_conf, sweep.span_dir):
                    raise ValueError("non-finite result")
            except Exception as e:  # noqa: BLE001
                wing_statuses.append(f"{side}:{e}")
                continue

            row[f"phi_{suffix}"] = sweep.phi
            row[f"theta_{suffix}"] = sweep.theta
            row[f"eta_{suffix}"] = mc_result.eta
            row[f"chord_conf_{suffix}"] = mc_result.chord_conf
            row[f"span_{suffix}_x"], row[f"span_{suffix}_y"], row[f"span_{suffix}_z"] = (
                float(sweep.span_dir[0]), float(sweep.span_dir[1]), float(sweep.span_dir[2])
            )
            prev_signed_chord[side] = mc_result.chord.copy()
            prev_span_tip[side] = mc_result.span_tip.copy()
            prev_body_cm[side] = np.asarray(frame.body_cm, dtype=float).copy()

        row["status"] = "ok" if not wing_statuses else ";".join(wing_statuses)
        rows.append(row)

    columns = io_schema.OUTPUT_COLUMNS + ["status"]
    out_df = pd.DataFrame(rows, columns=columns)
    out_df = out_df.sort_values("frame_id").reset_index(drop=True)

    dataset_name = dataset_root.name
    output_dir = Path(config.output_dir) if config.output_dir is not None else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_dir / f"kinematics_{dataset_name}.csv", index=False)

    return out_df
