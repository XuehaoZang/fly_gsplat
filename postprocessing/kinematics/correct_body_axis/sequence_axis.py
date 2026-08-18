"""Production sequence-level `x_body`: chain sign-determination + mandatory
anchor safety net, wired into `pipeline.py` (see
`pipeline.run_dataset_with_sequence_correction`).

Two stages, both mandatory whenever this module is used -- there is no flag
to run stage 1 without stage 2:

1. `build_guide_chain` restores `robust_body_axis.compute_robust_x_body`'s
   guide_axis chain propagation (previous frame's own axis as the next
   frame's guide) as the production sign-determination mechanism, replacing
   `body_frame.py`'s old per-frame-independent
   `orient_to_reference(PCA major axis, up)`. Measured on the real 640-frame
   dataset (`correct_body_axis/diag/h_robust_axis_timeseries.py`): 0/639
   adjacent-frame sign flips for the chain vs 39/639 for the old per-frame
   baseline -- this is the "0翻转基线" the chain is expected to recover.

2. `verify_and_flip_by_anchors` is the mandatory safety net the chain alone
   does not have. `build_guide_chain` only re-grounds off a trusted anchor
   axis at a sequence start / `frame_id` gap>1 / this method's own failure
   on the previous frame -- between those points it is one continuous
   causal chain, and `compute_robust_x_body` geometrically guarantees each
   step's axis has non-negative dot with its own guide (see that module's
   docstring), so the chain can never locally flip, but it also can never
   *notice* that an early seed landed on the globally wrong sign branch.
   Measured on the real dataset (`h_robust_axis_timeseries.py`'s
   `diagnose_global_sign`): the chain alone opposes the trustworthy
   per-anchor baseline-PCA sign at 100% of the 52 T-pose anchor frames
   (`g_anchors.csv`) -- i.e. the entire sequence, seeded once at frame 0,
   locked onto one coherent wrong branch and nothing downstream ever
   re-checked it. This is exactly the "种子从起点就错、无人能发现" failure
   mode this stage exists to catch: it walks the anchor frames in sequence
   order and, whenever the chain's (still-uncorrected) axis at an anchor
   disagrees in sign with that anchor's own trusted baseline axis, flips
   the *entire* segment since the previous checkpoint (previous anchor, a
   chain reset, or the sequence start) -- not just that one frame.

Both stages take already-loaded per-frame body point clouds and a
caller-supplied anchor table, so they have no I/O of their own beyond the
module-level cache write in `compute_sequence_x_body`. `compute_sequence_x_body`
is the orchestrator that also does the I/O (frame discovery, anchor
detection, body point cloud loading) via the read-only `correct_body_axis`
diagnostics already built for this: `build_sequence.build_sequence`,
`anchor_detect.compute_wing_geometry`/`detect_anchors`,
`diag.f_residual_jitter_attribution.compute_body_cm_all`.

Deliberately does not import `anchor_detect`/`f_residual_jitter_attribution`
at `pipeline.py`'s own module top level (see `pipeline.py`'s lazy import in
`run_dataset_with_sequence_correction`) -- those modules import
`postprocessing.calc_kinematics` for their `DEFAULT_DATASET_ROOT` default,
and `calc_kinematics.py` itself imports `pipeline`, so a top-level import
here would be circular.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import geometry as geo
from ..robust_body_axis import compute_robust_x_body
from .anchor_detect import compute_wing_geometry, detect_anchors
from .build_sequence import build_sequence, frame_body_xyz
from .diag.f_residual_jitter_attribution import compute_body_cm_all

OUT_DIR = Path(__file__).resolve().parent / "diag"
PRODUCTION_X_BODY_CSV = OUT_DIR / "m_production_x_body.csv"
FLIP_EVENTS_CSV = OUT_DIR / "m_anchor_flip_events.csv"

_RESET_METHODS = ("anchor_guide", "pca_up_fallback")
"""`compute_robust_x_body` `method` values that mean "this frame's guide_axis
came from something other than the previous frame's own chained output" --
i.e. a chain reset. `"prev_frame_guide"` is the only non-reset method."""


def _nearest_anchor_axis(frame_id: int, anchor_axes: dict[int, np.ndarray]) -> np.ndarray | None:
    if not anchor_axes:
        return None
    nearest_fid = min(anchor_axes, key=lambda fid: abs(fid - frame_id))
    return anchor_axes[nearest_fid]


def build_guide_chain(
    frame_ids: list[int],
    body_xyz_by_frame: dict[int, np.ndarray],
    anchor_axes: dict[int, np.ndarray],
    up: np.ndarray = (0.0, 0.0, 1.0),
) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    """Causal forward chain of `compute_robust_x_body` over `frame_ids` (in
    the given order), one call per frame present in `body_xyz_by_frame`.

    Guide policy mirrors `correct_body_axis/diag/h_robust_axis_timeseries.py`
    (already validated on the real dataset, 0/639 flips):
    - previous frame's own resulting axis, when available;
    - reset (`guide=None`, forcing the anchor tier) at a sequence start,
      after a `frame_id` gap > 1 versus the previous *processed* frame, or
      right after this method raised on the previous frame -- chaining a
      guide off of a just-broken frame would defeat continuity's purpose;
    - anchor tier (used automatically by `compute_robust_x_body` whenever
      `guide=None`): the nearest anchor frame's own baseline axis.

    Frames missing from `body_xyz_by_frame` are skipped entirely (not
    counted toward the gap computation, same convention `build_sequence.py`
    already uses for its own `frame_id_gap`). A frame where
    `compute_robust_x_body` itself raises gets `NaN` axis, `method` prefixed
    `"failed:"`, and forces a reset on the next processed frame.

    Returns `(axes, diags)`, each keyed by `frame_id`.
    """
    up_hat = geo.unit(np.asarray(up, dtype=float))
    axes: dict[int, np.ndarray] = {}
    diags: dict[int, dict] = {}
    prev_axis: np.ndarray | None = None
    prev_frame_id: int | None = None

    for frame_id in frame_ids:
        if frame_id not in body_xyz_by_frame:
            continue

        gap = None if prev_frame_id is None else frame_id - prev_frame_id
        if prev_frame_id is None or (gap is not None and gap > 1):
            prev_axis = None

        anchor_axis = None if prev_axis is not None else _nearest_anchor_axis(frame_id, anchor_axes)

        try:
            axis, diag = compute_robust_x_body(
                body_xyz_by_frame[frame_id], prev_axis, up=up_hat, anchor_axis=anchor_axis,
            )
            axes[frame_id] = axis
            diags[frame_id] = diag
            prev_axis = axis
        except Exception as e:  # noqa: BLE001
            axes[frame_id] = np.full(3, np.nan)
            diags[frame_id] = {"method": f"failed:{e}"}
            prev_axis = None  # forces a reset next frame, per spec

        prev_frame_id = frame_id

    return axes, diags


def verify_and_flip_by_anchors(
    frame_ids: list[int],
    chain_axes: dict[int, np.ndarray],
    chain_diag: dict[int, dict],
    anchor_axes: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Mandatory anchor safety net (module docstring stage 2).

    Checkpoints are, in sequence-position order: every anchor frame (`fid in
    anchor_axes`) and every chain-reset frame (`method in _RESET_METHODS`,
    including the sequence's own first processed frame, which is always a
    reset). An anchor checkpoint's `should_flip` is
    `dot(chain_axes[fid], anchor_axes[fid]) < 0` -- compared against the
    *original*, not yet corrected, chain value, so segments are judged
    independently of each other's corrections. A reset checkpoint's
    `should_flip` is always `False`: a reset already re-grounds its axis off
    a trusted source (the nearest anchor's raw axis, or in the last-resort
    case the same `pca_up_fallback` heuristic `body_frame.py` used to use
    everywhere), independent of whatever sign error accumulated before it.

    Each checkpoint owns the segment from just after the *previous*
    checkpoint through itself (inclusive); the very first checkpoint owns
    the segment from the sequence start through itself, so a bad seed at
    frame 0 is itself correctable (not just frames after the first anchor).
    The segment past the *last* checkpoint (no further anchor or reset ever
    occurs) inherits that last checkpoint's `should_flip` decision rather
    than being left uncorrected, since nothing about `compute_robust_x_body`'s
    causal chaining changes the sign branch on its own between checkpoints.

    Returns `(corrected_axes, events)`: `corrected_axes` has the same keys
    as `chain_axes` (NaN/failed frames pass through unchanged); `events` is
    one dict per segment actually flipped (`checkpoint_frame_id`,
    `checkpoint_kind`, `segment_start_frame_id`, `segment_end_frame_id`,
    `n_frames_flipped`) -- empty when every checkpoint already agreed with
    its trusted reference.
    """
    valid_positions = [
        (pos, fid) for pos, fid in enumerate(frame_ids)
        if fid in chain_axes and not np.any(np.isnan(chain_axes[fid]))
    ]
    corrected = dict(chain_axes)
    if not valid_positions:
        return corrected, []

    checkpoints: list[tuple[int, int, bool, str]] = []
    for pos, fid in valid_positions:
        method = chain_diag.get(fid, {}).get("method", "")
        if fid in anchor_axes:
            trusted_axis = geo.unit(np.asarray(anchor_axes[fid], dtype=float))
            should_flip = float(np.dot(chain_axes[fid], trusted_axis)) < 0.0
            checkpoints.append((pos, fid, should_flip, "anchor"))
        elif method in _RESET_METHODS:
            checkpoints.append((pos, fid, False, "reset"))

    if not checkpoints:
        return corrected, []
    checkpoints.sort(key=lambda c: c[0])

    events: list[dict] = []

    def _flip_segment(start_pos: int, end_pos: int, should_flip: bool, fid: int, kind: str) -> None:
        if not should_flip:
            return
        seg_fids = [
            frame_ids[p] for p in range(start_pos, end_pos + 1)
            if frame_ids[p] in corrected and not np.any(np.isnan(corrected[frame_ids[p]]))
        ]
        if not seg_fids:
            return
        for f in seg_fids:
            corrected[f] = -corrected[f]
        events.append({
            "checkpoint_frame_id": fid, "checkpoint_kind": kind,
            "segment_start_frame_id": seg_fids[0], "segment_end_frame_id": seg_fids[-1],
            "n_frames_flipped": len(seg_fids),
        })

    prev_boundary_pos = -1
    for pos, fid, should_flip, kind in checkpoints:
        _flip_segment(prev_boundary_pos + 1, pos, should_flip, fid, kind)
        prev_boundary_pos = pos

    last_pos, last_fid, last_should_flip, last_kind = checkpoints[-1]
    _flip_segment(last_pos + 1, len(frame_ids) - 1, last_should_flip, last_fid, f"{last_kind}_tail_extension")

    return corrected, events


def compute_sequence_x_body(
    root: Path, up: np.ndarray = (0.0, 0.0, 1.0),
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Orchestrator: build the chain (stage 1), verify/correct it against
    `anchor_detect.py`'s T-pose anchors (stage 2, mandatory), and return the
    final per-frame `x_body` table plus an audit `DataFrame` (also cached to
    `PRODUCTION_X_BODY_CSV`/`FLIP_EVENTS_CSV` for QA).

    `root` is required (no `DEFAULT_DATASET_ROOT` fallback here) so this
    module never needs to import `postprocessing.calc_kinematics` at its own
    top level -- see module docstring's circular-import note.
    """
    from postprocessing.labeling.motion.diag.identity_flip_stats import (
        discover_labeled_frames,
        load_labeled,
    )

    seq_df = build_sequence(root)
    cm_df = compute_body_cm_all(seq_df, root)
    geom_df = compute_wing_geometry(seq_df, cm_df, root)
    anchors_df = detect_anchors(seq_df, geom_df)

    ok = seq_df.loc[~seq_df["failed"]].sort_values("frame_id").reset_index(drop=True)
    frame_ids = ok["frame_id"].astype(int).tolist()

    frame_paths = discover_labeled_frames(root)
    body_xyz_by_frame: dict[int, np.ndarray] = {}
    for fid in frame_ids:
        if fid not in frame_paths:
            continue
        df = load_labeled(frame_paths[fid])
        body_xyz_by_frame[fid] = frame_body_xyz(df)

    anchor_frame_ids = anchors_df.loc[anchors_df["is_anchor"], "frame_id"].astype(int).tolist()
    seq_lookup = seq_df.set_index("frame_id")
    anchor_axes: dict[int, np.ndarray] = {}
    for fid in anchor_frame_ids:
        if fid in seq_lookup.index:
            row = seq_lookup.loc[fid]
            anchor_axes[fid] = np.array(
                [row["x_before_x"], row["x_before_y"], row["x_before_z"]], dtype=float
            )

    chain_axes, chain_diag = build_guide_chain(frame_ids, body_xyz_by_frame, anchor_axes, up=up)
    corrected_axes, flip_events = verify_and_flip_by_anchors(frame_ids, chain_axes, chain_diag, anchor_axes)

    rows = []
    for fid in frame_ids:
        axis = corrected_axes.get(fid)
        failed = axis is None or np.any(np.isnan(axis))
        rows.append({
            "frame_id": fid,
            "is_anchor": fid in anchor_axes,
            "method": chain_diag.get(fid, {}).get("method"),
            "x_x": np.nan if failed else float(axis[0]),
            "x_y": np.nan if failed else float(axis[1]),
            "x_z": np.nan if failed else float(axis[2]),
        })
    audit_df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(PRODUCTION_X_BODY_CSV, index=False)
    pd.DataFrame(flip_events).to_csv(FLIP_EVENTS_CSV, index=False)

    x_body_table = {
        fid: axis for fid, axis in corrected_axes.items()
        if axis is not None and not np.any(np.isnan(axis))
    }
    return x_body_table, audit_df
