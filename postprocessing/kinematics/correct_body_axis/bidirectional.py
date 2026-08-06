"""Step 2 (round 3, "stroke=90 anchor" task): bidirectional segment
reconstruction of `x_body` between anchor frames.

Generic / data-source-agnostic on purpose: it operates on a caller-supplied
ordered list of frame ids, per-frame body point clouds, a set of trusted
"anchor" `x_body` values, and a fallback per-frame causal-only `x_body`
series (round 1's `continuity.compute_continuous_x_body`, chained forward
across the whole sequence -- i.e. exactly `build_sequence.py`'s "after"
column for real data). This lets both the synthetic validation (`synthetic.py`
+ `diag/g_synthetic_validation.py`) and the real 640-frame run
(`diag/g_real_data_timeseries.py`) share one implementation. Does not modify
`continuity.py`/`build_sequence.py` -- only calls
`continuity.compute_continuous_x_body`, a read-only dependency.

Method (see task spec §2):
- Anchor frames keep their supplied trusted value verbatim, `coverage="anchor"`.
- Between two *position-adjacent* anchors (adjacent in the ordered
  `frame_ids` list, not necessarily adjacent `frame_id` values) whose
  position gap is `<= max_anchor_gap`: run `compute_continuous_x_body`
  forward from the segment's start anchor and, separately, backward (same
  function, just walking the frame list in reverse) from the segment's end
  anchor. Each interior frame ends up with one forward and one backward
  estimate. Merge by linear distance-to-anchor weighting: a frame closer to
  the *start* anchor trusts the forward estimate more (fewer propagation
  steps have accumulated error since that anchor), and symmetrically for the
  backward estimate near the *end* anchor. `coverage="bidirectional"`.
  Before merging, the backward estimate is flipped into the forward
  estimate's hemisphere if the two disagree by >90 deg (`fwd_bwd_disagree_deg`
  is recorded either way, pre-flip, as a diagnostic -- a large disagreement
  here means the two independent propagation directions didn't converge,
  which is itself useful diagnostic signal, not something to hide).
- Frames before the first anchor, after the last anchor, or inside a segment
  whose position gap exceeds `max_anchor_gap`: no bidirectional coverage,
  fall back to the supplied causal-only value verbatim,
  `coverage="no_anchor_causal_only"`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import geometry as geo
from .continuity import compute_continuous_x_body


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos_a = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _no_anchor_row(frame_id: int, causal_x_body: dict[int, np.ndarray]) -> dict:
    x = causal_x_body[frame_id]
    return {
        "frame_id": frame_id, "x_c_x": x[0], "x_c_y": x[1], "x_c_z": x[2],
        "coverage": "no_anchor_causal_only",
        "w_fwd": np.nan, "w_bwd": np.nan, "fwd_bwd_disagree_deg": np.nan,
    }


def _anchor_row(frame_id: int, anchor_x_body: dict[int, np.ndarray]) -> dict:
    x = anchor_x_body[frame_id]
    return {
        "frame_id": frame_id, "x_c_x": x[0], "x_c_y": x[1], "x_c_z": x[2],
        "coverage": "anchor",
        "w_fwd": np.nan, "w_bwd": np.nan, "fwd_bwd_disagree_deg": np.nan,
    }


def _forward_chain(frame_ids: list[int], start_pos: int, end_pos: int,
                    seed: np.ndarray, body_xyz_by_frame: dict[int, np.ndarray],
                    up: np.ndarray) -> dict[int, np.ndarray]:
    """`x_body` at every position in `(start_pos, end_pos]`, propagated
    forward from `seed` (the value *at* `start_pos`, not itself recomputed)."""
    out: dict[int, np.ndarray] = {}
    prev = seed
    for pos in range(start_pos + 1, end_pos + 1):
        fid = frame_ids[pos]
        x, _diag = compute_continuous_x_body(body_xyz_by_frame[fid], prev, up=up)
        out[pos] = x
        prev = x
    return out


def _backward_chain(frame_ids: list[int], start_pos: int, end_pos: int,
                     seed: np.ndarray, body_xyz_by_frame: dict[int, np.ndarray],
                     up: np.ndarray) -> dict[int, np.ndarray]:
    """`x_body` at every position in `[start_pos, end_pos)`, propagated
    backward from `seed` (the value *at* `end_pos`)."""
    out: dict[int, np.ndarray] = {}
    prev = seed
    for pos in range(end_pos - 1, start_pos - 1, -1):
        fid = frame_ids[pos]
        x, _diag = compute_continuous_x_body(body_xyz_by_frame[fid], prev, up=up)
        out[pos] = x
        prev = x
    return out


def bidirectional_reconstruct(
    frame_ids: list[int],
    body_xyz_by_frame: dict[int, np.ndarray],
    anchor_x_body: dict[int, np.ndarray],
    causal_x_body: dict[int, np.ndarray],
    up: np.ndarray = (0.0, 0.0, 1.0),
    max_anchor_gap: int = 50,
) -> pd.DataFrame:
    """Returns one row per `frame_ids` entry (same order), columns
    `frame_id, x_c_x, x_c_y, x_c_z, coverage, w_fwd, w_bwd,
    fwd_bwd_disagree_deg`. `x_c_*` is the reconstructed unit `x_body`.

    `causal_x_body` must have an entry for every `frame_id` in `frame_ids`
    (used verbatim for uncovered regions). `anchor_x_body` only needs entries
    for the frame ids that are actually anchors (a subset of `frame_ids`).
    `body_xyz_by_frame` only needs entries for frames strictly inside a
    covered (`<= max_anchor_gap`) segment -- anchors and uncovered frames
    don't need their point cloud here.
    """
    up_hat = geo.unit(np.asarray(up, dtype=float))
    n = len(frame_ids)
    positions = {fid: i for i, fid in enumerate(frame_ids)}
    anchor_positions = sorted(positions[fid] for fid in anchor_x_body if fid in positions)

    rows: list[dict] = [None] * n  # type: ignore[list-item]

    if len(anchor_positions) == 0:
        for pos, fid in enumerate(frame_ids):
            rows[pos] = _no_anchor_row(fid, causal_x_body)
        return pd.DataFrame(rows)

    for pos in range(0, anchor_positions[0]):
        rows[pos] = _no_anchor_row(frame_ids[pos], causal_x_body)

    for seg_i in range(len(anchor_positions) - 1):
        p0, p1 = anchor_positions[seg_i], anchor_positions[seg_i + 1]
        f0 = frame_ids[p0]
        rows[p0] = _anchor_row(f0, anchor_x_body)

        gap = p1 - p0
        if gap > max_anchor_gap:
            for pos in range(p0 + 1, p1):
                rows[pos] = _no_anchor_row(frame_ids[pos], causal_x_body)
            continue

        fwd = _forward_chain(frame_ids, p0, p1, anchor_x_body[f0], body_xyz_by_frame, up_hat)
        f1 = frame_ids[p1]
        bwd = _backward_chain(frame_ids, p0, p1, anchor_x_body[f1], body_xyz_by_frame, up_hat)

        for pos in range(p0 + 1, p1):
            x_fwd = fwd[pos]
            x_bwd = bwd[pos]
            disagree = _angle_deg(x_fwd, x_bwd)
            x_bwd_aligned = x_bwd if np.dot(x_fwd, x_bwd) >= 0 else -x_bwd

            w_fwd = (p1 - pos) / gap
            w_bwd = (pos - p0) / gap
            merged = w_fwd * x_fwd + w_bwd * x_bwd_aligned
            merged_norm = np.linalg.norm(merged)
            x_c = merged / merged_norm if merged_norm > 1e-9 else x_fwd

            rows[pos] = {
                "frame_id": frame_ids[pos], "x_c_x": x_c[0], "x_c_y": x_c[1], "x_c_z": x_c[2],
                "coverage": "bidirectional",
                "w_fwd": w_fwd, "w_bwd": w_bwd, "fwd_bwd_disagree_deg": disagree,
            }

    last_pos = anchor_positions[-1]
    rows[last_pos] = _anchor_row(frame_ids[last_pos], anchor_x_body)
    for pos in range(last_pos + 1, n):
        rows[pos] = _no_anchor_row(frame_ids[pos], causal_x_body)

    return pd.DataFrame(rows)
