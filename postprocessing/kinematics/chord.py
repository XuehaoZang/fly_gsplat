"""S4a/S4b wing chord & pitch (eta), single-frame, T4 kinematics.

Implements calc_kinematics.md §5's point-cloud chord method: bin the wing's
points along the leading-edge (span) axis, take each bin's in-plane
chordwise extremes, aggregate into one chord direction, then read off the
stroke-plane pitch `eta` via the `sp_chord`/`le_sp_normal` construction from
`reference/python_snippets.py` cell 3 (reference-only, never imported).
Reuses `wing_angles.estimate_leading_edge` for the leading-edge *and* wing
plane fit (no PCA/plane/line fitting is reimplemented here) and
`geometry.py` primitives for the rest.

**S4a baseline** (`robust=False, use_gaussian_normals=False`, the default):
a clean, single-frame, xyz-only estimate -- plain-mean bin aggregation, no
per-point normal/contaminant handling. This code path is byte-for-byte
unchanged from S4a and is the comparison baseline S4b is measured against
(see `tests/test_s4b.py`).

**S4b enhancements**, each strictly opt-in behind its own flag so the
baseline above never regresses:
  - `use_gaussian_normals=True` (needs `orientation`/`planarity`, §5 step 4):
    build a planarity-trust-weighted robust wing-plane normal `n_w` from each
    point's local normal proxy (`orientation_*`, precomputed upstream as
    `geometry.local_normal_from_gaussian` -- see `utils/gaussian_features.py`),
    then reject points whose own local normal disagrees with `n_w` beyond
    `_REJECT_ANGLE_DEG` (the opposite-wing contaminants that bleed in near
    stroke reversal). LE/binning are recomputed on the survivors. See
    `_robust_wing_normal_and_survivors`.
  - `robust=True` (§5 step 3): swap the plain-mean bin aggregator for a
    count/planarity-weighted, trimmed aggregate (`_aggregate_chords_robust`),
    and make `chord_conf` a real score -- bin agreement under those same
    weights, times the fraction of points that survived contaminant
    rejection (1.0 if `use_gaussian_normals` did no rejection). See
    `_chord_confidence_robust`.

Both flags accept `orientation`/`planarity` (`(N,3)`/`(N,)`, row-aligned with
`wing_xyz`, e.g. via `io_schema.get_part_columns`) as new trailing optional
kwargs -- existing call sites (S4a, `pipeline.py`) are untouched since they
never pass them and both new flags default to `False`. S5 hook: a future
`PipelineConfig` would grow `chord_robust`/`chord_use_gaussian_normals`
bools, and `_estimate_frame_impl` would fetch `orientation`/`planarity` via
`io_schema.get_part_columns(df, side, [...])` alongside the existing
`io_schema.get_part(df, side)` call, and pass all four through to
`estimate_chord`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .body_frame import BodyFrame
from .wing_angles import LeadingEdge, _SIGN_LEFT, _check_side, estimate_leading_edge

logger = logging.getLogger(__name__)

_EPS = 1e-12
_N_SPAN_BINS = 20
"""Span bins for chord extraction (§5 step 3); matches
`wing_angles.estimate_leading_edge`'s own `n_bins` default."""
_MIN_BIN_POINTS = 3
"""Minimum points in a span bin to trust its LE/TE extremes; matches
`wing_angles.estimate_leading_edge`'s own `min_bin_points` default."""

_PLANARITY_TRUST_LO = 0.3
_PLANARITY_TRUST_HI = 0.6
"""Soft planarity trust ramp (§5 step 4, S4b): a point's `orientation` is
untrusted (weight 0) at `planarity <= _LO`, fully trusted (weight 1) at
`planarity >= _HI`, linear in between. Per calc_kinematics.md §1, orientation
is only a reliable normal proxy on (near-)flat points; a hard threshold would
throw away a soft-good point right at the boundary and a frame's-worth of
otherwise-similar points to a marginal one. `mock.py`'s clean wing points
(planarity ~0.85) sit solidly above `_HI`; its low-planarity body-like noise
sits below `_LO` -- see `mock.scenario_noisy_orientation`.
"""
_MIN_TRUSTED_POINTS = 10
"""Below this many planarity-trusted points, `_robust_wing_normal_and_survivors`
can't form a stable weighted normal and falls back to the plane-fit normal
with no rejection at all (§5 step 4's documented fallback)."""
_REJECT_ANGLE_DEG = 45.0
"""Contaminant-rejection threshold (§5 step 4): a (trusted) point whose own
local normal disagrees with the robust wing-plane normal `n_w` by more than
this is treated as belonging to the other wing. 45 deg sits roughly midway
between "same plane, noisy fit" (a few degrees) and "opposite/perpendicular
wing plane" (order 90 deg at stroke reversal, per calc_kinematics.md §5's
mechanism paragraph), so it only fires on genuinely disagreeing points."""
_TRIM_ANGLE_DEG = 35.0
"""Robust-aggregator trim threshold (§5 step 3): a span bin whose chord
differs from the (count/planarity-weighted) mean chord by more than this is
treated as an outlier bin -- e.g. residual contamination that survived
per-point rejection, or a thin/noisy bin -- and excluded from the final
average."""


@dataclass
class ChordResult:
    """One wing's chord/pitch estimate, one frame (§5).

    `chord` is unit length, oriented leading-edge -> trailing-edge. It is
    built by removing only the *span* component from each bin's `te_i - le_i`
    (not the wing-plane-normal component) before averaging, so real
    camber/twist signal along the plane normal survives into `chord` and
    `per_bin_chords` rather than being projected away -- see
    `_bin_chords_core`. `eta` is degrees, the stroke-plane pitch (§5).

    `chord_conf` is in `[0, 1]`. Baseline (`robust=False`): the mean
    resultant length of `per_bin_chords` (how tightly the per-bin chords
    cluster in direction), unweighted -- unchanged from S4a. Enhanced
    (`robust=True`, see `_chord_confidence_robust`): that same agreement
    measure computed under the robust aggregator's weights, times the
    fraction of input points that survived `use_gaussian_normals`'s
    contaminant rejection (1.0 if that rejection didn't run).

    `per_bin_chords` is `(K, 3)`, one row per populated span bin (unit
    vectors, LE->TE oriented), in span order, for downstream twist analysis;
    computed over the (possibly contaminant-filtered) surviving points, not
    always the original `wing_xyz`. `n_bins_used` is `K`.

    `rejected_mask` is `(N,)` bool, aligned with the `wing_xyz` argument
    passed to `estimate_chord`: True for points `use_gaussian_normals`
    dropped as contaminants. All-False when that flag is off, orientation
    data wasn't supplied, or too few points were trustworthy to reject
    anything (S4b).
    """

    chord: np.ndarray
    eta: float
    chord_conf: float
    per_bin_chords: np.ndarray
    n_bins_used: int
    rejected_mask: np.ndarray


# ---------------------------------------------------------------------------
# LE->TE sign (kept separate from bin aggregation, per calc_kinematics.md §5:
# "chord sign is fixed by physical LE -> TE ordering", not the notebook's
# `psi < -100` patch)
# ---------------------------------------------------------------------------


def _oriented_chord_axis(
    wing_xyz: np.ndarray, plane_normal: np.ndarray, le_dir: np.ndarray, le_inlier_mask: np.ndarray
) -> np.ndarray:
    """Unit in-plane axis perpendicular to `le_dir`, oriented from the LE
    side toward the TE side (§5 step 4's sign fix).

    `le_inlier_mask` is `estimate_leading_edge`'s own leading-edge-line
    RANSAC inliers -- already the winning "which side is leading" call (§4
    docstring's straightness heuristic) -- so this only has to read off which
    sign of the candidate axis those points fall on, not re-decide LE vs TE.
    """
    axis = np.cross(plane_normal, le_dir)
    axis = axis / np.linalg.norm(axis)

    centroid = wing_xyz.mean(axis=0)
    le_side_mean = wing_xyz[le_inlier_mask].mean(axis=0) - centroid
    if np.dot(le_side_mean, axis) > 0.0:
        # `axis` currently points toward the LE side; flip so its positive
        # direction is TE, matching `_bin_chord_vectors`'s argmax = te_i.
        axis = -axis
    return axis


# ---------------------------------------------------------------------------
# Bin aggregation (kept separate from the sign logic above so S4b can swap
# in a trimmed/weighted aggregator without touching sign/eta code)
# ---------------------------------------------------------------------------


def _bin_chords_core(
    wing_xyz: np.ndarray,
    le_dir: np.ndarray,
    chord_axis_te: np.ndarray,
    n_bins: int,
    min_bin_points: int,
    planarity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Shared binning core behind `_bin_chord_vectors` (§5 step 2-3).

    Bins `wing_xyz` along `le_dir`; within each bin with >= `min_bin_points`
    points, `te_i`/`le_i` are the points with max/min projection onto
    `chord_axis_te` (already oriented LE->TE by `_oriented_chord_axis`).
    `chord_i = normalize(project_onto_plane(te_i - le_i, le_dir))` --
    removing only the span component (not the plane normal), so a bin whose
    `te_i`/`le_i` sit at slightly different out-of-plane heights (real
    camber/twist, not just binning noise) keeps that signal. Bins too small,
    or whose `te_i - le_i` is (near-)parallel to `le_dir` (degenerate: same
    point selected for both, or a pathological span-only pair), are skipped.

    Returns `(chords, counts, mean_planarity)`: `chords` is `(K,3)` (as
    `_bin_chord_vectors`); `counts` is `(K,)` float, each populated bin's
    point count; `mean_planarity` is `(K,)`, each bin's mean `planarity`, or
    `None` if `planarity` wasn't given -- both are S4b's robust-aggregator
    bin weights (`estimate_chord`), unused by the S4a baseline.
    """
    origin = wing_xyz.mean(axis=0)
    rel = wing_xyz - origin
    t = rel @ le_dir
    c = rel @ chord_axis_te

    bin_edges = np.linspace(t.min(), t.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(t, bin_edges[1:-1]), 0, n_bins - 1)

    chords, counts, mean_planarity = [], [], []
    for b in range(n_bins):
        idx = np.nonzero(bin_idx == b)[0]
        if idx.size < min_bin_points:
            continue
        te_i = wing_xyz[idx[np.argmax(c[idx])]]
        le_i = wing_xyz[idx[np.argmin(c[idx])]]
        diff = te_i - le_i
        if np.linalg.norm(diff) < _EPS:
            continue
        chord_i = geo.project_onto_plane(diff, le_dir)
        if not np.all(np.isfinite(chord_i)):
            continue  # diff (near-)parallel to le_dir -> no defined in-bin chord
        chords.append(chord_i)
        counts.append(idx.size)
        if planarity is not None:
            mean_planarity.append(float(np.mean(planarity[idx])))

    chords_arr = np.array(chords) if chords else np.zeros((0, 3))
    counts_arr = np.array(counts, dtype=float)
    planarity_arr = np.array(mean_planarity, dtype=float) if planarity is not None else None
    return chords_arr, counts_arr, planarity_arr


def _bin_chord_vectors(
    wing_xyz: np.ndarray,
    le_dir: np.ndarray,
    chord_axis_te: np.ndarray,
    n_bins: int,
    min_bin_points: int,
) -> np.ndarray:
    """Per span bin, take the chordwise extremes and return each bin's unit
    chord vector (§5 step 2-3, baseline). Thin wrapper over
    `_bin_chords_core` that drops the count/planarity bookkeeping S4b needs
    -- kept as its own function (byte-identical behavior to S4a) since
    `tests/test_s4a.py` exercises it directly.
    """
    chords, _counts, _planarity = _bin_chords_core(wing_xyz, le_dir, chord_axis_te, n_bins, min_bin_points)
    return chords


def _aggregate_chords(per_bin_chords: np.ndarray) -> np.ndarray:
    """§5 step 3, baseline aggregator: plain mean of per-bin chords,
    renormalized. See `_aggregate_chords_robust` for S4b's trimmed/weighted
    version, swapped in only when `robust=True`.
    """
    mean = per_bin_chords.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < _EPS:
        raise ValueError("_aggregate_chords: per-bin chords cancel to a near-zero mean")
    return mean / norm


def _aggregate_chords_robust(
    per_bin_chords: np.ndarray, bin_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """§5 step 3, S4b aggregator: weight each bin by `bin_weights` (point
    count, optionally x mean planarity -- see `estimate_chord`), then trim
    bins whose chord disagrees with the weighted mean by more than
    `_TRIM_ANGLE_DEG` before a final weighted average.

    Returns `(chord, weights_used)`: `weights_used` (post-trim) is reused by
    `_chord_confidence_robust`, so a trimmed-out bin also stops contributing
    to the reported confidence's agreement term. Falls back to the untrimmed
    weighted mean if trimming would remove every bin (all bins mutually
    disagree by more than the threshold -- nothing left to trim toward).
    """
    w = np.asarray(bin_weights, dtype=float)
    if w.sum() < _EPS:
        w = np.ones_like(w)

    mean0 = np.average(per_bin_chords, axis=0, weights=w)
    norm0 = np.linalg.norm(mean0)
    if norm0 < _EPS:
        raise ValueError("_aggregate_chords_robust: weighted per-bin chords cancel to a near-zero mean")
    mean0 = mean0 / norm0

    cos_ang = np.clip(per_bin_chords @ mean0, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_ang))
    keep = angle_deg <= _TRIM_ANGLE_DEG
    w_trimmed = np.where(keep, w, 0.0)
    if w_trimmed.sum() < _EPS:
        w_trimmed = w

    mean = np.average(per_bin_chords, axis=0, weights=w_trimmed)
    norm_ = np.linalg.norm(mean)
    if norm_ < _EPS:
        raise ValueError("_aggregate_chords_robust: trimmed per-bin chords cancel to a near-zero mean")
    return mean / norm_, w_trimmed


def _chord_confidence_robust(
    per_bin_chords: np.ndarray, bin_weights: np.ndarray, survival_frac: float
) -> float:
    """S4b `chord_conf` = `agreement * survival_frac`, both in `[0,1]` so the
    product is too.

    `agreement` is the mean resultant length of `per_bin_chords` under
    `bin_weights` (post-trim, from `_aggregate_chords_robust`) -- the same
    "how tightly do the per-bin chords cluster" quantity S4a used
    unweighted, just now consistent with the weights the chord itself was
    built from. `survival_frac` is the fraction of input points that passed
    `use_gaussian_normals`'s contaminant rejection (1.0 if that rejection
    didn't run), so a frame where a large fraction of points had to be
    thrown out is penalized even if the survivors happen to agree well.
    """
    w = np.asarray(bin_weights, dtype=float)
    if w.sum() < _EPS:
        w = np.ones_like(w)
    mean = np.average(per_bin_chords, axis=0, weights=w)
    agreement = float(np.clip(np.linalg.norm(mean), 0.0, 1.0))
    return float(np.clip(agreement * survival_frac, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Robust wing-plane normal + contaminant rejection (§5 step 4, S4b)
# ---------------------------------------------------------------------------


def _planarity_trust_weight(planarity: np.ndarray) -> np.ndarray:
    """Soft `[0,1]` trust weight from `planarity`, ramping over
    `[_PLANARITY_TRUST_LO, _PLANARITY_TRUST_HI]` (see those constants)."""
    planarity = np.asarray(planarity, dtype=float)
    lo, hi = _PLANARITY_TRUST_LO, _PLANARITY_TRUST_HI
    return np.clip((planarity - lo) / (hi - lo), 0.0, 1.0)


_NORMAL_RANSAC_ITERS = 100
"""RANSAC iterations for `_robust_wing_normal_and_survivors`'s normal
estimate. A single (non-RANSAC) weighted mean over *all* trusted points has
~0% breakdown point: a contaminant fraction as small as ~20-30% can drag the
mean far enough that those same contaminants end up back within
`_REJECT_ANGLE_DEG` of it, so nothing gets rejected (measured empirically
while developing `tests/test_s4b.py`). Minimal-sample (1-point) RANSAC over
the trusted points' own local normals -- pick one point as a candidate `n_w`,
count how many other trusted points fall within `_REJECT_ANGLE_DEG` of it,
keep the candidate with the most support -- finds the majority cluster
directly instead of averaging into the middle of two clusters; this is the
same trim-and-refit idea `geometry.fit_plane`/`fit_line` already use
elsewhere in this codebase, just with a 1-point minimal sample since a
"local normal" hypothesis needs only one point, not `fit_plane`'s 3 or
`fit_line`'s 2."""


def _robust_wing_normal_and_survivors(
    orientation: np.ndarray, planarity: np.ndarray, fallback_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """§5 step 4: a planarity-trust-weighted robust wing-plane normal `n_w`,
    plus a boolean survivor mask (True = kept, same length as `orientation`).

    Points with `_planarity_trust_weight` 0 never contribute to `n_w` and can
    never be rejected as contaminants either -- their own normal isn't
    trusted enough to make that call (calc_kinematics.md §1: orientation is
    only a reliable normal proxy on (near-)flat points). `orientation`'s
    per-point sign ambiguity is resolved once, against `fallback_normal` (the
    plane-fit normal from the *unfiltered*, possibly-contaminated cloud),
    before RANSAC.

    `n_w` itself comes from `_NORMAL_RANSAC_ITERS` rounds of minimal-sample
    (1-point) RANSAC over the trusted points' oriented local normals (see
    `_NORMAL_RANSAC_ITERS` docstring for why a plain weighted mean isn't
    robust enough), refined by a final planarity-trust-weighted mean over
    the winning candidate's inliers. Uses a fixed internal RNG seed, like
    `geometry.fit_plane`/`fit_line`'s own RANSAC default, for reproducible
    frame-to-frame output.

    Falls back to `fallback_normal` with no rejection (all-True survivor
    mask) if fewer than `_MIN_TRUSTED_POINTS` points are trusted, or if the
    winning cluster's weighted mean happens to cancel out (degenerate).
    """
    n = orientation.shape[0]
    fallback_normal = fallback_normal / np.linalg.norm(fallback_normal)
    trust_w = _planarity_trust_weight(planarity)
    trusted = trust_w > 0.0
    trusted_idx = np.nonzero(trusted)[0]

    if trusted_idx.size < _MIN_TRUSTED_POINTS:
        return fallback_normal, np.ones(n, dtype=bool)

    oriented = geo.orient_to_reference(orientation, fallback_normal)

    rng = np.random.default_rng(0)
    best_inliers, best_count = trusted_idx, -1
    for _ in range(_NORMAL_RANSAC_ITERS):
        seed_idx = trusted_idx[rng.integers(trusted_idx.size)]
        cos_ang = np.clip(oriented[trusted_idx] @ oriented[seed_idx], -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_ang))
        inlier_local = angle_deg <= _REJECT_ANGLE_DEG
        count = int(inlier_local.sum())
        if count > best_count:
            best_count = count
            best_inliers = trusted_idx[inlier_local]

    n_w = np.average(oriented[best_inliers], axis=0, weights=trust_w[best_inliers])
    norm_n_w = np.linalg.norm(n_w)
    if norm_n_w < _EPS:
        return fallback_normal, np.ones(n, dtype=bool)
    n_w = n_w / norm_n_w

    cos_ang = np.clip(oriented @ n_w, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_ang))
    reject = trusted & (angle_deg > _REJECT_ANGLE_DEG)
    return n_w, ~reject


# ---------------------------------------------------------------------------
# eta (§5, stroke-plane frame, cell 3 `sp_chord`/`le_sp_normal`)
# ---------------------------------------------------------------------------


def _le_sp_normal(n_sp: np.ndarray, le_dir: np.ndarray, sign_left: float) -> np.ndarray:
    v = np.cross(n_sp, le_dir) if sign_left > 0 else np.cross(le_dir, n_sp)
    return v / np.linalg.norm(v)


def _sp_chord_axis(le_dir: np.ndarray, le_sp_normal: np.ndarray) -> np.ndarray:
    v = np.cross(le_dir, le_sp_normal)
    return v / np.linalg.norm(v)


def _eta(chord: np.ndarray, le_dir: np.ndarray, n_sp: np.ndarray, sign_left: float) -> float:
    """Core §5 formula, isolated so it can be unit-tested directly against an
    inline reimplementation of `reference/python_snippets.py` cell 3
    (`sp_chord`/`le_sp_normal` + `psi`'s `atan2`, minus the rejected sign
    patch -- see module docstring / calc_kinematics.md §5, §7).
    """
    le_spn = _le_sp_normal(n_sp, le_dir, sign_left)
    sp_chord = _sp_chord_axis(le_dir, le_spn)
    x = float(np.dot(chord, le_spn))
    y = float(np.dot(chord, sp_chord))
    return math.degrees(math.atan2(sign_left * y, x))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def estimate_chord(
    wing_xyz: np.ndarray,
    body_frame: BodyFrame,
    side: str,
    leading_edge: LeadingEdge | None = None,
    weights: np.ndarray | None = None,
    robust: bool = False,
    use_gaussian_normals: bool = False,
    orientation: np.ndarray | None = None,
    planarity: np.ndarray | None = None,
) -> ChordResult:
    """One wing's chord direction and pitch (`eta`), one frame (§5).

    `leading_edge` reuses an already-computed `LeadingEdge` (e.g. from
    `wing_angles.stroke_deviation`) instead of re-fitting it; if omitted,
    `wing_angles.estimate_leading_edge(wing_xyz, body_frame, side,
    weights=weights)` is called. `weights` only affects LE/plane fitting, not
    bin aggregation.

    `orientation` (`(N,3)`) / `planarity` (`(N,)`) are new, optional, and
    row-aligned with `wing_xyz` (e.g. via `io_schema.get_part_columns`); only
    `use_gaussian_normals=True` reads them. Both `robust`/`use_gaussian_normals`
    default to `False`, and with both `False` this function is byte-identical
    to the S4a baseline (`_aggregate_chords`, unweighted `chord_conf`, no
    point rejection) regardless of `orientation`/`planarity` -- existing
    call sites (`pipeline.py`) are unaffected.

    `use_gaussian_normals=True`: builds a robust wing-plane normal `n_w` from
    planarity-trust-weighted `orientation` (§5 step 4,
    `_robust_wing_normal_and_survivors`), rejects points disagreeing with it
    beyond `_REJECT_ANGLE_DEG`, and recomputes the leading edge / bins on the
    survivors. Falls back to no rejection (with a warning) if `orientation`/
    `planarity` aren't supplied, too few points are planarity-trusted, or too
    few points survive to refit a leading edge.

    `robust=True`: swaps the plain-mean bin aggregator for a count/planarity
    -weighted, trimmed one (`_aggregate_chords_robust`) and makes `chord_conf`
    a real agreement-times-survival score (`_chord_confidence_robust`) instead
    of S4a's placeholder. Independent of `use_gaussian_normals` -- with no
    rejection, "survival" is just 1.0.

    Raises:
        ValueError: unknown `side`, or no span bin has >= `_MIN_BIN_POINTS`
            points (via `estimate_leading_edge` / `_bin_chords_core`), even
            after any `use_gaussian_normals` fallback.
    """
    _check_side(side)
    wing_xyz = np.asarray(wing_xyz, dtype=float)
    n = wing_xyz.shape[0]
    sign_left = _SIGN_LEFT[side]

    base_leading_edge = leading_edge if leading_edge is not None else estimate_leading_edge(
        wing_xyz, body_frame, side, weights=weights
    )

    active_xyz = wing_xyz
    active_planarity = None if planarity is None else np.asarray(planarity, dtype=float)
    survivors = np.ones(n, dtype=bool)
    active_leading_edge = base_leading_edge

    if use_gaussian_normals:
        if orientation is None or planarity is None:
            logger.warning(
                "estimate_chord(use_gaussian_normals=True): no orientation/planarity supplied; "
                "skipping contaminant rejection (see chord.py docstring for the expected inputs)."
            )
        else:
            orientation = np.asarray(orientation, dtype=float)
            _n_w, candidate_survivors = _robust_wing_normal_and_survivors(
                orientation, active_planarity, base_leading_edge.plane_normal
            )
            if not candidate_survivors.all():
                candidate_xyz = wing_xyz[candidate_survivors]
                candidate_weights = None if weights is None else np.asarray(weights, dtype=float)[candidate_survivors]
                try:
                    active_leading_edge = estimate_leading_edge(
                        candidate_xyz, body_frame, side, weights=candidate_weights
                    )
                except ValueError as e:
                    logger.warning(
                        "estimate_chord(use_gaussian_normals=True): contaminant rejection left too "
                        "few points to refit a leading edge (%s); keeping all points instead.", e
                    )
                else:
                    survivors = candidate_survivors
                    active_xyz = candidate_xyz
                    active_planarity = active_planarity[candidate_survivors]

    chord_axis_te = _oriented_chord_axis(
        active_xyz, active_leading_edge.plane_normal, active_leading_edge.le_dir, active_leading_edge.inlier_mask
    )
    per_bin_chords, bin_counts, bin_planarity = _bin_chords_core(
        active_xyz, active_leading_edge.le_dir, chord_axis_te,
        n_bins=_N_SPAN_BINS, min_bin_points=_MIN_BIN_POINTS, planarity=active_planarity,
    )
    if per_bin_chords.shape[0] == 0:
        raise ValueError(
            f"estimate_chord: no span bin had >= {_MIN_BIN_POINTS} points; cannot form a chord"
        )

    if robust:
        bin_weights = bin_counts if bin_planarity is None else bin_counts * np.clip(bin_planarity, 0.0, 1.0)
        chord, trimmed_weights = _aggregate_chords_robust(per_bin_chords, bin_weights)
        survival_frac = float(active_xyz.shape[0]) / n if n > 0 else 1.0
        chord_conf = _chord_confidence_robust(per_bin_chords, trimmed_weights, survival_frac)
    else:
        chord = _aggregate_chords(per_bin_chords)
        chord_conf = float(np.clip(np.linalg.norm(per_bin_chords.mean(axis=0)), 0.0, 1.0))

    eta = _eta(chord, active_leading_edge.le_dir, body_frame.n_sp, sign_left)

    return ChordResult(
        chord=chord,
        eta=eta,
        chord_conf=chord_conf,
        per_bin_chords=per_bin_chords,
        n_bins_used=per_bin_chords.shape[0],
        rejected_mask=~survivors,
    )
