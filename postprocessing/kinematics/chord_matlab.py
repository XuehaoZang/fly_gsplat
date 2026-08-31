"""Experimental alternative to `chord.py`'s eta (wing pitch) computation,
based on `reference/wing_pitch/find_chords_quad.m` instead of `chord.py`'s
own leading-edge RANSAC-straightness winner call.

**Why this exists**: `chord.py`'s `eta` depends on `wing_angles.
estimate_leading_edge`'s per-frame pos/neg winner call (which of two
candidate edge-point sets is "the leading edge", decided by comparing
RANSAC-line straightness). `correct_wing_pitch/` steps 08-12 (2026-08-31
investigation, see that directory's numbered diagnostics) found this winner
call flips frame-to-frame with no static per-frame predictor, producing
~180 deg eta jumps that `eta_unwrap.py`'s post-hoc `resolve_180_flip` can
only approximate-fix (assumes the two winner outcomes are exactly 180 deg
apart in eta, which step 11 measured is only true for ~85-88% of frames --
worse on wing_R, matching that side's worse post-unwrap results). Three
attempts to gate/fix a velocity-based cross-frame cue for the winner call
(steps 09, 11, 12: speed-only, speed+theta-rate, speed+phi-perpendicularity)
all landed close to a coin flip when they fired.

Step 13/14 tried a structurally different approach instead of patching the
existing one: MATLAB's `find_chords_quad.m` doesn't classify edge points into
LE/TE candidates by straightness at all. It takes a band of points near
mid-span, finds the single farthest-apart PAIR among the ones farthest from
the wing centroid (the "main" chord diagonal, `chord_hat`), then finds a
second "alternative" diagonal (`chord_alt_hat`) via the extremes
perpendicular to that first diagonal's own plane normal -- then picks
between them primarily by **diagonal length** (a real wing cross-section is
chordwise-elongated; if one diagonal is >=1.3x the other, MATLAB trusts it
outright), falling back to wingtip-velocity alignment only when the lengths
are close. On the real `ctrl_009_004_ratio3_sh0_dense_valid480` dataset
(step 13): the length ratio was decisive (>=1.3) on 78-84% of frames, agreed
with `chord.py`'s own pick 94% of the time when decisive (same physical
structure, sanity-checked), and (step 14) substituting it for `chord.py`'s
axis dropped wrap-crossings (|d(eta)|>90 deg) from ~160-177 to ~17-19 per
side, and plain `np.unwrap` span from 2000-4200 deg down to ~860-910 deg.
Re-testing with velocity actually enabled (step 14, after fixing a threshold
scale bug in the first run) did *not* improve on the length-only result --
consistent with steps 09/11/12's finding that this dataset's velocity signal
is unreliable -- so `use_velocity` defaults to `False` here.

**Status**: experimental, not validated to the depth `chord.py`'s S4a/S4b
baseline was (no unit tests, no synthetic-mock validation, only one real
dataset). Kept as an opt-in alternate pipeline entry point
(`pipeline.run_dataset_matlab_chord_eta`), not `chord.py`'s default.
`phi`/`theta` are untouched (still `wing_angles.stroke_deviation`) -- only
`eta`/`chord_conf` are replaced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform

from . import chord as ch
from . import wing_angles as wa
from .body_frame import BodyFrame
from .wing_angles import _SIGN_LEFT, _check_side

CHORD_FRACTION = 0.33
"""Top fraction of the mid-span band, by distance from wing centroid, used
to find the main diagonal -- matches MATLAB's `chordFraction`."""
DIAG_LENGTH_RATIO_DECISIVE = 1.3
"""MATLAB's own threshold: trust the longer diagonal outright above this ratio."""
MIN_BAND_POINTS = 5
VELOCITY_THRESHOLD_SCALE_DEFAULT = wa.VELOCITY_THRESHOLD_SCALE_DEFAULT


@dataclass
class MatlabChordResult:
    eta: float
    chord: np.ndarray
    """Unit, LE->TE oriented via frame-to-frame continuity (see `estimate_chord_matlab`)."""
    span_tip: np.ndarray
    """Winner-independent wingtip anchor (same construction as
    `wing_angles.LeadingEdge.span_tip`) -- pass as this side's own
    `prev_span_tip` on the next call to chain velocity/sign continuity."""
    diag1: float
    diag2: float
    ratio: float
    swapped: bool
    """Whether the "alternative" diagonal was chosen over the "main" one."""
    chord_conf: float
    """`clip(ratio / DIAG_LENGTH_RATIO_DECISIVE, 0, 1)` -- how decisively the
    length criterion favored the chosen diagonal (1.0 = at or past the
    "trust it outright" threshold). A different quantity from `chord.py`'s
    own `chord_conf` (per-bin agreement), not directly comparable."""


def _matlab_chords(wing_xyz: np.ndarray, span: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """`find_chords_quad.m`'s diagonal-finding logic (swap/velocity/sign
    handled by the caller). `delta` (mid-span band half-width) is a fraction
    of the wing's own span extent (0.12, widened to 0.36 on retry) rather
    than MATLAB's fixed absolute voxel units. Returns `None` on a degenerate
    band/plane (mirrors MATLAB's "empty chord" error path)."""
    centroid = wing_xyz.mean(axis=0)
    rel = wing_xyz - centroid
    t = rel @ span
    span_extent = t.max() - t.min()
    if span_extent < 1e-12:
        return None

    band_idx = None
    for frac in (0.12, 0.36):
        delta = frac * span_extent
        idx = np.nonzero(np.abs(t) < delta)[0]
        if idx.size >= MIN_BAND_POINTS:
            band_idx = idx
            break
    if band_idx is None:
        return None

    band_pts = wing_xyz[band_idx]
    dist_from_centroid = np.linalg.norm(band_pts - centroid, axis=1)
    order = np.argsort(dist_from_centroid)[::-1]
    n_select = max(2, int(np.ceil(band_idx.size * CHORD_FRACTION)))
    selected = band_pts[order[:n_select]]
    if selected.shape[0] < 2:
        return None

    dmat = squareform(pdist(selected))
    i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
    raw_main = selected[i] - selected[j]
    chord_hat = raw_main - span * np.dot(span, raw_main)
    diag1 = float(np.linalg.norm(chord_hat))
    if diag1 < 1e-12:
        return None
    chord_hat = chord_hat / diag1

    wing_norm = np.cross(span, chord_hat)
    wn_norm = np.linalg.norm(wing_norm)
    if wn_norm < 1e-12:
        return None
    wing_norm = wing_norm / wn_norm

    band_rel = band_pts - centroid
    proj = band_rel @ wing_norm
    i_max, i_min = int(np.argmax(proj)), int(np.argmin(proj))
    if proj[i_max] <= 0 or proj[i_min] >= 0:
        return None
    raw_alt = band_pts[i_max] - band_pts[i_min]
    chord_alt_hat = raw_alt - span * np.dot(span, raw_alt)
    diag2 = float(np.linalg.norm(chord_alt_hat))
    if diag2 < 1e-12:
        return None
    chord_alt_hat = chord_alt_hat / diag2

    return chord_hat, chord_alt_hat, diag1, diag2


def estimate_chord_matlab(
    wing_xyz: np.ndarray,
    body_frame: BodyFrame,
    side: str,
    span: np.ndarray,
    prev_signed_chord: np.ndarray | None = None,
    prev_span_tip: np.ndarray | None = None,
    prev_body_cm: np.ndarray | None = None,
    use_velocity: bool = False,
    velocity_threshold_scale: float = VELOCITY_THRESHOLD_SCALE_DEFAULT,
) -> MatlabChordResult:
    """One wing's chord/eta, MATLAB-diagonal style (see module docstring).

    `span` must be `wing_angles.estimate_span`'s output for this frame/side
    (winner-independent PCA axis; also stands in for `chord.py`'s `le_dir`
    when reading off `eta` via `chord._eta` -- that formula only needs *a*
    span-like axis to build its local in-plane basis, see that function).

    Sign is resolved by continuity: whichever sign of the chosen diagonal is
    closer to `prev_signed_chord` (this side's own previous-frame output,
    chain it through) is kept; with `prev_signed_chord=None` (first frame of
    a sequence), the sign is arbitrary (whichever the max-pairwise-distance
    search happened to produce) -- same caveat `eta_unwrap.resolve_180_flip`
    documents for its own frame-0 anchor.

    `use_velocity=False` (default, per module docstring's finding) skips
    `prev_span_tip`/`prev_body_cm` entirely, using only the length-ratio
    swap rule (`diag2/diag1 >= 1.3`; otherwise keeps the "main" diagonal, the
    same default MATLAB's own combination rule falls back to when velocity
    is unavailable). Set `True` to opt into the velocity fallback for the
    length-ambiguous case (needs `prev_span_tip`/`prev_body_cm`, both `None`
    on the first frame of a chained sequence).

    Raises:
        ValueError: unknown `side`, or the mid-span band/plane is degenerate
            (mirrors `find_chords_quad.m`'s "empty chord" error path).
    """
    _check_side(side)
    sign_left = _SIGN_LEFT[side]
    wing_xyz = np.asarray(wing_xyz, dtype=float)
    span = np.asarray(span, dtype=float)

    mc = _matlab_chords(wing_xyz, span)
    if mc is None:
        raise ValueError("estimate_chord_matlab: degenerate mid-span band/plane, cannot form a chord")
    chord_hat, chord_alt_hat, diag1, diag2 = mc
    ratio = max(diag1, diag2) / min(diag1, diag2)
    diag_swap_flag = (diag2 / diag1) >= DIAG_LENGTH_RATIO_DECISIVE

    wing_centroid = wing_xyz.mean(axis=0)
    span_tip = wing_xyz[np.argmax((wing_xyz - wing_centroid) @ span)]

    velocity_swap_flag = False
    speed = 0.0
    if use_velocity and prev_span_tip is not None and prev_body_cm is not None:
        tree_scale = _nn_scale(wing_xyz)
        threshold = velocity_threshold_scale * tree_scale
        raw_delta = (span_tip - prev_span_tip) - (np.asarray(body_frame.body_cm, dtype=float) - prev_body_cm)
        comp = float(np.dot(raw_delta, span))
        perp = raw_delta - comp * span
        speed = float(np.linalg.norm(perp))
        if speed > 1e-9:
            v_hat = perp / speed
            dot1 = float(np.dot(chord_hat, v_hat))
            dot2 = float(np.dot(chord_alt_hat, v_hat))
            if dot2 > dot1:
                velocity_swap_flag = True
            if dot1 < 0.0 and dot2 < 0.0 and speed >= threshold and not velocity_swap_flag:
                chord_hat = -chord_hat
        swap_flag = (velocity_swap_flag and speed >= threshold) or (diag_swap_flag and speed < threshold)
    else:
        swap_flag = diag_swap_flag

    chosen = chord_alt_hat if swap_flag else chord_hat
    if prev_signed_chord is not None and float(np.dot(chosen, prev_signed_chord)) < 0.0:
        chosen = -chosen

    eta = ch._eta(chosen, span, body_frame.n_sp, sign_left)
    chord_conf = float(np.clip(ratio / DIAG_LENGTH_RATIO_DECISIVE, 0.0, 1.0))

    return MatlabChordResult(
        eta=eta, chord=chosen, span_tip=span_tip,
        diag1=diag1, diag2=diag2, ratio=ratio, swapped=swap_flag, chord_conf=chord_conf,
    )


def _nn_scale(wing_xyz: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    n = wing_xyz.shape[0]
    tree = cKDTree(wing_xyz)
    nn_dist, _ = tree.query(wing_xyz, k=min(2, n))
    return float(np.median(nn_dist[:, -1])) if n > 1 else 0.0
