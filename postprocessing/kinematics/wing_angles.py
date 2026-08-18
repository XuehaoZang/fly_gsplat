"""S3 wing stroke & deviation angles (phi/theta), T4 kinematics.

Implements calc_kinematics.md §4 (wing stroke & deviation), building on
`geometry.py` primitives and the `BodyFrame` from `body_frame.py`
(specifically `x_body, y_body, n_sp, body_cm` -- nothing else from
`BodyFrame` is used here). Chord/eta (§5) is **not** implemented here -- see
S4.

Geometry reproduced from `reference/python_snippets.py` cell 3
(`project_on_plane` / `calculate_phi`), reference-only, never imported:
stroke-plane in-plane basis via plane projection, then
`phi = atan2(sign_left*(span.y_sp), span.x_sp)`, `theta = 90 - arccos(n_sp.span)`.
Angles are returned as raw single-frame degrees -- no `unwrap` (that is a
multi-frame concern the notebook applies across a whole trajectory; a lone
frame has nothing to unwrap against).

**S3 revision:** `phi`/`theta` are computed from `estimate_span`'s wing PCA
major axis (`spanHat` in MATLAB `calcAnglesRaw_Sam.m`,
`reference/matlab_snippets.m` lines ~190-195), not the leading edge -- see
`estimate_span` / `stroke_deviation`. `estimate_leading_edge` is unchanged
and still independently importable and side-symmetric; S4 (chord/eta) reuses
it as the source of the leading-edge axis `le_dir` (for chord LE->TE sign
only) and the wing-plane normal `plane_normal`.

**Velocity cue (opt-in, `correct_wing_pitch/diag_report.md` §8):** diagnosis
found that `estimate_leading_edge`'s pos/neg RANSAC-inlier-count "winner"
call is decided independently each frame and is unstable exactly at
near-tied counts (`winner_flip` -> chord-axis sign flip, OR=12.53,
p=3.3e-14 -- §8 of the report), and that none of the per-frame static
confidence measures tried (`margin_count`, `curvature_diff`, `axis_margin`)
predict which frames flip. `prev_tip`/`prev_body_cm` (both default `None`,
meaning "not supplied" -- with either omitted, `estimate_leading_edge`'s
output is byte-identical to before this revision) let a caller pass the
*previous* frame's own `LeadingEdge.span_tip` and body centroid so this
frame's winner call can be cross-checked against wing-tip velocity,
mirroring `reference/wing_pitch/find_chords_quad.m`'s `vWing`-based chord
swap: MATLAB's own `WingTip` there comes from the span direction
(PCA/farthest-point), *not* from whichever candidate the LE/TE call already
picked, so this implementation follows the same principle -- the velocity
anchor (`span_tip`, see `LeadingEdge`) is computed once, before the pos/neg
split, from the plane inliers' own initial span-axis guess, never from the
winning candidate's own RANSAC line fit. That independence matters: an
anchor that *did* depend on the winner (e.g. the final `le_dir`'s own tip)
would jump to a different physical edge of the wing exactly when the winner
call flips, manufacturing a spurious "high-speed" reading precisely on the
unstable frames this cue exists to fix -- an early version of this cue used
such an anchor and, measured on real data, made `winner_flip` rates *worse*
at high inferred speed, not better (`correct_wing_pitch/diag/09_velocity_cue_validation_summary.md`
records this). The anchor is also never a `chord.py` axis quantity -- that
axis is separately downstream of the winner call
(`chord.py::_oriented_chord_axis` reads `le_dir`'s inlier mask directly), so
using it here would close the loop back onto the unstable signal in a
different way. Also unlike the MATLAB reference, there is deliberately no
low-speed static fallback here: diagnosis already showed the available
static measures don't predict flips, so a low-speed frame's winner call is
left exactly as the count judge decided it (near stroke reversal, real
wingtip speed is close to zero anyway, so a velocity cue has nothing
reliable to say there -- that regime is left to a future cross-frame
hysteresis layer, out of scope here). See `_velocity_cue_winner`.

**Validated effect (`correct_wing_pitch/diag/09_velocity_cue_validation_summary.md`,
real 100-frame G2b_G9 dataset):** the low-speed no-op guarantee holds
exactly (0 mismatches on every ineligible frame, both sides). At high speed,
results are *not* a clean win: side L's cue-touched-transition winner_flip
rate and wrap-crossing count both improve (58.1%->51.2%, 31->27
wrap-crossings), but side R's get worse (51.1%->73.3%, 29->33
wrap-crossings) at the same default `velocity_threshold_scale`, and this
asymmetry persisted across a threshold sweep (10-15x), not just the default
8x -- i.e. the chordwise-alignment signal this cue relies on is only
weakly, inconsistently informative on this dataset, not a reliable fix.
This is why the cue is opt-in only and **not** wired into `pipeline.py`'s
default config; treat it as a diagnosed-but-unresolved mechanism, not a
validated correction, until further work (e.g. a true MATLAB-style
chord-vector comparison instead of the coarse `chord_axis`, or confidence
weighting on alignment magnitude) revisits it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as geo
from .body_frame import BodyFrame

_SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}
"""§4: sign_left = -1 for wing_L, +1 for wing_R."""

VELOCITY_THRESHOLD_SCALE_DEFAULT = 8.0
"""Default `velocity_threshold_scale` for `estimate_leading_edge`'s velocity
cue: the cue only overrides the count judge when the (body-motion-corrected,
span-perpendicular) wingtip displacement between `prev_tip` and this frame's
own tip exceeds this many multiples of the wing point cloud's own median
nearest-neighbor spacing (the same density-derived `scale` already used for
this function's RANSAC thresholds) -- a spatial-scale multiple rather than a
hardcoded absolute distance, so it stays meaningful across datasets shot at
different point densities. Tuned empirically against the real 100-frame
G2b_G9 dataset in `correct_wing_pitch/diag/09_velocity_cue_validation.py`
(see that script's summary for the sweep) to sit clear of near-reversal
noise while still firing on genuine mid-stroke motion."""


def _check_side(side: str) -> None:
    if side not in _SIGN_LEFT:
        raise ValueError(f"side must be 'wing_L' or 'wing_R', got {side!r}")


# ---------------------------------------------------------------------------
# Leading-edge estimation
# ---------------------------------------------------------------------------


@dataclass
class LeadingEdge:
    """One wing's estimated leading-edge / span geometry, one frame.

    `le_dir` is unit, span-like, oriented outward (away from the body,
    `dot(le_dir, wing_centroid - body_cm) >= 0`). `root`/`tip` are actual
    input points (not synthesized): the RANSAC-inlier leading-edge point with
    the smallest/largest projection onto `le_dir`. `inlier_mask` is boolean,
    shape matching the `wing_xyz` passed to `estimate_leading_edge`: True for
    points used as leading-edge-line RANSAC inliers, False for everything
    else (trailing-edge points, interior points, RANSAC-rejected outliers).
    `plane_normal` is the wing-plane unit normal from the *same* RANSAC plane
    fit `le_dir` was derived from (sign unspecified, per `geo.fit_plane`) --
    not consumed by phi/theta here, but exposed so S4 (`chord.py`) doesn't
    need a second, possibly-divergent plane fit.

    `span_tip` (module docstring's velocity cue) is the wing-plane-inlier
    point farthest along the *initial* span guess (the plane inliers' own
    PCA major axis, oriented outward) -- computed once, before the pos/neg
    winner split, exactly like `reference/wing_pitch/find_chords_quad.m`'s
    own `WingTip` (PCA/farthest-point, independent of LE/TE identity). This
    is deliberately a different point from `tip` (which *is* downstream of
    the winner call, by definition -- it's the winning candidate's own
    RANSAC-inlier extreme): a caller chaining `tip` frame-to-frame as a
    velocity anchor would see it jump to a different physical edge of the
    wing exactly when the winner call flips, producing a spurious
    "high-speed" reading precisely on the unstable frames the cue exists to
    fix. `span_tip` has no such dependency, so it's what `prev_tip` (this
    function's parameter) is meant to hold across calls.
    """

    le_dir: np.ndarray
    tip: np.ndarray
    root: np.ndarray
    inlier_mask: np.ndarray
    plane_normal: np.ndarray
    span_tip: np.ndarray


def _velocity_cue_winner(
    span_guess: np.ndarray,
    chord_axis: np.ndarray,
    span_tip: np.ndarray,
    use_pos_count: bool,
    prev_tip: np.ndarray,
    prev_body_cm: np.ndarray,
    body_cm: np.ndarray,
    scale: float,
    velocity_threshold_scale: float,
) -> bool:
    """Cross-frame override for the pos/neg winner call (module docstring's
    "Velocity cue").

    `span_tip` is the caller's winner-*independent* wingtip anchor (see
    `LeadingEdge.span_tip`'s docstring for why it -- not the winning
    candidate's own tip -- is what must be differenced frame-to-frame).
    `span_guess`/`chord_axis` are the caller's already-computed initial
    span-axis guess and `cross(plane_normal, span_guess)` in-plane axis; by
    construction (the caller's own per-bin `argmax`/`argmin` split that built
    `pos_idx_local`/`neg_idx_local`), positive `chord_axis` is the `pos` side
    and negative is `neg`, so it is a direct pos-vs-neg discriminator once a
    velocity direction is in hand.

    Only fires (i.e. can return something other than `use_pos_count`) when
    the estimated wingtip speed clears `velocity_threshold_scale * scale`;
    below that, returns `use_pos_count` unchanged (module docstring: no
    static low-speed fallback -- that regime is left as-is here).
    """
    if scale <= 0.0:
        return use_pos_count

    raw_delta = (span_tip - prev_tip) - (body_cm - prev_body_cm)
    comp = float(np.dot(raw_delta, span_guess))
    perp = raw_delta - comp * span_guess
    speed = float(np.linalg.norm(perp))

    if speed < velocity_threshold_scale * scale:
        return use_pos_count

    v_hat = perp / speed
    align = float(np.dot(v_hat, chord_axis))
    return align > 0.0


def estimate_leading_edge(
    wing_xyz: np.ndarray,
    body_frame: BodyFrame,
    side: str,
    weights: np.ndarray | None = None,
    *,
    n_bins: int = 20,
    min_bin_points: int = 3,
    plane_threshold: float | None = None,
    line_threshold: float | None = None,
    rng: int | np.random.Generator | None = 0,
    prev_tip: np.ndarray | None = None,
    prev_body_cm: np.ndarray | None = None,
    velocity_threshold_scale: float = VELOCITY_THRESHOLD_SCALE_DEFAULT,
) -> LeadingEdge:
    """Estimate one wing's leading-edge direction from its point cloud.

    For S3, "leading edge" is approximated by the wing's span axis: fit the
    wing **plane** (RANSAC, robust to stray/mislabeled points), then within
    that plane build two *edge-candidate* point sets by binning along an
    initial span-axis guess (the plane inliers' own PCA major axis) and, in
    each populated bin, taking the two points most extreme along the
    perpendicular in-plane (chordwise) axis -- one extreme set per side.

    Which side is "leading" is decided by **straightness**, not distance from
    the body: `mock.py::make_wing_points` documents (and real fly wings have)
    a structurally rigid, nearly-straight leading edge (costal vein) versus a
    flexible, gently-curved trailing edge. Concretely: a RANSAC line is fit
    to *each* candidate set independently, and whichever gets more inliers
    (ties broken by lower mean inlier residual) is taken as leading -- an
    inlier-*count* comparison, not a plain least-squares residual comparison,
    because a single contaminating point (e.g. a mislabeled point from the
    other wing, or a stray floater near the plane) can otherwise dominate an
    unweighted least-squares fit's direction even in an 18-20 point set (that
    single point's leverage was measured to flip the straighter-edge call in
    testing); RANSAC keeps that point from ever entering the fit instead of
    just discounting its residual after the fact. This is a single-frame,
    phase-invariant heuristic that does not depend on stroke direction,
    unlike a distance-from-body rule, which flips between up- and
    down-stroke; S4 refines LE vs TE further alongside chord extraction. The
    winning side's own RANSAC line *is* the final `le_dir` fit (no redundant
    second pass).

    `le_dir` is finally oriented outward via `dot(le_dir, wing_centroid -
    body_cm) >= 0` (§ spec). RANSAC thresholds default to a multiple of the
    whole `wing_xyz` cloud's median nearest-neighbor spacing (`plane`: 2x,
    `line`: 1.5x) -- a density-derived scale that stays meaningful under a
    modest fraction of far-flung outliers (each clean point's nearest
    neighbor is still almost always another clean point), unlike a PCA
    eigenvalue computed over the whole (possibly contaminated) cloud; both
    are overridable.

    `prev_tip`/`prev_body_cm` (both default `None`) opt into the module
    docstring's velocity cue: when both are supplied, the pos/neg winner call
    above is cross-checked against wing-tip velocity and can be overridden at
    high speed (see `_velocity_cue_winner`). `prev_tip` must be the
    *previous* frame's own `LeadingEdge.span_tip` (not `.tip` -- see that
    field's docstring for why), `prev_body_cm` the previous frame's
    `BodyFrame.body_cm`. With either omitted (the default), this function's
    output is unaffected -- same call sequence, same RNG draws,
    byte-identical to before the cue existed.

    Raises:
        ValueError: unknown `side`, or fewer than 3 populated bins on either
            candidate edge (can't compare candidates / fit a line).
    """
    _check_side(side)
    wing_xyz = np.asarray(wing_xyz, dtype=float)
    n = wing_xyz.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    tree = cKDTree(wing_xyz)
    nn_dist, _ = tree.query(wing_xyz, k=min(2, n))
    scale = float(np.median(nn_dist[:, -1])) if n > 1 else 0.0
    plane_thresh = plane_threshold if plane_threshold is not None else 2.0 * scale
    line_thresh = line_threshold if line_threshold is not None else 1.5 * scale

    normal, _, plane_mask = geo.fit_plane(wing_xyz, w, method="ransac", threshold=plane_thresh, rng=rng)

    idx_plane = np.nonzero(plane_mask)[0]
    pts_plane = wing_xyz[idx_plane]
    w_plane = w[idx_plane]

    wing_centroid = wing_xyz.mean(axis=0)
    out_ref = wing_centroid - np.asarray(body_frame.body_cm, dtype=float)

    _, eigvecs_plane, plane_centroid = geo.weighted_pca(pts_plane, w_plane)
    span_guess = geo.orient_to_reference(eigvecs_plane[:, -1], out_ref)
    chord_axis = np.cross(normal, span_guess)
    chord_axis = chord_axis / np.linalg.norm(chord_axis)

    t = (pts_plane - plane_centroid) @ span_guess
    c = (pts_plane - plane_centroid) @ chord_axis
    # Winner-independent velocity anchor (module docstring's velocity cue /
    # `LeadingEdge.span_tip`): the plane inlier farthest along `span_guess`,
    # computed before the pos/neg split below ever runs.
    span_tip = pts_plane[np.argmax(t)]

    bin_edges = np.linspace(t.min(), t.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(t, bin_edges[1:-1]), 0, n_bins - 1)

    pos_idx_local, neg_idx_local = [], []
    for b in range(n_bins):
        in_bin = np.nonzero(bin_idx == b)[0]
        if in_bin.size < min_bin_points:
            continue
        pos_idx_local.append(in_bin[np.argmax(c[in_bin])])
        neg_idx_local.append(in_bin[np.argmin(c[in_bin])])

    if len(pos_idx_local) < 3 or len(neg_idx_local) < 3:
        raise ValueError(
            "estimate_leading_edge: not enough populated span bins "
            f"({len(pos_idx_local)} pos / {len(neg_idx_local)} neg, need >=3 each) "
            "to disambiguate leading vs trailing edge"
        )
    pos_idx_local = np.array(pos_idx_local)
    neg_idx_local = np.array(neg_idx_local)

    def _ransac_candidate(local_idx: np.ndarray):
        orig_idx = idx_plane[local_idx]
        pts = wing_xyz[orig_idx]
        wts = w[orig_idx]
        direction, point_on_line, mask = geo.fit_line(
            pts, wts, method="ransac", threshold=line_thresh, min_inliers=2, rng=rng
        )
        rel = pts[mask] - point_on_line
        proj = rel @ direction
        perp = rel - np.outer(proj, direction)
        mean_resid = float(np.mean(np.linalg.norm(perp, axis=1)))
        return direction, point_on_line, mask, orig_idx, pts, int(mask.sum()), mean_resid

    pos_cand = _ransac_candidate(pos_idx_local)
    neg_cand = _ransac_candidate(neg_idx_local)
    _, _, _, _, _, pos_count, pos_resid = pos_cand
    _, _, _, _, _, neg_count, neg_resid = neg_cand
    use_pos = pos_count > neg_count or (pos_count == neg_count and pos_resid < neg_resid)

    if prev_tip is not None and prev_body_cm is not None:
        use_pos = _velocity_cue_winner(
            span_guess, chord_axis, span_tip, use_pos,
            prev_tip=np.asarray(prev_tip, dtype=float),
            prev_body_cm=np.asarray(prev_body_cm, dtype=float),
            body_cm=np.asarray(body_frame.body_cm, dtype=float),
            scale=scale,
            velocity_threshold_scale=velocity_threshold_scale,
        )

    direction, point_on_line, ransac_mask, le_orig_idx, le_points, _, _ = pos_cand if use_pos else neg_cand
    le_dir = geo.orient_to_reference(direction, out_ref)

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[le_orig_idx[ransac_mask]] = True

    inlier_points = le_points[ransac_mask]
    t_final = (inlier_points - point_on_line) @ le_dir
    root = inlier_points[np.argmin(t_final)]
    tip = inlier_points[np.argmax(t_final)]

    return LeadingEdge(
        le_dir=le_dir, tip=tip, root=root, inlier_mask=inlier_mask, plane_normal=normal, span_tip=span_tip
    )


# ---------------------------------------------------------------------------
# Span estimation (wing PCA major axis -- MATLAB `spanHat`)
# ---------------------------------------------------------------------------


def estimate_span(
    wing_xyz: np.ndarray,
    body_frame: BodyFrame,
    side: str,
    weights: np.ndarray | None = None,
    *,
    plane_threshold: float | None = None,
    rng: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Wing span direction (root -> tip), one wing, one frame.

    This is MATLAB `calcAnglesRaw_Sam.m`'s `spanHat` (`reference/matlab_snippets.m`
    lines ~190-195): the wing's own PCA major axis, **not** the leading edge.
    `phi`/`theta` (§4) are computed from this vector (see `stroke_deviation`);
    `estimate_leading_edge`'s `le_dir` remains the source of the chord LE->TE
    sign in `chord.py` and is untouched by this function.

    Fits the wing plane (RANSAC, the same construction `estimate_leading_edge`
    uses for its own plane fit -- a density-derived threshold that is robust
    to a modest fraction of far-flung outlier/contaminant points, unlike a
    raw whole-cloud PCA), then takes the plane inliers' PCA major axis,
    oriented outward via `dot(span_dir, wing_centroid - body_cm) >= 0` (same
    convention as `le_dir`).

    Deliberately independent of `estimate_leading_edge` -- no shared internal
    state -- so this stays a separate, directly testable helper per the wing
    span/leading-edge distinction this revision introduces.
    """
    _check_side(side)
    wing_xyz = np.asarray(wing_xyz, dtype=float)
    n = wing_xyz.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    tree = cKDTree(wing_xyz)
    nn_dist, _ = tree.query(wing_xyz, k=min(2, n))
    scale = float(np.median(nn_dist[:, -1])) if n > 1 else 0.0
    thresh = plane_threshold if plane_threshold is not None else 2.0 * scale

    _, _, plane_mask = geo.fit_plane(wing_xyz, w, method="ransac", threshold=thresh, rng=rng)
    idx_plane = np.nonzero(plane_mask)[0]
    pts_plane = wing_xyz[idx_plane]
    w_plane = w[idx_plane]

    _, eigvecs_plane, _ = geo.weighted_pca(pts_plane, w_plane)
    span_axis = eigvecs_plane[:, -1]

    wing_centroid = wing_xyz.mean(axis=0)
    out_ref = wing_centroid - np.asarray(body_frame.body_cm, dtype=float)
    return geo.orient_to_reference(span_axis, out_ref)


# ---------------------------------------------------------------------------
# Stroke / deviation angles (phi, theta)
# ---------------------------------------------------------------------------


@dataclass
class WingSweep:
    """One wing's stroke-plane angles, one frame (§4).

    `phi`/`theta` are degrees, raw (not unwrapped -- unwrapping needs a
    trajectory across frames, not a single frame), computed from `span_dir`
    (`estimate_span`'s wing-PCA major axis -- MATLAB `spanHat`), not the
    leading edge. `leading_edge` is still the `LeadingEdge` fit (kept for
    `chord.py`'s LE->TE sign, unaffected by this change); `span_dir` is the
    vector `phi`/`theta` were actually computed from.
    """

    phi: float
    theta: float
    leading_edge: LeadingEdge
    span_dir: np.ndarray


def _phi_theta(
    le_dir: np.ndarray,
    x_body: np.ndarray,
    y_body: np.ndarray,
    n_sp: np.ndarray,
    sign_left: float,
) -> tuple[float, float]:
    """Core §4 formula, isolated from LE estimation so it can be unit-tested
    directly against an inline reimplementation of `reference/python_snippets.py`
    cell 3 (`project_on_plane` + `calculate_phi`).

    `x_body`/`y_body` are projected onto the stroke plane (`n_sp`) to get the
    in-plane basis `x_sp, y_sp` (`geo.project_onto_plane` normalizes,
    matching the reference's `project_on_plane`), then `le_dir` is projected
    the same way for `phi`. `theta` uses the un-projected `le_dir` against
    `n_sp` directly (also matching the reference, which computes `theta`
    from the raw LE vector, not its in-plane projection).
    """
    x_sp = geo.project_onto_plane(x_body, n_sp)
    y_sp = geo.project_onto_plane(y_body, n_sp)
    le_sp = geo.project_onto_plane(le_dir, n_sp)

    xle = float(np.dot(le_sp, x_sp))
    yle = float(np.dot(le_sp, y_sp))
    phi = math.degrees(math.atan2(sign_left * yle, xle))

    cos_theta = np.clip(np.dot(n_sp, le_dir), -1.0, 1.0)
    theta = 90.0 - math.degrees(math.acos(cos_theta))
    return phi, theta


def stroke_deviation(
    wing_xyz: np.ndarray,
    body_frame: BodyFrame,
    side: str,
    weights: np.ndarray | None = None,
    **le_kwargs,
) -> WingSweep:
    """Stroke-plane azimuth (`phi`) and deviation (`theta`) for one wing, §4.

    `phi`/`theta` are computed from `estimate_span`'s wing PCA major axis
    (MATLAB `spanHat`, per `calcAnglesRaw_Sam.m`), not the leading edge -- see
    `_phi_theta` for the angle formulas (reproduced from
    `reference/python_snippets.py` cell 3 `calculate_phi`, with `le` replaced
    by `span_dir`). `leading_edge` is still fit (via `estimate_leading_edge`)
    and returned so `chord.py` can keep reusing it for the LE->TE chord sign,
    which stays leading-edge-based and untouched by this change. `weights`
    and any `**le_kwargs` (e.g. `n_bins`, `plane_threshold`, or the velocity
    cue's `prev_tip`/`prev_body_cm`/`velocity_threshold_scale` -- module
    docstring) pass straight through to `estimate_leading_edge` only --
    `estimate_span` takes just `weights`.
    """
    _check_side(side)
    le = estimate_leading_edge(wing_xyz, body_frame, side, weights=weights, **le_kwargs)
    span_dir = estimate_span(wing_xyz, body_frame, side, weights=weights)
    phi, theta = _phi_theta(
        span_dir, body_frame.x_body, body_frame.y_body, body_frame.n_sp, _SIGN_LEFT[side]
    )
    return WingSweep(phi=phi, theta=theta, leading_edge=le, span_dir=span_dir)
