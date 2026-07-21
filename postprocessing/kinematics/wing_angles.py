"""S3 wing stroke & deviation angles (phi/theta), T4 kinematics.

Implements calc_kinematics.md §4 (wing stroke & deviation), building on
`geometry.py` primitives and the `BodyFrame` from `body_frame.py`
(specifically `x_body, y_body, n_sp, body_cm` -- nothing else from
`BodyFrame` is used here). Chord/eta (§5) is **not** implemented here -- see
S4.

Geometry reproduced from `reference/python_snippets.py` cell 3
(`project_on_plane` / `calculate_phi`), reference-only, never imported:
stroke-plane in-plane basis via plane projection, then
`phi = atan2(sign_left*(le.y_sp), le.x_sp)`, `theta = 90 - arccos(n_sp.le)`.
Angles are returned as raw single-frame degrees -- no `unwrap` (that is a
multi-frame concern the notebook applies across a whole trajectory; a lone
frame has nothing to unwrap against).

`estimate_leading_edge` is independently importable and side-symmetric;
S4 (chord/eta) reuses it as the source of the span axis `le_dir` and the
wing-plane normal `plane_normal`.
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
    """

    le_dir: np.ndarray
    tip: np.ndarray
    root: np.ndarray
    inlier_mask: np.ndarray
    plane_normal: np.ndarray


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

    direction, point_on_line, ransac_mask, le_orig_idx, le_points, _, _ = pos_cand if use_pos else neg_cand
    le_dir = geo.orient_to_reference(direction, out_ref)

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[le_orig_idx[ransac_mask]] = True

    inlier_points = le_points[ransac_mask]
    t_final = (inlier_points - point_on_line) @ le_dir
    root = inlier_points[np.argmin(t_final)]
    tip = inlier_points[np.argmax(t_final)]

    return LeadingEdge(le_dir=le_dir, tip=tip, root=root, inlier_mask=inlier_mask, plane_normal=normal)


# ---------------------------------------------------------------------------
# Stroke / deviation angles (phi, theta)
# ---------------------------------------------------------------------------


@dataclass
class WingSweep:
    """One wing's stroke-plane angles, one frame (§4).

    `phi`/`theta` are degrees, raw (not unwrapped -- unwrapping needs a
    trajectory across frames, not a single frame). `leading_edge` is the
    `LeadingEdge` `phi`/`theta` were computed from.
    """

    phi: float
    theta: float
    leading_edge: LeadingEdge


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

    See `_phi_theta` for the angle formulas (reproduced from
    `reference/python_snippets.py` cell 3 `calculate_phi`). `weights` and any
    `**le_kwargs` (e.g. `n_bins`, `plane_threshold`) pass straight through to
    `estimate_leading_edge`.
    """
    _check_side(side)
    le = estimate_leading_edge(wing_xyz, body_frame, side, weights=weights, **le_kwargs)
    phi, theta = _phi_theta(
        le.le_dir, body_frame.x_body, body_frame.y_body, body_frame.n_sp, _SIGN_LEFT[side]
    )
    return WingSweep(phi=phi, theta=theta, leading_edge=le)
