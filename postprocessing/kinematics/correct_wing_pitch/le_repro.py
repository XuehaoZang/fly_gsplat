"""Independent reproduction of `wing_angles.estimate_leading_edge`'s internal
pos/neg RANSAC candidate-selection logic, instrumented to expose diagnostics
the real function computes but never returns: RANSAC inlier counts/margin,
tie-break residuals, plane-fit inlier fraction, and each candidate set's
pre-RANSAC arc-chord (straight vs curved) ratio.

This is a *re-implementation*, not an import of `estimate_leading_edge`'s
locals (those aren't module-level state, so there is nothing to import) --
every line below mirrors `wing_angles.py::estimate_leading_edge` step for
step, same call order, same default RANSAC `rng` handling, so that
`check_consistency.py` can verify byte-for-byte agreement of `le_dir`
between this module and the real one. Do not "fix" or diverge behavior here
even if something looks fixable -- this module exists to *observe* the
existing algorithm, not to change it (see task constraints in
`correct_wing_pitch/`'s governing task description).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics import geometry as geo  # noqa: E402
from postprocessing.kinematics.body_frame import BodyFrame  # noqa: E402

_SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}
_EPS = 1e-12


def _arc_chord_ratio(points: np.ndarray, order_key: np.ndarray) -> float:
    """arc length / chord length of `points`, ordered by ascending `order_key`
    (the span coordinate `t`). 1.0 for a perfectly straight, monotonically
    ordered set; >1.0 the more the polyline wiggles/backtracks. NaN if fewer
    than 2 points or the endpoints coincide (undefined chord).

    Computed on whichever point set is passed in -- callers must pass the
    pre-RANSAC candidate set (not a RANSAC-inlier subset) to avoid measuring
    "straightness of the points RANSAC already decided were straight."
    """
    if points.shape[0] < 2:
        return float("nan")
    order = np.argsort(order_key)
    pts = points[order]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = float(seg.sum())
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if chord < _EPS:
        return float("nan")
    return arc / chord


@dataclass
class LEDiag:
    """One wing/frame's leading-edge RANSAC candidate diagnostics.

    `le_dir`/`tip`/`root`/`inlier_mask`/`plane_normal` mirror
    `wing_angles.LeadingEdge` exactly (same values, for the consistency
    check). Everything else is new: `pos_*`/`neg_*` describe the two
    chordwise-extreme candidate edges *before* the winner is picked;
    `pos_orig_idx`/`neg_orig_idx` are each candidate's point indices into the
    original `wing_xyz` passed in (so external ground truth can be looked up
    per point without re-deriving the binning).
    """

    le_dir: np.ndarray
    tip: np.ndarray
    root: np.ndarray
    inlier_mask: np.ndarray
    plane_normal: np.ndarray
    use_pos: bool
    """True if the `pos` candidate (positive side of `chord_axis`) won."""

    pos_count: int
    neg_count: int
    margin_count: int
    """winner_count - loser_count (>=0)."""
    margin_ratio: float
    """winner_count / max(loser_count, 1) (>=1)."""
    pos_resid: float
    neg_resid: float
    """Mean RANSAC-inlier perpendicular residual for each candidate's own
    line fit -- the tie-break metric `estimate_leading_edge` falls back to
    when `pos_count == neg_count`."""

    pos_arc_chord: float
    neg_arc_chord: float
    """Each candidate's pre-RANSAC arc-chord ratio (see `_arc_chord_ratio`);
    computed on the *whole* candidate set (one point per populated span
    bin), never the RANSAC-winner's inlier subset -- so this cannot be
    circular with the count-based winner call."""
    winner_arc_chord: float
    loser_arc_chord: float
    curvature_diff: float
    """`neg_arc_chord - pos_arc_chord`. >0 means the `pos` candidate set is
    straighter (lower arc/chord) than `neg`."""

    plane_inlier_frac: float
    n_plane_points: int
    n_pos_candidates: int
    n_neg_candidates: int
    pos_orig_idx: np.ndarray
    neg_orig_idx: np.ndarray


def estimate_leading_edge_diag(
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
) -> LEDiag:
    """Re-implementation of `wing_angles.estimate_leading_edge` with extra
    diagnostics exposed. Same signature, same defaults, same exceptions.
    """
    if side not in _SIGN_LEFT:
        raise ValueError(f"side must be 'wing_L' or 'wing_R', got {side!r}")
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
    plane_inlier_frac = float(idx_plane.size) / n if n > 0 else float("nan")

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
            "estimate_leading_edge_diag: not enough populated span bins "
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
        arc_chord = _arc_chord_ratio(pts, t[local_idx])  # pre-RANSAC, full candidate set
        return direction, point_on_line, mask, orig_idx, pts, int(mask.sum()), mean_resid, arc_chord

    pos_cand = _ransac_candidate(pos_idx_local)
    neg_cand = _ransac_candidate(neg_idx_local)
    _, _, _, pos_orig_idx, _, pos_count, pos_resid, pos_arc_chord = pos_cand
    _, _, _, neg_orig_idx, _, neg_count, neg_resid, neg_arc_chord = neg_cand
    use_pos = pos_count > neg_count or (pos_count == neg_count and pos_resid < neg_resid)

    direction, point_on_line, ransac_mask, le_orig_idx, le_points, _, _, _ = pos_cand if use_pos else neg_cand
    le_dir = geo.orient_to_reference(direction, out_ref)

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[le_orig_idx[ransac_mask]] = True

    inlier_points = le_points[ransac_mask]
    t_final = (inlier_points - point_on_line) @ le_dir
    root = inlier_points[np.argmin(t_final)]
    tip = inlier_points[np.argmax(t_final)]

    winner_count, loser_count = (pos_count, neg_count) if use_pos else (neg_count, pos_count)
    winner_arc_chord, loser_arc_chord = (pos_arc_chord, neg_arc_chord) if use_pos else (neg_arc_chord, pos_arc_chord)

    return LEDiag(
        le_dir=le_dir, tip=tip, root=root, inlier_mask=inlier_mask, plane_normal=normal,
        use_pos=use_pos,
        pos_count=pos_count, neg_count=neg_count,
        margin_count=winner_count - loser_count,
        margin_ratio=winner_count / max(loser_count, 1),
        pos_resid=pos_resid, neg_resid=neg_resid,
        pos_arc_chord=pos_arc_chord, neg_arc_chord=neg_arc_chord,
        winner_arc_chord=winner_arc_chord, loser_arc_chord=loser_arc_chord,
        curvature_diff=neg_arc_chord - pos_arc_chord,
        plane_inlier_frac=plane_inlier_frac, n_plane_points=int(idx_plane.size),
        n_pos_candidates=int(pos_idx_local.size), n_neg_candidates=int(neg_idx_local.size),
        pos_orig_idx=pos_orig_idx, neg_orig_idx=neg_orig_idx,
    )
