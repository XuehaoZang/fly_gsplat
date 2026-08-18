""""Plan A" body axis estimator: head/tail centroid method, no voxelization.

Port of the old MATLAB pipeline's `reference/body_axis/findBodyAxis_mk2.m`
(head/tail voxel centroids + largest-connected-component filtering) to the
current point-cloud (no-voxel-grid) representation.

`compute_x_body_zhead` was *tried* as a production `x_body` (`body_frame.py`)
and reverted -- **not currently wired in**. It is a self-contained,
single-frame method with no cross-frame state: it reuses this module's
`_head_tail_centroids` machinery, but sources its own `guide_axis` from the
*current frame's own* PCA major axis (sign-agnostic -- only used to split
the cloud into two candidate end-clusters, never as the final direction) and
disambiguates which cluster is the head by comparing the two resulting
centroids' `up`-axis component (higher = head), instead of
`orient_to_reference`-ing a raw eigenvector.

`correct_body_axis/diag/k_zhead_axis_timeseries.py` measured this on the
real 640-frame dataset and found it does **not** solve the instability it
was meant to fix -- it makes the adjacent-frame self-continuity flip count
*worse* than the raw-PCA baseline it would have replaced (65 vs 39 flips /
639 pairs), and the flips are still concentrated on the same near-degenerate
(`eigval_ratio` top decile) frames as the baseline's own flips (26% flip
rate there vs 7% elsewhere). Root cause: sourcing `guide_axis` from the
*current frame's own* PCA reintroduces exactly the direction-wobble problem
this whole module exists to sidestep (module docstring above) -- on a
near-disc body cloud, that guide is itself unstable, and an unstable guide
produces an unstable split even though `_head_tail_centroids` is a much
better *final*-axis estimator than a raw eigenvector once given a stable
guide. Only `compute_robust_x_body`'s cross-frame guide (previous frame's
own output, or a nearby anchor) has actually been shown (`h_robust_axis_timeseries.py`,
0/639 flips) to give the split a guide worth trusting.

`compute_robust_x_body` (below) is that three-tier design (guide from the
previous frame's own output, or a nearby T-pose anchor, or the same PCA-up
fallback) and remains a *diagnostic-only* comparison point -- same status as
`correct_body_axis/continuity.py` and friends, not wired into `pipeline.py`.
Wiring *this* method into production is the next real candidate, but it
needs sequence-level orchestration (previous-frame state, an anchor table)
that `pipeline.py`'s current per-frame-stateless design (`pipeline.py`'s own
module docstring) does not have yet -- not a drop-in one-line swap the way
`compute_x_body_zhead` was.

Why this exists (see `correct_body_axis/continuity.py` and
`postprocessing/labeling/motion/diag/flip_root_cause_check.py`, both already
diagnosed): `body_frame.py::estimate_body_frame`'s `x_body` is
`orient_to_reference(weighted_pca(body_xyz)[1][:, -1], up)` -- a PCA major
axis whose *direction itself* (not just its sign) goes unstable whenever the
body point cloud's first two eigenvalues are close (`eigval_ratio` near 1,
i.e. a near-disc/near-spherical cloud), which happens on ~6% of frames.
`continuity.py` patches this with cross-frame projection but is still built
on the same PCA eigenvector, so it inherits the eigenvector's own
instability on frames where the *plane itself* wobbles, not just its sign.

This module sidesteps PCA's eigenvector entirely. The body long axis is
instead defined geometrically: find the points furthest from the body
centroid, split them into a "head" cluster and a "tail" cluster along a
*guide* direction (`guide_axis` -- doesn't need to be precise, just roughly
right), keep only the largest spatially-connected blob on each end (so a
stray point near the wrong end, or split across e.g. an occluded gap, can't
drag the centroid), and take `axis = unit(head_cm - tail_cm)`. The sign
comes for free from which cluster is physically "head" vs "tail" -- there is
no post-hoc `orient_to_reference` sign guess anywhere in this algorithm,
which is the entire point: `guide_axis` only has to pick the right *half* of
the point cloud, a much easier problem than picking the right eigenvector
direction on a near-degenerate PCA.

`guide_axis` itself still has to come from somewhere every frame; that
sourcing policy (previous frame's already-good axis, or a nearby T-pose
anchor frame, or the same PCA-up fallback `body_frame.py` uses as a last
resort) is `compute_robust_x_body`'s job, not this module's problem to
solve once and for all -- see its docstring. Sequence-level orchestration
(which frame gets which kind of guide) lives in
`correct_body_axis/diag/h_robust_axis_timeseries.py`, mirroring how
`build_sequence.py` orchestrates `continuity.compute_continuous_x_body`.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import geometry as geo

_EPS = 1e-12

FAR_FRACTION = 0.15
"""Fraction of body points (by distance to `body_cm`) kept as head/tail
candidates, mirroring `findBodyAxis_mk2.m`'s `fraction = 0.15`."""

MIN_CANDIDATE_POOL = 6
"""Floor on the candidate-pool size (`max(MIN_CANDIDATE_POOL, ceil(N *
FAR_FRACTION))`, clipped to N). `FAR_FRACTION` alone gives a pool too small
to be meaningful on sparse frames (e.g. 15% of 20 points = 3) -- the MATLAB
original never had this problem because voxel counts were always in the
hundreds; our per-frame point clouds are not guaranteed to be."""

DELTA_FRACTION = 0.15
"""`findBodyAxis_mk2.m` uses a fixed `delta` in pixels (`8 * pixPerCM/232`)
tied to camera zoom. We have no pixel/zoom concept here, so `delta` is
redefined as a fraction of the candidate pool's own projection span
(`(proj.max() - proj.min()) * DELTA_FRACTION`) -- scale-free, and adapts
to how spread-out the candidates already are on this specific frame."""

CONNECT_RADIUS_FACTOR = 2.5
"""Connectivity radius for the head/tail largest-connected-component filter
= `CONNECT_RADIUS_FACTOR * median(nearest-neighbor distance)` within that
end's candidate set. Adaptive (not a fixed physical radius) so it tracks
whatever point density this frame's reconstruction happens to have, per the
task's "不要写死绝对物理半径" requirement."""

MIN_CC_POINTS = 3
"""If the largest connected component on one end has fewer than this many
points, the CC filter is judged to have been too aggressive (over-pruned a
genuinely small-but-valid cluster down to nothing/near-nothing) and we fall
back to the *unfiltered* candidate set for that end instead of returning a
near-empty/empty centroid."""


def _weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (points * weights[:, None]).sum(axis=0) / weights.sum()


def _largest_connected_component(
    points: np.ndarray,
    weights: np.ndarray,
    connect_radius_factor: float,
    min_cc_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`findLargestHullCC` equivalent: keep only the largest spatially
    connected blob in `points` (radius graph + `connected_components`,
    mirroring `postprocessing/cleaning/eda_features.py::knn_component_labels`'s
    `coo_matrix` + `connected_components` pattern, but with a radius graph
    instead of a k-NN graph -- a head/tail candidate cluster's natural
    connectivity is "close together in space", not "has >=k neighbors").

    Falls back to returning `(points, weights)` unfiltered when there are
    too few points to build a meaningful graph (`< 2`) or when the largest
    component has fewer than `min_cc_points` points -- see `MIN_CC_POINTS`.
    """
    n = points.shape[0]
    if n < 2:
        return points, weights

    tree = cKDTree(points)
    nn_dist = tree.query(points, k=2)[0][:, 1]  # nearest *other* point per point
    median_nn = float(np.median(nn_dist))
    radius = connect_radius_factor * max(median_nn, _EPS)

    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        adjacency = coo_matrix((n, n))
    else:
        data = np.ones(len(pairs))
        adjacency = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _n_components, labels = connected_components(adjacency, directed=False)

    sizes = np.bincount(labels)
    largest_label = int(np.argmax(sizes))
    mask = labels == largest_label
    if int(mask.sum()) < min_cc_points:
        return points, weights
    return points[mask], weights[mask]


def _head_tail_centroids(
    body_xyz: np.ndarray,
    body_cm: np.ndarray,
    guide_axis: np.ndarray,
    weights: np.ndarray,
    far_fraction: float,
    min_candidate_pool: int,
    delta_fraction: float,
    connect_radius_factor: float,
    min_cc_points: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """`head_cm`, `tail_cm`, and each end's post-CC-filter point count."""
    n = body_xyz.shape[0]
    dist_to_cm = np.linalg.norm(body_xyz - body_cm, axis=1)
    n_pool = min(max(min_candidate_pool, int(np.ceil(n * far_fraction))), n)
    pool_idx = np.argsort(dist_to_cm)[::-1][:n_pool]
    pool_xyz = body_xyz[pool_idx]
    pool_w = weights[pool_idx]

    proj = (pool_xyz - body_cm) @ guide_axis
    span = float(proj.max() - proj.min())
    delta = span * delta_fraction
    # Inclusive (`>=`/`<=`) rather than strict: on a fully degenerate pool
    # (`span == 0`, every candidate projects to the same point) `delta == 0`
    # too, and a strict `>`/`<` would leave both masks empty. `>=`/`<=`
    # guarantees the extremal point(s) are always claimed by their end.
    head_mask = proj >= proj.max() - delta
    tail_mask = proj <= proj.min() + delta

    head_xyz, head_w = _largest_connected_component(
        pool_xyz[head_mask], pool_w[head_mask], connect_radius_factor, min_cc_points
    )
    tail_xyz, tail_w = _largest_connected_component(
        pool_xyz[tail_mask], pool_w[tail_mask], connect_radius_factor, min_cc_points
    )

    head_cm = _weighted_centroid(head_xyz, head_w)
    tail_cm = _weighted_centroid(tail_xyz, tail_w)
    return head_cm, tail_cm, head_xyz.shape[0], tail_xyz.shape[0]


def compute_robust_x_body(
    body_xyz: np.ndarray,
    x_axis_prev: np.ndarray | None,
    up: np.ndarray = (0.0, 0.0, 1.0),
    weights: np.ndarray | None = None,
    *,
    anchor_axis: np.ndarray | None = None,
    far_fraction: float = FAR_FRACTION,
    min_candidate_pool: int = MIN_CANDIDATE_POOL,
    delta_fraction: float = DELTA_FRACTION,
    connect_radius_factor: float = CONNECT_RADIUS_FACTOR,
    min_cc_points: int = MIN_CC_POINTS,
) -> tuple[np.ndarray, dict]:
    """One frame's head/tail-centroid body axis (`axis`) plus a diagnostic dict.

    Signature deliberately mirrors `correct_body_axis/continuity.py`'s
    `compute_continuous_x_body(body_xyz, x_body_prev, up, weights)`, with one
    addition (`anchor_axis`, keyword-only) -- this method's guide-axis
    sourcing has three tiers instead of continuity.py's two:

    1. `x_axis_prev` given (not `None`): use it as `guide_axis`.
       `method="prev_frame_guide"` -- true cross-frame continuity, since
       `x_axis_prev` should be the *previous frame's already-disambiguated*
       output of this same function (or of any equivalent method), not a raw
       PCA eigenvector.
    2. Else, `anchor_axis` given: use it as `guide_axis`.
       `method="anchor_guide"` -- for sequence starts, frames right after a
       gap, or frames where the previous frame's estimate itself failed;
       callers are expected to supply the nearest T-pose anchor frame's axis
       here (see `anchor_detect.py` / `g_anchors.csv`) since `guide_axis`
       only needs to roughly split the cloud into head/tail halves, not be
       exact -- a nearby anchor's axis is more than good enough, and unlike
       `x_axis_prev` requires no assumption that the previous frame was
       itself successfully estimated.
    3. Else (no previous frame, no anchor nearby): fall back to the same
       heuristic `body_frame.py::estimate_body_frame` /
       `continuity.py::compute_continuous_x_body` already use --
       `orient_to_reference(weighted_pca(body_xyz)[1][:, -1], up)`.
       `method="pca_up_fallback"`. This is the one place PCA's own sign
       ambiguity can leak in here, but only as a *guide* for candidate-pool
       splitting (a coarse problem), never as the final axis direction
       itself (a precise problem) -- see module docstring.

    Which tier actually gets used every frame (including *finding* the
    nearest anchor, which requires sequence position and is not a
    single-frame concept) is the caller's job, not this function's --
    exactly as `build_sequence.py` owns `continuity.py`'s reset policy while
    `compute_continuous_x_body` itself only reacts to `x_body_prev is None`.
    See `correct_body_axis/diag/h_robust_axis_timeseries.py`.

    Given whichever `guide_axis` results, the actual axis computation never
    looks at `x_axis_prev`/`anchor_axis`/`up` again: it takes the
    `far_fraction` of body points furthest from the (weighted) centroid,
    splits them into head/tail halves by their projection onto `guide_axis`
    (`delta_fraction` of that projection's span at each extreme), keeps only
    the largest spatially-connected component on each end, and returns
    `unit(head_cm - tail_cm)`. See module docstring / `_head_tail_centroids`
    / `_largest_connected_component` for the geometric detail.

    Returns `(axis, diag)`. `diag` keys: `head_n`/`tail_n` (each end's
    point count *after* CC filtering -- both dropping toward
    `min_cc_points` or hugging the raw candidate-pool size are useful
    degeneracy signals), `method` (one of the three above),
    `body_cm`/`head_cm`/`tail_cm`/`guide_axis` (for downstream plotting/QA).

    Raises `ValueError` (via `geo.unit`) if `head_cm == tail_cm`, i.e. the
    head and tail clusters collapsed onto the same point -- a genuinely
    degenerate frame, not something to paper over with a fallback here
    (callers should catch this per-frame, same as `pipeline.py` catches
    per-stage exceptions into a `status` string).
    """
    body_xyz = np.asarray(body_xyz, dtype=float)
    n = body_xyz.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    up_hat = geo.unit(np.asarray(up, dtype=float))
    body_cm = _weighted_centroid(body_xyz, w)

    if x_axis_prev is not None:
        guide_axis = geo.unit(np.asarray(x_axis_prev, dtype=float))
        method = "prev_frame_guide"
    elif anchor_axis is not None:
        guide_axis = geo.unit(np.asarray(anchor_axis, dtype=float))
        method = "anchor_guide"
    else:
        _eigvals, eigvecs, _centroid = geo.weighted_pca(body_xyz, w)
        guide_axis = geo.orient_to_reference(eigvecs[:, -1], up_hat)
        method = "pca_up_fallback"

    head_cm, tail_cm, head_n, tail_n = _head_tail_centroids(
        body_xyz, body_cm, guide_axis, w,
        far_fraction=far_fraction,
        min_candidate_pool=min_candidate_pool,
        delta_fraction=delta_fraction,
        connect_radius_factor=connect_radius_factor,
        min_cc_points=min_cc_points,
    )
    axis = geo.unit(head_cm - tail_cm)

    return axis, {
        "head_n": head_n,
        "tail_n": tail_n,
        "method": method,
        "body_cm": body_cm,
        "head_cm": head_cm,
        "tail_cm": tail_cm,
        "guide_axis": guide_axis,
    }


def compute_x_body_zhead(
    body_xyz: np.ndarray,
    up: np.ndarray = (0.0, 0.0, 1.0),
    weights: np.ndarray | None = None,
    *,
    far_fraction: float = FAR_FRACTION,
    min_candidate_pool: int = MIN_CANDIDATE_POOL,
    delta_fraction: float = DELTA_FRACTION,
    connect_radius_factor: float = CONNECT_RADIUS_FACTOR,
    min_cc_points: int = MIN_CC_POINTS,
) -> tuple[np.ndarray, dict]:
    """Single-frame, no-cross-frame-state `x_body`: current frame's own PCA
    axis picks the split, the split's own centroids' `up`-component picks
    the head.

    **Measured worse than the raw-PCA baseline on real data, not currently
    used in production** -- see module docstring's summary of
    `correct_body_axis/diag/k_zhead_axis_timeseries.py` before wiring this
    in anywhere; its per-frame-independent guide reintroduces the same
    direction-wobble problem this module otherwise sidesteps.

    Unlike `compute_robust_x_body`, this never takes a previous-frame or
    anchor guide -- there is nothing to reset, no anchor table to build, no
    sequence orchestration. Two steps, both scoped to this one frame:

    1. Split: the current frame's own `weighted_pca(body_xyz)` major
       eigenvector is passed to `_head_tail_centroids` purely to divide the
       far-from-centroid candidate pool into two end-clusters. Its *sign* is
       never used -- only the line it defines, so the PCA-direction wobble
       that makes near-degenerate frames unreliable for a final axis
       (module docstring) doesn't need to be resolved here, only "roughly
       which half is which," a coarser problem the split tolerates fine.
    2. Head call: of the two resulting (largest-connected-component-filtered)
       centroids, whichever has the larger `dot(., up)` is `head_cm`, the
       other `tail_cm`; `axis = unit(head_cm - tail_cm)`. This is the same
       "head points up-ish" assumption `body_frame.py::estimate_body_frame`
       already documented for its old raw-eigenvector heuristic, just
       applied to two actual point-cluster centroids (which only move
       smoothly frame-to-frame as real points enter/leave each cluster)
       instead of to the eigenvector itself (which can swing sharply on a
       near-disc cloud even though the underlying points barely changed).

    Returns `(axis, diag)`, same `diag` shape as `compute_robust_x_body`
    (`head_n`/`tail_n`/`method="z_head"`/`body_cm`/`head_cm`/`tail_cm`), plus
    `split_axis` (the PCA line used for step 1, replacing `guide_axis` --
    there is no guide here, just a split).

    Raises `ValueError` (via `geo.unit`) on a degenerate frame (head/tail
    collapse to the same point, or `weighted_pca` itself fails on too few /
    coincident points) -- same non-papering-over policy as
    `compute_robust_x_body`.
    """
    body_xyz = np.asarray(body_xyz, dtype=float)
    n = body_xyz.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    up_hat = geo.unit(np.asarray(up, dtype=float))
    body_cm = _weighted_centroid(body_xyz, w)

    _eigvals, eigvecs, _centroid = geo.weighted_pca(body_xyz, w)
    split_axis = eigvecs[:, -1]

    end_a_cm, end_b_cm, end_a_n, end_b_n = _head_tail_centroids(
        body_xyz, body_cm, split_axis, w,
        far_fraction=far_fraction,
        min_candidate_pool=min_candidate_pool,
        delta_fraction=delta_fraction,
        connect_radius_factor=connect_radius_factor,
        min_cc_points=min_cc_points,
    )

    a_is_head = np.dot(end_a_cm, up_hat) >= np.dot(end_b_cm, up_hat)
    head_cm, tail_cm = (end_a_cm, end_b_cm) if a_is_head else (end_b_cm, end_a_cm)
    head_n, tail_n = (end_a_n, end_b_n) if a_is_head else (end_b_n, end_a_n)

    axis = geo.unit(head_cm - tail_cm)

    return axis, {
        "head_n": head_n,
        "tail_n": tail_n,
        "method": "z_head",
        "body_cm": body_cm,
        "head_cm": head_cm,
        "tail_cm": tail_cm,
        "split_axis": split_axis,
    }


def compute_wing_hinge_far_cc(
    wing_xyz: np.ndarray,
    body_cm: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    far_fraction: float = FAR_FRACTION,
    min_candidate_pool: int = MIN_CANDIDATE_POOL,
    delta_fraction: float = DELTA_FRACTION,
    connect_radius_factor: float = CONNECT_RADIUS_FACTOR,
    min_cc_points: int = MIN_CC_POINTS,
) -> tuple[np.ndarray, dict]:
    """One wing's hinge (root) point: far-from-centroid + connected-component
    method, scoped to a single wing/single frame -- the same idea this
    module uses for the body's head/tail, applied to `body_frame.py`'s
    `_wing_hinge()` in place of that function's old PCA-span-axis extreme-
    point selection.

    `_wing_hinge`'s old method PCAs the wing's own points to get a span
    axis, then picks whichever of that axis's two *extreme* points is
    nearer `body_cm` -- a single sampled point, and one whose direction (not
    just its sign) is exactly the kind of PCA eigenvector estimate this
    module's docstring already diagnosed as unstable on near-degenerate
    point clouds (here, a wing folded flat or foreshortened toward the
    camera is a near-disc cloud, not just the body). This function instead
    reuses `_head_tail_centroids` verbatim, exactly as `compute_robust_x_body`
    does for the body: pool the `far_fraction` of the wing's own points
    farthest from the wing's own centroid, split that pool by projection onto
    a *guide* direction, largest-connected-component filter each end, and
    return a centroid rather than a single extreme point -- averaging over
    several root-region points instead of trusting one sampled extreme is
    itself part of why this is more stable, independent of the PCA-vs-CC
    question.

    `guide_axis = unit(body_cm - wing_cm)` (from the wing's own centroid
    toward the body): unlike `compute_x_body_zhead`'s split axis (which only
    needs to divide the cloud into two halves, sign-agnostic), here the
    split axis *is* already correctly oriented for the head/tail call --
    "toward the body" is by construction the root/hinge end, so
    `_head_tail_centroids`'s `head_cm` (the high-projection end) is directly
    the hinge, with no separate disambiguation step needed the way
    `compute_x_body_zhead` needs its `up`-component comparison. This also
    means, unlike `compute_x_body_zhead`/`compute_robust_x_body`, there is
    no PCA anywhere in this function at all -- `guide_axis` comes from two
    centroids, not an eigenvector.

    Validated on the real 640-frame dataset (`correct_body_axis/diag/
    i_roll_source_isolation.py`, method 3: body-axis method held fixed,
    only this wing-hinge method swapped in for the old one) before being
    promoted here: adjacent-frame roll jumps (>90 deg) dropped from 25 to 13
    (48% reduction) -- the one clean, unconfounded comparison in that
    diagnostic (see its module docstring / printed conclusion). This
    function is a straight formalization of that diagnostic script's
    `_wing_hinge_root_far_cc` prototype (same constants, same math -- not a
    retune), promoted into a reusable module function so `body_frame.py` can
    call it directly instead of every caller re-deriving it.

    Returns `(hinge_cm, diag)`. `diag` mirrors `compute_robust_x_body`'s
    shape: `hinge_n`/`tip_n` (each end's point count after CC filtering),
    `wing_cm`, `tip_cm` (the low-projection/away-from-body end -- not the
    wing's true tip in general, since `far_fraction` only pools points *far
    from the wing centroid*, but a useful QA/plotting companion to
    `hinge_cm`), `guide_axis`.

    Raises `ValueError` (via `geo.unit`) if `wing_cm == body_cm`, or if the
    hinge cluster collapses to a single point exactly at `wing_cm` -- both
    genuinely degenerate frames the caller is expected to catch, same
    contract as `compute_x_body_zhead`/`compute_robust_x_body`.
    """
    wing_xyz = np.asarray(wing_xyz, dtype=float)
    n = wing_xyz.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    body_cm = np.asarray(body_cm, dtype=float)
    wing_cm = _weighted_centroid(wing_xyz, w)
    guide_axis = geo.unit(body_cm - wing_cm)

    hinge_cm, tip_cm, hinge_n, tip_n = _head_tail_centroids(
        wing_xyz, wing_cm, guide_axis, w,
        far_fraction=far_fraction,
        min_candidate_pool=min_candidate_pool,
        delta_fraction=delta_fraction,
        connect_radius_factor=connect_radius_factor,
        min_cc_points=min_cc_points,
    )

    return hinge_cm, {
        "hinge_n": hinge_n,
        "tip_n": tip_n,
        "wing_cm": wing_cm,
        "tip_cm": tip_cm,
        "guide_axis": guide_axis,
    }
