"""In-memory T3 segmentation for `simulate_gt` scenes.

`segment_frame_kmeans_v2` is the primary method: a faithful in-memory replica
of `postprocessing/labeling/labeling.py::process_frame`'s actual production
algorithm (`kmeans_split.py`'s v2 seed-guided KMeans + rule-A semantic
mapping + wing connectivity/merge handling + body-PCA L/R anchoring), built
by importing `kmeans_split.py`'s self-contained functions directly (they
take arrays/DataFrames, no real-dataset I/O) rather than calling
`labeling.py::process_frame` itself, which only knows how to read/write real
per-frame CSVs from `DATASET_DIR` and pulls in a much heavier import chain
(reprojection viewers, `cv2`, camera calibration) for that I/O layer alone.

`segment_frame_motion` is the *other* production method: a faithful
in-memory replica of `postprocessing/labeling/motion/label.py::process_frame`
(cross-frame voxel-density body/wing split, `HALF_WINDOW=36` frames each
side -- body stays put across the window so its voxels get hit by many
distinct frames, wings sweep through each voxel only briefly). Reuses
`density.py`'s self-contained voxel functions
(`compute_voxel_frame_counts`/`extract_body_voxels`/`points_to_voxel_keys`,
none of which touch disk) plus `label.py`'s own
`split_wing_candidates`/`check_wing_merged`/`forced_wing_split` directly
(imported, not reimplemented -- unlike `kmeans_split.py`'s functions,
`label.py` itself is already self-contained enough at the per-function level
to import safely). Needs a *window* of frames' xyz (not just one frame), so
callers must build `window_xyz_by_frame` themselves (see
`run_step2_motion.py`) -- a frame within `HALF_WINDOW` of a sequence
boundary has no valid window and this function will raise.

`segment_frame_binary_threshold` is the earlier, much weaker placeholder
this module used before comparing against real data
(`postprocessing.labeling.binary.binary_split.classify_body_wing_quantile`,
a single planarity/dist_to_principal_axis threshold) -- kept only for
reference/comparison, not used by `evaluate.py` anymore. Real data showed
`planarity` barely separates body from wing at all (`kmeans_split.py`'s own
v2 docstring: "planarity/scale_ratio/sphericity/linearity在真实阈值下不鲁棒,
本次剔除"), which is exactly why production moved to `[x,y,z,opacity,R]`
KMeans instead of a planarity threshold -- `segment_frame_kmeans_v2`/
`segment_frame_motion` are the ones whose accuracy is actually informative
about real-world T3 performance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics.geometry import orient_to_reference, weighted_pca  # noqa: E402
from postprocessing.labeling.binary.binary_split import classify_body_wing_quantile  # noqa: E402
from postprocessing.labeling.fusion import motion_body_veto, motion_is_body_for_window  # noqa: E402
from postprocessing.labeling.kmeans.kmeans_split import (  # noqa: E402
    label_by_rule_a, run_kmeans, run_kmeans_v2, secondary_axis, seed_mask, standardize_v2,
)
from postprocessing.labeling.motion import density as motion_density  # noqa: E402
from postprocessing.labeling.motion.label import (  # noqa: E402
    check_wing_merged as _motion_check_wing_merged,
    forced_wing_split as _motion_forced_wing_split,
    split_wing_candidates as _motion_split_wing_candidates,
)
from utils.ply import connected_component_labels  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])

AUX_WEIGHT_FINAL = 1
"""Matches `labeling.py::AUX_WEIGHT_FINAL` -- production's locked-in v2
config ("定版配置: v2, aux_weight=1x(w1)")."""
MIN_BODY_SEED = 5
"""Matches `labeling.py::MIN_BODY_SEED` -- below this many `seed_mask` hits,
production falls back to unseeded `run_kmeans` (degraded, unstable init)."""
MAIN_RANDOM_STATE = 0
"""Matches `labeling.py`'s own `MAIN_RANDOM_STATE` (imported from
`kmeans_split.py` there; re-declared here to avoid importing `labeling.py`
itself, see module docstring)."""
WING_CC_K, WING_CC_PERCENTILE = 10, 75.0
"""Matches `labeling.py::WING_CC_K`/`WING_CC_PERCENTILE`."""
WING_MERGE_MIN_FRAC, WING_MERGE_MIN_ABS = 0.05, 5
"""Matches `labeling.py::WING_MERGE_MIN_FRAC`/`WING_MERGE_MIN_ABS`."""


def _body_axes_and_right_axis(body_xyz: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, eigvecs, _ = weighted_pca(body_xyz)
    x_body = orient_to_reference(eigvecs[:, -1], up)
    right_axis = np.cross(x_body, up)
    right_axis = right_axis / np.linalg.norm(right_axis)
    body_cm = body_xyz.mean(axis=0)
    return x_body, right_axis, body_cm


def segment_frame_binary_threshold(
    df: pd.DataFrame,
    up: np.ndarray = UP,
    cc_k: int = 10,
    cc_dist_percentile: float = 75.0,
) -> pd.Series:
    """Predict `part_label` (`body`/`wing_L`/`wing_R`) for every row of an
    *unlabeled* per-point `df` (see `scene.make_unlabeled_frame`).

    Superseded by `segment_frame_kmeans_v2` -- kept for reference/comparison
    only, see module docstring.

    Two stages: (1) `classify_body_wing_quantile` splits body vs "wing"
    (both sides pooled) by thresholding `planarity`/`dist_to_principal_axis`;
    (2) connected-component labeling on the wing points splits the two sides
    into spatially disjoint components -- valid as long as the two wings'
    point clouds don't touch (true for well-separated scenes; a
    near-stroke-reversal scene would need `labeling.py`'s real
    merge-detection + forced median split, not implemented here). Each of
    the two largest wing components is assigned `wing_L`/`wing_R` by its
    centroid's `right_axis` projection sign (see module docstring); any wing
    point outside those two components is labeled `wing_unassigned` (not a
    real `io_schema.PART_LABELS` value) so it shows up as a segmentation
    error downstream rather than being silently forced into some part.

    Raises `ValueError` if the body/wing split or the wing connected-component
    split degenerates (too few points, or fewer than 2 wing components) --
    callers should treat that as a segmentation failure for this frame.
    """
    up = np.asarray(up, dtype=float)
    is_wing = classify_body_wing_quantile(df)

    body_xyz = df.loc[~is_wing, ["x", "y", "z"]].to_numpy()
    if body_xyz.shape[0] < 3:
        raise ValueError(f"segment_frame: only {body_xyz.shape[0]} body points after binary split")
    x_body, right_axis, body_cm = _body_axes_and_right_axis(body_xyz, up)

    wing_idx = df.index[is_wing].to_numpy()
    wing_xyz = df.loc[wing_idx, ["x", "y", "z"]].to_numpy()
    if wing_xyz.shape[0] < 3:
        raise ValueError(f"segment_frame: only {wing_xyz.shape[0]} wing points after binary split")

    k_use = min(cc_k, wing_xyz.shape[0] - 1)
    comp_labels = connected_component_labels(wing_xyz, k=k_use, dist_percentile=cc_dist_percentile)
    comp_ids, comp_sizes = np.unique(comp_labels, return_counts=True)
    if len(comp_ids) < 2:
        raise ValueError(
            f"segment_frame: wing points formed only {len(comp_ids)} connected component(s), expected 2"
        )
    top2 = comp_ids[np.argsort(comp_sizes)[::-1][:2]]

    labels = pd.Series("body", index=df.index, dtype=object)
    proj_by_comp = {}
    for comp_id in top2:
        comp_mask = comp_labels == comp_id
        comp_xyz = wing_xyz[comp_mask]
        proj_by_comp[comp_id] = (comp_mask, float(np.dot(comp_xyz.mean(axis=0) - body_cm, right_axis)))

    ordered = sorted(proj_by_comp.items(), key=lambda kv: kv[1][1])
    (_, (mask_lo, _)), (_, (mask_hi, _)) = ordered
    labels.loc[wing_idx[mask_lo]] = "wing_L"
    labels.loc[wing_idx[mask_hi]] = "wing_R"

    leftover = np.ones(wing_xyz.shape[0], dtype=bool)
    for mask, _ in proj_by_comp.values():
        leftover &= ~mask
    if leftover.any():
        labels.loc[wing_idx[leftover]] = "wing_unassigned"

    return labels.rename("part_label")


# ---------------------------------------------------------------------------
# segment_frame_kmeans_v2: in-memory replica of labeling.py::process_frame
# ---------------------------------------------------------------------------


def _wing_merged(
    xyz: np.ndarray, semantic: np.ndarray, k: int = WING_CC_K, percentile: float = WING_CC_PERCENTILE,
) -> bool:
    """Mirrors `labeling.py::check_wing_merged`: are `wing_A`+`wing_B`
    (pooled, ignoring the cluster boundary) really 2+ spatially separate
    blobs, or did KMeans cut a single physically-merged blob in feature
    space only?"""
    idx = np.where((semantic == "wing_A") | (semantic == "wing_B"))[0]
    if len(idx) < 2:
        return False
    sub = xyz[idx]
    k_use = min(k, len(sub) - 1)
    if k_use < 1:
        return False
    comp_labels = connected_component_labels(sub, k=k_use, dist_percentile=percentile)
    comp_sizes = np.bincount(comp_labels)
    n_total = len(idx)
    n_significant = int(np.sum((comp_sizes >= WING_MERGE_MIN_ABS) & (comp_sizes >= WING_MERGE_MIN_FRAC * n_total)))
    return n_significant <= 1


def _forced_wing_split(
    xyz: np.ndarray, semantic: np.ndarray, right_axis: np.ndarray, body_cm: np.ndarray,
) -> np.ndarray:
    """Mirrors `labeling.py::forced_wing_split`: when the two wings can't be
    physically separated, fall back to a median split on `right_axis`
    projection."""
    semantic = semantic.copy()
    idx = np.where((semantic == "wing_A") | (semantic == "wing_B"))[0]
    proj = (xyz[idx] - body_cm) @ right_axis
    median = np.median(proj)
    semantic[idx] = np.where(proj > median, "wing_A", "wing_B")
    return semantic


def _fix_wing_connectivity(
    xyz: np.ndarray, semantic: np.ndarray, k: int = WING_CC_K, percentile: float = WING_CC_PERCENTILE,
) -> np.ndarray:
    """Mirrors `labeling.py::fix_wing_connectivity`: within each of
    `wing_A`/`wing_B`, keep only the largest spatial connected component as
    that cluster's "main" block; every other (fragment) point is reassigned
    to whichever of `{body, wing_A-main, wing_B-main}` it is spatially
    nearest to (1-NN), not left in its original (feature-space) cluster."""
    semantic = semantic.copy()
    main_xyz: dict[str, np.ndarray] = {}
    fragments: list[int] = []
    for wname in ("wing_A", "wing_B"):
        idx = np.where(semantic == wname)[0]
        if len(idx) == 0:
            main_xyz[wname] = np.empty((0, 3))
            continue
        k_use = min(k, len(idx) - 1)
        if k_use < 1:
            main_xyz[wname] = xyz[idx]
            continue
        comp_labels = connected_component_labels(xyz[idx], k=k_use, dist_percentile=percentile)
        comp_sizes = np.bincount(comp_labels)
        main_mask = comp_labels == int(np.argmax(comp_sizes))
        main_xyz[wname] = xyz[idx][main_mask]
        fragments.extend(idx[~main_mask].tolist())

    if not fragments:
        return semantic

    candidates = {"body": xyz[semantic == "body"], "wing_A": main_xyz["wing_A"], "wing_B": main_xyz["wing_B"]}
    trees = {name: cKDTree(pts) for name, pts in candidates.items() if len(pts) > 0}
    for i in fragments:
        best_name, best_dist = None, np.inf
        for name, tree in trees.items():
            dist, _ = tree.query(xyz[i], k=1)
            if dist < best_dist:
                best_dist, best_name = dist, name
        semantic[i] = best_name
    return semantic


def segment_frame_kmeans_v2(
    df: pd.DataFrame,
    up: np.ndarray = UP,
    random_state: int = MAIN_RANDOM_STATE,
) -> pd.Series:
    """Predict `part_label` for every row of an *unlabeled* per-point `df`
    via a faithful in-memory replica of `labeling.py::process_frame`'s
    actual production algorithm (see module docstring): `seed_mask` + v2
    seeded KMeans (`standardize_v2`/`build_seed_init`/`run_kmeans_v2`, degraded
    fallback to plain `run_kmeans` below `MIN_BODY_SEED`) -> rule-A semantic
    mapping (`label_by_rule_a`) -> wing-merge check/forced split or
    connectivity fixup -> body-PCA `right_axis` L/R anchoring.

    Never raises: KMeans always assigns every point to one of 3 clusters and
    the connectivity fixup reassigns every wing fragment to one of the three
    main blocks, so the result is always a full `body`/`wing_L`/`wing_R`
    labeling (no `wing_unassigned` residue, unlike
    `segment_frame_binary_threshold`). Unlike production, there is no
    `if_keep=False` NN-propagation step (`finalize_part_labels`'s other
    job) -- `simulate_gt` frames have no dropped points to propagate to.
    """
    up = np.asarray(up, dtype=float)
    xyz = df[["x", "y", "z"]].to_numpy()

    seeds = seed_mask(df)
    X = standardize_v2(df, AUX_WEIGHT_FINAL)
    labels = run_kmeans(X, random_state) if seeds.sum() < MIN_BODY_SEED else run_kmeans_v2(X, seeds, random_state)

    axis, centroid = secondary_axis(xyz)
    mapping = label_by_rule_a(df, labels, axis, centroid)
    semantic = np.array([mapping[c] for c in labels], dtype=object)

    if _wing_merged(xyz, semantic):
        body_xyz = xyz[semantic == "body"]
        _, right_axis, body_cm = _body_axes_and_right_axis(body_xyz, up)
        semantic = _forced_wing_split(xyz, semantic, right_axis, body_cm)
    else:
        semantic = _fix_wing_connectivity(xyz, semantic)

    body_xyz = xyz[semantic == "body"]
    _, right_axis, body_cm = _body_axes_and_right_axis(body_xyz, up)
    has_a, has_b = np.any(semantic == "wing_A"), np.any(semantic == "wing_B")
    proj_a = float(np.dot(xyz[semantic == "wing_A"].mean(axis=0) - body_cm, right_axis)) if has_a else -np.inf
    proj_b = float(np.dot(xyz[semantic == "wing_B"].mean(axis=0) - body_cm, right_axis)) if has_b else -np.inf
    lr_map = {"body": "body"}
    if proj_a > proj_b:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_R", "wing_L"
    else:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_L", "wing_R"

    final = np.array([lr_map[s] for s in semantic], dtype=object)
    return pd.Series(final, index=df.index, name="part_label")


def segment_frame_kmeans_motion_fusion(
    df: pd.DataFrame,
    window_xyz_by_frame: dict[int, np.ndarray],
    center_frame_idx: int,
    up: np.ndarray = UP,
    random_state: int = MAIN_RANDOM_STATE,
    half_window: int | None = None,
) -> pd.Series:
    """`segment_frame_kmeans_v2` plus one new step: `postprocessing.labeling.
    fusion.motion_body_veto`, applied right after KMeans's raw cluster labels
    are mapped to body/wing_A/wing_B semantics and before the wing-merge
    check -- see `fusion.py`'s module docstring for the joint-error-analysis
    derivation (phase 4.3 in `segmentation_fusion_progress.md`) of why this
    is a one-directional *veto* on kmeans's `body` cluster (via motion's
    windowed voxel-density evidence), not a two-way label fusion.

    `window_xyz_by_frame`/`center_frame_idx` have the same contract as
    `segment_frame_motion`'s (a full `+/-half_window` window of frames' xyz
    must be buildable), EXCEPT this function never raises when the window
    isn't available for `center_frame_idx` -- it silently degrades to plain
    `segment_frame_kmeans_v2` behavior for that frame (motion has nothing to
    contribute outside its window; see `fusion.motion_is_body_for_window`).
    `df` (the center frame's own unlabeled points) must be the same frame
    whose xyz equals `window_xyz_by_frame[center_frame_idx]`, same row order
    -- this is the caller's responsibility (mirrors `segment_frame_motion`'s
    own contract), not re-derived/re-checked here.
    """
    up = np.asarray(up, dtype=float)
    half_window = HALF_WINDOW if half_window is None else half_window
    xyz = df[["x", "y", "z"]].to_numpy()

    seeds = seed_mask(df)
    X = standardize_v2(df, AUX_WEIGHT_FINAL)
    labels = run_kmeans(X, random_state) if seeds.sum() < MIN_BODY_SEED else run_kmeans_v2(X, seeds, random_state)

    axis, centroid = secondary_axis(xyz)
    mapping = label_by_rule_a(df, labels, axis, centroid)
    semantic = np.array([mapping[c] for c in labels], dtype=object)

    is_body_motion = motion_is_body_for_window(window_xyz_by_frame, center_frame_idx, half_window)
    semantic = motion_body_veto(xyz, semantic, is_body_motion)

    if _wing_merged(xyz, semantic):
        body_xyz = xyz[semantic == "body"]
        _, right_axis, body_cm = _body_axes_and_right_axis(body_xyz, up)
        semantic = _forced_wing_split(xyz, semantic, right_axis, body_cm)
    else:
        semantic = _fix_wing_connectivity(xyz, semantic)

    body_xyz = xyz[semantic == "body"]
    _, right_axis, body_cm = _body_axes_and_right_axis(body_xyz, up)
    has_a, has_b = np.any(semantic == "wing_A"), np.any(semantic == "wing_B")
    proj_a = float(np.dot(xyz[semantic == "wing_A"].mean(axis=0) - body_cm, right_axis)) if has_a else -np.inf
    proj_b = float(np.dot(xyz[semantic == "wing_B"].mean(axis=0) - body_cm, right_axis)) if has_b else -np.inf
    lr_map = {"body": "body"}
    if proj_a > proj_b:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_R", "wing_L"
    else:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_L", "wing_R"

    final = np.array([lr_map[s] for s in semantic], dtype=object)
    return pd.Series(final, index=df.index, name="part_label")


# ---------------------------------------------------------------------------
# segment_frame_motion: in-memory replica of motion/label.py::process_frame
# ---------------------------------------------------------------------------

HALF_WINDOW = motion_density.HALF_WINDOW
"""Re-exported from `density.py` -- `segment_frame_motion` needs frames
`[center-HALF_WINDOW, center+HALF_WINDOW]` all present."""


def segment_frame_motion(
    window_xyz_by_frame: dict[int, np.ndarray],
    center_frame_idx: int,
    up: np.ndarray = UP,
    half_window: int = HALF_WINDOW,
) -> pd.Series:
    """Predict `part_label` for `window_xyz_by_frame[center_frame_idx]` using
    the cross-frame voxel-density method (see module docstring). Unlike the
    single-frame methods above, this needs *every* frame in
    `[center_frame_idx - half_window, center_frame_idx + half_window]`
    present as a key in `window_xyz_by_frame` -- raises `ValueError` if any
    are missing (matching `density.py`'s own "don't silently truncate the
    window" stance, just turned into a hard error here since `simulate_gt`
    sequences are short and every frame should exist, unlike real T2 gaps).

    Body voxels are extracted once from the *whole window* pooled together
    (`compute_voxel_frame_counts` + `extract_body_voxels`, `density.py`'s own
    constants -- `VOXEL_SIZE_M`/`BODY_VOXEL_COUNT_THRESH`/etc., unchanged);
    only the *center* frame's own points are then classified against that
    voxel set and returned.
    """
    up = np.asarray(up, dtype=float)
    needed = range(center_frame_idx - half_window, center_frame_idx + half_window + 1)
    missing = [f for f in needed if f not in window_xyz_by_frame]
    if missing:
        raise ValueError(
            f"segment_frame_motion: frame {center_frame_idx} needs a full "
            f"+/-{half_window}-frame window, missing {len(missing)} frame(s): {missing[:5]}..."
        )

    window_dfs = []
    for f in needed:
        xyz_f = window_xyz_by_frame[f]
        window_dfs.append(pd.DataFrame({"x": xyz_f[:, 0], "y": xyz_f[:, 1], "z": xyz_f[:, 2], "frame_idx": f}))
    window_df = pd.concat(window_dfs, ignore_index=True)

    voxel_counts = motion_density.compute_voxel_frame_counts(window_df)
    body_voxels = motion_density.extract_body_voxels(voxel_counts)

    xyz = window_xyz_by_frame[center_frame_idx]
    voxel_keys = motion_density.points_to_voxel_keys(xyz)
    is_body = np.array([tuple(vk) in body_voxels for vk in voxel_keys])

    semantic, comp_sizes, _diag = _motion_split_wing_candidates(xyz, is_body)
    n_wing_total = int((~is_body).sum())
    if _motion_check_wing_merged(comp_sizes, n_wing_total):
        body_xyz = xyz[semantic == "body"]
        _, right_axis, body_cm = _body_axes_and_right_axis(body_xyz, up)
        semantic = _motion_forced_wing_split(xyz, semantic, right_axis, body_cm)

    body_xyz = xyz[semantic == "body"]
    _, right_axis, body_cm = _body_axes_and_right_axis(body_xyz, up)
    has_a, has_b = np.any(semantic == "wing_A"), np.any(semantic == "wing_B")
    proj_a = float(np.dot(xyz[semantic == "wing_A"].mean(axis=0) - body_cm, right_axis)) if has_a else -np.inf
    proj_b = float(np.dot(xyz[semantic == "wing_B"].mean(axis=0) - body_cm, right_axis)) if has_b else -np.inf
    lr_map = {"body": "body"}
    if proj_a > proj_b:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_R", "wing_L"
    else:
        lr_map["wing_A"], lr_map["wing_B"] = "wing_L", "wing_R"

    final = np.array([lr_map.get(s, s) for s in semantic], dtype=object)
    return pd.Series(final, index=range(len(xyz)), name="part_label")
