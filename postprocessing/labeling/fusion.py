"""
Motion + KMeans segmentation fusion -- the shared orchestration glue.

Background (see postprocessing/kinematics/reference/segmentation_fusion_progress.md
phase 4.3 for the full joint-error-analysis derivation): on
`postprocessing/kinematics/simulate_gt`'s synthetic ground truth, on the
`[36,63]`-frame window where both `kmeans_split.py`'s v2 seeded-KMeans method
and `motion/label.py`'s cross-frame voxel-density method are independently
scorable against ground truth:

- Overriding kmeans's `wing` verdict with motion's `body` verdict wherever
  they disagree is net-HARMFUL (fixes 679 points, breaks 2019, net -1340) --
  motion's precision on its own "body" prediction is only ~76%, not perfect.
- Overriding kmeans's `body` verdict with motion's label wherever motion
  says NOT-body is net-STRICTLY-POSITIVE (fixes 510, breaks 0) -- in this
  window, motion's "not body" prediction is *never* wrong (0/5709 such
  points have GT=body), because motion's recall on the body class is 1.0
  (it never mislabels a true body point as wing; its errors all run the
  other way, wing points that stay in a persistently-occupied voxel getting
  mislabeled as body).

So the fusion rule implemented here is deliberately ONE-DIRECTIONAL: motion
is used only as a veto on kmeans's `body` cluster (points kmeans put in
`body` that motion's windowed voxel-density evidence says are NOT
persistently occupied get reassigned to whichever wing cluster they're
spatially nearest to), never as a positive override of kmeans's `wing`
calls. This is intentionally the ONLY new mechanism added this round (see
brief section 4.4's "single variable first" instruction) -- no continuity
layer beyond motion's own `HALF_WINDOW` cross-frame evidence is implemented
here.

Reused, not reimplemented: kmeans primitives from
`.legacy/kmeans/kmeans_split.py`, motion voxel primitives
from `postprocessing/labeling/motion/density.py`. This module only adds (a)
the veto step itself (`motion_body_veto`, genuinely new -- reassigning a
vetoed body point to its nearest surviving wing cluster didn't exist
anywhere before) and (b) two small adapters that compute the per-point
`is_body_motion` mask motion produces, one for `simulate_gt`'s in-memory
window dict, one for the real disk-backed dataset -- both delegate the
actual voxel computation to `density.py`, they don't reimplement it.

Caller contract for `motion_body_veto`: must run on kmeans's raw
body/wing_A/wing_B semantic labels, BEFORE the wing-merge check / forced
split / connectivity fixup that both `kmeans_split.py`-based orchestrations
(`labeling.py::process_frame`, `simulate_gt/segment.py::segment_frame_kmeans_v2`)
already run -- so the (possibly enlarged, by vetoed points) wing point set
gets processed by that same existing downstream machinery, not treated as a
separate special case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.labeling.motion import density as motion_density  # noqa: E402


MAX_VETO_FRAC = 0.30
"""Safety cap added after the phase-4.6 real-data sanity check (see
segmentation_fusion_progress.md): on `simulate_gt`'s synthetic scenario, the
veto fraction (points vetoed / total kept points in the frame) is small on
almost all frames (median 1.4%, p90 7.5%) except the 3 already-diagnosed
catastrophic frames (36/39/40, whole-cluster body/wing confusion, see phase
4.2), where a *large* veto fraction (18-21%) is exactly the beneficial
correction. On the real G2b_G9 dataset, veto fraction runs systematically
higher overall (median 11%, p90 19%) -- consistent with `density.py`'s own
documented caveats (no rigid-motion alignment, body jitter, a voxel-count
threshold that "isn't a clean bimodal split" per that module's own
docstring) -- and two frames (f0052 40%, f0055 46%) are far outside even the
synthetic catastrophic-frame range, and are exactly the frames responsible
for the two largest, least-plausible roll swings (up to ~168 deg
frame-to-frame) found in the real-data sanity check. `MAX_VETO_FRAC=0.30`
sits with margin above the synthetic catastrophic-frame ceiling (0.21) and
with margin below the problematic real-data frames (0.40+): above this
fraction, motion's disagreement with kmeans is treated as motion's own
signal being unreliable for that specific frame (not a targeted correction)
and the veto is skipped entirely for that frame, falling back to plain
kmeans_v2. This is a magnitude safety guard on the SAME single mechanism,
not a second mechanism -- it does not change what a veto does when it fires,
only adds one more condition for whether it fires at all."""


def motion_body_veto(
    xyz: np.ndarray, semantic: np.ndarray, is_body_motion: np.ndarray | None,
    max_veto_frac: float = MAX_VETO_FRAC,
) -> np.ndarray:
    """Reassign every point currently labeled `body` in `semantic` whose
    `is_body_motion` is False to whichever of the current `wing_A`/`wing_B`
    clusters it is spatially nearest to (1-NN on `xyz`). Returns a new array
    (input `semantic` untouched).

    No-op (returns `semantic` unchanged) when:
    - `is_body_motion` is None -- motion has no valid signal for this frame
      (outside its `HALF_WINDOW` window), the standard "degrade to plain
      kmeans_v2 behavior outside the window" fallback.
    - there is nothing to veto, or vetoing would leave zero points in
      `body` or zero points in the wing pool to reassign into -- a
      defensive guard so this step can never degenerate a frame that kmeans
      handled reasonably on its own.
    - the fraction of ALL kept points that would be vetoed exceeds
      `max_veto_frac` (default `MAX_VETO_FRAC`, see its own docstring) --
      motion disagreeing with kmeans on that large a slice of the frame is
      treated as motion's own signal being unreliable for this specific
      frame, not a targeted correction.
    """
    if is_body_motion is None:
        return semantic
    semantic = semantic.copy()
    body_idx = np.where(semantic == "body")[0]
    if len(body_idx) == 0:
        return semantic
    veto_idx = body_idx[~is_body_motion[body_idx]]
    if len(veto_idx) == 0:
        return semantic
    if len(veto_idx) == len(body_idx):
        # Motion disagrees with kmeans on the ENTIRE body cluster -- almost
        # certainly a degenerate/garbage frame for one method or the other;
        # vetoing everything would leave no body point at all. Don't.
        return semantic
    if len(veto_idx) / len(xyz) > max_veto_frac:
        return semantic

    wing_a_idx = np.where(semantic == "wing_A")[0]
    wing_b_idx = np.where(semantic == "wing_B")[0]
    trees = {}
    if len(wing_a_idx) > 0:
        trees["wing_A"] = cKDTree(xyz[wing_a_idx])
    if len(wing_b_idx) > 0:
        trees["wing_B"] = cKDTree(xyz[wing_b_idx])
    if not trees:
        return semantic

    for i in veto_idx:
        best_name, best_dist = None, np.inf
        for name, tree in trees.items():
            dist, _ = tree.query(xyz[i], k=1)
            if dist < best_dist:
                best_dist, best_name = dist, name
        semantic[i] = best_name
    return semantic


def motion_is_body_for_window(
    window_xyz_by_frame: dict[int, np.ndarray], center_frame_idx: int, half_window: int,
) -> np.ndarray | None:
    """In-memory adapter (`simulate_gt` path): per-point `is_body` mask
    (aligned to `window_xyz_by_frame[center_frame_idx]`'s own row order) via
    `density.py`'s voxel primitives, or `None` if the full
    `+/-half_window`-frame window isn't available in `window_xyz_by_frame`
    (mirrors `segment.segment_frame_motion`'s own availability check, but
    never raises -- returns `None` so the caller can fall back to plain
    kmeans_v2 instead of failing the whole frame).
    """
    needed = range(center_frame_idx - half_window, center_frame_idx + half_window + 1)
    if any(f not in window_xyz_by_frame for f in needed):
        return None

    window_dfs = []
    for f in needed:
        xyz_f = window_xyz_by_frame[f]
        window_dfs.append(pd.DataFrame({"x": xyz_f[:, 0], "y": xyz_f[:, 1], "z": xyz_f[:, 2], "frame_idx": f}))
    window_df = pd.concat(window_dfs, ignore_index=True)

    voxel_counts = motion_density.compute_voxel_frame_counts(window_df)
    body_voxels = motion_density.extract_body_voxels(voxel_counts)

    xyz_center = window_xyz_by_frame[center_frame_idx]
    voxel_keys = motion_density.points_to_voxel_keys(xyz_center)
    return np.array([tuple(vk) in body_voxels for vk in voxel_keys])


def motion_is_body_for_frame_idx(
    frame_idx: int, xyz_kept: np.ndarray, dataset_dir: Path = motion_density.DATASET_DIR,
) -> np.ndarray | None:
    """Production adapter (real, disk-backed dataset path): per-point
    `is_body` mask (aligned to `xyz_kept`'s row order) for frame `frame_idx`,
    via `density.py::compute_body_voxels_for_frame` (does its own T2 window
    I/O). Returns `None` -- never raises -- if `frame_idx` falls outside
    `density.valid_frame_range()` (within `HALF_WINDOW` of a sequence
    boundary, or any window frame missing T2 output), matching the in-memory
    adapter's fallback contract.
    """
    if frame_idx not in motion_density.valid_frame_range():
        return None
    info = motion_density.compute_body_voxels_for_frame(frame_idx, dataset_dir)
    if info["n_frames_used"] < 2 * motion_density.HALF_WINDOW + 1:
        # Same "don't trust a truncated window" stance density.py itself
        # documents -- some frames in-range can still be missing T2 output
        # on a partially-processed real dataset; a truncated window's voxel
        # counts are not comparable to a full-window one, so treat it the
        # same as "no signal" rather than silently vetoing off a weaker
        # basis than the validated full-window case.
        return None
    voxel_keys = motion_density.points_to_voxel_keys(xyz_kept)
    body_voxels = info["body_voxels"]
    return np.array([tuple(vk) in body_voxels for vk in voxel_keys])
