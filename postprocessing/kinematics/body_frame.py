"""S2 body frame + body angles (T4 kinematics), single frame only.

Implements calc_kinematics.md §2 (body frame construction) and §3 (body
angles yaw/pitch/roll). Built entirely on `geometry.py` primitives — no PCA /
Rodrigues / signed-angle math is reimplemented here.

Roll uses the cell-2 `calculate_roll` construction from
`reference/python_snippets.py` (authoritative; see module docstring there).
S3/S4 (wing phi/theta/eta, chord) depend only on this module's
`x_body, y_body, z_body, n_sp` — nothing else here is part of their contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from . import robust_body_axis as rba


@dataclass
class BodyFrame:
    """One frame's body pose (§2 frame + §3 angles).

    `x_body/y_body/z_body` are unit, mutually orthogonal, right-handed
    (`z_body = x_body x y_body`). `n_sp` is the stroke-plane normal (§2 step
    4). `yaw/pitch/roll` are degrees (§3). `hinge_L/hinge_R` are the wing-root
    points used to build `y_body` (§2 step 2); `body_cm` is the body
    centroid.
    """

    x_body: np.ndarray
    y_body: np.ndarray
    z_body: np.ndarray
    n_sp: np.ndarray
    yaw: float
    pitch: float
    roll: float
    hinge_L: np.ndarray
    hinge_R: np.ndarray
    body_cm: np.ndarray


def _wing_hinge(wing_xyz: np.ndarray, body_cm: np.ndarray, root_mode: str) -> np.ndarray:
    """Proximal (nearest-body) end of one wing's span (§2 step 2).

    `root_mode="root"`: `robust_body_axis.compute_wing_hinge_far_cc` --
    far-from-wing-centroid + connected-component root cluster, oriented by
    `guide_axis = unit(body_cm - wing_cm)` (see that function's docstring).
    Replaces this mode's earlier PCA-span-axis extreme-point pick (PCA the
    wing's own points, take whichever of the axis's two extreme points is
    closer to `body_cm`): that method's axis *direction* — not just sign —
    is unstable on a near-degenerate (folded/foreshortened) wing cloud, the
    same PCA failure mode this codebase already diagnosed for the body, and
    it returns a single sampled point rather than a centroid. Measured on
    the real 640-frame dataset (`correct_body_axis/diag/
    i_roll_source_isolation.py`'s method 3, body axis held fixed): swapping
    just this wing-hinge method dropped adjacent-frame roll jumps (>90 deg)
    from 25 to 13 (48%) -- see `compute_wing_hinge_far_cc`'s docstring.
    `root_mode="centroid"`: fallback per §2 step 2 — the wing's point
    centroid, used when root-region points aren't reliably separable from
    the tip (e.g. T3 mislabeling near the body).
    """
    if root_mode == "centroid":
        return wing_xyz.mean(axis=0)
    if root_mode != "root":
        raise ValueError(f"_wing_hinge: unknown root_mode {root_mode!r}, expected 'root' or 'centroid'")

    hinge_cm, _diag = rba.compute_wing_hinge_far_cc(wing_xyz, body_cm)
    return hinge_cm


def _calculate_roll(yaw_rad: float, pitch_rad: float, y_body: np.ndarray) -> float:
    """Mirrors `reference/python_snippets.py` cell 2 `calculate_roll` exactly
    (single-frame, non-vectorized). Builds the zero-roll frame `(e_y, e_z)`
    from `yaw_rad`/`pitch_rad` alone — this assumes the standard lab
    `up = +z` convention (§0), not an arbitrary `up`, since that is what the
    reference formula's `e_z` component layout encodes. `roll` is then the
    signed angle carrying `y_body` from `e_y` toward `e_z`.
    """
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    e_y = np.array([-sy, cy, 0.0])
    e_z = np.array([-sp * cy, -sp * sy, cp])
    return math.atan2(float(np.dot(y_body, e_z)), float(np.dot(y_body, e_y)))


def estimate_body_frame(
    body_xyz: np.ndarray,
    wingL_xyz: np.ndarray,
    wingR_xyz: np.ndarray,
    up: np.ndarray = (0.0, 0.0, 1.0),
    stroke_plane_pitch_deg: float = 45.0,
    stroke_plane_normal: np.ndarray | None = None,
    root_mode: str = "root",
    x_body: np.ndarray | None = None,
) -> BodyFrame:
    """Estimate one frame's `BodyFrame` from labeled body/wing points (§2/§3).

    `body_xyz`, `wingL_xyz`, `wingR_xyz` are `(N,3)` arrays (see
    `io_schema.body_xyz` / `wingL_xyz` / `wingR_xyz`). `up` is the lab-frame
    up vector (§0, default `+z`), used both as the head-sign fallback and as
    the `pitch` reference axis. `root_mode` selects the §2-step-2 wing-hinge
    method (`"root"` or `"centroid"`, see `_wing_hinge`).

    `x_body`, if given, is used verbatim (re-normalized) as the body long
    axis instead of this function's own single-frame head-sign heuristic
    below — this is how a sequence-level caller
    (`correct_body_axis.sequence_axis.compute_sequence_x_body`, wired in via
    `pipeline.run_dataset_with_sequence_correction`) supplies a
    continuity-chained, anchor-verified axis per frame instead of the
    per-frame-independent PCA guess. `hinge_L`/`hinge_R`/`y_body`/`n_sp`/
    angles are unaffected either way — only where `x_body` itself comes from
    changes.

    Head-sign heuristic (§2 step 1, used only when `x_body` is not supplied):
    the body point cloud's PCA major axis has no head/tail sign of its own,
    so it is oriented via `dot(x_body, up) > 0` — i.e. assuming the fly's
    head points up-ish. This is a single-frame fallback, not a real head
    detector: it fails (picks the tail-as-head) whenever the true body pitch
    is negative enough that the nose is not the "up" end (e.g. a steep
    dive), and — measured on the real 640-frame dataset,
    `correct_body_axis/diag/h_robust_axis_timeseries.py` — flips sign on
    ~6% of adjacent frame pairs whenever the body point cloud is near a
    disc shape (PCA's own major-axis *direction* going unstable, not just
    this heuristic's sign guess). Downstream `yaw`/`pitch`/`roll` inherit
    whichever error. Not fixed here for the no-`x_body` case — see
    `correct_body_axis/sequence_axis.py` for the sequence-level fix.
    """
    body_xyz = np.asarray(body_xyz, dtype=float)
    wingL_xyz = np.asarray(wingL_xyz, dtype=float)
    wingR_xyz = np.asarray(wingR_xyz, dtype=float)
    up_hat = geo.unit(np.asarray(up, dtype=float))

    body_cm = body_xyz.mean(axis=0)

    if x_body is not None:
        x_body = geo.unit(np.asarray(x_body, dtype=float))
    else:
        _, eigvecs, _ = geo.weighted_pca(body_xyz)
        x_body = geo.orient_to_reference(eigvecs[:, -1], up_hat)

    hinge_L = _wing_hinge(wingL_xyz, body_cm, root_mode)
    hinge_R = _wing_hinge(wingR_xyz, body_cm, root_mode)
    y_body = geo.project_onto_plane(hinge_L - hinge_R, x_body)

    z_body = np.cross(x_body, y_body)
    z_body = z_body / np.linalg.norm(z_body)

    if stroke_plane_normal is not None:
        n_sp = geo.unit(np.asarray(stroke_plane_normal, dtype=float))
    else:
        n_sp = geo.rodrigues_rotate(x_body, y_body, math.radians(-stroke_plane_pitch_deg))

    yaw = math.degrees(math.atan2(x_body[1], x_body[0]))
    pitch = 90.0 - math.degrees(math.acos(np.clip(np.dot(x_body, up_hat), -1.0, 1.0)))
    roll = math.degrees(_calculate_roll(math.radians(yaw), math.radians(pitch), y_body))

    return BodyFrame(
        x_body=x_body,
        y_body=y_body,
        z_body=z_body,
        n_sp=n_sp,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        hinge_L=hinge_L,
        hinge_R=hinge_R,
        body_cm=body_cm,
    )
