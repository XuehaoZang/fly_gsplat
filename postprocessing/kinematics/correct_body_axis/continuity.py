"""Core per-frame update: cross-frame-continuous `x_body`.

Root cause (see task background, already diagnosed by
`postprocessing/labeling/motion/diag/flip_root_cause_check.py` and
`flip_point_cloud_diag.py`): ~6% of frames have a body point cloud whose
PCA is genuinely near-degenerate (eigval_ratio = 2nd/largest eigenvalue
close to 1, i.e. a flat-disc shape) — not a point-cloud-quality artifact.
On those frames the single-frame `orient_to_reference(x_body, up)`
disambiguation used by `body_frame.py::estimate_body_frame` picks a sign
from noise, causing the long axis to flip ~180 deg frame-to-frame.

This module does not touch `body_frame.py`/`pipeline.py`/`geometry.py` —
it only calls `geometry.weighted_pca` / `geometry.project_onto_plane` /
`geometry.orient_to_reference`, which are read-only dependencies.
"""
from __future__ import annotations

import numpy as np

from .. import geometry as geo


def compute_continuous_x_body(
    body_xyz: np.ndarray,
    x_body_prev: np.ndarray | None,
    up: np.ndarray = (0.0, 0.0, 1.0),
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """One frame's continuity-corrected `x_body`.

    `body_xyz` is `(N,3)`. `x_body_prev` is the previous frame's *already
    disambiguated* `x_body`, or `None` for a sequence-start frame (first
    frame overall, or the frame right after a frame_id gap / a failed
    frame — see `build_sequence.py`, which owns that reset policy; this
    function only reacts to `x_body_prev is None`, it does not decide when
    that should be the case).

    Method:
    - `x_body_prev is None`: same single-frame heuristic as
      `body_frame.py::estimate_body_frame` — `orient_to_reference(major
      axis, up)`. `method="anchor_up"`.
    - otherwise: project `x_body_prev` onto the plane spanned by the
      current frame's two largest-variance axes (i.e. perpendicular to
      `eigvecs[:, 0]`, the smallest-variance/plane-normal axis) via
      `geo.project_onto_plane`. This is the direction *within that plane*
      closest to `x_body_prev` — since `project_onto_plane` operates on
      `x_body_prev` itself (a vector with an already-fixed sign), the
      result inherits a consistent sign for free; no separate
      disambiguation step is needed. `method="continuity"`.
      - `project_onto_plane` returns NaN when `x_body_prev` is (near)
        parallel to `eigvecs[:, 0]` (degenerate: the previous direction is
        almost exactly the current frame's plane normal, so its in-plane
        residual is ~0). Falls back to the `anchor_up` heuristic in that
        case, `method="fallback_up_degenerate"`.

    Returns `(x_body, diag)` where `diag` has keys `eigval_ratio` (2nd
    largest / largest PCA eigenvalue — near 1 means a near-degenerate,
    flat-disc body point cloud), `method` (one of `"anchor_up"`,
    `"continuity"`, `"fallback_up_degenerate"`), and `angle_to_prev_deg`
    (angle in degrees between the returned `x_body` and `x_body_prev`;
    `NaN` when `x_body_prev is None`).
    """
    body_xyz = np.asarray(body_xyz, dtype=float)
    up_hat = geo.unit(np.asarray(up, dtype=float))

    eigvals, eigvecs, _centroid = geo.weighted_pca(body_xyz, weights)
    eigval_ratio = float(eigvals[-2] / eigvals[-1]) if eigvals[-1] > 0 else float("nan")

    if x_body_prev is None:
        x_body = geo.orient_to_reference(eigvecs[:, -1], up_hat)
        return x_body, {
            "eigval_ratio": eigval_ratio,
            "method": "anchor_up",
            "angle_to_prev_deg": float("nan"),
        }

    x_body_prev = np.asarray(x_body_prev, dtype=float)
    plane_normal = eigvecs[:, 0]
    x_body = geo.project_onto_plane(x_body_prev, normal=plane_normal)

    if np.any(np.isnan(x_body)):
        x_body = geo.orient_to_reference(eigvecs[:, -1], up_hat)
        method = "fallback_up_degenerate"
    else:
        method = "continuity"

    cos_angle = float(np.clip(np.dot(x_body, x_body_prev), -1.0, 1.0))
    angle_to_prev_deg = float(np.degrees(np.arccos(cos_angle)))

    return x_body, {
        "eigval_ratio": eigval_ratio,
        "method": method,
        "angle_to_prev_deg": angle_to_prev_deg,
    }
