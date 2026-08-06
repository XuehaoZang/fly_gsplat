"""Step 3 (round 3, "stroke=90 anchor" task): synthetic ground-truth generator
for validating the bidirectional reconstruction (`bidirectional.py`) against
baseline (per-frame independent) and causal-only (round 1's `continuity.py`)
`x_body` estimates.

Deliberately independent of `mock.py` (the `correct_wing_pitch` synthetic
generator) -- this only needs a *body* point cloud with a controllable
degeneracy schedule and known-anchor-position labels, not a full wing/hinge
kinematic model.

Design (see task spec §3):
- Ground-truth body direction `x_body(t)` is a slow sinusoidal yaw drift plus
  a faster, smaller wobble, both in the world x-y plane, applied to a body
  axis with a small constant pitch tilt (`PITCH_DEG`) out of that plane. The
  small tilt is deliberate: it reproduces the real-data failure mode
  documented in `identity_flip_stats.py` ("body长轴近似水平,跟UP(+z)接近垂直,
  disambiguation本身容易picked wrong sign") -- `dot(x_body, UP)` stays small
  (not exactly 0, so `orient_to_reference` is *usually* right, but the
  margin is thin enough that per-frame PCA noise can flip it, especially
  when the point cloud is also near-degenerate).
- Each frame's body point cloud is Gaussian, axis-aligned to the true local
  frame `(a=x_body(t), b=cross(UP,a) normalized, c=cross(a,b))`, with
  per-axis std `(A_LEN, B_LEN(zone), C_LEN)`. `A_LEN` and `C_LEN` are fixed;
  `B_LEN` depends on the frame's zone within a repeating `PERIOD_LEN`-frame
  cycle:
  - `zone="anchor"` (`t % PERIOD_LEN == 0`, one frame per period): small
    `B_LEN` -> the cloud is cleanly elongated (expected eigval_ratio
    `(B_LEN/A_LEN)**2 ~= 0.04`), i.e. a trustworthy T-pose-equivalent anchor.
    Matches task spec §3 "在形状最细长/最典型的时刻插入模拟锚点".
  - `zone="degenerate"` (a contiguous `DEGEN_LEN`-frame window per period):
    larger `B_LEN` (expected ratio `~=0.36`) -- deliberately in the 0.1-0.4
    "moderately, persistently degenerate" band the task background calls out
    (f0513-520/f0313/f0317), not an extreme >0.7 flat-disc.
  - `zone="normal"` (everything else): intermediate `B_LEN` (expected ratio
    `~=0.16`), representative of a well-conditioned, non-degenerate frame.
  Actual per-frame eigval_ratio (measured from the sampled points, not the
  expected value above) varies frame to frame from sampling noise, same as
  real data -- callers should read the measured value off the point cloud,
  not assume the zone's nominal ratio.

This module only *generates* data; it does not itself compute any x_body
estimate or run any of the three comparison methods (that's
`diag/g_synthetic_validation.py`'s job) or write any diagnostic output.
"""
from __future__ import annotations

import numpy as np

UP = np.array([0.0, 0.0, 1.0])

N_PERIODS = 10
PERIOD_LEN = 30
N_FRAMES = N_PERIODS * PERIOD_LEN

DEGEN_START_OFFSET = 10
DEGEN_LEN = 10
"""Degenerate window is `[DEGEN_START_OFFSET, DEGEN_START_OFFSET+DEGEN_LEN)`
within each period -- 10 consecutive frames, comparable in length to the
real f0513-520 step (~8 frames) that motivated this task."""

DRIFT_AMP_DEG = 25.0
DRIFT_PERIOD_FRAMES = 250.0
"""Slow drift period deliberately not a multiple of `PERIOD_LEN` (250 vs 30)
so the degeneracy cycle and the "true motion" cycle are not accidentally
synchronized."""
WOBBLE_AMP_DEG = 6.0
WOBBLE_PERIOD_FRAMES = 17.0

PITCH_DEG = 6.0
"""Constant tilt of the body axis out of the world x-y plane -- see module
docstring: keeps `dot(x_body, UP)` small but not exactly 0, reproducing the
real near-horizontal-body sign-ambiguity regime."""

A_LEN = 1.0
C_LEN = 0.15
B_LEN_BY_ZONE = {"anchor": 0.2, "normal": 0.4, "degenerate": 0.85}
"""Expected eigval_ratio = (B_LEN/A_LEN)**2 per zone: anchor~=0.04,
normal~=0.16, degenerate~=0.72 -- a genuinely near-degenerate "flat disc"
(task spec §3 literally says "扁盘"/high eigval_ratio for the synthetic
degenerate frames), comparable in magnitude to `flip_root_cause_check.py`'s
measured flip-group eigval_ratio median of 0.548 (see `b_jitter_by_bucket.py`
docstring) -- i.e. large enough that the top-2 eigenvectors' angular estimate
is genuinely unstable under point-sampling noise, not just moderately noisy.
(The real f0513-520 case that motivated this task sits in a milder 0.1-0.4
band -- see `diag/g_synthetic_validation.py`'s summary table, which also
reports the "normal"-zone error away from this extreme window, showing the
causal chain already drifts measurably from ordinary per-step noise alone,
well before reaching this synthetic zone's more extreme ratio.)"""

N_POINTS_PER_FRAME = 200

_PITCH_RAD = np.radians(PITCH_DEG)
_V0 = np.array([np.cos(_PITCH_RAD), 0.0, np.sin(_PITCH_RAD)])
"""Body axis at yaw=0: mostly along world +x, tilted `PITCH_DEG` up out of
the x-y plane."""


def zone_at(t: int) -> str:
    phase = t % PERIOD_LEN
    if phase == 0:
        return "anchor"
    if DEGEN_START_OFFSET <= phase < DEGEN_START_OFFSET + DEGEN_LEN:
        return "degenerate"
    return "normal"


def true_yaw_deg(t: int) -> float:
    return (DRIFT_AMP_DEG * np.sin(2.0 * np.pi * t / DRIFT_PERIOD_FRAMES)
            + WOBBLE_AMP_DEG * np.sin(2.0 * np.pi * t / WOBBLE_PERIOD_FRAMES))


def true_x_body(t: int) -> np.ndarray:
    """Unit ground-truth body axis at frame `t`: `_V0` rotated about world
    `UP` by `true_yaw_deg(t)`."""
    yaw = np.radians(true_yaw_deg(t))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    v = rot @ _V0
    return v / np.linalg.norm(v)


def _local_frame(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.cross(UP, a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    c = c / np.linalg.norm(c)
    return a, b, c


def make_frame_points(t: int, rng: np.random.Generator) -> np.ndarray:
    """`(N_POINTS_PER_FRAME, 3)` synthetic body point cloud for frame `t`,
    centered at the origin (body_cm is not modeled -- only direction
    matters for this validation)."""
    a, b, c = _local_frame(true_x_body(t))
    b_len = B_LEN_BY_ZONE[zone_at(t)]
    coeffs = rng.normal(0.0, 1.0, size=(N_POINTS_PER_FRAME, 3))
    coeffs[:, 0] *= A_LEN
    coeffs[:, 1] *= b_len
    coeffs[:, 2] *= C_LEN
    axes = np.stack([a, b, c], axis=0)  # (3,3), row i = axis i
    return coeffs @ axes


def generate_synthetic_sequence(seed: int = 0) -> dict:
    """Full `N_FRAMES`-frame synthetic sequence. Returns a dict with parallel
    per-frame structures keyed by `frame_id` (== `t`, `0..N_FRAMES-1`):
    `frame_ids` (list), `body_xyz_by_frame`, `gt_x_body`, `zone`,
    `is_anchor_gt`.
    """
    rng = np.random.default_rng(seed)
    frame_ids = list(range(N_FRAMES))
    body_xyz_by_frame = {}
    gt_x_body = {}
    zone = {}
    is_anchor_gt = {}
    for t in frame_ids:
        body_xyz_by_frame[t] = make_frame_points(t, rng)
        gt_x_body[t] = true_x_body(t)
        zone[t] = zone_at(t)
        is_anchor_gt[t] = zone[t] == "anchor"
    return {
        "frame_ids": frame_ids,
        "body_xyz_by_frame": body_xyz_by_frame,
        "gt_x_body": gt_x_body,
        "zone": zone,
        "is_anchor_gt": is_anchor_gt,
    }
