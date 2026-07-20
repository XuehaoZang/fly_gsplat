"""Synthetic per-point clouds with known ground truth, for T4 development.

Builds a mock fly (ellipsoid body + two thin flat-sheet wings) directly from
chosen body/wing angles using the *forward* geometry of
`reference/calc_kinematics.md` §2-§5 (body frame, stroke plane, phi/theta/eta
conventions). This is construction, not estimation — no angle is ever fit
back out of points here; that is S2+'s job. Units are meters throughout (§0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import io_schema

# ---------------------------------------------------------------------------
# Scale constants (§0: wing length ~= 2.5-3mm)
# ---------------------------------------------------------------------------

WING_LENGTH_M = 2.7e-3
BODY_LENGTH_M = 2.5e-3
WING_THICKNESS_M = 30e-6
WING_MAX_CHORD_M = 0.35 * WING_LENGTH_M

UP = np.array([0.0, 0.0, 1.0])
"""§0: up = +z (lab frame)."""


# ---------------------------------------------------------------------------
# Ground truth containers
# ---------------------------------------------------------------------------


@dataclass
class WingGroundTruth:
    """Ground truth placement for one wing.

    `span_dir` is the true leading-edge / span direction (`le` in §4/§5),
    root -> tip, unit length. `deviation_deg` is the elevation of `span_dir`
    out of the stroke plane (i.e. the true `theta`, §4). `eta_deg` is the true
    chord/pitch angle about `span_dir` (§5). Both angles are stored alongside
    `span_dir` for convenience even though `span_dir` alone determines them
    geometrically (given the body ground truth) — see `deviation_of` /
    `stroke_plane_normal` for the direct (non-estimator) check.
    """

    root: np.ndarray
    span_dir: np.ndarray
    deviation_deg: float
    eta_deg: float
    length_m: float = WING_LENGTH_M


@dataclass
class GroundTruth:
    """Full mock ground truth for one frame (§1: body yaw/pitch/roll + per-side)."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    body_center: np.ndarray
    wing_L: WingGroundTruth
    wing_R: WingGroundTruth
    body_length_m: float = BODY_LENGTH_M
    up: np.ndarray = field(default_factory=lambda: UP.copy())


# ---------------------------------------------------------------------------
# Forward geometry helpers (§2-§5) — construction only, never inverted here
# ---------------------------------------------------------------------------


def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of `v` about unit `axis` by `angle_deg`."""
    theta = math.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    return (
        v * math.cos(theta)
        + np.cross(axis, v) * math.sin(theta)
        + axis * np.dot(axis, v) * (1 - math.cos(theta))
    )


def body_axes(gt: GroundTruth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x_body, y_body, z_body) from (yaw, pitch, roll), per §2/§3.

    `x_body` is placed at azimuth `yaw` and elevation `pitch` above the
    horizontal (consistent with §3's `yaw = atan2(x_body_y, x_body_x)` and
    `pitch = 90 - arccos(x_body . up)`). The zero-roll frame is built by
    projecting `up` perpendicular to `x_body` (`z0`) and taking
    `y0 = z0 x_body` (sign convention chosen here, since §3 only names the
    frame, not its handedness); `roll` then rotates `y_body` from `y0` toward
    `z0`, matching `roll = atan2(y_body . z0, y_body . y0)`.
    """
    yaw, pitch, roll = (math.radians(a) for a in (gt.yaw_deg, gt.pitch_deg, gt.roll_deg))
    x_body = np.array(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    )
    x_body /= np.linalg.norm(x_body)

    up = gt.up / np.linalg.norm(gt.up)
    z0 = up - np.dot(up, x_body) * x_body
    z0 /= np.linalg.norm(z0)
    y0 = np.cross(z0, x_body)
    y0 /= np.linalg.norm(y0)

    y_body = math.cos(roll) * y0 + math.sin(roll) * z0
    y_body /= np.linalg.norm(y_body)
    z_body = np.cross(x_body, y_body)
    return x_body, y_body, z_body


def stroke_plane_normal(gt: GroundTruth) -> np.ndarray:
    """`n_sp` = `x_body` rotated -45 deg about `y_body` (§0, §2 step 4)."""
    x_body, y_body, _ = body_axes(gt)
    n_sp = _rotate_about_axis(x_body, y_body, -45.0)
    return n_sp / np.linalg.norm(n_sp)


def _stroke_plane_axes(
    x_body: np.ndarray, n_sp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes (x_sp, y_sp): x_body projected onto the stroke plane and
    completed to a right-handed pair with `n_sp` (§4 uses `x_sp`/`y_sp`
    implicitly via `le . x_sp`, `le . y_sp`; the exact construction is not
    spelled out there, so this is the convention this module commits to).
    """
    x_sp = x_body - np.dot(x_body, n_sp) * n_sp
    x_sp /= np.linalg.norm(x_sp)
    y_sp = np.cross(n_sp, x_sp)
    return x_sp, y_sp


def _span_dir_from_angles(
    x_sp: np.ndarray, y_sp: np.ndarray, n_sp: np.ndarray, phi_deg: float, theta_deg: float, sign_left: float
) -> np.ndarray:
    """Invert §4's `phi = atan2(sign_left*(le.y_sp), le.x_sp)`,
    `theta = 90 - arccos(n_sp.le)` to build `le` from chosen angles.
    """
    phi, theta = math.radians(phi_deg), math.radians(theta_deg)
    le = (
        math.cos(theta) * math.cos(phi) * x_sp
        + sign_left * math.cos(theta) * math.sin(phi) * y_sp
        + math.sin(theta) * n_sp
    )
    return le / np.linalg.norm(le)


def deviation_of(span_dir: np.ndarray, n_sp: np.ndarray) -> float:
    """Direct (non-estimator) read-off of theta/deviation from a span vector:
    `90 - degrees(arccos(n_sp . span_dir))`, i.e. exactly §4's `theta` formula
    applied to a known vector — used by tests to check ground-truth
    self-consistency, not to fit anything from noisy points.
    """
    dot = np.clip(np.dot(n_sp, span_dir), -1.0, 1.0)
    return 90.0 - math.degrees(math.acos(dot))


def _le_sp_normal(n_sp: np.ndarray, le: np.ndarray, sign_left: float) -> np.ndarray:
    v = np.cross(n_sp, le) if sign_left > 0 else np.cross(le, n_sp)
    return v / np.linalg.norm(v)


def _sp_chord(le: np.ndarray, le_sp_normal: np.ndarray) -> np.ndarray:
    v = np.cross(le, le_sp_normal)
    return v / np.linalg.norm(v)


def _chord_dir(le: np.ndarray, n_sp: np.ndarray, eta_deg: float, sign_left: float) -> np.ndarray:
    """§5: chord (LE->TE) at pitch `eta` about the span axis `le`."""
    le_spn = _le_sp_normal(n_sp, le, sign_left)
    spc = _sp_chord(le, le_spn)
    eta = math.radians(eta_deg)
    chord = math.cos(eta) * le_spn + sign_left * math.sin(eta) * spc
    return chord / np.linalg.norm(chord)


_SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}
"""§4: sign_left = -1 for wing_L, +1 for wing_R."""


# ---------------------------------------------------------------------------
# Default ground truth factory
# ---------------------------------------------------------------------------


def default_ground_truth(
    *,
    yaw_deg: float = 0.0,
    pitch_deg: float = 10.0,
    roll_deg: float = 0.0,
    phi_L_deg: float = 140.0,
    phi_R_deg: float = 40.0,
    theta_L_deg: float = 10.0,
    theta_R_deg: float = 10.0,
    eta_L_deg: float = 25.0,
    eta_R_deg: float = 25.0,
    root_lateral_scale: float = 1.0,
    body_center: np.ndarray | None = None,
) -> GroundTruth:
    """Build a `GroundTruth` from body pose + per-wing stroke angles.

    `phi_*` (stroke azimuth) is only used here to place `span_dir` — it is
    not stored on `GroundTruth` (§1's per-side ground truth is `span_dir`,
    `deviation` (theta), `eta`; `phi` is recoverable from `span_dir` and the
    body ground truth via the same §4 formula, not re-derived here since that
    would be an estimator). `root_lateral_scale` shrinks the hinge-to-center
    offset (< 1) to bring the two wing roots closer together, e.g. for a
    near-stroke-reversal scenario.
    """
    body_center = np.zeros(3) if body_center is None else np.asarray(body_center, dtype=float)
    stub = GroundTruth(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        body_center=body_center,
        wing_L=None,  # type: ignore[arg-type]
        wing_R=None,  # type: ignore[arg-type]
    )
    x_body, y_body, _ = body_axes(stub)
    n_sp = stroke_plane_normal(stub)
    x_sp, y_sp = _stroke_plane_axes(x_body, n_sp)

    hinge_half_span = 0.35 * BODY_LENGTH_M * root_lateral_scale
    hinge_fore = -0.1 * BODY_LENGTH_M
    root_L = body_center + hinge_fore * x_body + hinge_half_span * y_body
    root_R = body_center + hinge_fore * x_body - hinge_half_span * y_body

    span_L = _span_dir_from_angles(x_sp, y_sp, n_sp, phi_L_deg, theta_L_deg, _SIGN_LEFT["wing_L"])
    span_R = _span_dir_from_angles(x_sp, y_sp, n_sp, phi_R_deg, theta_R_deg, _SIGN_LEFT["wing_R"])

    wing_L = WingGroundTruth(root=root_L, span_dir=span_L, deviation_deg=theta_L_deg, eta_deg=eta_L_deg)
    wing_R = WingGroundTruth(root=root_R, span_dir=span_R, deviation_deg=theta_R_deg, eta_deg=eta_R_deg)
    return GroundTruth(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        body_center=body_center,
        wing_L=wing_L,
        wing_R=wing_R,
    )


# ---------------------------------------------------------------------------
# Point-cloud synthesis
# ---------------------------------------------------------------------------


def _knn_local_density(xyz: np.ndarray, k: int = 8) -> np.ndarray:
    """Placeholder `local_density` proxy: `1 / mean_dist_to_k_nn ** 3`."""
    n = xyz.shape[0]
    if n <= 1:
        return np.ones(n)
    k = min(k, n - 1)
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=k + 1)
    mean_dist = dist[:, 1:].mean(axis=1)
    return 1.0 / np.maximum(mean_dist, 1e-9) ** 3


def _empty_columns(n: int) -> dict:
    return {c: np.zeros(n) for c in io_schema.INPUT_COLUMNS if c not in ("part_label",)}


def make_body_points(gt: GroundTruth, n_points: int, rng: np.random.Generator) -> pd.DataFrame:
    """Elongated ellipsoid surface, oriented by (yaw, pitch, roll) (§2/§3).

    `orientation_*` is the true local outward surface normal of the ellipsoid
    (analytic, since the body is a smooth quadric — not a flat sheet, so no
    single global normal applies). `planarity` is set low / `linearity` high,
    matching an elongated (rather than flat) shape (§1 optional fields).
    """
    x_body, y_body, z_body = body_axes(gt)
    R = np.stack([x_body, y_body, z_body], axis=1)  # body-local -> world

    a = gt.body_length_m / 2
    b = c = gt.body_length_m * 0.18

    u = rng.uniform(0, 2 * math.pi, n_points)
    v = np.arccos(rng.uniform(-1, 1, n_points))
    local = np.stack(
        [a * np.sin(v) * np.cos(u), b * np.sin(v) * np.sin(u), c * np.cos(v)], axis=1
    )
    world = gt.body_center + local @ R.T

    normal_local = local / np.array([a ** 2, b ** 2, c ** 2])
    normal_local /= np.linalg.norm(normal_local, axis=1, keepdims=True)
    normal_world = normal_local @ R.T

    n = n_points
    cols = _empty_columns(n)
    cols["x"], cols["y"], cols["z"] = world[:, 0], world[:, 1], world[:, 2]
    centroid = world.mean(axis=0)
    cols["dist_to_centroid"] = np.linalg.norm(world - centroid, axis=1)
    axis_pt = gt.body_center
    to_pt = world - axis_pt
    proj_len = to_pt @ x_body
    perp = to_pt - np.outer(proj_len, x_body)
    cols["dist_to_principal_axis"] = np.linalg.norm(perp, axis=1)
    cols["R"], cols["G"], cols["B"] = (rng.uniform(60, 100, n), rng.uniform(40, 80, n), rng.uniform(30, 60, n))
    cols["color_oob"] = np.zeros(n, dtype=bool)
    cols["opacity"] = rng.uniform(0.7, 1.0, n)

    eig = np.array([1.0, 0.15, 0.10])
    eig = eig * (1 + rng.uniform(-0.05, 0.05, size=(n, 3)))
    cols["scale_phys_0"] = a * eig[:, 0] * 0.05
    cols["scale_phys_1"] = b * eig[:, 1] * 0.5
    cols["scale_phys_2"] = c * eig[:, 2] * 0.5
    scales = np.stack([cols["scale_phys_0"], cols["scale_phys_1"], cols["scale_phys_2"]], axis=1)
    cols["scale_ratio"] = scales.max(axis=1) / np.maximum(scales.min(axis=1), 1e-12)
    l1, l2, l3 = 1.0, 0.15, 0.10
    cols["linearity"] = np.full(n, (l1 - l2) / l1)
    cols["planarity"] = np.full(n, (l2 - l3) / l1)
    cols["sphericity"] = np.full(n, l3 / l1)
    cols["orientation_x"], cols["orientation_y"], cols["orientation_z"] = (
        normal_world[:, 0], normal_world[:, 1], normal_world[:, 2]
    )
    cols["local_density"] = _knn_local_density(world)
    cols["if_keep"] = np.ones(n, dtype=bool)

    df = pd.DataFrame(cols)
    df["part_label"] = "body"
    return df[io_schema.INPUT_COLUMNS]


def make_wing_points(
    gt: GroundTruth,
    side: str,
    n_points: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Thin flat-sheet wing: straight leading edge, gently curved trailing
    edge, thickness << chord << span (§5's "point cloud is a thin sheet").

    `planarity` is set high, `scale_phys_*` smallest axis is along the sheet
    normal, and `orientation_*` = that normal (sign: `normalize(le x chord)`,
    documented here since §1 only says orientation is "a local normal proxy"
    without fixing its sign).
    """
    wing_gt: WingGroundTruth = getattr(gt, side)
    n_sp = stroke_plane_normal(gt)
    sign_left = _SIGN_LEFT[side]
    le = wing_gt.span_dir
    chord = _chord_dir(le, n_sp, wing_gt.eta_deg, sign_left)
    normal = np.cross(le, chord)
    normal /= np.linalg.norm(normal)

    s = rng.uniform(0.0, 1.0, n_points)  # normalized span position, root(0)->tip(1)
    width = WING_MAX_CHORD_M * (0.3 + 0.7 * np.sin(math.pi * s))
    u = rng.uniform(0.0, 1.0, n_points)  # chordwise position, LE(0)->TE(1)
    t = rng.normal(0.0, WING_THICKNESS_M / 6, n_points)

    span_term = np.outer(s * wing_gt.length_m, le)
    chord_term = np.outer(u * width, chord)
    thick_term = np.outer(t, normal)
    world = wing_gt.root + span_term + chord_term + thick_term

    n = n_points
    cols = _empty_columns(n)
    cols["x"], cols["y"], cols["z"] = world[:, 0], world[:, 1], world[:, 2]
    centroid = world.mean(axis=0)
    cols["dist_to_centroid"] = np.linalg.norm(world - centroid, axis=1)
    to_pt = world - wing_gt.root
    proj_len = to_pt @ le
    perp = to_pt - np.outer(proj_len, le)
    cols["dist_to_principal_axis"] = np.linalg.norm(perp, axis=1)
    if side == "wing_L":
        cols["R"], cols["G"], cols["B"] = (rng.uniform(150, 200, n), rng.uniform(150, 200, n), rng.uniform(150, 200, n))
    else:
        cols["R"], cols["G"], cols["B"] = (rng.uniform(180, 220, n), rng.uniform(180, 220, n), rng.uniform(200, 240, n))
    cols["color_oob"] = np.zeros(n, dtype=bool)
    cols["opacity"] = rng.uniform(0.5, 0.9, n)

    eig = np.array([1.0, 0.9, 0.05])
    eig = eig * (1 + rng.uniform(-0.03, 0.03, size=(n, 3)))
    in_plane_extent = width.mean() / 10
    cols["scale_phys_0"] = in_plane_extent * eig[:, 0]
    cols["scale_phys_1"] = in_plane_extent * eig[:, 1]
    cols["scale_phys_2"] = (WING_THICKNESS_M / 2) * eig[:, 2]
    scales = np.stack([cols["scale_phys_0"], cols["scale_phys_1"], cols["scale_phys_2"]], axis=1)
    cols["scale_ratio"] = scales.max(axis=1) / np.maximum(scales.min(axis=1), 1e-12)
    l1, l2, l3 = 1.0, 0.9, 0.05
    cols["linearity"] = np.full(n, (l1 - l2) / l1)
    cols["planarity"] = np.full(n, (l2 - l3) / l1)
    cols["sphericity"] = np.full(n, l3 / l1)
    cols["orientation_x"] = np.full(n, normal[0])
    cols["orientation_y"] = np.full(n, normal[1])
    cols["orientation_z"] = np.full(n, normal[2])
    cols["local_density"] = _knn_local_density(world)
    cols["if_keep"] = np.ones(n, dtype=bool)

    df = pd.DataFrame(cols)
    df["part_label"] = side
    return df[io_schema.INPUT_COLUMNS]


def make_frame(
    gt: GroundTruth, seed: int = 0, n_body: int = 300, n_wing: int = 400
) -> tuple[pd.DataFrame, GroundTruth]:
    """Assemble body + wing_L + wing_R points for one frame matching
    `io_schema.INPUT_COLUMNS`, plus the `GroundTruth` used to build them.
    """
    rng = np.random.default_rng(seed)
    body_df = make_body_points(gt, n_body, rng)
    wl_df = make_wing_points(gt, "wing_L", n_wing, rng)
    wr_df = make_wing_points(gt, "wing_R", n_wing, rng)
    df = pd.concat([body_df, wl_df, wr_df], ignore_index=True)
    return df, gt


# ---------------------------------------------------------------------------
# Scenario factories
# ---------------------------------------------------------------------------


def scenario_clean(seed: int = 0) -> tuple[pd.DataFrame, GroundTruth]:
    """Normal mid-stroke pose, wings well separated."""
    gt = default_ground_truth()
    return make_frame(gt, seed=seed)


def scenario_reversal_contaminated(
    overlap: float = 0.9, contam_frac: float = 0.15, seed: int = 0
) -> tuple[pd.DataFrame, GroundTruth]:
    """Wings brought close together (near stroke reversal) with `contam_frac`
    of wing_L's points mislabeled as wing_R — the key stress case for the S4
    chord method (§5: "fragile exactly where two wings' hulls merge").

    `overlap` in [0, 1]: 0 = `scenario_clean`-like separation, 1 = both wings'
    stroke azimuth driven to the same value and roots pulled together. xyz
    are left untouched by the mislabeling — only `part_label` is corrupted,
    matching a T3 labeling error rather than a geometry change.
    """
    overlap = float(np.clip(overlap, 0.0, 1.0))
    phi_mid = 90.0
    phi_L = 140.0 + (phi_mid - 140.0) * overlap
    phi_R = 40.0 + (phi_mid - 40.0) * overlap
    gt = default_ground_truth(
        phi_L_deg=phi_L,
        phi_R_deg=phi_R,
        root_lateral_scale=1.0 - 0.6 * overlap,
    )
    df, gt = make_frame(gt, seed=seed)

    rng = np.random.default_rng(seed + 1)
    wl_idx = df.index[df["part_label"] == "wing_L"].to_numpy()
    n_contam = int(round(contam_frac * len(wl_idx)))
    contam_idx = rng.choice(wl_idx, size=n_contam, replace=False) if n_contam > 0 else np.array([], dtype=int)
    df.loc[contam_idx, "part_label"] = "wing_R"
    return df, gt


def scenario_noisy(
    pos_noise_std: float = 5e-5, density_imbalance: float = 0.6, seed: int = 0
) -> tuple[pd.DataFrame, GroundTruth]:
    """Positional noise (`pos_noise_std`, meters, isotropic Gaussian) plus
    uneven point density along each wing's span: keep-probability decays
    linearly from the root (1.0) to the tip (`1 - density_imbalance`),
    `density_imbalance` in [0, 1).
    """
    gt = default_ground_truth()
    df, gt = make_frame(gt, seed=seed)
    rng = np.random.default_rng(seed + 2)

    xyz = df[["x", "y", "z"]].to_numpy()
    xyz = xyz + rng.normal(0.0, pos_noise_std, size=xyz.shape)
    df[["x", "y", "z"]] = xyz

    keep_mask = np.ones(len(df), dtype=bool)
    for side in ("wing_L", "wing_R"):
        wing_gt: WingGroundTruth = getattr(gt, side)
        idx = df.index[df["part_label"] == side].to_numpy()
        pts = xyz[idx]
        s = np.clip((pts - wing_gt.root) @ wing_gt.span_dir / wing_gt.length_m, 0.0, 1.0)
        keep_prob = 1.0 - density_imbalance * s
        keep_mask[idx] = rng.uniform(0.0, 1.0, size=idx.shape[0]) < keep_prob

    df = df.loc[keep_mask].reset_index(drop=True)
    xyz_kept = df[["x", "y", "z"]].to_numpy()
    df["local_density"] = _knn_local_density(xyz_kept)
    return df, gt
