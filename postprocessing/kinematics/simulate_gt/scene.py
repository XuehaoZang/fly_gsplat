"""Synthetic scenes for end-to-end T3(segmentation)+T4(kinematics) validation.

Reuses `mock.py`'s forward geometry (`mock.default_ground_truth`,
`mock.make_body_points`, `mock.make_wing_points`, `mock.body_axes`,
`mock.stroke_plane_normal`, `mock._chord_dir`) verbatim -- this module does
not reimplement any construction geometry, it only:

1. Assembles body + wing_L + wing_R into one cloud and recomputes
   `dist_to_centroid` / `dist_to_principal_axis` / `local_density` over the
   *whole* assembled cloud, using the exact formula
   `utils/gaussian_features.py` (T1) uses on real data. `mock.py` computes
   these per-part (each wing's `dist_to_principal_axis` is measured from
   that wing's own span axis, not the body's) -- correct for `mock.py`'s own
   single-part unit tests, but not representative of what a T3 segmentation
   step actually receives (one undifferentiated cloud), which is the whole
   point of this package.
2. Keeps `part_label` out of the returned per-point frame (as `FrameGroundTruth
   .part_label`, row-aligned with the returned DataFrame) instead of handing
   it to whatever consumes the frame -- so a segmentation step run on the
   returned DataFrame is genuinely unlabeled, and its predictions can be
   scored against ground truth afterwards.
3. Carries forward the ground-truth vectors/angles T4 estimates
   (`x_body`/`y_body`/`z_body`/`n_sp`, per-wing `span`/`chord`, and every
   `io_schema.OUTPUT_COLUMNS` angle) as `FrameGroundTruth`, computed the same
   *construction* way `mock.py` itself does (never inverted out of the noisy
   points) so scoring never contaminates itself with estimator logic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics import mock  # noqa: E402

REF_N_BODY = 327
REF_N_WING_L = 161
REF_N_WING_R = 147
"""Reference per-part point counts, from `outputs/ctrl_009_002_ratio3_sh0_dense/
ratio3_sh0_dense/f0265/splatfacto/2026-07-23_134359/gaussian_features_f0265_labeled.csv`
(`if_keep=True` rows only, `part_label` value counts) -- a real T3-labeled
frame, picked as the point-count ratio reference so `mock.py`'s per-part
generators produce a cloud whose body/wing_L/wing_R proportions resemble an
actual capture (body ~51%, wing_L ~25%, wing_R ~23%) instead of the
wing-majority default (`mock.make_frame`'s own `n_body=300, n_wing=400`
each side, i.e. wings are 73% of the cloud) that `binary_split.py`'s
quantile thresholds and `dist_to_principal_axis`'s "whole-cloud PCA axis ~=
body axis" assumption are not calibrated for -- see `run_step1.py`'s first
run notes."""

ROOT_LATERAL_SCALE = 0.5
"""`mock.default_ground_truth`'s own `root_lateral_scale` parameter, used
here (not left at its default `1.0`) to fix a real bug found by watching the
ground-truth flight video: `mock.py`'s default `hinge_half_span` (`0.35 *
BODY_LENGTH_M` = 0.875mm) is *larger* than the body ellipsoid's own lateral
semi-axis (`b = c = 0.18 * BODY_LENGTH_M` = 0.45mm), so a wing root sits
~0.4mm off the body surface instead of on it -- visually "the wing roots
aren't attached to the body." `0.5` brings `hinge_half_span` to ~0.44mm,
approximately the body's own radius, so the root lands at the body surface.
Only overridden here (not in `mock.py` itself) to avoid changing behavior
for `mock.py`'s other, already-established consumers (`correct_wing_pitch/
synthetic_validation.py`, the `test_s*.py` suite, etc.), several of which
were written and tuned against the `1.0` default."""



@dataclass
class FrameGroundTruth:
    """Everything `evaluate.py` needs to score one frame's segmentation +
    T4 output against ground truth. Angles in degrees, vectors unit `(3,)`,
    `part_label` an `(N,)` string array row-aligned with the sibling
    unlabeled DataFrame `scene.make_unlabeled_frame` returns alongside this.
    """

    frame_id: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    phi_L_deg: float
    phi_R_deg: float
    theta_L_deg: float
    theta_R_deg: float
    eta_L_deg: float
    eta_R_deg: float
    x_body: np.ndarray
    y_body: np.ndarray
    z_body: np.ndarray
    n_sp: np.ndarray
    hinge_L: np.ndarray
    hinge_R: np.ndarray
    body_cm: np.ndarray
    span_L: np.ndarray
    span_R: np.ndarray
    chord_L: np.ndarray
    chord_R: np.ndarray
    part_label: np.ndarray


def _frame_ground_truth(
    frame_id: int, gt: mock.GroundTruth, phi_L_deg: float, phi_R_deg: float, part_label: np.ndarray,
) -> FrameGroundTruth:
    x_body, y_body, z_body = mock.body_axes(gt)
    n_sp = mock.stroke_plane_normal(gt)
    chord_L = mock._chord_dir(gt.wing_L.span_dir, n_sp, gt.wing_L.eta_deg, mock._SIGN_LEFT["wing_L"])
    chord_R = mock._chord_dir(gt.wing_R.span_dir, n_sp, gt.wing_R.eta_deg, mock._SIGN_LEFT["wing_R"])
    return FrameGroundTruth(
        frame_id=frame_id,
        yaw_deg=gt.yaw_deg, pitch_deg=gt.pitch_deg, roll_deg=gt.roll_deg,
        phi_L_deg=phi_L_deg, phi_R_deg=phi_R_deg,
        theta_L_deg=gt.wing_L.deviation_deg, theta_R_deg=gt.wing_R.deviation_deg,
        eta_L_deg=gt.wing_L.eta_deg, eta_R_deg=gt.wing_R.eta_deg,
        x_body=x_body, y_body=y_body, z_body=z_body, n_sp=n_sp,
        hinge_L=gt.wing_L.root, hinge_R=gt.wing_R.root, body_cm=gt.body_center,
        span_L=gt.wing_L.span_dir, span_R=gt.wing_R.span_dir,
        chord_L=chord_L, chord_R=chord_R,
        part_label=part_label,
    )


_APPEARANCE_REF_PATH = Path(__file__).resolve().parent / "real_appearance_reference.csv"
_appearance_ref_cache: dict[str, pd.DataFrame] | None = None


def _load_appearance_reference() -> dict[str, pd.DataFrame]:
    """Pooled real per-part `(lam1, lam2, lam3, opacity, R, G, B)` rows, see
    `calibrate_appearance.py`'s docstring for provenance. Cached at module
    level (loaded once per process, ~400k rows) since every generated frame
    bootstraps from it.
    """
    global _appearance_ref_cache
    if _appearance_ref_cache is None:
        if not _APPEARANCE_REF_PATH.exists():
            raise FileNotFoundError(
                f"{_APPEARANCE_REF_PATH} not found -- run `python -m postprocessing."
                f"kinematics.simulate_gt.calibrate_appearance` once to build it from "
                f"real _labeled.csv files under outputs/."
            )
        ref = pd.read_csv(_APPEARANCE_REF_PATH)
        _appearance_ref_cache = {
            part: ref.loc[ref["part_label"] == part].reset_index(drop=True)
            for part in ("body", "wing_L", "wing_R")
        }
    return _appearance_ref_cache


def _resample_appearance_features(
    df: pd.DataFrame, part_label: np.ndarray, rng: np.random.Generator,
) -> pd.DataFrame:
    """Overwrite each point's Gaussian-shape (`scale_phys_0/1/2` and the
    `linearity`/`planarity`/`sphericity`/`scale_ratio` derived from them) and
    `opacity`/`R`/`G`/`B` by bootstrapping (with replacement) a real row of
    the matching part from `_load_appearance_reference()`.

    Why: comparing `mock.py`'s synthetic points against real `f0265` showed
    `mock.py`'s idealized-flat-sheet wing assumption is far from what a real
    Gaussian-splat reconstruction of a flapping wing looks like -- real
    planarity is body~0.20 vs wing~0.23 (barely separable) vs `mock.py`'s
    body~0.18 vs wing~0.89; real `R` is body~0.17 vs wing~0.33-0.36 (the
    actual signal `kmeans_split.py`'s v2 seed rule uses) vs `mock.py`'s `R`
    being on a 60-220 scale instead of the real `[0,1]` range at all.
    Bootstrapping real rows (rather than hand-fit target means) automatically
    reproduces the real *distribution* -- including e.g. the ~5% tail of real
    body points with `opacity>=0.98` that `kmeans_split.seed_mask` keys off
    of, which a mean-only match would not preserve.

    `xyz`/`orientation_*`/`dist_to_centroid`/`dist_to_principal_axis`/
    `local_density` are untouched -- those come from `mock.py`'s forward
    kinematics (validated against real root/tip/hinge geometry, see chat
    history) and `_recompute_whole_cloud_features`, not from this reference
    table.
    """
    ref = _load_appearance_reference()
    df = df.copy()
    n = len(df)
    lam = np.empty((n, 3))
    for part in ("body", "wing_L", "wing_R"):
        idx = np.where(part_label == part)[0]
        if len(idx) == 0:
            continue
        pool = ref[part]
        draw = pool.iloc[rng.integers(0, len(pool), size=len(idx))]
        lam[idx, 0] = draw["lam1"].to_numpy()
        lam[idx, 1] = draw["lam2"].to_numpy()
        lam[idx, 2] = draw["lam3"].to_numpy()
        row_idx = df.index[idx]
        df.loc[row_idx, "opacity"] = draw["opacity"].to_numpy()
        df.loc[row_idx, "R"] = draw["R"].to_numpy()
        df.loc[row_idx, "G"] = draw["G"].to_numpy()
        df.loc[row_idx, "B"] = draw["B"].to_numpy()

    df["scale_phys_0"], df["scale_phys_1"], df["scale_phys_2"] = lam[:, 0], lam[:, 1], lam[:, 2]
    linearity, planarity, sphericity = mock._linearity_planarity_sphericity(lam)
    df["linearity"], df["planarity"], df["sphericity"] = linearity, planarity, sphericity
    df["scale_ratio"] = lam[:, 0] / np.maximum(lam[:, 2], 1e-12)
    return df


def _recompute_whole_cloud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite `dist_to_centroid`/`dist_to_principal_axis`/`local_density`
    in place (on a copy) using the whole-cloud PCA formula
    `utils/gaussian_features.py` uses on real T1 output -- see module
    docstring point 1.
    """
    xyz = df[["x", "y", "z"]].to_numpy()
    centroid = xyz.mean(axis=0)
    dist_to_centroid = np.linalg.norm(xyz - centroid, axis=1)

    cov = np.cov((xyz - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal_axis = eigvecs[:, order[0]]
    rel = xyz - centroid
    proj_len = rel @ principal_axis
    proj_vec = np.outer(proj_len, principal_axis)
    dist_to_principal_axis = np.linalg.norm(rel - proj_vec, axis=1)

    df = df.copy()
    df["dist_to_centroid"] = dist_to_centroid
    df["dist_to_principal_axis"] = dist_to_principal_axis
    df["local_density"] = mock._knn_local_density(xyz)
    return df


def make_unlabeled_frame(
    gt: mock.GroundTruth,
    frame_id: int,
    phi_L_deg: float,
    phi_R_deg: float,
    seed: int = 0,
    n_body: int = REF_N_BODY,
    n_wing_L: int = REF_N_WING_L,
    n_wing_R: int = REF_N_WING_R,
    resample_appearance: bool = True,
) -> tuple[pd.DataFrame, FrameGroundTruth]:
    """One frame's unlabeled point cloud (`part_label` column dropped) plus
    the `FrameGroundTruth` needed to score a segmentation/T4 estimate
    against it. `phi_L_deg`/`phi_R_deg` must be the same values used to
    build `gt` (via `mock.default_ground_truth`) -- passed in separately
    because `mock.GroundTruth` itself does not store `phi` (see `mock.py`'s
    own `default_ground_truth` docstring on why), so the caller carries it.
    `n_body`/`n_wing_L`/`n_wing_R` default to `REF_N_BODY`/`REF_N_WING_L`/
    `REF_N_WING_R` (a real labeled frame's point-count ratio, see that
    constant's docstring). `resample_appearance` (default `True`) replaces
    `mock.py`'s idealized Gaussian-shape/opacity/color with values
    bootstrapped from real data, see `_resample_appearance_features`; set
    `False` only to reproduce `mock.py`'s own (unrealistic but analytically
    clean) defaults, e.g. for a test that wants the old behavior.
    """
    rng = np.random.default_rng(seed)
    body_df = mock.make_body_points(gt, n_body, rng)
    wl_df = mock.make_wing_points(gt, "wing_L", n_wing_L, rng)
    wr_df = mock.make_wing_points(gt, "wing_R", n_wing_R, rng)
    df = pd.concat([body_df, wl_df, wr_df], ignore_index=True)

    gt_part_label = df["part_label"].to_numpy(copy=True)
    if resample_appearance:
        df = _resample_appearance_features(df, gt_part_label, rng)
    df = _recompute_whole_cloud_features(df)
    df_unlabeled = df.drop(columns=["part_label"])

    frame_gt = _frame_ground_truth(frame_id, gt, phi_L_deg, phi_R_deg, gt_part_label)
    return df_unlabeled, frame_gt


def scenario_step1_static(
    n_frames: int = 10,
    pitch_deg: float = 45.0,
    n_body: int = REF_N_BODY,
    n_wing_L: int = REF_N_WING_L,
    n_wing_R: int = REF_N_WING_R,
    seed: int = 0,
) -> list[tuple[pd.DataFrame, FrameGroundTruth]]:
    """Step 1 (simplest end-to-end case): one static body pose
    (`yaw=0, pitch=pitch_deg, roll=0`) + two flat elliptical wings at
    `mock.default_ground_truth`'s own default stroke angles, repeated for
    `n_frames` independently point-sampled frames. No positional noise, no
    density imbalance, no time-varying angles yet -- those are later steps;
    this one only exercises the full segment -> T4 -> compare chain.
    `n_body`/`n_wing_L`/`n_wing_R` default to the real-frame-calibrated
    `REF_N_*` counts (see that constant's docstring) so the point-count
    ratio a segmentation step sees resembles an actual capture. `phi_L_deg`/
    `phi_R_deg` are negative (`-140`/`-40`, not `mock.py`'s own positive
    `140`/`40` defaults) and `root_lateral_scale=ROOT_LATERAL_SCALE` -- see
    `PHI_MEAN_DEG`'s and `ROOT_LATERAL_SCALE`'s docstrings for the two real
    bugs (wings crossing the midline; wing roots floating off the body
    surface) this fixes.
    """
    phi_L_deg, phi_R_deg = -140.0, -40.0
    gt = mock.default_ground_truth(
        yaw_deg=0.0, pitch_deg=pitch_deg, roll_deg=0.0,
        phi_L_deg=phi_L_deg, phi_R_deg=phi_R_deg,
        root_lateral_scale=ROOT_LATERAL_SCALE,
    )
    frames = []
    for frame_id in range(n_frames):
        df, frame_gt = make_unlabeled_frame(
            gt, frame_id, phi_L_deg, phi_R_deg,
            seed=seed + frame_id, n_body=n_body, n_wing_L=n_wing_L, n_wing_R=n_wing_R,
        )
        frames.append((df, frame_gt))
    return frames


# ---------------------------------------------------------------------------
# Step 2: flapping sequence, calibrated against literature + real data
# ---------------------------------------------------------------------------

FPS = 16000.0
"""User-confirmed camera rate (`reference/s6b_real_data_diagnostics_findings.md`
line 17), reused here so `WINGBEAT_PERIOD_FRAMES` below is expressed in the
same frame units this codebase's own real-data diagnostics use."""

WINGBEAT_HZ = 200.0
"""Drosophila hovering/free-flight wingbeat frequency, Dickinson/Fry-era
literature range ~180-220 Hz -- same band `diagnostics.py`'s own
`ANGLE_BANDS` docstring cites as its source. `s6b_real_data_diagnostics_findings
.md` §2/§3 could not itself confirm a wingbeat frequency from the real
100-frame clip (FFT resolution too coarse, naive reversal count 4-5x the ~2-3
a single ~200 Hz cycle would give) -- so this constant is literature, not
measured from this repo's data."""

WINGBEAT_PERIOD_FRAMES = FPS / WINGBEAT_HZ
"""80 frames/cycle at the constants above -- a 100-frame clip is ~1.25
cycles, matching the real `f0000-f0099` clip length used throughout T4's own
real-data diagnostics, so a `simulate_gt` run at `n_frames=100` is directly
comparable in scale."""

BODY_DRIFT_PERIOD_FRAMES = 10.0 * WINGBEAT_PERIOD_FRAMES
"""Body yaw/pitch/roll drift period, 10x the wingbeat period (800 frames) --
"slowly varying" relative to the wingbeat, i.e. a 100-frame clip only covers
~1/8 of one drift cycle (a gentle trend, not a full swing), consistent with
a body that doesn't complete a maneuver within one ~6ms clip."""

YAW_BASE_DEG, YAW_AMP_DEG = 0.0, 10.0
PITCH_BASE_DEG, PITCH_AMP_DEG = 25.0, 8.0
"""`PITCH_BASE_DEG=25.0` matches the real dataset's observed mean pitch
(25.8 deg, `s6b_real_data_diagnostics_findings.md` §5)."""
ROLL_BASE_DEG, ROLL_AMP_DEG = 0.0, 15.0

PHI_MEAN_DEG, PHI_AMP_DEG = -90.0, 50.0
"""Stroke amplitude range `[-140, -40]` deg (peak-to-peak 100 deg), swept by
both wings in phase (mirrors typical straight hovering/forward flight; L/R
asymmetry only shows up during a turn, out of scope here). Literature
peak-to-peak stroke amplitude for Drosophila runs somewhat higher (~130-160
deg); 100 deg is a deliberately more conservative range so wings stay well
short of a fully vertical stroke, which is where `chord.py`'s LE/TE
disambiguation is documented to be most fragile (see that module's own
docstring) -- not something this first flapping-validation pass is trying
to stress-test yet.

**Sign is deliberately negative** -- a real bug the ground-truth flight
video surfaced (`render_flight_video.py`): `wing_angles.py`/`mock.py`'s
shared convention is `phi = atan2(sign_left * span.y_sp, span.x_sp)`,
`sign_left=-1` for `wing_L`/`+1` for `wing_R`, and `y_sp == y_body` exactly
at roll=0 (verified numerically). `y_body` points from `hinge_R` toward
`hinge_L` (`calc_kinematics.md` §2), so `wing_L`'s root sits on the
`+y_body` side -- but `span_L . y_body = -sin(phi_L)`, which is *negative*
(pointing across the midline, toward the `wing_R` side) for **any**
`phi_L` in `(0, 180)` deg. Checked directly: `mock.default_ground_truth()`'s
own untouched defaults (`phi_L=140, phi_R=40`, positive) already put *both*
wingtips on the wrong side of the body -- this had never been caught
because nothing in this repo had rendered the combined body+wing_L+wing_R
cloud and looked at which side each wing ended up on; every other `mock.py`
consumer scores one wing against its own known ground-truth vectors in
isolation, never both wings + body together. `phi` in `(-180, 0)` deg makes
`sign_left * sin(phi)` positive for both sides, keeping each wingtip on its
own root's side for the entire range (verified across the whole interval,
not just at these endpoints). Not changed in `mock.py` itself or its
defaults -- only here -- to avoid affecting `mock.py`'s other, already
-established consumers (`correct_wing_pitch/synthetic_validation.py`, the
`test_s*.py` suite, etc.)."""

THETA_MEAN_DEG, THETA_AMP_DEG = 10.0, 15.0
"""Stroke-plane deviation, oscillating at *2x* the wingbeat frequency (the
classic dipteran figure-eight wingtip path: two deviation peaks per stroke
cycle, one per half-stroke) -- amplitude keeps `theta` within the
`ANGLE_BANDS["theta_L"/"theta_R"]` "concerning" band (+-40 deg,
`diagnostics.py`) with room to spare."""

ETA_MEAN_DEG, ETA_AMP_DEG = 45.0, 35.0
"""Feathering/pitch angle, swept `cos`-phase relative to `phi`'s `sin` (a
quarter-cycle offset) so `eta` is mid-swing -- fastest-changing -- exactly
at each `phi` reversal, matching the real supination/pronation timing
(feathering flips near each stroke reversal, not mid-stroke) even though a
plain sinusoid is a much softer transition than the real near-trapezoidal
waveform -- a simplification flagged here, not hidden."""


def _flap_angles_deg(t: float) -> dict:
    """Ground-truth body pose + per-wing stroke angles at frame time `t`
    (float frame index, same units as `frame_id`) -- see the `*_BASE_DEG`/
    `*_AMP_DEG`/`*_PERIOD_FRAMES` constants above for the literature/real-data
    provenance of every number here.
    """
    body_phase = 2.0 * np.pi * t / BODY_DRIFT_PERIOD_FRAMES
    wing_phase = 2.0 * np.pi * t / WINGBEAT_PERIOD_FRAMES
    return dict(
        yaw_deg=YAW_BASE_DEG + YAW_AMP_DEG * np.sin(body_phase),
        pitch_deg=PITCH_BASE_DEG + PITCH_AMP_DEG * np.sin(body_phase + 0.5),
        roll_deg=ROLL_BASE_DEG + ROLL_AMP_DEG * np.sin(body_phase + 1.0),
        phi_L_deg=PHI_MEAN_DEG + PHI_AMP_DEG * np.sin(wing_phase),
        phi_R_deg=PHI_MEAN_DEG + PHI_AMP_DEG * np.sin(wing_phase),
        theta_L_deg=THETA_MEAN_DEG + THETA_AMP_DEG * np.sin(2.0 * wing_phase),
        theta_R_deg=THETA_MEAN_DEG + THETA_AMP_DEG * np.sin(2.0 * wing_phase),
        eta_L_deg=ETA_MEAN_DEG + ETA_AMP_DEG * np.cos(wing_phase),
        eta_R_deg=ETA_MEAN_DEG + ETA_AMP_DEG * np.cos(wing_phase),
    )


def scenario_step2_flapping(
    n_frames: int = 100,
    n_body: int = REF_N_BODY,
    n_wing_L: int = REF_N_WING_L,
    n_wing_R: int = REF_N_WING_R,
    seed: int = 0,
) -> list[tuple[pd.DataFrame, FrameGroundTruth]]:
    """Step 2: slowly-drifting body yaw/pitch/roll + sinusoidally flapping
    wings (`phi`/`theta`/`eta` all time-varying, see `_flap_angles_deg`),
    `n_frames` frames (default 100, matching the real `f0000-f0099` clip
    scale) -- unlike `scenario_step1_static`, the wings actually move frame
    to frame. Still no positional noise / density imbalance (later step).
    """
    frames = []
    for frame_id in range(n_frames):
        angles = _flap_angles_deg(float(frame_id))
        gt = mock.default_ground_truth(
            yaw_deg=angles["yaw_deg"], pitch_deg=angles["pitch_deg"], roll_deg=angles["roll_deg"],
            phi_L_deg=angles["phi_L_deg"], phi_R_deg=angles["phi_R_deg"],
            theta_L_deg=angles["theta_L_deg"], theta_R_deg=angles["theta_R_deg"],
            eta_L_deg=angles["eta_L_deg"], eta_R_deg=angles["eta_R_deg"],
            root_lateral_scale=ROOT_LATERAL_SCALE,
        )
        df, frame_gt = make_unlabeled_frame(
            gt, frame_id, angles["phi_L_deg"], angles["phi_R_deg"],
            seed=seed + frame_id, n_body=n_body, n_wing_L=n_wing_L, n_wing_R=n_wing_R,
        )
        frames.append((df, frame_gt))
    return frames
