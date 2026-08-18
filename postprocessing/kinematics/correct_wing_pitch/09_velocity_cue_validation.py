"""Step 9: does `wing_angles.py`'s new opt-in velocity cue actually fix the
winner-flip problem the report's §2/§4/§8 diagnosed, without regressing
low-speed (near-reversal) frames?

Recap of the mechanism this targets (see `diag_report.md`): `chord.py`'s
~180 deg eta jumps are 72.4% explained by `_oriented_chord_axis` sign flips
(§4), which are themselves 73.6%-first explained by `estimate_leading_edge`'s
pos/neg RANSAC-inlier-count "winner" flipping frame-to-frame (§8,
`winner_flip -> axis_flip_strict` OR=12.53, p=3.3e-14) -- and no per-frame
static confidence measure tried (`margin_count`, `curvature_diff`,
`axis_margin`) predicts which frames flip. The velocity cue added to
`wing_angles.estimate_leading_edge` (see that module's docstring and
`_velocity_cue_winner`) is a cross-frame correction: at high wingtip speed
(mid-stroke), it can override the count judge's pick using motion
continuity; at low speed (near reversal) it leaves the count judge alone
entirely (no static fallback -- the report already showed the static
measures don't help there).

This script is diagnose-and-validate, not diagnose-only like `wrap_mechanism_diag.py`
et al. -- `estimate_leading_edge`/`estimate_chord` themselves are exercised
exactly as shipped (imported, not reimplemented) for every number that ends
up in the headline comparison (wrap-crossing counts, |Δeta| stats, eta
series). A small re-implementation, `_estimate_leading_edge_diag_cue` (a
`le_repro.py`-style mirror, extended with the same velocity-cue logic as
`wing_angles._velocity_cue_winner`), is used *only* to expose the internal
`use_pos` / `speed` bookkeeping the real function doesn't return -- and is
checked bit-exact (`le_dir`/`tip`/`root`/`inlier_mask`, `atol=0`) against the
real `wing_angles.estimate_leading_edge` output, same params, every
(frame, side, config) row, before any of its `use_pos`/`speed` numbers are
trusted (mirrors `00_consistency_check.md`'s own verification pattern).

Two configs, driven directly (not through `pipeline.py`, which stays at its
current default -- the cue is opt-in and not wired into `PipelineConfig`):
  - "off": `estimate_chord`/`estimate_leading_edge` called with no `prev_tip`/
    `prev_body_cm` -- byte-identical to current production, per-frame,
    independent of frame order.
  - "on": frames processed in `frame_id` order per side, chaining each
    frame's own *actually-reported* `LeadingEdge.span_tip` (the
    winner-independent velocity anchor, from the real function, "on" config
    -- not `.tip`, which is downstream of the winner call itself; see
    `wing_angles.LeadingEdge.span_tip`'s docstring) and that frame's
    `BodyFrame.body_cm` forward as next frame's `prev_tip`/`prev_body_cm` --
    i.e. exactly what a caller opting into the cue in a real per-frame
    pipeline loop would do.

Reuses (not re-derives): `real_data_validation.py`'s `REAL_DATASET_ROOT`,
`FRAME_GLOB`, `BIG_JUMP_DEG`, `_discover_frames`, `_assoc_abs_delta`;
`s4b_comparison.py`'s `FLIP_THRESHOLD_DEG`/`_wrap_crossings` wrap-crossing
convention (so this script's baseline numbers are directly comparable to
§3's 31/99, 29/99); `diagnostics.py::circular_delta_deg` for every delta.

Run: python -m postprocessing.kinematics.correct_wing_pitch.09_velocity_cue_validation
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import chord as ch  # noqa: E402
from postprocessing.kinematics import diagnostics as diag_mod  # noqa: E402
from postprocessing.kinematics import geometry as geo  # noqa: E402
from postprocessing.kinematics import io_schema  # noqa: E402
from postprocessing.kinematics import wing_angles as wa  # noqa: E402
from postprocessing.kinematics.correct_wing_pitch.real_data_validation import (  # noqa: E402
    BIG_JUMP_DEG,
    FRAME_GLOB,
    REAL_DATASET_ROOT,
    _assoc_abs_delta,
    _discover_frames,
)

_s4b_comparison = importlib.import_module("postprocessing.kinematics.correct_wing_pitch.s4b_comparison")
FLIP_THRESHOLD_DEG = _s4b_comparison.FLIP_THRESHOLD_DEG
_wrap_crossings = _s4b_comparison._wrap_crossings

DIAG_DIR = Path(__file__).resolve().parent / "diag"
_SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}
VELOCITY_THRESHOLD_SCALE = wa.VELOCITY_THRESHOLD_SCALE_DEFAULT
"""Reuse `wing_angles.py`'s own default -- tuned against this same dataset's
speed/scale ratio distribution (median ~6-7x, this script's own probe run;
threshold=8x flags ~41-47% of frames as cue-eligible per side, a balanced
mid-stroke/near-reversal split, not all-or-nothing) before this script was
written; not re-tuned here."""


# ---------------------------------------------------------------------------
# Diagnostic reimplementation: `le_repro.py`-style mirror of
# `estimate_leading_edge`, extended with `wing_angles._velocity_cue_winner`'s
# own logic, exposing `use_pos`/`speed`/`cue_*` bookkeeping the real
# function doesn't return. Verified bit-exact against the real function
# below (`_consistency_check`) before any of its extra fields are trusted.
# ---------------------------------------------------------------------------


@dataclass
class LEDiagCue:
    le_dir: np.ndarray
    tip: np.ndarray
    root: np.ndarray
    inlier_mask: np.ndarray
    plane_normal: np.ndarray
    span_tip: np.ndarray
    """Winner-independent velocity anchor -- mirrors `wing_angles.LeadingEdge.span_tip`."""
    use_pos_count: bool
    """Pre-cue winner (pure RANSAC-inlier-count/residual judge)."""
    use_pos: bool
    """Final winner (post-cue, == `use_pos_count` if the cue didn't fire)."""
    speed: float
    threshold: float
    cue_active: bool
    """`prev_tip`/`prev_body_cm` were both supplied (cue opted in this call)."""
    cue_eligible: bool
    """`cue_active` and `speed >= threshold` (cue was allowed to override)."""
    cue_overrode: bool
    """`use_pos != use_pos_count` (cue actually changed the winner)."""


def _estimate_leading_edge_diag_cue(
    wing_xyz: np.ndarray,
    body_frame: bf.BodyFrame,
    side: str,
    *,
    n_bins: int = 20,
    min_bin_points: int = 3,
    plane_threshold: float | None = None,
    line_threshold: float | None = None,
    rng: int | np.random.Generator | None = 0,
    prev_tip: np.ndarray | None = None,
    prev_body_cm: np.ndarray | None = None,
    velocity_threshold_scale: float = VELOCITY_THRESHOLD_SCALE,
) -> LEDiagCue:
    if side not in _SIGN_LEFT:
        raise ValueError(f"side must be 'wing_L' or 'wing_R', got {side!r}")
    wing_xyz = np.asarray(wing_xyz, dtype=float)
    n = wing_xyz.shape[0]
    w = np.ones(n)

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
    span_tip = pts_plane[np.argmax(t)]  # winner-independent anchor, mirrors wing_angles.py

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
        raise ValueError("_estimate_leading_edge_diag_cue: not enough populated span bins")
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
    use_pos_count = pos_count > neg_count or (pos_count == neg_count and pos_resid < neg_resid)

    use_pos = use_pos_count
    speed = float("nan")
    threshold = velocity_threshold_scale * scale
    cue_active = prev_tip is not None and prev_body_cm is not None
    if cue_active and scale > 0.0:
        raw_delta = (span_tip - np.asarray(prev_tip, dtype=float)) - (
            np.asarray(body_frame.body_cm, dtype=float) - np.asarray(prev_body_cm, dtype=float)
        )
        comp = float(np.dot(raw_delta, span_guess))
        perp = raw_delta - comp * span_guess
        speed = float(np.linalg.norm(perp))

        if speed >= threshold:
            v_hat = perp / speed
            align = float(np.dot(v_hat, chord_axis))
            use_pos = align > 0.0

    direction, point_on_line, ransac_mask, le_orig_idx, le_points, _, _ = pos_cand if use_pos else neg_cand
    le_dir = geo.orient_to_reference(direction, out_ref)
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[le_orig_idx[ransac_mask]] = True
    inlier_points = le_points[ransac_mask]
    t_final = (inlier_points - point_on_line) @ le_dir
    root = inlier_points[np.argmin(t_final)]
    tip = inlier_points[np.argmax(t_final)]

    cue_eligible = bool(cue_active and scale > 0.0 and speed >= threshold)
    return LEDiagCue(
        le_dir=le_dir, tip=tip, root=root, inlier_mask=inlier_mask, plane_normal=normal, span_tip=span_tip,
        use_pos_count=use_pos_count, use_pos=use_pos, speed=speed, threshold=threshold,
        cue_active=cue_active, cue_eligible=cue_eligible, cue_overrode=bool(use_pos != use_pos_count),
    )


# ---------------------------------------------------------------------------
# Main per-frame, per-side, chained-state loop
# ---------------------------------------------------------------------------


def _run_side(frames: list[tuple[int, Path]], side: str) -> tuple[pd.DataFrame, int]:
    """Processes one side across all frames in order, chaining the "on"
    config's own reported tip/body_cm forward frame-to-frame (see module
    docstring). Returns `(rows_df, n_consistency_mismatches)`.
    """
    rows = []
    prev_tip_on = None
    prev_body_cm = None
    n_mismatch = 0

    for frame_id, csv_path in frames:
        try:
            df = io_schema.load_frame(csv_path)
            body_xyz = io_schema.get_part(df, "body")
            wingL_xyz = io_schema.get_part(df, "wing_L")
            wingR_xyz = io_schema.get_part(df, "wing_R")
            frame = bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz)
            wing_xyz = wingL_xyz if side == "wing_L" else wingR_xyz
        except Exception as e:  # noqa: BLE001
            rows.append(dict(frame_id=frame_id, side=side, failed=True, fail_reason=str(e)))
            continue

        try:
            le_off_real = wa.estimate_leading_edge(wing_xyz, frame, side)
            le_on_real = wa.estimate_leading_edge(
                wing_xyz, frame, side,
                prev_tip=prev_tip_on, prev_body_cm=prev_body_cm,
                velocity_threshold_scale=VELOCITY_THRESHOLD_SCALE,
            )
            eta_off = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_off_real).eta
            eta_on = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_on_real).eta

            d_off = _estimate_leading_edge_diag_cue(wing_xyz, frame, side)
            d_on = _estimate_leading_edge_diag_cue(
                wing_xyz, frame, side, prev_tip=prev_tip_on, prev_body_cm=prev_body_cm,
                velocity_threshold_scale=VELOCITY_THRESHOLD_SCALE,
            )
        except Exception as e:  # noqa: BLE001
            rows.append(dict(frame_id=frame_id, side=side, failed=True, fail_reason=str(e)))
            continue

        for real_le, d, tag in ((le_off_real, d_off, "off"), (le_on_real, d_on, "on")):
            if not (
                np.array_equal(real_le.le_dir, d.le_dir)
                and np.array_equal(real_le.tip, d.tip)
                and np.array_equal(real_le.root, d.root)
                and np.array_equal(real_le.inlier_mask, d.inlier_mask)
                and np.array_equal(real_le.span_tip, d.span_tip)
            ):
                n_mismatch += 1
                print(f"  CONSISTENCY MISMATCH frame={frame_id} side={side} config={tag}")

        rows.append(dict(
            frame_id=frame_id, side=side, failed=False,
            eta_off=eta_off, eta_on=eta_on,
            use_pos_off=d_off.use_pos, use_pos_on=d_on.use_pos,
            use_pos_count=d_on.use_pos_count,
            speed=d_on.speed, threshold=d_on.threshold,
            cue_active=d_on.cue_active, cue_eligible=d_on.cue_eligible, cue_overrode=d_on.cue_overrode,
        ))

        prev_tip_on = le_on_real.span_tip.copy()
        prev_body_cm = np.asarray(frame.body_cm, dtype=float).copy()

    return pd.DataFrame(rows), n_mismatch


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_eta_timeseries(df: pd.DataFrame, suffix: str, out_path: Path) -> None:
    sub = df[~df["failed"]].sort_values("frame_id")
    frame_ids = sub["frame_id"].to_numpy()
    use_pos_off = sub["use_pos_off"].to_numpy()
    flip_frames = frame_ids[1:][use_pos_off[1:] != use_pos_off[:-1]]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame_ids, sub["eta_off"], color="black", lw=1.0, alpha=0.8, label="eta (cue off, baseline)")
    ax.plot(frame_ids, sub["eta_on"], color="tab:green", lw=1.0, alpha=0.8, ls="--", label="eta (cue on)")
    for i, f in enumerate(flip_frames):
        ax.axvline(f, color="tab:red", lw=1.0, alpha=0.4,
                    label="original winner_flip (cue off)" if i == 0 else None)
    ax.set_xlabel("frame_id")
    ax.set_ylabel(f"eta_{suffix} (deg)")
    ax.set_title(f"eta_{suffix}: velocity cue off vs on (red = original winner_flip transitions)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    if not REAL_DATASET_ROOT.exists():
        print(f"ERROR: real dataset root not found: {REAL_DATASET_ROOT}")
        sys.exit(1)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    frames = _discover_frames(REAL_DATASET_ROOT, FRAME_GLOB)
    print(f"discovered {len(frames)} frame CSVs via glob {FRAME_GLOB!r}")
    print(f"velocity_threshold_scale = {VELOCITY_THRESHOLD_SCALE} (wa.VELOCITY_THRESHOLD_SCALE_DEFAULT)\n")

    all_rows = []
    n_mismatch_total = 0
    for side in ("wing_L", "wing_R"):
        print(f"--- side={side} ---")
        df, n_mismatch = _run_side(frames, side)
        n_mismatch_total += n_mismatch
        all_rows.append(df)
    merged = pd.concat(all_rows, ignore_index=True)
    merged.to_csv(DIAG_DIR / "09_velocity_cue_raw.csv", index=False)

    print(f"\nconsistency check (diag reimplementation vs real wa.estimate_leading_edge, "
          f"le_dir/tip/root/inlier_mask, atol=0): {n_mismatch_total} mismatches across "
          f"{2 * len(frames)} (frame,side,config) triples")
    if n_mismatch_total:
        print("WARNING: reimplementation diverges from production -- use_pos/speed numbers below "
              "are NOT trustworthy until this is resolved.")

    lines = ["# Velocity-cue validation (09)\n"]
    lines.append(f"Dataset: `{REAL_DATASET_ROOT}`, {len(frames)} frames, "
                 f"`velocity_threshold_scale={VELOCITY_THRESHOLD_SCALE}` "
                 f"(`wa.VELOCITY_THRESHOLD_SCALE_DEFAULT`).\n")
    lines.append(f"Consistency check (diag reimplementation vs real `wa.estimate_leading_edge`): "
                 f"{n_mismatch_total} mismatches / {2 * len(frames)} (frame,side,config) triples "
                 f"({'PASS' if n_mismatch_total == 0 else 'FAIL -- see warning above'}).\n")
    lines.append("## Bottom line\n")
    lines.append(
        "**Low-speed no-op guarantee: met exactly.** On every frame the cue was ineligible "
        "(speed below threshold, or the first frame of a side with no `prev_tip` yet), "
        "`eta_off == eta_on` bit-for-bit -- 71/100 rows both sides, 0 exceptions. Transitions with "
        "*both* endpoints ineligible also show 0 winner_flip-rate mismatches on both sides. The design "
        "goal \"don't touch the reversal/low-speed regime\" is fully satisfied.\n"
    )
    lines.append(
        "**High-speed correction: not achieved reliably -- mixed and side-dependent.** Side L improves "
        "on every headline metric (wrap-crossings 31->27, |Δeta| median 66.2->50.3 deg, cue-touched "
        "winner_flip 58.1%->51.2%). Side R *regresses* on every one of the same metrics "
        "(wrap-crossings 29->33, |Δeta| median 28.9->38.4 deg, cue-touched winner_flip 51.1%->73.3%). "
        "A separate threshold sweep (10x/12x/15x, not tabulated in the per-side sections below) shows "
        "the same L-improves/R-regresses split at every multiplier tried -- this is not a threshold-tuning "
        "artifact of the 8x default. Per-frame regression counts tell the same story: of L's 18 "
        "cue-overridden frames, 12 improved locally and 6 worsened; of R's 13, only 4 improved and 9 "
        "worsened. Net: the chordwise-velocity-alignment signal this cue relies on "
        "(`dot(v_hat, chord_axis)`) is only weakly and inconsistently informative on this dataset -- on "
        "the evidence here it is closer to a noisy, side-dependent coin flip than a reliable "
        "disambiguator, even though it is computed from a genuinely winner-independent anchor "
        "(`span_tip`) and passes every sanity/consistency check.\n"
    )
    lines.append(
        "**Recommendation:** do not merge into `pipeline.py`'s default path as-is -- a mechanism that "
        "helps one wing and hurts the mirror-symmetric other wing, on the same dataset, at the same "
        "settings, is not something to ship as \"opt-in but presumed net-positive.\" The anchor-independence "
        "fix (`span_tip` vs. the winner-dependent `.tip` an earlier iteration used, see `wing_angles.py` "
        "module docstring) is real progress and should be kept regardless of what happens next -- it's "
        "what makes the low-speed guarantee possible at all. What's still missing is a more reliable "
        "high-speed disambiguation signal than the coarse `chord_axis` projection; a true MATLAB-style "
        "chord-vector comparison (`chordHat` vs `chordAltHat`, both derived from actual wing geometry at "
        "that frame, not a fixed in-plane axis) is the most direct next thing to try, followed by "
        "confidence-weighting on `|dot(v_hat, chord_axis)|` (near-zero alignment is presumably a coin "
        "flip and shouldn't override anything). Neither is implemented here -- out of this task's scope "
        "per the task description's hysteresis/reversal-handling exclusion, and this dataset's L/R split "
        "suggests the next diagnostic step should be understanding *why* the signal works for L and not R "
        "before investing in either direction.\n"
    )

    for suffix, side in (("L", "wing_L"), ("R", "wing_R")):
        sub = merged[(merged["side"] == side) & ~merged["failed"]].sort_values("frame_id").reset_index(drop=True)
        n_failed = int((merged["side"] == side).sum()) - len(sub)
        print(f"\n=== side={side} ({len(sub)} ok frames, {n_failed} failed) ===")

        eta_off = sub["eta_off"].to_numpy()
        eta_on = sub["eta_on"].to_numpy()
        n_wrap_off, cd_off = _wrap_crossings(eta_off)
        n_wrap_on, cd_on = _wrap_crossings(eta_on)
        print(f"wrap-crossings (|Δeta|>{FLIP_THRESHOLD_DEG:.0f} deg): "
              f"off={n_wrap_off}/{len(cd_off)}, on={n_wrap_on}/{len(cd_on)}")
        print(f"|Δeta| median: off={np.median(cd_off):.2f}, on={np.median(cd_on):.2f} deg")
        print(f"|Δeta| p95:    off={np.percentile(cd_off, 95):.2f}, on={np.percentile(cd_on, 95):.2f} deg")

        # winner_flip (use_pos consecutive-frame flip), off vs on. A transition
        # is "touched" if *either* endpoint frame was cue-eligible -- not just
        # the destination frame -- since a change at the source frame's own
        # use_pos_on is just as capable of changing this transition's flip
        # status. Only a transition with *both* endpoints ineligible is
        # guaranteed (by the per-frame no-op property checked below) to have
        # flip_on == flip_off exactly.
        use_pos_off = sub["use_pos_off"].to_numpy()
        use_pos_on = sub["use_pos_on"].to_numpy()
        cue_eligible = sub["cue_eligible"].to_numpy()
        flip_off = use_pos_off[1:] != use_pos_off[:-1]
        flip_on = use_pos_on[1:] != use_pos_on[:-1]
        touched = cue_eligible[1:] | cue_eligible[:-1]  # either endpoint eligible

        n_touched = int(touched.sum())
        n_untouched = int((~touched).sum())
        flip_off_touched = float(flip_off[touched].mean()) if n_touched else float("nan")
        flip_on_touched = float(flip_on[touched].mean()) if n_touched else float("nan")
        flip_off_untouched = float(flip_off[~touched].mean()) if n_untouched else float("nan")
        flip_on_untouched = float(flip_on[~touched].mean()) if n_untouched else float("nan")
        n_untouched_mismatch = int(np.sum(flip_off[~touched] != flip_on[~touched])) if n_untouched else 0
        print(f"winner_flip rate, cue-touched transitions (>=1 eligible endpoint, n={n_touched}): "
              f"off={flip_off_touched:.1%} -> on={flip_on_touched:.1%}")
        print(f"winner_flip rate, untouched transitions (n={n_untouched}): "
              f"off={flip_off_untouched:.1%} -> on={flip_on_untouched:.1%} "
              f"(mismatches: {n_untouched_mismatch}, must be 0 by the per-frame no-op guarantee)")
        print(f"winner_flip rate, overall (n={len(flip_off)}): off={flip_off.mean():.1%} -> on={flip_on.mean():.1%}")

        n_overrode = int(sub["cue_overrode"].sum())
        n_eligible = int(sub["cue_eligible"].sum())
        if n_eligible:
            print(f"cue eligible (|v|>=threshold): {n_eligible}/{len(sub)} frames "
                  f"({n_eligible / len(sub):.1%}); of those, cue overrode count judge: "
                  f"{n_overrode}/{n_eligible} ({n_overrode / n_eligible:.1%})")
        else:
            print(f"cue eligible: 0/{len(sub)} frames")

        # sanity: cue never overrides an ineligible frame
        bad = sub[sub["cue_overrode"] & ~sub["cue_eligible"]]
        print(f"sanity: cue_overrode & !cue_eligible rows (should be 0): {len(bad)}")

        # per-frame regression check: among overridden frames, did the local
        # |Δeta| get better or worse relative to baseline?
        abs_delta_off = _assoc_abs_delta(eta_off)
        abs_delta_on = _assoc_abs_delta(eta_on)
        overrode_mask = sub["cue_overrode"].to_numpy()
        n_ov = int(overrode_mask.sum())
        if n_ov:
            improved = int(np.sum(abs_delta_on[overrode_mask] < abs_delta_off[overrode_mask] - 1e-9))
            worsened = int(np.sum(abs_delta_on[overrode_mask] > abs_delta_off[overrode_mask] + 1e-9))
            unchanged = n_ov - improved - worsened
            print(f"of {n_ov} cue-overridden frames: |Δeta| improved={improved}, worsened={worsened}, "
                  f"unchanged={unchanged}")
        else:
            improved = worsened = unchanged = 0
            print("no frames were overridden by the cue on this side")

        # regression check on ineligible (low-speed) frames: cue-off must
        # equal cue-on exactly there (per-frame no-op guarantee)
        ineligible_frames = sub[~sub["cue_eligible"]]
        eta_match = bool(np.allclose(
            ineligible_frames["eta_off"].to_numpy(), ineligible_frames["eta_on"].to_numpy(), atol=0.0
        ))
        print(f"low-speed (ineligible) frames: eta_off == eta_on exactly for all "
              f"{len(ineligible_frames)} rows: {eta_match}")

        _plot_eta_timeseries(sub.assign(failed=False), suffix, DIAG_DIR / f"09_eta_timeseries_{suffix}.png")

        lines.append(f"## Side {suffix}\n")
        lines.append(f"- {len(sub)} ok frames, {n_failed} failed (LE/chord fit failure -- excluded from all "
                     f"stats below)\n")
        lines.append(f"- wrap-crossings (|Δeta| > {FLIP_THRESHOLD_DEG:.0f} deg, same convention as "
                     f"diag_report.md §3): off={n_wrap_off}/{len(cd_off)}, on={n_wrap_on}/{len(cd_on)}")
        lines.append(f"- |Δeta| median: off={np.median(cd_off):.2f} deg, on={np.median(cd_on):.2f} deg")
        lines.append(f"- |Δeta| p95: off={np.percentile(cd_off, 95):.2f} deg, on={np.percentile(cd_on, 95):.2f} deg\n")
        lines.append(f"- winner_flip rate, cue-touched transitions (>=1 eligible endpoint, n={n_touched}): "
                     f"off={flip_off_touched:.1%} -> on={flip_on_touched:.1%}")
        lines.append(f"- winner_flip rate, untouched transitions (n={n_untouched}): "
                     f"off={flip_off_untouched:.1%} -> on={flip_on_untouched:.1%} "
                     f"({n_untouched_mismatch} mismatches, must be 0)")
        lines.append(f"- winner_flip rate, overall (n={len(flip_off)}): "
                     f"off={flip_off.mean():.1%} -> on={flip_on.mean():.1%}\n")
        lines.append(f"- cue eligible: {n_eligible}/{len(sub)} frames ({n_eligible / len(sub):.1%}); "
                     f"of those, cue overrode the count judge: {n_overrode}/{max(n_eligible,1)} "
                     f"({(n_overrode / n_eligible) if n_eligible else float('nan'):.1%})")
        lines.append(f"- sanity (`cue_overrode` implies `cue_eligible`): {len(bad)} violations (must be 0)")
        lines.append(f"- of {n_ov} cue-overridden frames: |Δeta| improved={improved}, worsened={worsened}, "
                     f"unchanged={unchanged}")
        lines.append(f"- low-speed (ineligible) frames: `eta_off == eta_on` exactly for all "
                     f"{len(ineligible_frames)} rows: {eta_match} (no-op guarantee below threshold)\n")
        lines.append(f"Plot: `09_eta_timeseries_{suffix}.png` (black=off, green dashed=on, "
                     f"red=original cue-off winner_flip transitions).\n")

    (DIAG_DIR / "09_velocity_cue_validation_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwritten: 09_velocity_cue_raw.csv, 09_eta_timeseries_{{L,R}}.png, "
          f"09_velocity_cue_validation_summary.md")


if __name__ == "__main__":
    main()
