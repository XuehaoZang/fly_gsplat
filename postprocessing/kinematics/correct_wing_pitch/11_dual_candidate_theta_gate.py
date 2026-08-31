"""Step 11: two questions raised while rethinking the eta post-processing
strategy (2026-08-31 conversation, valid480 unwrap investigation), before
touching any production code:

Q1. `eta_unwrap.py`'s whole `resolve_180_flip`/`resolve_180_flip_dp` machinery
    assumes the two ambiguous LE/TE winner outcomes at a given frame produce
    eta values exactly 180 deg apart, so it only ever tries `eta[i]` vs
    `eta[i]+180`. But `chord.py` only ever computes eta from the *winning*
    candidate -- the loser's actual eta (from its own, independently RANSAC-fit
    line and point set) is never computed. Is "+180" actually a good
    approximation of `eta_neg - eta_pos`, or is the spread wide enough to
    explain the residual mid-chunk jumps found when the chunked-DP unwrap was
    tried on `ctrl_009_004_ratio3_sh0_dense_valid480`?

Q2. `09_velocity_cue_validation.py` found the existing velocity cue
    (`wing_angles._velocity_cue_winner`) gated on wingtip *speed* alone is
    unreliable (helps L, hurts R; winner_flip rate rises, not falls, with
    speed -- see `10_winner_flip_phase_correlation_summary.md`). The user's
    proposal: gate the cue on speed *and* a "wing is genuinely broadside /
    mid-stroke, not currently pitching" condition -- operationalized here as
    low frame-to-frame |d(theta)/dt| (`theta` from `wa.stroke_deviation`,
    computed from `estimate_span`'s PCA axis, independent of the winner call
    and of `plane_normal`'s sign ambiguity, so this gate can't be circular
    with the thing it's gating). Does adding this gate change the flip-rate
    result found in step 09?

Both are diagnose-only (no production code touched). Dataset: the same
`ctrl_009_004_ratio3_sh0_dense_valid480` real dataset the unwrap investigation
was run on (453 status=ok frames, longer than the 100-frame G2b_G9 set 09/10
used -- more representative of the dataset the eta drift was actually
reported on).

Run: python -m postprocessing.kinematics.correct_wing_pitch.11_dual_candidate_theta_gate
"""
from __future__ import annotations

import re
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
from postprocessing.kinematics import geometry as geo  # noqa: E402
from postprocessing.kinematics import io_schema  # noqa: E402
from postprocessing.kinematics import wing_angles as wa  # noqa: E402

DATASET_ROOT = REPO_ROOT / "outputs" / "ctrl_009_004_ratio3_sh0_dense_valid480" / "ratio3_sh0_dense"
FRAME_GLOB = "f*/*/*/*_labeled.csv"
BIG_JUMP_DEG = 90.0
VELOCITY_THRESHOLD_SCALE = wa.VELOCITY_THRESHOLD_SCALE_DEFAULT
_SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}

DIAG_DIR = Path(__file__).resolve().parent / "diag"
_FRAME_DIR_RE = re.compile(r"^f(\d+)$")


def _discover_frames(dataset_root: Path, frame_glob: str) -> list[tuple[int, Path]]:
    out = []
    for csv_path in dataset_root.glob(frame_glob):
        rel = csv_path.relative_to(dataset_root)
        m = _FRAME_DIR_RE.match(rel.parts[0])
        if not m:
            continue
        out.append((int(m.group(1)), csv_path))
    out.sort(key=lambda t: t[0])
    return out


def _wrap_crossings(eta: np.ndarray, threshold: float = BIG_JUMP_DEG) -> tuple[int, np.ndarray]:
    d = np.diff(eta)
    cd = ((d + 180.0) % 360.0) - 180.0
    return int(np.sum(np.abs(cd) > threshold)), cd


# ---------------------------------------------------------------------------
# Q1: both pos/neg candidates as real, fully-formed `wa.LeadingEdge` objects
# (mirrors `le_repro.py`'s reimplementation, extended to keep *both*
# candidates instead of collapsing to the winner) -- then each is fed through
# the *real* `ch.estimate_chord(..., leading_edge=...)` to get its own eta.
# ---------------------------------------------------------------------------


@dataclass
class DualLE:
    pos_le: wa.LeadingEdge
    neg_le: wa.LeadingEdge
    use_pos_count: bool
    pos_count: int
    neg_count: int


def _dual_leading_edge(
    wing_xyz: np.ndarray, body_frame: bf.BodyFrame, side: str,
    n_bins: int = 20, min_bin_points: int = 3, rng: int | np.random.Generator | None = 0,
) -> DualLE:
    """Mirrors `wing_angles.estimate_leading_edge` step for step (same as
    `le_repro.py`), but builds a full `LeadingEdge` for *both* the pos and
    neg candidate, not just the winner."""
    wing_xyz = np.asarray(wing_xyz, dtype=float)
    n = wing_xyz.shape[0]
    w = np.ones(n)

    tree = cKDTree(wing_xyz)
    nn_dist, _ = tree.query(wing_xyz, k=min(2, n))
    scale = float(np.median(nn_dist[:, -1])) if n > 1 else 0.0
    plane_thresh = 2.0 * scale
    line_thresh = 1.5 * scale

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
    span_tip = pts_plane[np.argmax(t)]

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
        raise ValueError("_dual_leading_edge: not enough populated span bins")
    pos_idx_local = np.array(pos_idx_local)
    neg_idx_local = np.array(neg_idx_local)

    def _candidate(local_idx: np.ndarray) -> tuple[wa.LeadingEdge, int, float]:
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

        le_dir = geo.orient_to_reference(direction, out_ref)
        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[orig_idx[mask]] = True
        inlier_points = pts[mask]
        t_final = (inlier_points - point_on_line) @ le_dir
        root = inlier_points[np.argmin(t_final)]
        tip = inlier_points[np.argmax(t_final)]
        le = wa.LeadingEdge(
            le_dir=le_dir, tip=tip, root=root, inlier_mask=inlier_mask,
            plane_normal=normal, span_tip=span_tip,
        )
        return le, int(mask.sum()), mean_resid

    pos_le, pos_count, pos_resid = _candidate(pos_idx_local)
    neg_le, neg_count, neg_resid = _candidate(neg_idx_local)
    use_pos_count = pos_count > neg_count or (pos_count == neg_count and pos_resid < neg_resid)

    return DualLE(pos_le=pos_le, neg_le=neg_le, use_pos_count=use_pos_count,
                  pos_count=pos_count, neg_count=neg_count)


def _circ_gap_deg(a: float, b: float) -> float:
    """Shortest signed distance from `b` to `a`, degrees."""
    return ((a - b + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Q2: theta-rate gate on top of the existing speed-only velocity cue
# ---------------------------------------------------------------------------


def _velocity_cue_winner_gated(
    span_guess: np.ndarray, chord_axis: np.ndarray, span_tip: np.ndarray, use_pos_count: bool,
    prev_tip: np.ndarray, prev_body_cm: np.ndarray, body_cm: np.ndarray, scale: float,
    velocity_threshold_scale: float, theta_rate: float, theta_rate_threshold: float,
) -> tuple[bool, bool]:
    """Same as `wing_angles._velocity_cue_winner`, plus an additional
    eligibility requirement: `|theta_rate| <= theta_rate_threshold` (this
    frame's `theta` must not have moved far from the previous frame's --
    "wing is not currently pitching/rotating out of the stroke plane").
    Returns `(use_pos, eligible)`.
    """
    if scale <= 0.0 or not np.isfinite(theta_rate) or abs(theta_rate) > theta_rate_threshold:
        return use_pos_count, False

    raw_delta = (span_tip - prev_tip) - (body_cm - prev_body_cm)
    comp = float(np.dot(raw_delta, span_guess))
    perp = raw_delta - comp * span_guess
    speed = float(np.linalg.norm(perp))
    if speed < velocity_threshold_scale * scale:
        return use_pos_count, False

    v_hat = perp / speed
    align = float(np.dot(v_hat, chord_axis))
    return (align > 0.0), True


def main() -> None:
    if not DATASET_ROOT.exists():
        print(f"ERROR: dataset root not found: {DATASET_ROOT}")
        sys.exit(1)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    frames = _discover_frames(DATASET_ROOT, FRAME_GLOB)
    print(f"discovered {len(frames)} frame CSVs via glob {FRAME_GLOB!r}\n")

    # First pass: theta rate distribution needs a first pass over the dataset
    # (theta itself is per-frame, no chaining needed for it).
    theta_by_side_frame: dict[str, dict[int, float]] = {"wing_L": {}, "wing_R": {}}
    for frame_id, csv_path in frames:
        try:
            df = io_schema.load_frame(csv_path)
            body_xyz = io_schema.get_part(df, "body")
            wingL_xyz = io_schema.get_part(df, "wing_L")
            wingR_xyz = io_schema.get_part(df, "wing_R")
            frame = bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz)
        except Exception:  # noqa: BLE001
            continue
        for side, wing_xyz in (("wing_L", wingL_xyz), ("wing_R", wingR_xyz)):
            try:
                sweep = wa.stroke_deviation(wing_xyz, frame, side)
            except Exception:  # noqa: BLE001
                continue
            theta_by_side_frame[side][frame_id] = sweep.theta

    # theta_rate = |theta[t] - theta[t-1]| over *consecutive available* frames
    # (not necessarily contiguous frame_id, matches how the chained loop below
    # walks status=ok frames only).
    all_rates = []
    for side in ("wing_L", "wing_R"):
        items = sorted(theta_by_side_frame[side].items())
        for (f0, th0), (f1, th1) in zip(items[:-1], items[1:]):
            all_rates.append(abs(th1 - th0))
    theta_rate_threshold = float(np.median(all_rates))
    print(f"theta_rate_threshold (median |d(theta)/dt| across both sides, n={len(all_rates)}): "
          f"{theta_rate_threshold:.2f} deg/frame\n")

    rows = []
    for side in ("wing_L", "wing_R"):
        print(f"--- side={side} ---")
        prev_tip = None
        prev_body_cm = None
        prev_theta = None
        n_consistency_mismatch = 0

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

            theta = theta_by_side_frame[side].get(frame_id, float("nan"))
            theta_rate = abs(theta - prev_theta) if prev_theta is not None and np.isfinite(theta) else float("nan")

            try:
                dual = _dual_leading_edge(wing_xyz, frame, side)
                eta_pos = ch.estimate_chord(wing_xyz, frame, side, leading_edge=dual.pos_le).eta
                eta_neg = ch.estimate_chord(wing_xyz, frame, side, leading_edge=dual.neg_le).eta

                # off (production, no cue)
                le_off = wa.estimate_leading_edge(wing_xyz, frame, side)
                eta_off = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_off).eta

                # consistency: production's winner must match one of our two
                # candidates exactly (le_dir/tip/root/inlier_mask).
                for cand_le, tag in ((dual.pos_le, "pos"), (dual.neg_le, "neg")):
                    if (np.array_equal(le_off.le_dir, cand_le.le_dir)
                            and np.array_equal(le_off.tip, cand_le.tip)
                            and np.array_equal(le_off.root, cand_le.root)
                            and np.array_equal(le_off.inlier_mask, cand_le.inlier_mask)):
                        matched = tag
                        break
                else:
                    matched = None
                    n_consistency_mismatch += 1

                # speed-only cue (production default, chained)
                le_speed = wa.estimate_leading_edge(
                    wing_xyz, frame, side, prev_tip=prev_tip, prev_body_cm=prev_body_cm,
                    velocity_threshold_scale=VELOCITY_THRESHOLD_SCALE,
                )
                eta_speed = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_speed).eta

                # speed+theta-gate cue (this script's proposal), built from
                # the same span_guess/chord_axis/span_tip our dual candidates
                # already computed (winner-independent quantities).
                use_pos_gated = dual.use_pos_count
                gate_eligible = False
                if prev_tip is not None and prev_body_cm is not None:
                    tree = cKDTree(wing_xyz)
                    nn_dist, _ = tree.query(wing_xyz, k=min(2, wing_xyz.shape[0]))
                    scale = float(np.median(nn_dist[:, -1])) if wing_xyz.shape[0] > 1 else 0.0
                    out_ref = wing_xyz.mean(axis=0) - np.asarray(frame.body_cm, dtype=float)
                    # recompute span_guess/chord_axis exactly as _dual_leading_edge did
                    _, _, plane_mask2 = geo.fit_plane(wing_xyz, np.ones(wing_xyz.shape[0]), method="ransac",
                                                       threshold=2.0 * scale, rng=0)
                    idx_plane2 = np.nonzero(plane_mask2)[0]
                    _, eigvecs2, _ = geo.weighted_pca(wing_xyz[idx_plane2], np.ones(idx_plane2.size))
                    span_guess = geo.orient_to_reference(eigvecs2[:, -1], out_ref)
                    chord_axis = np.cross(dual.pos_le.plane_normal, span_guess)
                    chord_axis = chord_axis / np.linalg.norm(chord_axis)
                    use_pos_gated, gate_eligible = _velocity_cue_winner_gated(
                        span_guess, chord_axis, dual.pos_le.span_tip, dual.use_pos_count,
                        prev_tip=prev_tip, prev_body_cm=prev_body_cm, body_cm=np.asarray(frame.body_cm, dtype=float),
                        scale=scale, velocity_threshold_scale=VELOCITY_THRESHOLD_SCALE,
                        theta_rate=theta_rate, theta_rate_threshold=theta_rate_threshold,
                    )
                le_gated = dual.pos_le if use_pos_gated else dual.neg_le
                eta_gated = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_gated).eta

            except Exception as e:  # noqa: BLE001
                rows.append(dict(frame_id=frame_id, side=side, failed=True, fail_reason=str(e)))
                continue

            rows.append(dict(
                frame_id=frame_id, side=side, failed=False,
                theta=theta, theta_rate=theta_rate,
                eta_pos=eta_pos, eta_neg=eta_neg, gap_pos_minus_neg=_circ_gap_deg(eta_pos, eta_neg),
                use_pos_count=dual.use_pos_count, matched_production=matched,
                eta_off=eta_off, eta_speed=eta_speed, eta_gated=eta_gated,
                use_pos_gated=use_pos_gated, gate_eligible=gate_eligible,
            ))

            prev_tip = dual.pos_le.span_tip.copy()
            prev_body_cm = np.asarray(frame.body_cm, dtype=float).copy()
            if np.isfinite(theta):
                prev_theta = theta

        print(f"  consistency mismatches (production winner not matching either candidate): "
              f"{n_consistency_mismatch}/{len(frames)}")

    merged = pd.DataFrame(rows)
    merged.to_csv(DIAG_DIR / "11_dual_candidate_theta_gate_raw.csv", index=False)

    # ------------------------------------------------------------------ Q1 --
    print("\n" + "=" * 70)
    print("Q1: is eta_neg - eta_pos really ~180 deg?")
    print("=" * 70)
    ok = merged[~merged["failed"]]
    for side in ("wing_L", "wing_R"):
        sub = ok[ok["side"] == side]
        gap = sub["gap_pos_minus_neg"].to_numpy()
        dist_from_180 = np.abs(np.abs(gap) - 180.0)
        print(f"\n[{side}] n={len(sub)}")
        print(f"  gap median={np.median(gap):.1f}, |gap| median={np.median(np.abs(gap)):.1f}, "
              f"p95={np.percentile(np.abs(gap), 95):.1f}")
        print(f"  |dist from +/-180| median={np.median(dist_from_180):.1f}, "
              f"p95={np.percentile(dist_from_180, 95):.1f}, max={np.max(dist_from_180):.1f}")
        print(f"  frac within 10 deg of +/-180: {np.mean(dist_from_180 <= 10.0):.1%}; "
              f"within 30 deg: {np.mean(dist_from_180 <= 30.0):.1%}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(dist_from_180, bins=40, color="tab:blue", alpha=0.8)
        ax.set_xlabel("|  |eta_neg - eta_pos|  -  180 deg  |")
        ax.set_ylabel("frame count")
        ax.set_title(f"{side}: how far the pos/neg candidate gap deviates from a clean 180 deg flip")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DIAG_DIR / f"11_gap_from_180_hist_{side}.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ Q2 --
    print("\n" + "=" * 70)
    print("Q2: does a theta-rate gate improve the velocity cue over speed-only?")
    print("=" * 70)
    for side in ("wing_L", "wing_R"):
        sub = ok[ok["side"] == side].sort_values("frame_id").reset_index(drop=True)
        eta_off = sub["eta_off"].to_numpy()
        eta_speed = sub["eta_speed"].to_numpy()
        eta_gated = sub["eta_gated"].to_numpy()

        n_wrap_off, cd_off = _wrap_crossings(eta_off)
        n_wrap_speed, cd_speed = _wrap_crossings(eta_speed)
        n_wrap_gated, cd_gated = _wrap_crossings(eta_gated)

        # off has no cue, so its winner IS use_pos_count
        use_pos_off = sub["use_pos_count"].to_numpy()
        use_pos_gated = sub["use_pos_gated"].to_numpy()
        flip_off = use_pos_off[1:] != use_pos_off[:-1]
        flip_gated = use_pos_gated[1:] != use_pos_gated[:-1]

        n_gate_eligible = int(sub["gate_eligible"].sum())
        print(f"\n[{side}] n={len(sub)}, gate_eligible frames: {n_gate_eligible}/{len(sub)} "
              f"({n_gate_eligible/len(sub):.1%})")
        print(f"  wrap-crossings (|d(eta)|>{BIG_JUMP_DEG:.0f}): off={n_wrap_off}, speed-only={n_wrap_speed}, "
              f"speed+theta-gate={n_wrap_gated}")
        print(f"  |d(eta)| median: off={np.median(np.abs(cd_off)):.1f}, speed-only={np.median(np.abs(cd_speed)):.1f}, "
              f"speed+theta-gate={np.median(np.abs(cd_gated)):.1f}")
        print(f"  winner_flip rate: off={flip_off.mean():.1%}, speed+theta-gate={flip_gated.mean():.1%}")

    print(f"\nwritten: 11_dual_candidate_theta_gate_raw.csv, 11_gap_from_180_hist_{{wing_L,wing_R}}.png")


if __name__ == "__main__":
    main()
