"""Step 12: re-test the velocity cue's phase gate using a better
operationalization of the user's "T-shape" intuition than step 11 tried.

Step 11's Q2 gated the speed-only velocity cue on low frame-to-frame
`|d(theta)/dt|` ("wing not currently pitching out of the stroke plane") and
found it barely fired (~7% of frames after ANDing with the speed threshold)
and moved nothing. The user's actual description was more specific: the LE
"leads" the velocity direction *when the wing is fully extended and
perpendicular to the body* (fly + wing forming a "T"), not merely "theta is
stable". `_phi_theta` (`wing_angles.py`) computes `phi` from
`x_body`/`y_body`: `y_body = project_onto_plane(hinge_L - hinge_R, x_body)`
(`body_frame.py`) is the wing-hinge-to-wing-hinge axis, i.e. the body's
*lateral* direction -- so a wing pointing straight out sideways (the "T")
projects almost entirely onto `y_sp`, almost none onto `x_sp`, which is
exactly `phi = +/-90 deg` in `phi = atan2(sign_left*yle, xle)`. That is the
correct gate variable for "T-shape", not `theta` (deviation out of the
stroke plane, a different axis entirely).

Gate metric: `phi_perp_score = |cos(radians(phi))|` -- 0 exactly at `phi =
+/-90` (perpendicular to body, "T"), 1 at `phi = 0/180` (wing swept forward/
back along the body axis). `phi` is computed via `wa.stroke_deviation`
(`estimate_span`'s PCA axis), independent of the LE/TE winner call and of
`plane_normal`'s sign ambiguity -- same non-circularity argument as step 11's
theta gate.

Same dataset as step 11 (`ctrl_009_004_ratio3_sh0_dense_valid480`, 453
status=ok frames), same three-way comparison (off / speed-only / speed+gate),
same infra (`_dual_leading_edge`, wrap-crossing / winner_flip metrics)
copied from `11_dual_candidate_theta_gate.py` rather than imported, since
that module's name starts with a digit (not importable) -- project convention
in this directory (see e.g. `09_velocity_cue_validation.py` re-deriving
rather than importing `08_...`).

Run: python -m postprocessing.kinematics.correct_wing_pitch.12_phi_perp_gate
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
    """Mirrors `wing_angles.estimate_leading_edge` step for step, keeping a
    full `LeadingEdge` for *both* the pos and neg candidate (copied from
    `11_dual_candidate_theta_gate.py`, see that file for provenance)."""
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


def _velocity_cue_winner_gated(
    span_guess: np.ndarray, chord_axis: np.ndarray, span_tip: np.ndarray, use_pos_count: bool,
    prev_tip: np.ndarray, prev_body_cm: np.ndarray, body_cm: np.ndarray, scale: float,
    velocity_threshold_scale: float, phi_perp_score: float, phi_perp_threshold: float,
) -> tuple[bool, bool]:
    """Same as `wing_angles._velocity_cue_winner`, plus an additional
    eligibility requirement: `phi_perp_score <= phi_perp_threshold` (this
    frame's wing must be near-perpendicular to the body -- the "T-shape"
    condition -- not just "moving fast"). Returns `(use_pos, eligible)`.
    """
    if scale <= 0.0 or not np.isfinite(phi_perp_score) or phi_perp_score > phi_perp_threshold:
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

    # First pass: phi (and phi_perp_score) per frame/side -- no chaining needed.
    phi_by_side_frame: dict[str, dict[int, float]] = {"wing_L": {}, "wing_R": {}}
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
            phi_by_side_frame[side][frame_id] = sweep.phi

    all_scores = [
        abs(np.cos(np.radians(phi)))
        for side in ("wing_L", "wing_R")
        for phi in phi_by_side_frame[side].values()
    ]
    phi_perp_threshold = float(np.median(all_scores))
    print(f"phi distribution (both sides, n={len(all_scores)}): "
          f"phi_perp_score = |cos(phi)|, median={phi_perp_threshold:.3f} "
          f"(0=wing perpendicular to body/'T', 1=wing along body axis)")
    print(f"phi_perp_threshold (median split): {phi_perp_threshold:.3f}\n")

    rows = []
    for side in ("wing_L", "wing_R"):
        print(f"--- side={side} ---")
        prev_tip = None
        prev_body_cm = None
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

            phi = phi_by_side_frame[side].get(frame_id, float("nan"))
            phi_perp_score = abs(np.cos(np.radians(phi))) if np.isfinite(phi) else float("nan")

            try:
                dual = _dual_leading_edge(wing_xyz, frame, side)

                le_off = wa.estimate_leading_edge(wing_xyz, frame, side)
                eta_off = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_off).eta

                for cand_le, tag in ((dual.pos_le, "pos"), (dual.neg_le, "neg")):
                    if (np.array_equal(le_off.le_dir, cand_le.le_dir)
                            and np.array_equal(le_off.tip, cand_le.tip)
                            and np.array_equal(le_off.root, cand_le.root)
                            and np.array_equal(le_off.inlier_mask, cand_le.inlier_mask)):
                        break
                else:
                    n_consistency_mismatch += 1

                le_speed = wa.estimate_leading_edge(
                    wing_xyz, frame, side, prev_tip=prev_tip, prev_body_cm=prev_body_cm,
                    velocity_threshold_scale=VELOCITY_THRESHOLD_SCALE,
                )
                eta_speed = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_speed).eta

                use_pos_gated = dual.use_pos_count
                gate_eligible = False
                if prev_tip is not None and prev_body_cm is not None:
                    tree = cKDTree(wing_xyz)
                    nn_dist, _ = tree.query(wing_xyz, k=min(2, wing_xyz.shape[0]))
                    scale = float(np.median(nn_dist[:, -1])) if wing_xyz.shape[0] > 1 else 0.0
                    out_ref = wing_xyz.mean(axis=0) - np.asarray(frame.body_cm, dtype=float)
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
                        phi_perp_score=phi_perp_score, phi_perp_threshold=phi_perp_threshold,
                    )
                le_gated = dual.pos_le if use_pos_gated else dual.neg_le
                eta_gated = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_gated).eta

            except Exception as e:  # noqa: BLE001
                rows.append(dict(frame_id=frame_id, side=side, failed=True, fail_reason=str(e)))
                continue

            rows.append(dict(
                frame_id=frame_id, side=side, failed=False,
                phi=phi, phi_perp_score=phi_perp_score,
                use_pos_count=dual.use_pos_count,
                eta_off=eta_off, eta_speed=eta_speed, eta_gated=eta_gated,
                use_pos_gated=use_pos_gated, gate_eligible=gate_eligible,
            ))

            prev_tip = dual.pos_le.span_tip.copy()
            prev_body_cm = np.asarray(frame.body_cm, dtype=float).copy()

        print(f"  consistency mismatches (production winner not matching either candidate): "
              f"{n_consistency_mismatch}/{len(frames)}")

    merged = pd.DataFrame(rows)
    merged.to_csv(DIAG_DIR / "12_phi_perp_gate_raw.csv", index=False)

    print("\n" + "=" * 70)
    print("does a phi-perpendicularity ('T-shape') gate improve the velocity cue?")
    print("=" * 70)
    ok = merged[~merged["failed"]]
    for side in ("wing_L", "wing_R"):
        sub = ok[ok["side"] == side].sort_values("frame_id").reset_index(drop=True)
        eta_off = sub["eta_off"].to_numpy()
        eta_speed = sub["eta_speed"].to_numpy()
        eta_gated = sub["eta_gated"].to_numpy()

        n_wrap_off, cd_off = _wrap_crossings(eta_off)
        n_wrap_speed, cd_speed = _wrap_crossings(eta_speed)
        n_wrap_gated, cd_gated = _wrap_crossings(eta_gated)

        use_pos_off = sub["use_pos_count"].to_numpy()
        use_pos_gated = sub["use_pos_gated"].to_numpy()
        flip_off = use_pos_off[1:] != use_pos_off[:-1]
        flip_gated = use_pos_gated[1:] != use_pos_gated[:-1]

        n_gate_eligible = int(sub["gate_eligible"].sum())
        n_overrode = int(np.sum(sub["gate_eligible"].to_numpy() & (use_pos_gated != use_pos_off)))
        print(f"\n[{side}] n={len(sub)}, gate_eligible frames: {n_gate_eligible}/{len(sub)} "
              f"({n_gate_eligible/len(sub):.1%}); of those, cue overrode count judge: {n_overrode}")
        print(f"  wrap-crossings (|d(eta)|>{BIG_JUMP_DEG:.0f}): off={n_wrap_off}, speed-only={n_wrap_speed}, "
              f"speed+phi-gate={n_wrap_gated}")
        print(f"  |d(eta)| median: off={np.median(np.abs(cd_off)):.1f}, speed-only={np.median(np.abs(cd_speed)):.1f}, "
              f"speed+phi-gate={np.median(np.abs(cd_gated)):.1f}")
        print(f"  |d(eta)| p95:    off={np.percentile(np.abs(cd_off),95):.1f}, "
              f"speed-only={np.percentile(np.abs(cd_speed),95):.1f}, "
              f"speed+phi-gate={np.percentile(np.abs(cd_gated),95):.1f}")
        print(f"  winner_flip rate: off={flip_off.mean():.1%}, speed+phi-gate={flip_gated.mean():.1%}")

        # per-frame regression check among gate-overridden frames only
        abs_delta_off = np.array([
            max(
                abs(((eta_off[i] - eta_off[i-1] + 180) % 360) - 180) if i > 0 else 0.0,
                abs(((eta_off[i] - eta_off[i+1] + 180) % 360) - 180) if i < len(eta_off)-1 else 0.0,
            ) for i in range(len(eta_off))
        ])
        abs_delta_gated = np.array([
            max(
                abs(((eta_gated[i] - eta_gated[i-1] + 180) % 360) - 180) if i > 0 else 0.0,
                abs(((eta_gated[i] - eta_gated[i+1] + 180) % 360) - 180) if i < len(eta_gated)-1 else 0.0,
            ) for i in range(len(eta_gated))
        ])
        overrode_mask = sub["gate_eligible"].to_numpy() & (use_pos_gated != use_pos_off)
        n_ov = int(overrode_mask.sum())
        if n_ov:
            improved = int(np.sum(abs_delta_gated[overrode_mask] < abs_delta_off[overrode_mask] - 1e-9))
            worsened = int(np.sum(abs_delta_gated[overrode_mask] > abs_delta_off[overrode_mask] + 1e-9))
            print(f"  of {n_ov} gate-overridden frames: |d(eta)| improved={improved}, worsened={worsened}")

    print(f"\nwritten: 12_phi_perp_gate_raw.csv")


if __name__ == "__main__":
    main()
