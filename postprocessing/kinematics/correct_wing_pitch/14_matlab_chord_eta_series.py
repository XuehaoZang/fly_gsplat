"""Step 14: turn step 13's MATLAB-style chord-diagonal selection into an
actual eta series and compare wrap-crossings/winner_flip against current
production -- the definitive head-to-head test, not just the univariate
checks step 13 ran.

Correction from step 13's docstring: re-reading `find_chords_quad.m`'s
combination logic more carefully than the comment above it suggests, the
*implemented* swap rule is NOT "trust length unless ambiguous, then
velocity" -- it's the other way round:

    swapFlag = (velocitySwapFlag && nrm>=threshold) || (diagSwapFlag && nrm<threshold)

i.e. **trust velocity's pick whenever wingtip speed clears the threshold
(mid-stroke, moving), and only fall back to the length-ratio's pick when
speed is low** (near reversal, where translational velocity is a noisy
signal anyway but the wing's static cross-section shape is presumably more
stable). This script implements that combination rule exactly, chained
frame-to-frame for both `vWing` (as before) and the chosen chord's own SIGN
(MATLAB's `chordHat(3)<0` flip is a lab-frame-specific convention that
doesn't translate to our coordinate system; substituted here with continuity
to the previous frame's own resolved, signed chord -- same idea
`eta_unwrap.resolve_180_flip` uses, just applied one layer upstream, on the
actual geometry instead of the collapsed eta scalar).

`le_dir` in our `chord._eta(chord, le_dir, n_sp, sign_left)` formula only
needs to be *a* span-like axis to build the local in-plane basis (see that
function) -- substituted here with `wa.estimate_span`'s robust, winner-
independent PCA axis (`span`, already used for the diagonal extraction
itself), not `wing_angles.estimate_leading_edge`'s RANSAC-fit line.

Run: python -m postprocessing.kinematics.correct_wing_pitch.14_matlab_chord_eta_series
"""
from __future__ import annotations

import re
import sys
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
from scipy.spatial.distance import pdist, squareform

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import chord as ch  # noqa: E402
from postprocessing.kinematics import io_schema  # noqa: E402
from postprocessing.kinematics import wing_angles as wa  # noqa: E402

DATASET_ROOT = REPO_ROOT / "outputs" / "ctrl_009_004_ratio3_sh0_dense_valid480" / "ratio3_sh0_dense"
FRAME_GLOB = "f*/*/*/*_labeled.csv"
DIAG_DIR = Path(__file__).resolve().parent / "diag"
_FRAME_DIR_RE = re.compile(r"^f(\d+)$")
_SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}

CHORD_FRACTION = 0.33
DIAG_LENGTH_RATIO_DECISIVE = 1.3
MIN_BAND_POINTS = 5
WING_TIP_VELOCITY_THRESHOLD_SCALE = wa.VELOCITY_THRESHOLD_SCALE_DEFAULT
BIG_JUMP_DEG = 90.0


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


def _matlab_chords(wing_xyz: np.ndarray, span: np.ndarray) -> dict | None:
    """Same as `13_matlab_chord_diagonal.py::_matlab_chords` (copied, not
    imported -- this directory's own convention for numeric-prefixed
    modules, see e.g. `09_velocity_cue_validation.py` re-deriving rather
    than importing `08_...`)."""
    centroid = wing_xyz.mean(axis=0)
    rel = wing_xyz - centroid
    t = rel @ span
    span_extent = t.max() - t.min()
    if span_extent < 1e-12:
        return None

    band_idx = None
    for frac in (0.12, 0.36):
        delta = frac * span_extent
        idx = np.nonzero(np.abs(t) < delta)[0]
        if idx.size >= MIN_BAND_POINTS:
            band_idx = idx
            break
    if band_idx is None:
        return None

    band_pts = wing_xyz[band_idx]
    dist_from_centroid = np.linalg.norm(band_pts - centroid, axis=1)
    order = np.argsort(dist_from_centroid)[::-1]
    n_select = max(2, int(np.ceil(band_idx.size * CHORD_FRACTION)))
    selected = band_pts[order[:n_select]]
    if selected.shape[0] < 2:
        return None

    dmat = squareform(pdist(selected))
    i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
    raw_main = selected[i] - selected[j]
    chord_hat = raw_main - span * np.dot(span, raw_main)
    diag1 = float(np.linalg.norm(chord_hat))
    if diag1 < 1e-12:
        return None
    chord_hat = chord_hat / diag1

    wing_norm = np.cross(span, chord_hat)
    wn_norm = np.linalg.norm(wing_norm)
    if wn_norm < 1e-12:
        return None
    wing_norm = wing_norm / wn_norm

    band_rel = band_pts - centroid
    proj = band_rel @ wing_norm
    i_max, i_min = int(np.argmax(proj)), int(np.argmin(proj))
    if proj[i_max] <= 0 or proj[i_min] >= 0:
        return None
    raw_alt = band_pts[i_max] - band_pts[i_min]
    chord_alt_hat = raw_alt - span * np.dot(span, raw_alt)
    diag2 = float(np.linalg.norm(chord_alt_hat))
    if diag2 < 1e-12:
        return None
    chord_alt_hat = chord_alt_hat / diag2

    return dict(chord_hat=chord_hat, chord_alt_hat=chord_alt_hat, diag1=diag1, diag2=diag2)


def main() -> None:
    if not DATASET_ROOT.exists():
        print(f"ERROR: dataset root not found: {DATASET_ROOT}")
        sys.exit(1)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    frames = _discover_frames(DATASET_ROOT, FRAME_GLOB)
    print(f"discovered {len(frames)} frame CSVs via glob {FRAME_GLOB!r}\n")

    rows = []
    for side in ("wing_L", "wing_R"):
        print(f"--- side={side} ---")
        prev_span_tip = None
        prev_body_cm = None
        prev_signed_chord = None
        sign_left = _SIGN_LEFT[side]
        n_disagree_prod = 0

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
                span = wa.estimate_span(wing_xyz, frame, side)
                mc = _matlab_chords(wing_xyz, span)
                if mc is None:
                    raise ValueError("matlab chord extraction failed")

                le_prod = wa.estimate_leading_edge(wing_xyz, frame, side)
                prod_result = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_prod)
                eta_off = prod_result.eta
                span_tip = le_prod.span_tip
            except Exception as e:  # noqa: BLE001
                rows.append(dict(frame_id=frame_id, side=side, failed=True, fail_reason=str(e)))
                continue

            chord_hat, chord_alt_hat = mc["chord_hat"], mc["chord_alt_hat"]
            diag1, diag2 = mc["diag1"], mc["diag2"]
            diag_swap_flag = (diag2 / diag1) >= DIAG_LENGTH_RATIO_DECISIVE

            # velocity (winner-independent anchor, chained) -- mirrors vWing.
            # Threshold is `WING_TIP_VELOCITY_THRESHOLD_SCALE * scale`, `scale`
            # = this frame's own median nearest-neighbor spacing (same
            # density-derived convention `wing_angles.py` uses for its own
            # RANSAC thresholds and velocity cue) -- NOT a bare absolute
            # constant (that was step 14's first-run bug: comparing raw
            # `speed` to `WING_TIP_VELOCITY_THRESHOLD_SCALE`=8.0 directly
            # meant the velocity branch never fired, since real speeds are
            # tiny in this point cloud's physical units).
            tree = cKDTree(wing_xyz)
            nn_dist, _ = tree.query(wing_xyz, k=min(2, wing_xyz.shape[0]))
            scale = float(np.median(nn_dist[:, -1])) if wing_xyz.shape[0] > 1 else 0.0
            velocity_threshold = WING_TIP_VELOCITY_THRESHOLD_SCALE * scale

            speed = 0.0
            velocity_swap_flag = False
            if prev_span_tip is not None and prev_body_cm is not None:
                raw_delta = (span_tip - prev_span_tip) - (np.asarray(frame.body_cm, dtype=float) - prev_body_cm)
                comp = float(np.dot(raw_delta, span))
                perp = raw_delta - comp * span
                speed = float(np.linalg.norm(perp))
                if speed > 1e-9:
                    v_hat = perp / speed
                    dot1 = float(np.dot(chord_hat, v_hat))
                    dot2 = float(np.dot(chord_alt_hat, v_hat))
                    if dot2 > dot1:
                        velocity_swap_flag = True
                    if dot1 < 0.0 and dot2 < 0.0 and speed >= velocity_threshold and not velocity_swap_flag:
                        chord_hat = -chord_hat

            # MATLAB's exact combination rule (see module docstring)
            swap_flag = (velocity_swap_flag and speed >= velocity_threshold) or (
                diag_swap_flag and speed < velocity_threshold
            )
            chosen = chord_alt_hat if swap_flag else chord_hat

            # sign continuity (substitute for MATLAB's lab-frame chordHat(3)<0
            # convention, which doesn't translate to our coordinate system)
            if prev_signed_chord is None:
                # seed: orient toward production's own chord this frame
                if float(np.dot(chosen, prod_result.chord)) < 0.0:
                    chosen = -chosen
            else:
                if float(np.dot(chosen, prev_signed_chord)) < 0.0:
                    chosen = -chosen

            eta_matlab = ch._eta(chosen, span, frame.n_sp, sign_left)

            disagrees_with_prod = abs(float(np.dot(chosen, prod_result.chord))) < 0.7
            if disagrees_with_prod:
                n_disagree_prod += 1

            rows.append(dict(
                frame_id=frame_id, side=side, failed=False,
                eta_off=eta_off, eta_matlab=eta_matlab,
                diag_swap_flag=diag_swap_flag, velocity_swap_flag=velocity_swap_flag,
                swap_flag=swap_flag, speed=speed, velocity_threshold=velocity_threshold,
                velocity_eligible=speed >= velocity_threshold,
                disagrees_with_prod=disagrees_with_prod,
            ))

            prev_span_tip = span_tip.copy()
            prev_body_cm = np.asarray(frame.body_cm, dtype=float).copy()
            prev_signed_chord = chosen.copy()

        print(f"  frames where MATLAB-chosen axis diverges from production chord (|cos|<0.7): "
              f"{n_disagree_prod}/{len(frames)}")

    merged = pd.DataFrame(rows)
    merged.to_csv(DIAG_DIR / "14_matlab_chord_eta_series_raw.csv", index=False)
    ok = merged[~merged["failed"]].copy()
    print(f"\n{len(ok)} ok rows, {len(merged)-len(ok)} failed\n")

    print("=" * 70)
    print("head-to-head: production (no cue) vs MATLAB-style chord-diagonal eta")
    print("=" * 70)
    for side in ("wing_L", "wing_R"):
        sub = ok[ok["side"] == side].sort_values("frame_id").reset_index(drop=True)
        eta_off = sub["eta_off"].to_numpy()
        eta_matlab = sub["eta_matlab"].to_numpy()

        n_wrap_off, cd_off = _wrap_crossings(eta_off)
        n_wrap_matlab, cd_matlab = _wrap_crossings(eta_matlab)

        print(f"\n[{side}] n={len(sub)}")
        print(f"  wrap-crossings (|d(eta)|>{BIG_JUMP_DEG:.0f}): production={n_wrap_off}, matlab-style={n_wrap_matlab}")
        print(f"  |d(eta)| median: production={np.median(np.abs(cd_off)):.1f}, matlab-style={np.median(np.abs(cd_matlab)):.1f}")
        print(f"  |d(eta)| p95:    production={np.percentile(np.abs(cd_off),95):.1f}, "
              f"matlab-style={np.percentile(np.abs(cd_matlab),95):.1f}")
        print(f"  swap_flag rate: {sub['swap_flag'].astype(bool).mean():.1%}  "
              f"(velocity-driven: {(sub['velocity_swap_flag'].astype(bool) & sub['velocity_eligible'].astype(bool)).mean():.1%}, "
              f"length-driven: {(sub['diag_swap_flag'].astype(bool) & ~sub['velocity_eligible'].astype(bool)).mean():.1%})")
        print(f"  velocity-eligible frames (speed>=threshold): {sub['velocity_eligible'].astype(bool).mean():.1%}")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(sub["frame_id"], eta_off, color="black", lw=1.0, alpha=0.7, label="production (no cue)")
        ax.plot(sub["frame_id"], eta_matlab, color="tab:orange", lw=1.0, alpha=0.8, ls="--", label="MATLAB-style chord-diagonal")
        ax.set_xlabel("frame_id")
        ax.set_ylabel(f"eta_{side[-1]} (deg, raw wrapped)")
        ax.set_title(f"{side}: production vs MATLAB-style chord-diagonal eta (both raw, no unwrap)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DIAG_DIR / f"14_eta_comparison_{side}.png", dpi=150)
        plt.close(fig)

    print(f"\nwritten: 14_matlab_chord_eta_series_raw.csv, 14_eta_comparison_{{wing_L,wing_R}}.png")


if __name__ == "__main__":
    main()
