"""Step 13: try the actual MATLAB `find_chords_quad.m` chord-selection
strategy (`reference/wing_pitch/find_chords_quad.m`) on real data, instead of
our current `chord.py`/`wing_angles.py` machinery's RANSAC-line-straightness
winner call + coarse in-plane `chord_axis` velocity cue -- both of which
steps 08-12 showed are unreliable on this dataset (winner_flip correlates
with big jumps but no static predictor found it; the velocity cue, gated
three different ways (none/speed/speed+theta-rate/speed+phi-perp), stays
close to a coin flip when it fires).

MATLAB's algorithm is structurally different, not just a re-tuned version of
ours:
  1. Take points in a band near mid-span (perpendicular to `span`), select
     the ones farthest from the wing centroid (top third), and find the
     single farthest-apart PAIR among them -- that pair's separation is the
     "main" chord diagonal, `chordHat` (`diag1` = its length).
  2. Using `chordHat`'s own wing-plane normal (`cross(span, chordHat)`),
     project the *whole* mid-span band onto that normal and take the
     extremes on each side -- that gives the "alternative" diagonal,
     `chordAltHat` (`diag2`).
  3. **Primary disambiguator: diagonal LENGTH.** A real wing cross-section is
     elongated chordwise; if one diagonal is >=1.3x the other, MATLAB trusts
     the longer one outright and never touches velocity.
  4. **Only when the lengths are close** (`diag2/diag1 < 1.3`, genuinely
     ambiguous) does it fall back to wingtip velocity -- and even then it
     compares actual `chordHat`/`chordAltHat` (real wing-shape vectors), not
     a fixed in-plane axis like our `chord_axis`.

This script does NOT reimplement the full swap+sign pipeline or wire
anything into `chord.py` -- it only tests whether the two premises behind
MATLAB's design hold on our real point clouds:
  Q1. Is the diagonal-length ratio usually decisive (>=1.3) on its own,
      without ever touching velocity?
  Q2. When it *is* decisive, does it broadly agree with the axis our
      production `ch.estimate_chord` already picked (sanity: same physical
      structure, different selection mechanism)?
  Q3. In the genuinely ambiguous (length-tied) minority, is
      `|dot(chordHat, v_hat)| - |dot(chordAltHat, v_hat)|` more separated
      (i.e. does velocity carry a real signal) than it looked in steps 09/12,
      where it had to adjudicate on *every* frame including easy ones?

Uses `wa.estimate_span` (winner-independent PCA span axis, already
production code) for `span`, and production `ch.estimate_chord(...).chord`
+ `wa.estimate_leading_edge(...).span_tip` as the comparison references --
no re-derivation of those.

Run: python -m postprocessing.kinematics.correct_wing_pitch.13_matlab_chord_diagonal
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
from scipy.spatial.distance import pdist, squareform

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import chord as ch  # noqa: E402
from postprocessing.kinematics import io_schema  # noqa: E402
from postprocessing.kinematics import wing_angles as wa  # noqa: E402

DATASET_ROOT = REPO_ROOT / "outputs" / "ctrl_009_004_ratio3_sh0_dense_valid480" / "ratio3_sh0_dense"
FRAME_GLOB = "f*/*/*/*_labeled.csv"
DIAG_DIR = Path(__file__).resolve().parent / "diag"
_FRAME_DIR_RE = re.compile(r"^f(\d+)$")

CHORD_FRACTION = 0.33
"""Top fraction of the mid-span band, by distance from wing centroid, used
to find the main diagonal -- matches MATLAB's `chordFraction`."""
DIAG_LENGTH_RATIO_DECISIVE = 1.3
"""MATLAB's own threshold: trust the longer diagonal outright above this ratio."""
MIN_BAND_POINTS = 5


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


def _matlab_chords(wing_xyz: np.ndarray, span: np.ndarray) -> dict | None:
    """Direct translation of `find_chords_quad.m`'s diagonal-finding logic
    (chordHat/chordAltHat/diag1/diag2 only -- swap/velocity handled by the
    caller, not here). `delta` (mid-span band half-width) is expressed as a
    fraction of the wing's own span extent (0.12, then 0.36 as a widen-and-
    retry, mirroring MATLAB's "first try delta, then 3*delta") since our
    point clouds aren't on MATLAB's fixed absolute voxel grid.

    Returns None if even the widened band has < MIN_BAND_POINTS points
    (mirrors MATLAB's "empty chord" error path).
    """
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
        return None  # degenerate: all band points on one side of the plane
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
                    raise ValueError("matlab chord extraction failed (band/degenerate)")

                le_prod = wa.estimate_leading_edge(wing_xyz, frame, side)
                prod_chord = ch.estimate_chord(wing_xyz, frame, side, leading_edge=le_prod).chord
                span_tip = le_prod.span_tip  # winner-independent anchor, reused as WingTip
            except Exception as e:  # noqa: BLE001
                rows.append(dict(frame_id=frame_id, side=side, failed=True, fail_reason=str(e)))
                continue

            diag1, diag2 = mc["diag1"], mc["diag2"]
            ratio = max(diag1, diag2) / min(diag1, diag2)
            length_decisive = ratio >= DIAG_LENGTH_RATIO_DECISIVE
            length_winner = "main" if diag1 >= diag2 else "alt"
            length_winner_vec = mc["chord_hat"] if length_winner == "main" else mc["chord_alt_hat"]

            # sign-agnostic agreement with production chord (same physical axis?)
            cos_main = abs(float(np.dot(mc["chord_hat"], prod_chord)))
            cos_alt = abs(float(np.dot(mc["chord_alt_hat"], prod_chord)))
            matlab_matches_prod = "main" if cos_main >= cos_alt else "alt"
            agree_len_vs_prod = (length_winner == matlab_matches_prod)

            # velocity (winner-independent anchor, chained)
            v_score_main = v_score_alt = float("nan")
            speed = float("nan")
            if prev_span_tip is not None and prev_body_cm is not None:
                raw_delta = (span_tip - prev_span_tip) - (np.asarray(frame.body_cm, dtype=float) - prev_body_cm)
                comp = float(np.dot(raw_delta, span))
                perp = raw_delta - comp * span
                speed = float(np.linalg.norm(perp))
                if speed > 1e-9:
                    v_hat = perp / speed
                    v_score_main = abs(float(np.dot(mc["chord_hat"], v_hat)))
                    v_score_alt = abs(float(np.dot(mc["chord_alt_hat"], v_hat)))

            rows.append(dict(
                frame_id=frame_id, side=side, failed=False,
                diag1=diag1, diag2=diag2, ratio=ratio, length_decisive=length_decisive,
                length_winner=length_winner, cos_main_vs_prod=cos_main, cos_alt_vs_prod=cos_alt,
                agree_len_vs_prod=agree_len_vs_prod, speed=speed,
                v_score_main=v_score_main, v_score_alt=v_score_alt,
            ))

            prev_span_tip = span_tip.copy()
            prev_body_cm = np.asarray(frame.body_cm, dtype=float).copy()

    merged = pd.DataFrame(rows)
    merged.to_csv(DIAG_DIR / "13_matlab_chord_diagonal_raw.csv", index=False)
    ok = merged[~merged["failed"]].copy()
    n_failed = len(merged) - len(ok)
    print(f"\n{len(ok)} ok rows, {n_failed} failed (band/degenerate/LE-fit failures)\n")

    print("=" * 70)
    print("Q1: is the diagonal length ratio usually decisive (>=1.3) on its own?")
    print("=" * 70)
    for side in ("wing_L", "wing_R"):
        sub = ok[ok["side"] == side]
        ratio = sub["ratio"].to_numpy()
        frac_decisive = float(sub["length_decisive"].astype(bool).mean())
        print(f"\n[{side}] n={len(sub)}")
        print(f"  ratio median={np.median(ratio):.2f}, p25={np.percentile(ratio,25):.2f}, "
              f"p75={np.percentile(ratio,75):.2f}")
        print(f"  frac length_decisive (ratio>={DIAG_LENGTH_RATIO_DECISIVE}): {frac_decisive:.1%}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ratio, bins=50, range=(1.0, 5.0), color="tab:blue", alpha=0.8)
        ax.axvline(DIAG_LENGTH_RATIO_DECISIVE, color="tab:red", ls="--", label=f"decisive threshold={DIAG_LENGTH_RATIO_DECISIVE}")
        ax.set_xlabel("diag ratio = max(diag1,diag2)/min(diag1,diag2)")
        ax.set_ylabel("frame count")
        ax.set_title(f"{side}: MATLAB chord-diagonal length ratio distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DIAG_DIR / f"13_diag_ratio_hist_{side}.png", dpi=150)
        plt.close(fig)

    print("\n" + "=" * 70)
    print("Q2: among length-decisive frames, does the winning diagonal match")
    print("    production's chord.py chord axis (same physical structure)?")
    print("=" * 70)
    for side in ("wing_L", "wing_R"):
        sub = ok[(ok["side"] == side) & ok["length_decisive"].astype(bool)]
        n = len(sub)
        n_agree = int(sub["agree_len_vs_prod"].astype(bool).sum())
        cos_when_agree = sub.loc[sub["agree_len_vs_prod"].astype(bool)].apply(
            lambda r: max(r["cos_main_vs_prod"], r["cos_alt_vs_prod"]), axis=1)
        print(f"\n[{side}] length-decisive n={n}")
        print(f"  agrees with production chord axis: {n_agree}/{n} ({n_agree/n:.1%} if n>0 else n/a)"
              if n else "  n=0")
        if len(cos_when_agree):
            print(f"  |cos angle| to production chord, when agreeing: median={cos_when_agree.median():.3f} "
                  f"(1.0 = identical axis)")

    print("\n" + "=" * 70)
    print("Q3: in the length-AMBIGUOUS minority, is velocity more separating")
    print("    than it looked when forced to adjudicate every frame (09/12)?")
    print("=" * 70)
    for side in ("wing_L", "wing_R"):
        sub = ok[(ok["side"] == side) & ~ok["length_decisive"].astype(bool)]
        sub = sub[np.isfinite(sub["v_score_main"]) & np.isfinite(sub["v_score_alt"])]
        n = len(sub)
        print(f"\n[{side}] length-ambiguous & velocity-available n={n} "
              f"({n / max(1, (ok['side']==side).sum()):.1%} of all ok frames)")
        if n == 0:
            continue
        sep = (sub["v_score_main"] - sub["v_score_alt"]).abs().to_numpy()
        print(f"  |v_score_main - v_score_alt| median={np.median(sep):.3f}, p25={np.percentile(sep,25):.3f} "
              f"(0=no separation/coin flip, 1=maximally decisive)")
        print(f"  frac with separation >= 0.2: {np.mean(sep >= 0.2):.1%}; >= 0.4: {np.mean(sep >= 0.4):.1%}")

    print(f"\nwritten: 13_matlab_chord_diagonal_raw.csv, 13_diag_ratio_hist_{{wing_L,wing_R}}.png")


if __name__ == "__main__":
    main()
