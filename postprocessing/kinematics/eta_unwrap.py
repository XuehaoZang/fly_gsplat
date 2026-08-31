"""Quick-and-dirty post-processing for `eta_L`/`eta_R` (wing pitch): circular
median filter to kill isolated (bad-frame) outliers, then `np.unwrap` to
remove the atan2 wrap artifact. See task brief for the full root-cause
writeup; short version:

1. `eta` comes out of `chord.py`/`wing_angles.py` as an `atan2`-derived angle
   folded into `(-180, 180]`. A real wingbeat cycle sweeps close to 360 deg,
   so it gets a spurious jump every time it crosses the wrap boundary --
   `diagnostics.delta_report` calls these out as "wrap-artifact frames".
2. Separately, some frames have bad T3 segmentation and `eta` on those
   frames is a genuine outlier, not a wrap artifact -- naively unwrapping
   raw eta lets a single bad frame permanently shift every later sample.

This module does NOT touch the per-frame eta formula in `chord.py`/
`wing_angles.py` -- it is a whole-sequence post-pass over the already-written
per-frame `eta_L`/`eta_R` (any array, not just this dataset's). It IS wired
into production: `pipeline.py::run_dataset_with_eta_unwrap` calls
`process_eta` on the `status == "ok"` rows of the per-frame table and
overwrites `eta_L`/`eta_R` before the CSV is (re-)written; that's the
function `calc_kinematics.py` calls for T4.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Step 0 (discovered during validation, not in the original task brief):
# resolve a per-frame +/-180 deg branch ambiguity BEFORE filtering/unwrapping.
#
# Plain circular-median-filter + unwrap (steps 1-2 below) turned out to leave
# a huge (~1900-3000 deg over 630 frames) spurious drift. Root cause, found
# by inspecting the worst frame-to-frame jumps: many frames flip by ~180 deg
# from one frame to the next and back (e.g. raw eta_L around frame 281:
# -129.5, 45.6, 55.0, -132.3, 57.8, -127.4, 56.6, ...) *at high chord_conf*
# (0.93-0.99 on both sides of the flip) -- so this is not sensor noise or a
# low-confidence bad frame (chord_conf does NOT flag these), and a 3-5-frame
# median filter cannot fix it because the flip recurs almost every other
# frame, not as an isolated outlier. This looks like a real leading/trailing
# edge sign-disambiguation instability in the upstream chord estimate (see
# `chord.py`), independent of the two causes named in the task brief -- flagged
# here, NOT fixed upstream (task scope explicitly excludes touching chord.py).
# ---------------------------------------------------------------------------


def resolve_180_flip(eta_deg: np.ndarray, ref_window: int = 15) -> np.ndarray:
    """Greedily pick, for each frame, whichever of `{eta[i], eta[i]+180}`
    (both wrapped to `(-180, 180]`) is circularly closer to the circular
    median of the last `ref_window` already-resolved frames. Frame 0 is
    taken as-is (arbitrary anchor -- shifts the whole resolved series by at
    most a constant 180 deg branch choice, which does not affect downstream
    delta/unwrap/correlation statistics).

    This is a real fix for a real, discovered-during-validation problem, but
    it is a greedy/local heuristic, not a rigorous solution: results are
    somewhat sensitive to `ref_window` (spot-checked 5-30 on the real
    dataset; L/R correlation ranged ~-0.4 to ~0.8, non-monotonic), so treat
    this as a documented caveat, not a fully validated production fix.
    """
    x = np.asarray(eta_deg, dtype=float)
    n = x.size
    out = np.empty(n)
    if n == 0:
        return out
    out[0] = x[0]
    for i in range(1, n):
        lo = max(0, i - ref_window)
        ref = _circular_median_deg(out[lo:i])
        cand0 = x[i]
        cand1 = _wrap180(x[i] + 180.0)
        d0 = abs(_wrap180(cand0 - ref))
        d1 = abs(_wrap180(cand1 - ref))
        out[i] = cand0 if d0 <= d1 else cand1
    return out


def _wrap180(x: np.ndarray | float) -> np.ndarray | float:
    return ((x + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Step 0, alternative: global (Viterbi-style, two-state) branch selection.
#
# `resolve_180_flip` above is a local greedy heuristic -- it anchors each
# frame's +/-180 choice to the circular median of the last `ref_window`
# *already-resolved* frames. On short (~200-frame), noisier sequences (the
# per-movie `mid200` sweep outputs, as opposed to the ~630-frame dev set the
# greedy version was validated on) this can "lose lock": a few
# consecutively-wrong local choices drag the reference window itself off
# track, and every subsequent frame keeps agreeing with the now-wrong
# reference -- producing a few-hundred-degree unbounded drift after unwrap
# instead of a bounded, per-wingbeat sawtooth.
#
# This version reframes the same per-frame binary choice
# (`eta[i]` vs `eta[i]+180`) as a global shortest-path problem: pick one
# branch per frame to minimize the TOTAL frame-to-frame circular distance
# over the whole sequence, via a standard two-state Viterbi DP. Unlike the
# greedy version it can't get stuck -- a locally-attractive wrong choice is
# only taken if it's cheaper for the sequence as a whole, so an isolated bad
# stretch can't permanently drag later frames off. O(n) (2 states).
# ---------------------------------------------------------------------------


def resolve_180_flip_dp(eta_deg: np.ndarray) -> np.ndarray:
    """Global two-state Viterbi alternative to `resolve_180_flip`: choose,
    per frame, `eta[i]` or `wrap180(eta[i] + 180)` to minimize the sum of
    circular frame-to-frame jumps over the whole sequence.

    Quick/dirty compared to `resolve_180_flip`: no `ref_window` to tune, no
    per-frame local reference to lose lock on -- just least total motion,
    globally. Same caveat as the greedy version applies in spirit (this is
    still a heuristic proxy for "real wingbeat continuity", not a physical
    model), but it does not exhibit the same open-ended drift failure mode
    on short/noisy sequences.
    """
    x = np.asarray(eta_deg, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()

    cand = np.stack([x, _wrap180(x + 180.0)], axis=1)  # (n, 2): branch 0 / branch 1
    cost = np.zeros((n, 2))
    back = np.zeros((n, 2), dtype=int)
    for i in range(1, n):
        for s in range(2):
            d0 = abs(_wrap180(cand[i, s] - cand[i - 1, 0]))
            d1 = abs(_wrap180(cand[i, s] - cand[i - 1, 1]))
            if cost[i - 1, 0] + d0 <= cost[i - 1, 1] + d1:
                cost[i, s] = cost[i - 1, 0] + d0
                back[i, s] = 0
            else:
                cost[i, s] = cost[i - 1, 1] + d1
                back[i, s] = 1

    state = int(np.argmin(cost[-1]))
    states = np.empty(n, dtype=int)
    states[-1] = state
    for i in range(n - 1, 0, -1):
        state = back[i, state]
        states[i - 1] = state

    return cand[np.arange(n), states]


def _circular_median_deg(vals: np.ndarray) -> float:
    rad = np.deg2rad(np.asarray(vals, dtype=float))
    return float(np.rad2deg(np.arctan2(np.median(np.sin(rad)), np.median(np.cos(rad)))))


# ---------------------------------------------------------------------------
# Step 1: circular median filter (kill isolated outliers without touching
# the ±180 wrap boundary -- filtering the raw angle directly would corrupt
# any window straddling the boundary, e.g. median([179, -179, 178]) != ~179).
# ---------------------------------------------------------------------------


def circular_median_filter(eta_deg: np.ndarray, window: int = 5) -> np.ndarray:
    """Median-filter an angle series in (cos, sin) space and convert back.

    `window` must be odd; edges are handled by clamping the window (no
    wraparound / no padding-induced artifacts at the sequence ends).
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    x = np.asarray(eta_deg, dtype=float)
    n = x.size
    rad = np.deg2rad(x)
    cos_x, sin_x = np.cos(rad), np.sin(rad)
    half = window // 2
    cos_f = np.empty(n)
    sin_f = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        cos_f[i] = np.median(cos_x[lo:hi])
        sin_f[i] = np.median(sin_x[lo:hi])
    return np.rad2deg(np.arctan2(sin_f, cos_f))


# ---------------------------------------------------------------------------
# Step 2: unwrap the cleaned series (period = 360 deg).
# ---------------------------------------------------------------------------


def unwrap_deg(eta_deg: np.ndarray) -> np.ndarray:
    """`np.unwrap` with a 360 deg period, operating directly on degrees."""
    return np.unwrap(np.asarray(eta_deg, dtype=float), period=360.0)


# ---------------------------------------------------------------------------
# Step 2, alternative: chunked unwrap + overlap-stitch.
#
# `unwrap_deg` runs `np.unwrap`'s cumulative sum over the *entire* sequence
# in one pass. That's the right tool for a signal whose phase genuinely
# accumulates -- but eta is a bounded, periodic wingbeat angle: real data
# never keeps circling past +-190 deg, it just re-visits the same ~[0, 200]
# band every cycle (confirmed against `Expr_112_mov_005_cleaned.mat`). Once
# a sequence runs long enough (~450+ frames, the `valid480` datasets vs. the
# ~200-frame `mid200` ones), the occasional local branch-resolution error
# that a single-pass cumulative unwrap can't tell apart from a genuine wrap
# stops being a bounded nuisance and becomes a biased random walk: each
# error adds a permanent +/-360 deg offset that every later frame inherits,
# so the whole trace ends up climbing a monotonic staircase to 800-1000+ deg
# (see `diagnostics_output/*/valid480` span numbers) instead of oscillating
# in place like the reference.
#
# This function keeps `np.unwrap`'s cumulative sum *local*: run it inside
# short overlapping chunks (each well under the ~450-frame span where drift
# becomes visible, but several wingbeat cycles long so within-chunk unwrap
# still has enough context to resolve real wrap crossings), then splice
# chunks together by picking, per chunk, whichever multiple of 360 deg
# makes its overlap region agree with the already-stitched result so far.
# A single bad chunk can still misalign by 360 relative to its neighbors,
# but the error can no longer compound across the whole recording -- it's
# confined to that one splice.
# ---------------------------------------------------------------------------


def unwrap_deg_chunked(eta_deg: np.ndarray, chunk: int = 80, overlap: int = 30) -> np.ndarray:
    """Chunked alternative to `unwrap_deg`: `np.unwrap` inside overlapping
    windows of `chunk` frames (stride `chunk - overlap`), stitched by
    aligning each new chunk's `overlap`-frame head against the
    already-stitched tail (least-squares-in-multiples-of-360 sense: shift
    by `round(median(stitched_overlap - new_chunk_overlap) / 360) * 360`).

    Falls back to a single `unwrap_deg` call when `eta_deg` doesn't exceed
    one chunk. `overlap` must be < `chunk` and non-trivial (>= ~10 frames)
    so the alignment estimate isn't just noise.
    """
    x = np.asarray(eta_deg, dtype=float)
    n = x.size
    if n <= chunk:
        return unwrap_deg(x)
    if overlap >= chunk:
        raise ValueError(f"overlap ({overlap}) must be < chunk ({chunk})")

    step = chunk - overlap
    starts = list(range(0, n - chunk + 1, step))
    if starts[-1] != n - chunk:
        starts.append(n - chunk)

    result = np.empty(n)
    prev_end = 0
    for i, s in enumerate(starts):
        e = s + chunk
        seg = unwrap_deg(x[s:e])
        if i > 0:
            ov_start, ov_end = s, prev_end
            seg_ov = seg[: ov_end - ov_start]
            res_ov = result[ov_start:ov_end]
            k = round(float(np.median(res_ov - seg_ov)) / 360.0)
            seg = seg + 360.0 * k
        result[s:e] = seg
        prev_end = e

    return result


# ---------------------------------------------------------------------------
# Step 3 (optional): re-interpolate low-confidence frames in the now-linear
# (unwrapped) space, using an existing chord_conf column as the mask.
# ---------------------------------------------------------------------------


def interpolate_low_conf(
    eta_unwrapped: np.ndarray, chord_conf: np.ndarray, conf_threshold: float = 0.9,
) -> np.ndarray:
    """Linearly interpolate over frames where `chord_conf < conf_threshold`,
    in the unwrapped (continuous) angle space. Endpoints below threshold are
    left as-is (nothing to interpolate from/to)."""
    x = np.asarray(eta_unwrapped, dtype=float).copy()
    conf = np.asarray(chord_conf, dtype=float)
    bad = conf < conf_threshold
    if not bad.any():
        return x
    idx = np.arange(x.size)
    good = ~bad
    if good.sum() < 2:
        return x
    x[bad] = np.interp(idx[bad], idx[good], x[good])
    return x


# ---------------------------------------------------------------------------
# Step 3b (DP path only): fix the residual global +/-180 branch ambiguity
# that both `resolve_180_flip` and `resolve_180_flip_dp` leave unresolved.
#
# Both only optimize *frame-to-frame* consistency (circular distance), and
# that objective is exactly invariant under adding 180 deg to every frame at
# once -- so neither can pick an absolute branch, only a self-consistent
# relative one. Each of eta_L/eta_R independently lands on whichever branch
# its own arbitrary anchor (frame 0, for the greedy version) happened to
# fall on. Validated against `Expr_112_mov_005_cleaned.mat` (different
# experiment, shape-only comparison): the wingbeat eta waveform spends most
# of the cycle in a "high plateau" state near ~90-190 deg with only a brief
# dip near 0, not the mirrored (~-190..-90) shape -- so nudging the resolved
# median toward a fixed positive target is a cheap, dataset-independent way
# to pick the physically-plausible branch instead of an arbitrary one.
# ---------------------------------------------------------------------------


def anchor_global_branch(eta_unwrapped: np.ndarray, target_center: float = 90.0) -> np.ndarray:
    """Shift the whole (already-unwrapped) series by the multiple of 180 deg
    that brings its median closest to `target_center`.

    Quick/dirty: a single global additive correction, not a per-frame
    decision -- assumes the whole input series is already internally
    consistent (true for `resolve_180_flip`/`resolve_180_flip_dp` output)
    and only its absolute branch is in question.
    """
    x = np.asarray(eta_unwrapped, dtype=float)
    if x.size == 0:
        return x.copy()
    k = round((target_center - float(np.median(x))) / 180.0)
    return x + 180.0 * k


@dataclass
class EtaUnwrapResult:
    raw: np.ndarray
    filtered: np.ndarray
    """After circular median filter, still wrapped to (-180, 180]."""
    unwrapped: np.ndarray
    """Final output: filtered + unwrapped (+ optional low-conf interpolation)."""


def process_eta(
    eta_deg: np.ndarray,
    chord_conf: np.ndarray | None = None,
    window: int = 5,
    conf_threshold: float = 0.9,
    resolve_flip: bool = True,
    flip_ref_window: int = 15,
) -> EtaUnwrapResult:
    """Full pipeline: (optional) 180-flip resolution -> circular median
    filter -> unwrap -> (optional) low-confidence interpolation.

    `resolve_flip=True` (default) runs `resolve_180_flip` first -- required
    to get a bounded, plausible result on this dataset (see that function's
    docstring for why); set `resolve_flip=False` to reproduce the task
    brief's literal minimum recipe (median filter + unwrap only), which is
    known to drift unboundedly here. Pass `chord_conf=None` to skip step 3.
    """
    raw = np.asarray(eta_deg, dtype=float)
    pre = resolve_180_flip(raw, ref_window=flip_ref_window) if resolve_flip else raw
    filtered = circular_median_filter(pre, window=window)
    unwrapped = unwrap_deg(filtered)
    if chord_conf is not None:
        unwrapped = interpolate_low_conf(unwrapped, chord_conf, conf_threshold=conf_threshold)
    return EtaUnwrapResult(raw=raw, filtered=filtered, unwrapped=unwrapped)


def process_eta_dp(
    eta_deg: np.ndarray,
    chord_conf: np.ndarray | None = None,
    window: int = 5,
    conf_threshold: float = 0.9,
    anchor_branch: bool = True,
    anchor_target_center: float = 90.0,
    chunk: int | None = 80,
    overlap: int = 30,
) -> EtaUnwrapResult:
    """Same recipe as `process_eta`, but with `resolve_180_flip_dp` (global
    DP) instead of `resolve_180_flip` (local greedy) for the 180-flip
    pre-pass, `unwrap_deg_chunked` (default `chunk=80`) instead of a single
    whole-sequence `unwrap_deg` call -- needed on longer (~450+ frame)
    sequences, where a single-pass cumulative unwrap turns occasional local
    branch-resolution noise into an unbounded staircase drift (see that
    function's docstring) -- and (`anchor_branch=True`, default) a final
    `anchor_global_branch` pass to pick the absolute +/-180 branch neither
    DP nor chunked-unwrap can determine on their own (see that function's
    docstring). Pass `chunk=None` to fall back to plain whole-sequence
    `unwrap_deg` (the original short-sequence behavior). Experimental/
    standalone -- not called by `pipeline.py`; use `process_eta` for the
    production T4 path."""
    raw = np.asarray(eta_deg, dtype=float)
    pre = resolve_180_flip_dp(raw)
    filtered = circular_median_filter(pre, window=window)
    unwrapped = (
        unwrap_deg_chunked(filtered, chunk=chunk, overlap=overlap)
        if chunk is not None
        else unwrap_deg(filtered)
    )
    if anchor_branch:
        unwrapped = anchor_global_branch(unwrapped, target_center=anchor_target_center)
    if chord_conf is not None:
        unwrapped = interpolate_low_conf(unwrapped, chord_conf, conf_threshold=conf_threshold)
    return EtaUnwrapResult(raw=raw, filtered=filtered, unwrapped=unwrapped)


# ---------------------------------------------------------------------------
# CLI: run against the real ratio3_sh0_dense CSV, report before/after
# delta_report + symmetry_report numbers, and plot raw vs processed vs the
# reference .mat's clean eta.
# ---------------------------------------------------------------------------

DEFAULT_CSV = (
    Path(__file__).resolve().parent
    / "diagnostics_output" / "ratio3_sh0_dense" / "kinematics_ratio3_sh0_dense.csv"
)
DEFAULT_REF_MAT = Path(__file__).resolve().parent / "diagnostics_output" / "Expr_112_mov_005_cleaned.mat"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "diagnostics_output" / "ratio3_sh0_dense"


def _load_ref_eta(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import scipy.io as sio

    d = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    abf = d["data"].anglesBodyFrame  # (N,8): col2=phiR,3=thetaR,4=etaR,5=phiL,6=thetaL,7=etaL
    eta_r_ref = pd.Series(abf[:, 4]).interpolate().to_numpy()
    eta_l_ref = pd.Series(abf[:, 7]).interpolate().to_numpy()
    return eta_l_ref, eta_r_ref


def main() -> None:
    from postprocessing.kinematics import diagnostics as diag

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"frames: total={len(df)}, status=ok={len(ok)}")

    print("\n== BEFORE (raw eta) ==")
    for suffix in ("L", "R"):
        diag.delta_report(f"eta_{suffix} (raw)", ok[f"eta_{suffix}"].to_numpy())
    diag.symmetry_report("eta (raw)", ok["eta_L"], ok["eta_R"], period_frames=None)

    # Minimal recipe from the task brief: median filter -> unwrap only.
    # Kept and reported for the record -- it does clear the named wrap
    # artifacts, but see the printed span below: it does NOT bound the
    # result (see `resolve_180_flip` docstring for why).
    print("\n== AFTER, minimal recipe (median filter window=5 -> unwrap, no flip-resolution) ==")
    minimal: dict[str, EtaUnwrapResult] = {}
    for suffix in ("L", "R"):
        res = process_eta(ok[f"eta_{suffix}"].to_numpy(), chord_conf=None, window=5, resolve_flip=False)
        minimal[suffix] = res
        diag.delta_report(f"eta_{suffix} (minimal)", res.unwrapped)
        print(f"  [{suffix}] span = {res.unwrapped.max() - res.unwrapped.min():.1f} deg (unbounded drift if >> ~600 deg)")
    corr_minimal = float(np.corrcoef(minimal["L"].unwrapped, minimal["R"].unwrapped)[0, 1])
    print(f"  [eta (minimal) L vs R] pearson r={corr_minimal:.3f}")

    # Full recipe: + resolve_180_flip pre-pass (discovered-during-validation fix).
    print("\n== AFTER, full recipe (+ resolve_180_flip, ref_window=15) ==")
    results: dict[str, EtaUnwrapResult] = {}
    for suffix in ("L", "R"):
        res = process_eta(ok[f"eta_{suffix}"].to_numpy(), chord_conf=None, window=5, resolve_flip=True, flip_ref_window=15)
        results[suffix] = res
        diag.delta_report(f"eta_{suffix} (processed)", res.unwrapped)
        print(f"  [{suffix}] span = {res.unwrapped.max() - res.unwrapped.min():.1f} deg")
    corr = float(np.corrcoef(results["L"].unwrapped, results["R"].unwrapped)[0, 1])
    print(f"  [eta (processed) L vs R] pearson r={corr:.3f}")

    # Comparison plot: raw vs processed vs reference mat -----------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_available = DEFAULT_REF_MAT.exists()
    if ref_available:
        eta_l_ref, eta_r_ref = _load_ref_eta(DEFAULT_REF_MAT)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for ax, suffix, ref in (
        (axes[0], "L", eta_l_ref if ref_available else None),
        (axes[1], "R", eta_r_ref if ref_available else None),
    ):
        frame_id = ok["frame_id"].to_numpy()
        ax.plot(frame_id, results[suffix].raw, ".", ms=2, lw=0.6, color="tab:gray", alpha=0.6, label="raw (wrapped)")
        ax.plot(frame_id, results[suffix].unwrapped, "-", lw=1.2, color="tab:blue", label="processed (flip-resolved+filtered+unwrapped)")
        if ref is not None:
            ref_frame_idx = np.arange(len(ref)) * 2  # reference fps is half ours -- stretch onto the same axis
            ax.plot(ref_frame_idx, ref, "-", lw=1.0, color="tab:green", alpha=0.7, label="reference .mat (frame idx *2, same axis)")
        ax.set_ylabel(f"eta_{suffix} (deg)")
        ax.set_xlabel("frame_id (ratio3_sh0_dense) / reference-frame*2, shared axis")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("eta_L / eta_R: raw vs processed (flip-resolve + circular-median + unwrap) vs reference shape")
    fig.tight_layout()
    plot_path = out_dir / "07_eta_unwrap_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\ncomparison plot written to: {plot_path}")


if __name__ == "__main__":
    main()
