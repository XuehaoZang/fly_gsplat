"""S6b: real-data accuracy diagnostics for T4 kinematics output.

No ground truth exists for real fly angles, so accuracy is judged
*indirectly* (per task spec): a single-frame estimator that produces
smooth, periodic, physically-plausible time series is very likely tracking
the real quantity, and specific failures (jumps, asymmetry, non-periodicity,
implausible ranges) are diagnostic even without ground truth.

Diagnose only -- this module never changes an algorithm/parameter and never
smooths or unwraps `pipeline.py`'s output; it only reads the raw per-frame
CSV `pipeline.run_dataset` already produces (§ single-frame scope, see
`pipeline.py`'s own docstring) and reports on it.

Reusable/importable: every `plot_*`/`*_stats`/`*_check` function takes plain
arrays or the output DataFrame and is usable standalone; `run_diagnostics`
is the orchestrating entry point. The `if __name__ == "__main__"` block runs
it against the real 100-frame `outputs/ctrl_009_002_8groups_100frames/G2b_G9`
dataset (see `reference/s6a_real_data_smoke_test_findings.md`) and writes
figures + `report.md` to a scratch directory (never into tracked `outputs/`).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics import pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Physically-plausible bands -- stated assumptions, not hard constraints.
# Sources: Dickinson/Fry/Muijres-era Drosophila free-flight kinematics
# (typical hovering/forward-flight ranges); NOT derived from this repo's data.
# `fail` bands are values the angle *formula itself* cannot produce (a hit
# there means a bug, not just an unusual maneuver); `concerning` bands are
# generous behavioral envelopes -- a hit there flags "worth a look", not "wrong".
# ---------------------------------------------------------------------------

ANGLE_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "pitch": {"concerning": (-60.0, 60.0), "fail": (-90.0, 90.0)},
    "roll": {"concerning": (-120.0, 120.0), "fail": (-180.0, 180.0)},
    "theta_L": {"concerning": (-40.0, 40.0), "fail": (-90.0, 90.0)},
    "theta_R": {"concerning": (-40.0, 40.0), "fail": (-90.0, 90.0)},
}
"""`yaw` (unbounded heading), `phi_*` (stroke amplitude judged by range, not
absolute value) and `eta_*` (feathering angle legitimately sweeps most of
its cyclic range near reversal) are checked separately, not via a fixed band."""

CHORD_CONF_BAND = (0.0, 1.0)
"""Hard range by construction (§5) -- any excursion is a bug, not a maneuver."""


# ---------------------------------------------------------------------------
# Delta / smoothness statistics
# ---------------------------------------------------------------------------


def circular_delta_deg(x: np.ndarray) -> np.ndarray:
    """Frame-to-frame change for a `(-180, 180]`-valued angle, wrapped to the
    shortest signed distance. This is the physically correct measure of
    angular change across a wrap boundary; raw `np.diff` would report a
    spurious ~360 deg "jump" exactly there (see `delta_report`'s `wrap_frames`)."""
    d = np.diff(x)
    return ((d + 180.0) % 360.0) - 180.0


@dataclass
class DeltaReport:
    name: str
    median: float
    p95: float
    max: float
    jump_threshold: float
    jump_frames: list[int]
    wrap_frames: list[int]
    """Frames where the raw difference wrapped near +/-360 deg but the
    circular delta was small -- a wrap *artifact*, not a real jump. Reported
    separately per the task's "note wrap artifacts, don't fix them"."""


def delta_report(name: str, x: np.ndarray, jump_factor: float = 5.0) -> DeltaReport:
    """Smoothness stats for one angle column: median/p95/max of the
    frame-to-frame circular delta, plus frames whose delta exceeds
    `jump_factor` times the median (a "jump"), and frames where the *raw*
    diff differed from the circular diff by a full wrap (an artifact of the
    angle's `atan2` range, not a tracking error).
    """
    x = np.asarray(x, dtype=float)
    raw_d = np.diff(x)
    circ_d = circular_delta_deg(x)
    abs_circ = np.abs(circ_d)
    median = float(np.median(abs_circ))
    p95 = float(np.percentile(abs_circ, 95))
    mx = float(np.max(abs_circ))
    threshold = max(jump_factor * median, 1e-6)
    jump_frames = [int(i + 1) for i in np.where(abs_circ > threshold)[0]]
    wrap_frames = [
        int(i + 1)
        for i in np.where((np.abs(raw_d - circ_d) > 1.0) & (abs_circ <= threshold))[0]
    ]
    rep = DeltaReport(name, median, p95, mx, threshold, jump_frames, wrap_frames)
    print(
        f"  [{name}] delta deg: median={median:.3f} p95={p95:.3f} max={mx:.3f} "
        f"| jump(>{jump_factor:.0f}x median={threshold:.3f}): {len(jump_frames)} frames {jump_frames} "
        f"| wrap-artifact frames: {wrap_frames}"
    )
    return rep


# ---------------------------------------------------------------------------
# Periodicity (FFT) and stroke-reversal detection
# ---------------------------------------------------------------------------


@dataclass
class SpectrumResult:
    freqs: np.ndarray
    power: np.ndarray
    dominant_freq: float
    """Cycles/frame, or Hz if `fps` was supplied to `fft_power_spectrum`."""
    power_fraction: float
    """Dominant bin's power / total AC power (DC excluded)."""
    unit: str


def fft_power_spectrum(x: np.ndarray, fps: float | None = None) -> SpectrumResult:
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - np.mean(x)
    spec = np.fft.rfft(xc)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=(1.0 / fps if fps else 1.0))
    ac_power = power[1:]
    if ac_power.size == 0 or not np.any(np.isfinite(ac_power)) or ac_power.sum() == 0:
        return SpectrumResult(freqs, power, float("nan"), float("nan"), "Hz" if fps else "cycles/frame")
    peak_i = int(np.argmax(ac_power)) + 1
    dominant = float(freqs[peak_i])
    frac = float(power[peak_i] / ac_power.sum())
    return SpectrumResult(freqs, power, dominant, frac, "Hz" if fps else "cycles/frame")


def find_reversals(phi: np.ndarray, prominence: float = 60.0, distance: int = 5) -> np.ndarray:
    """Frame indices of stroke reversals: both maxima and minima of raw
    (non-unwrapped) `phi`, found by prominence-filtered peak detection.

    `prominence=60` is a diagnostic-plot choice (this module's own parameter,
    not a T4/pipeline.py setting), picked to sit above phi's own per-frame
    jump threshold (~61-63 deg, see `delta_report`) so single-frame noise
    spikes aren't double-counted as reversals. Even so, on the real dataset
    this still finds far more "reversals" than the ~2-3 a clean ~200 Hz
    wingbeat would produce over 100 frames at the assumed fps -- see
    `run_diagnostics`'s printed reversal count, called out in the report as
    evidence phi itself is noisier than a single-frequency oscillator here.
    """
    x = np.asarray(phi, dtype=float)
    peaks, _ = find_peaks(x, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-x, prominence=prominence, distance=distance)
    return np.sort(np.concatenate([peaks, troughs]))


# ---------------------------------------------------------------------------
# L/R symmetry
# ---------------------------------------------------------------------------


@dataclass
class SymmetryResult:
    correlation: float
    phase_offset_frames: int
    phase_offset_frac_cycle: float | None


def symmetry_report(name: str, a: np.ndarray, b: np.ndarray, period_frames: float | None) -> SymmetryResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    corr = float(np.corrcoef(a, b)[0, 1])
    ac = a - a.mean()
    bc = b - b.mean()
    xcorr = np.correlate(ac, bc, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    best_lag = int(lags[np.argmax(xcorr)])
    frac = float(best_lag / period_frames) if period_frames else None
    print(
        f"  [{name} L vs R] pearson r={corr:.3f}, best-lag={best_lag} frames"
        + (f" ({frac:.2%} of one cycle)" if frac is not None else "")
    )
    return SymmetryResult(corr, best_lag, frac)


# ---------------------------------------------------------------------------
# Angle-range plausibility
# ---------------------------------------------------------------------------


def angle_range_check(name: str, x: np.ndarray) -> str:
    """Returns "pass" / "concerning" / "fail" against `ANGLE_BANDS[name]`,
    printing the observed min/max/mean either way."""
    x = np.asarray(x, dtype=float)
    lo_obs, hi_obs, mean_obs = float(np.min(x)), float(np.max(x)), float(np.mean(x))
    bands = ANGLE_BANDS.get(name)
    if bands is None:
        print(f"  [{name}] min={lo_obs:.2f} max={hi_obs:.2f} mean={mean_obs:.2f} (no fixed band checked)")
        return "n/a"
    c_lo, c_hi = bands["concerning"]
    f_lo, f_hi = bands["fail"]
    if lo_obs < f_lo or hi_obs > f_hi:
        verdict = "fail"
    elif lo_obs < c_lo or hi_obs > c_hi:
        verdict = "concerning"
    else:
        verdict = "pass"
    print(
        f"  [{name}] min={lo_obs:.2f} max={hi_obs:.2f} mean={mean_obs:.2f} "
        f"vs concerning-band {bands['concerning']}, fail-band {bands['fail']} -> {verdict}"
    )
    return verdict


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_body_angles(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, col in zip(axes, ("yaw", "pitch", "roll")):
        ax.plot(df["frame_id"], df[col], marker=".", ms=3, lw=1)
        ax.set_ylabel(f"{col} (deg)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame_id")
    fig.suptitle("Body angles vs frame (raw, single-frame estimates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lr_overlay(
    df: pd.DataFrame, col_l: str, col_r: str, ylabel: str, title: str, out_path: Path,
    vlines: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["frame_id"], df[col_l], marker=".", ms=3, lw=1, label=col_l, color="tab:blue")
    ax.plot(df["frame_id"], df[col_r], marker=".", ms=3, lw=1, label=col_r, color="tab:orange")
    if vlines is not None and len(vlines):
        for v in df["frame_id"].to_numpy()[vlines]:
            ax.axvline(v, color="gray", lw=0.6, alpha=0.5, zorder=0)
    ax.set_xlabel("frame_id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fft(spec_l: SpectrumResult, spec_r: SpectrumResult, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(spec_l.freqs[1:], spec_l.power[1:], label="phi_L", color="tab:blue")
    ax.plot(spec_r.freqs[1:], spec_r.power[1:], label="phi_R", color="tab:orange")
    ax.axvline(spec_l.dominant_freq, color="tab:blue", ls="--", lw=0.8)
    ax.axvline(spec_r.dominant_freq, color="tab:orange", ls="--", lw=0.8)
    ax.set_xlabel(f"frequency ({spec_l.unit})")
    ax.set_ylabel("power")
    ax.set_title("Power spectrum of wing stroke phi (L/R)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticsResult:
    delta_reports: dict[str, DeltaReport] = field(default_factory=dict)
    range_verdicts: dict[str, str] = field(default_factory=dict)
    spectra: dict[str, SpectrumResult] = field(default_factory=dict)
    symmetry: dict[str, SymmetryResult] = field(default_factory=dict)
    reversals_l: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    reversals_r: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    chord_conf_at_reversal: dict[str, float] = field(default_factory=dict)
    chord_conf_baseline: dict[str, float] = field(default_factory=dict)
    eta_continuity_at_reversal: dict[str, float] = field(default_factory=dict)


def run_diagnostics(df: pd.DataFrame, out_dir: Path, fps: float | None = None) -> DiagnosticsResult:
    """Run every S6b check/plot against an already-loaded T4 output table
    (`pipeline.OUTPUT_COLUMNS` + `status`) and write PNGs + `report.md` into
    `out_dir`. Prints the same numbers to stdout. Rows with `status != "ok"`
    are dropped first (their angle fields are NaN by construction, see
    `pipeline._estimate_frame_impl`) -- reported as a count, not silently.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = DiagnosticsResult()

    n_total = len(df)
    ok = df[df["status"] == "ok"].reset_index(drop=True)
    n_bad = n_total - len(ok)
    print(f"frames total={n_total}, status=ok: {len(ok)}, dropped (non-ok): {n_bad}")
    if n_bad:
        print(f"  non-ok frames: {df.loc[df['status'] != 'ok', 'frame_id'].tolist()}")

    # 1. Body angles ---------------------------------------------------
    print("\n== Body angles ==")
    plot_body_angles(ok, out_dir / "01_body_angles.png")
    for col in ("yaw", "pitch", "roll"):
        result.delta_reports[col] = delta_report(col, ok[col].to_numpy())
        result.range_verdicts[col] = angle_range_check(col, ok[col].to_numpy())

    # 2/3/4. Wing series (phi, theta, eta) ------------------------------
    print("\n== Wing stroke phi ==")
    plot_lr_overlay(ok, "phi_L", "phi_R", "phi (deg)", "Wing stroke phi vs frame", out_dir / "02_phi.png")
    for col in ("phi_L", "phi_R"):
        result.delta_reports[col] = delta_report(col, ok[col].to_numpy())

    print("\n== Deviation theta ==")
    plot_lr_overlay(ok, "theta_L", "theta_R", "theta (deg)", "Deviation theta vs frame", out_dir / "03_theta.png")
    for col in ("theta_L", "theta_R"):
        result.delta_reports[col] = delta_report(col, ok[col].to_numpy())
        result.range_verdicts[col] = angle_range_check(col, ok[col].to_numpy())

    print("\n== Wing pitch eta (headline quantity) ==")
    plot_lr_overlay(ok, "eta_L", "eta_R", "eta (deg)", "Wing pitch eta vs frame", out_dir / "04_eta.png")
    for col in ("eta_L", "eta_R"):
        result.delta_reports[col] = delta_report(col, ok[col].to_numpy())

    # Reversals from raw phi extrema
    result.reversals_l = find_reversals(ok["phi_L"].to_numpy())
    result.reversals_r = find_reversals(ok["phi_R"].to_numpy())
    print(f"\n== Stroke reversals (phi extrema) ==")
    print(f"  phi_L reversal frames: {ok['frame_id'].to_numpy()[result.reversals_l].tolist()}")
    print(f"  phi_R reversal frames: {ok['frame_id'].to_numpy()[result.reversals_r].tolist()}")

    # 5. chord_conf with reversal markers --------------------------------
    print("\n== chord_conf vs frame (reversals marked) ==")
    all_reversals = np.union1d(result.reversals_l, result.reversals_r)
    plot_lr_overlay(
        ok, "chord_conf_L", "chord_conf_R", "chord_conf", "chord_conf vs frame (gray = phi reversal)",
        out_dir / "05_chord_conf.png", vlines=all_reversals,
    )
    for suffix, rev_idx in (("L", result.reversals_l), ("R", result.reversals_r)):
        conf = ok[f"chord_conf_{suffix}"].to_numpy()
        eta = ok[f"eta_{suffix}"].to_numpy()
        at_rev = conf[rev_idx] if len(rev_idx) else np.array([])
        baseline_mask = np.ones(len(conf), dtype=bool)
        baseline_mask[rev_idx] = False
        baseline = conf[baseline_mask]
        result.chord_conf_at_reversal[suffix] = float(np.mean(at_rev)) if len(at_rev) else float("nan")
        result.chord_conf_baseline[suffix] = float(np.mean(baseline)) if len(baseline) else float("nan")
        # eta continuity through reversal: circular delta at the reversal frame itself
        eta_cd = np.abs(circular_delta_deg(eta))
        rev_deltas = eta_cd[np.clip(rev_idx - 1, 0, len(eta_cd) - 1)] if len(rev_idx) else np.array([])
        result.eta_continuity_at_reversal[suffix] = float(np.mean(rev_deltas)) if len(rev_deltas) else float("nan")
        print(
            f"  [{suffix}] chord_conf at reversal={result.chord_conf_at_reversal[suffix]:.3f} "
            f"vs baseline={result.chord_conf_baseline[suffix]:.3f}; "
            f"eta |delta| at reversal={result.eta_continuity_at_reversal[suffix]:.2f} deg "
            f"(median eta |delta| overall={result.delta_reports[f'eta_{suffix}'].median:.2f} deg)"
        )
        result.range_verdicts[f"chord_conf_{suffix}"] = (
            "pass" if (conf.min() >= CHORD_CONF_BAND[0] and conf.max() <= CHORD_CONF_BAND[1]) else "fail"
        )

    # 6. FFT --------------------------------------------------------------
    print("\n== FFT of phi (periodicity) ==")
    spec_l = fft_power_spectrum(ok["phi_L"].to_numpy(), fps=fps)
    spec_r = fft_power_spectrum(ok["phi_R"].to_numpy(), fps=fps)
    result.spectra["phi_L"] = spec_l
    result.spectra["phi_R"] = spec_r
    unit = spec_l.unit
    print(f"  phi_L dominant freq = {spec_l.dominant_freq:.5f} {unit}, power fraction = {spec_l.power_fraction:.2%}")
    print(f"  phi_R dominant freq = {spec_r.dominant_freq:.5f} {unit}, power fraction = {spec_r.power_fraction:.2%}")
    plot_fft(spec_l, spec_r, out_dir / "06_fft_phi.png")

    if not spec_l.dominant_freq or not np.isfinite(spec_l.dominant_freq):
        period_frames = None
    elif fps:
        period_frames = fps / spec_l.dominant_freq  # dominant_freq is Hz here; convert back to frames
    else:
        period_frames = 1.0 / spec_l.dominant_freq  # dominant_freq is already cycles/frame
    if fps:
        print(f"  assumption: fps={fps} and dataset frames are consecutive raw camera frames (1:1), so "
              f"frequency-in-Hz = frequency-in-cycles/frame * fps.")

    # Symmetry --------------------------------------------------------------
    print("\n== L/R symmetry ==")
    result.symmetry["phi"] = symmetry_report("phi", ok["phi_L"], ok["phi_R"], period_frames)
    result.symmetry["theta"] = symmetry_report("theta", ok["theta_L"], ok["theta_R"], period_frames)
    result.symmetry["eta"] = symmetry_report("eta", ok["eta_L"], ok["eta_R"], period_frames)

    _write_report_md(out_dir / "report.md", ok, result, fps, n_total, n_bad)
    return result


def _verdict_line(name: str, verdict: str, detail: str) -> str:
    tag = {"pass": "PASS", "concerning": "CONCERNING", "fail": "FAIL", "n/a": "N/A"}[verdict]
    return f"- **{name}**: {tag} -- {detail}"


def _write_report_md(
    path: Path, ok: pd.DataFrame, r: DiagnosticsResult, fps: float | None, n_total: int, n_bad: int,
) -> None:
    lines = ["# S6b real-data kinematics diagnostics\n"]
    lines.append(f"Frames: {n_total} total, {len(ok)} status=ok, {n_bad} dropped.\n")

    lines.append("## Smoothness (frame-to-frame delta)\n")
    for name, d in r.delta_reports.items():
        lines.append(
            f"- `{name}`: median={d.median:.3f} deg, p95={d.p95:.3f} deg, max={d.max:.3f} deg, "
            f"jumps(>5x median)={len(d.jump_frames)} {d.jump_frames}, wrap-artifacts={d.wrap_frames}"
        )
    lines.append("")

    lines.append("## Periodicity (phi FFT)\n")
    for name, s in r.spectra.items():
        lines.append(f"- `{name}`: dominant freq={s.dominant_freq:.5f} {s.unit}, power fraction={s.power_fraction:.2%}")
    if fps:
        lines.append(f"- assumption: fps={fps}, dataset frames == consecutive raw camera frames (1:1)")
    lines.append("")

    lines.append("## L/R symmetry\n")
    for name, s in r.symmetry.items():
        frac = f", lag={s.phase_offset_frac_cycle:.2%} of one cycle" if s.phase_offset_frac_cycle is not None else ""
        lines.append(f"- `{name}`: r={s.correlation:.3f}, phase lag={s.phase_offset_frames} frames{frac}")
    lines.append("")

    lines.append("## Stroke reversals vs chord_conf / eta continuity\n")
    for suffix in ("L", "R"):
        lines.append(
            f"- `{suffix}`: chord_conf at reversal={r.chord_conf_at_reversal[suffix]:.3f} vs "
            f"baseline={r.chord_conf_baseline[suffix]:.3f}; eta |delta| at reversal="
            f"{r.eta_continuity_at_reversal[suffix]:.2f} deg (overall median="
            f"{r.delta_reports[f'eta_{suffix}'].median:.2f} deg)"
        )
    lines.append("")

    lines.append("## Angle-range plausibility\n")
    for name, v in r.range_verdicts.items():
        lines.append(f"- `{name}`: {v}")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI: run against the real 100-frame dataset
# ---------------------------------------------------------------------------

REAL_DATASET_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
SCRATCH_OUT = Path(__file__).resolve().parent / "diagnostics_output" / "s6b_g2b_g9"
FPS = 16000.0
"""Camera frame rate, user-confirmed (not discoverable anywhere in the repo)."""


def main() -> None:
    if not REAL_DATASET_ROOT.exists():
        print(f"SKIP  real dataset root not found: {REAL_DATASET_ROOT}")
        return

    SCRATCH_OUT.mkdir(parents=True, exist_ok=True)
    config = pipeline.PipelineConfig(
        min_points=10,
        output_dir=SCRATCH_OUT,
        write_debug=False,
        frame_glob="f*/splatfacto-checkpoint/*/*_labeled.csv",
    )
    df = pipeline.run_dataset(REAL_DATASET_ROOT, config)
    run_diagnostics(df, SCRATCH_OUT, fps=FPS)
    print(f"\nfigures + report.md written to: {SCRATCH_OUT}")


if __name__ == "__main__":
    main()
