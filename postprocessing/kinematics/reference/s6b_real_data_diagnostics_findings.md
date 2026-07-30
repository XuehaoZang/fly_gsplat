# S6b — real-data accuracy diagnostics (T4 kinematics)

Dataset: `outputs/ctrl_009_002_8groups_100frames/G2b_G9`, 100 frames (f0000-f0099),
100/100 `status = "ok"` (per S6a). No ground truth exists for real fly angles, so
this is an **indirect** validation: a single-frame estimator that produces smooth,
periodic, physically-plausible output is very likely tracking the real quantity;
conversely, specific jumps/asymmetries/implausible ranges are diagnostic even
without ground truth. **Diagnose only** — no algorithm or parameter change was
made to `pipeline.py`/`body_frame.py`/`wing_angles.py`/`chord.py`, and the output
was not smoothed or unwrapped.

Reproduce with `python -m postprocessing.kinematics.diagnostics` (writes PNGs +
a numeric `report.md` to `postprocessing/kinematics/diagnostics_output/s6b_g2b_g9/`,
gitignored, never into tracked `outputs/`). Full console output has every number
below; this file is the qualitative read.

**fps assumption**: user-confirmed camera rate = 16000 fps, and `f0000..f0099`
are assumed consecutive raw camera frames (1:1) — `run/serial/batch_8groups_100frames.py`
passes `target_frame` directly as a raw frame index with no visible stride/subsampling,
consistent with this. All Hz numbers below depend on this assumption; if wrong,
scale accordingly (see periodicity finding below, which is itself in tension with it).

---

## 1. Smoothness (frame-to-frame delta)

| angle | median Δ | p95 Δ | max Δ | jump frames (>5x median) |
|---|---|---|---|---|
| yaw | 3.47° | 108.0° | 177.1° | 24 (incl. clusters at 63-72, 81-90) |
| pitch | 4.16° | 26.6° | 33.9° | 9 |
| roll | 11.9° | 105.5° | 178.2° | 12 |
| phi_L | 12.2° | 151.7° | 175.9° | 21 |
| phi_R | 12.6° | 118.2° | 168.5° | 14 |
| theta_L | 10.8° | 75.2° | 105.2° | 14 |
| theta_R | 11.3° | 51.6° | 103.4° | 5 |
| eta_L | 66.1° | 178.1° | 180.0° | **0** (threshold itself blows up to 331° because the baseline delta is already huge — the "5x median" rule breaks down here, see below) |
| eta_R | 28.9° | 178.0° | 180.0° | 31 |

**Verdict: CONCERNING.** `pitch` is genuinely smooth (median 4.2°, few jumps).
Everything else has a heavy-tailed jump population — `yaw`/`roll`/`phi`/`theta`
jump 20-100+° in a single 1/16000s frame in ~10-25% of transitions, which is not
physically possible for real body/wing rotation at that timescale. `eta`'s
median delta (66° for eta_L) is itself larger than most "jump" thresholds used
for other angles — i.e. eta doesn't have an occasional-jump problem, it has a
*baseline* noise problem (see §4).

## 2. Periodicity (FFT of phi)

Both `phi_L` and `phi_R` have their FFT's single largest AC-power bin at
**160 Hz** — which is the *lowest resolvable frequency* given `df = fps/n =
16000/100 = 160 Hz`. Power fraction there: 44.7% (L), 56.9% (R); no secondary
peak visible anywhere else in the spectrum (see `06_fft_phi.png`).

**Verdict: INCONCLUSIVE, leaning CONCERNING.** A peak at the very lowest bin
reads as "most non-DC power is in the slowest possible trend across the 6.25 ms
window," not as evidence of a genuine oscillator — a true ~200 Hz Drosophila
wingbeat would need a much longer clip (100 frames / 16000 fps ≈ 1.25 cycles is
far too short for 160 Hz frequency resolution to distinguish 160 Hz from 200 Hz).
Compounding this: naive reversal-counting on raw `phi` (prominence=60°,
distance=5 frames — a diagnostic-plot parameter, not a pipeline setting) finds
**13 reversals in phi_L, 10 in phi_R** over 100 frames — 4-5x more than the ~2-3
a single ~200 Hz cycle would produce at 16 kHz. Either the fps assumption is
off, the true wingbeat frequency is far higher than the literature range, or
`phi` carries enough frame-level noise to manufacture extra apparent reversals.
This tension is itself worth resolving before trusting any Hz number here.

## 3. L/R symmetry

| pair | r (all frames) | r (excl. flagged jump frames) | phase lag* |
|---|---|---|---|
| phi_L vs phi_R | 0.485 | 0.613 | -8 frames (-8% of the 100-frame "cycle") |
| theta_L vs theta_R | -0.016 | -0.055 | +10 frames (+10%) |
| eta_L vs eta_R | 0.250 | 0.317 | +3 frames (+3%) |

*phase lag uses the FFT's 100-frame "period," which §2 flags as low-confidence — treat as indicative only.

**Verdict: CONCERNING.** `phi` shows moderate correlation that improves only
modestly once the ~29 already-flagged jump frames are excluded (0.49→0.61) —
so bad frames explain part but not all of the weakness. `theta` stays at
essentially zero correlation with or without those frames excluded — a
genuine finding, not an artifact of the bad-frame contamination. `eta` is weak
throughout (0.25-0.32).

## 4. Stroke-reversal behavior (direct evidence for the S4b chord method)

`chord_conf` at `phi`-reversal frames vs. all other frames:

| side | conf at reversal | conf baseline | Δ |
|---|---|---|---|
| L | 0.944 | 0.974 | -0.030 |
| R | 0.923 | 0.971 | -0.048 |

**This is the direct evidence the task asks to call out**: `chord_conf` *is*
systematically lower at reversal (as expected — two wings closest together,
hardest chord fit), but it stays **above 0.92 even at the hardest frames** —
it does not collapse toward 0 the way the old voxel-hull method would when the
two wings' hulls fuse (§5's stated failure mode for `find_chords_quad`). That
the Gaussian-normal-filtered, segmented chord fit keeps >90% confidence through
reversal on **real** data (not just the mock scenarios in `test_s4b.py`) is
good, concrete support for the S4b design rationale.

`eta` continuity through reversal is the opposite of clean, though: mean
`|Δeta|` at reversal is **88.7°** (L) / **95.6°** (R), both *higher* than each
side's own (already large) overall median delta of 66.1° / 28.9°. So `eta`
does not stay continuous through reversal here — see §5's chord-sign-bistability
hypothesis for why that's more likely a sign artifact than genuine per-frame
rotation.

## 5. Angle-range plausibility

Assumed bands (Dickinson/Fry-era free-flight Drosophila kinematics literature,
*not* derived from this dataset — generous "concerning" envelopes, "fail" bands
are the formula's own mathematical range):

| angle | observed range (mean) | concerning band | fail band | verdict |
|---|---|---|---|---|
| pitch | 0.02° to 35.0° (25.8°) | ±60° | ±90° | **PASS** |
| roll | -175.1° to 78.8° (-18.2°) | ±120° | ±180° | **CONCERNING** |
| theta_L | -84.0° to 85.7° (-7.7°) | ±40° | ±90° | **CONCERNING** |
| theta_R | -57.2° to 88.2° (-8.8°) | ±40° | ±90° | **CONCERNING** |
| chord_conf_L/R | within [0,1] both sides | — | [0,1] | **PASS** |

`yaw` (unbounded heading — no band checked) and `phi`/`eta` (judged by
smoothness/periodicity above, not absolute value) are excluded from this table
by design (see `diagnostics.py::ANGLE_BANDS`).

---

## Top issues to investigate next

1. **Body-frame axis flips, frames 63-72 and 81-90 (+ isolated 13, 39, 52, 55, 97
   throughout).** `yaw`/`roll` (and correlated `phi`/`theta`) alternate between a
   baseline value and a wildly different one on consecutive frames, then snap
   back — not physically possible motion at 16 kHz. This is consistent with the
   **already-documented** head/tail sign heuristic in `body_frame.py:101-113`
   (`dot(x_body, up) > 0`, flagged failure-mode-tested in
   `tests/test_s2.py::test_negative_pitch_head_sign_heuristic_documented_failure`):
   the sign choice gets fragile as the body axis nears horizontal (`pitch` near
   0°). Some flagged frames do show low `pitch` right at the flip (69: 6.6°, 81:
   2.7°, 85: 1.6°, 88: 8.9°), but others don't (63: 20.8°, 66: 30.9°) and the
   documented yaw+180°/pitch-negation signature doesn't cleanly match every
   flagged frame (e.g. frame 63) — so this is a strong lead, not a confirmed
   sole cause. Next step: pull `x_body`/`hinge_L`/`hinge_R` from the pipeline's
   debug pickle (`write_debug=True`) at these exact frames to see whether it's
   this heuristic, a different sign ambiguity in `y_body`, or upstream T3
   part-labeling quality at those specific frames.

2. **`eta` LE->TE chord-sign bistability, most of the clip (not just the bad
   windows above).** `eta_L`'s frame-to-frame delta crosses the ±180° wrap
   boundary on 30 of 99 transitions (frames 17, 24, 26, 27, 29, 32, 34, 41, 42,
   52, 53, 57-61, 65-68, 72-74, 81, 86, 87, 91, 96, 98, 99) — `eta` is
   bouncing between two levels ~180° apart even where `chord_conf` stays high
   and body/`phi` are otherwise stable. That pattern (high fit confidence, but
   value alternating by ~180°) points at the LE→TE sign disambiguation (§5) in
   `chord.py` occasionally picking the wrong edge assignment, not a low-quality
   chord fit. Next step: inspect `chord_L`/`le_dir_L` debug sidecars at a sample
   of these frames against the LE→TE ordering rule.

3. **Periodicity can't be confirmed at this clip length/fps.** §2's FFT peak
   sits at the lowest resolvable bin and the naive reversal count (13/10) is
   4-5x the ~2-3 a single ~200 Hz wingbeat would give over 100 frames at 16 kHz.
   Confirm the actual camera fps and whether `f0000-f0099` really are
   consecutive raw frames (no stride) — if so, a longer run (many more frames)
   is needed before periodicity claims are meaningful; if not, the whole Hz
   framing above needs to be redone with the correct spacing.

## Overall judgment

**Mixed — the pipeline is tracking something real some of the time, but its raw
single-frame output isn't reliable across the whole clip yet.** Positive
evidence: `pitch` is smooth and stays in a physically plausible range for all
100 frames; in the frames 0-50 window (excluding two single-frame outliers at
13 and 39), `phi_L`/`phi_R`/`theta_L`/`theta_R` move together in a
visually clean, plausible, roughly-covarying pattern; `chord_conf` stays above
0.92 even at the hardest (reversal) frames, which is concrete support for the
S4b chord method's design goal. Negative evidence: roughly half the clip
(the 63-72 and 81-90 windows especially) shows non-physical frame-to-frame
flips across yaw/roll/phi/theta simultaneously, `eta` is bistable through most
of the clip via an apparent sign issue rather than genuine noise, `theta` L/R
are essentially uncorrelated even outside the bad frames, and periodicity
can't be confirmed at this clip length. None of this rules out the underlying
per-frame geometry being basically correct — items #1 and #2 above look like
discrete sign-disambiguation bugs layered on top of a plausible estimate,
which is a very different (and more fixable) problem than the estimator being
fundamentally wrong. Multi-frame smoothing/robustness is explicitly out of
T4's single-frame scope (`calc_kinematics.md` §0/§6) — this diagnostic's
purpose was to establish whether that future work is warranted; it is.
