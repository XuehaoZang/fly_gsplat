# LE/TE judge diagnosis — wing pitch (eta) 180° wrap crossings

Diagnose-only. No change to `wing_angles.py` / `chord.py` / `pipeline.py` /
`diagnostics.py`. No unwrap/smoothing/cross-frame logic introduced. All
RANSAC randomness uses fixed seeds. Starting point: Top issue #2 of
`reference/s6b_real_data_diagnostics_findings.md` (eta_L crosses the ±180°
wrap boundary on 30/99 real-data transitions while `chord_conf` stays
>0.92) — not re-proven here, only investigated further.

Scripts (in this directory) and their outputs (in `diag/`):
- `check_consistency.py` → `00_consistency_check.md`
- `synthetic_validation.py` → `02_synthetic_*`
- `real_data_validation.py` → `03_real_data_*`
- `s4b_comparison.py` → `04_s4b_*`

---

## 0. Reproduction fidelity (`le_repro.py`)

`estimate_leading_edge_diag`'s `le_dir`/`tip`/`root`/`inlier_mask`/`plane_normal`
matched `wing_angles.estimate_leading_edge`'s own output **exactly** (bit-for-bit,
`atol=0`) on all 20 tested (frame, side) pairs: 5 `mock.py` scenarios × 2 sides,
plus 5 real frames (f0000–f0004) × 2 sides. See `00_consistency_check.md`. All
downstream numbers in this report are computed from this verified-identical
reproduction.

---

## 1. Synthetic (ground-truth) validation: does curvature beat count?

**Setup**: `mock.py::default_ground_truth` + `make_wing_points`, clean (u=0
exactly at the true leading edge) geometry as ground truth, with positional
Gaussian noise added post-hoc (std swept 0–100% of `WING_MAX_CHORD_M`) to
manufacture boundary/near-tied cases. Ground truth for "which candidate set is
the true LE" is read from the *clean* (pre-noise) point positions projected
onto the true chord direction, never from the noisy positions used for
candidate selection itself — so the noise that makes the call hard cannot also
corrupt the answer key. Two point-density levels tested (120 and 400 points/wing,
the latter matching `mock.py`'s own default); 40 trials × 9 noise levels × 2
sides × 2 densities = 1440 trials, 37 failed with `ValueError` (excluded).

**Overall accuracy (all noise levels pooled):**

| density | count judge | curvature judge |
|---|---|---|
| 120 pts/wing | **0.7625** | 0.5417 |
| 400 pts/wing | **0.8346** | 0.7394 |

**Accuracy restricted to near-tied trials (`margin_ratio <= 1.3`):**

| density | n | count judge | curvature judge |
|---|---|---|---|
| 120 pts/wing | 382 | **0.6623** | 0.4241 |
| 400 pts/wing | 267 | **0.6517** | 0.5468 |

**Rescue analysis** (of trials where the count judge was wrong, how often was
the curvature judge right instead):

| density | n count-wrong | rescued by curvature | rescue rate |
|---|---|---|---|
| 120 pts/wing | 171 | 30 | 17.5% |
| 400 pts/wing | 113 | 39 | 34.5% |

**Finding: the curvature (pre-RANSAC arc-chord ratio) judge is NOT more
accurate than the current count judge in this synthetic setup — at every
noise level tested, at both point densities, overall and restricted to
near-tied trials, count-judge accuracy is equal to or higher than
curvature-judge accuracy.** The curvature judge only "rescues" 17.5–34.5% of
the count judge's wrong calls (denser point clouds rescue more often, but
still a minority). Per-noise-level breakdown and plots: `02_synthetic_accuracy.png`,
`02_synthetic_accuracy_table.csv`, `02_synthetic_validation_summary.md`.

A secondary observation surfaced while building this: even in the noiseless
case, the pre-RANSAC arc-chord ratio of the true-LE candidate set is not close
to 1.0 (median ≈1.2–1.6 in the raw trials, see `02_synthetic_trials_raw.csv`)
— the per-bin chordwise-extreme point selection itself introduces zigzag even
along the true straight edge when few points populate a bin, since the
extreme-of-a-few-random-points selection is a noisy order statistic. This
compresses the gap between the true-LE and true-TE candidate sets' arc-chord
ratios and is a plausible mechanism for why the curvature signal underperforms
here.

---

## 2. Real-data validation: does the LE/TE judge track eta wrap crossings?

**Setup**: production `pipeline.run_dataset` on `outputs/ctrl_009_002_8groups_100frames/G2b_G9`
(100/100 frames status=ok, fps=16000 per `diagnostics.py::FPS`), `le_repro`
diagnostics computed independently per (frame, side) on the same input CSVs
(200 rows, 0 LE-fit failures). Per-frame `|Δeta|` is the larger of that
frame's two neighboring `diagnostics.circular_delta_deg` transitions (reused,
not re-derived) — see `real_data_validation.py::_assoc_abs_delta` for the
exact definition.

**Correlation with `|Δeta|`** (Spearman, n=200):

| quantity | r | p |
|---|---|---|
| `margin_count` | -0.1958 | 0.0055 |
| `\|curvature_diff\|` | 0.0172 | 0.8086 |

`margin_count` (count judge's confidence margin) has a small but
statistically significant negative correlation with `|Δeta|` — larger margin
associates with smaller eta jumps. `|curvature_diff|` (the straight-vs-curved
gap between the two candidate edges) shows no significant correlation with
`|Δeta|` at all.

**2x2 contingency — count vs. curvature judge agreement × big jump (`|Δeta| > 90°`)**,
Fisher exact test:

```
big_jump      False  True
judges_agree
True             56    99
False            13    32
```

odds_ratio = 1.39, **p = 0.476 (not significant)**. Whether the count and
curvature judges agree on the winner is not associated with whether that
frame/side has a large eta jump.

**Wilcoxon signed-rank — is the count judge's winner's candidate set
systematically straighter (lower arc-chord ratio) than the loser's?**
n=200, 0 ties: statistic=3305.0, **p = 9.4e-17**. Winner median arc-chord
ratio = 1.5073 vs. loser median = 1.8516; the winner is straighter than the
loser in 77.5% of (frame, side) rows.

**Finding: on real data, the "straight edge = leading edge" assumption
itself holds up in aggregate** (count-winner is straighter than count-loser
77.5% of the time, highly significant), **but neither `margin_count` nor
`|curvature_diff|` meaningfully predicts which specific frames have large
`|Δeta|` jumps** — the margin_count correlation is real but small (r≈-0.20),
the curvature correlation is statistically indistinguishable from zero, and
judge agreement/disagreement does not predict big-jump status (Fisher
p=0.48). Scatter plots: `03_margin_count_vs_delta_eta.png`,
`03_curvature_diff_vs_delta_eta.png`; jump-size distribution:
`03_delta_eta_histogram.png`.

---

## 3. S4b contamination guard — independent parallel check

**Setup**: same 100-frame real dataset, `chord.estimate_chord` run twice per
frame/side directly (not through `pipeline.py`, which never sets these
flags): baseline (`robust=False, use_gaussian_normals=False`, S4a, current
production behavior) vs. enhanced (`robust=True, use_gaussian_normals=True`,
using the real `orientation_*`/`planarity` columns, present in all tested
CSVs). "Wrap-crossing" = a frame-to-frame `|circular_delta_deg| > 150°`.

| side | wrap-crossings baseline | wrap-crossings enhanced | \|Δeta\| median baseline | \|Δeta\| median enhanced | \|Δeta\| p95 baseline | \|Δeta\| p95 enhanced |
|---|---|---|---|---|---|---|
| L | 31/99 | 26/99 | 66.15° | 37.23° | 178.06° | 179.13° |
| R | 29/99 | 29/99 | 28.88° | 25.63° | 177.97° | 178.87° |

Contamination rejection was active on half the frames (50/100 for both
sides, 225–229 total points rejected across the clip); mean `chord_conf`
dropped slightly under the enhanced config (L: 0.9698→0.9574, R:
0.9664→0.9574).

**Finding: enabling the S4b contamination guard changes the eta series
(median `|Δeta|` drops materially on side L, is roughly flat on side R) and
reduces wrap-crossing count on side L (31→26) but leaves it essentially
unchanged on side R (29→29) and the p95/max `|Δeta|` stay ~178–180° on both
sides in both configs.** The guard is not independently sufficient to
resolve the eta bistability at the tail (worst-case jumps stay near the full
180° wrap in both configs); its effect is a partial, side-asymmetric
reduction in typical-case jump size and crossing count, not an elimination.
Series/delta plot: `04_s4b_eta_and_delta.png`; full numbers:
`04_s4b_comparison_summary.md`.

---

## 4. Mechanism attribution: `le_dir` flip vs `chord.py::_oriented_chord_axis` sign flip vs unexplained

**Setup**: `wrap_mechanism_diag.py`, same 100-frame real dataset (100/100
`status=ok`). `chord.py::_oriented_chord_axis` is reimplemented (not
imported) as `_oriented_chord_axis_diag`, exposing `raw_projection` (the
LE-side mean's projection onto the candidate axis before the sign-flip
check) and `axis_margin = |raw_projection| / norm(le_side_mean)`.
**Consistency gate**: the reimplemented axis, fed through the real
(unmodified) `chord.py::_bin_chords_core` / `_aggregate_chords` / `_eta`,
reproduces `chord.estimate_chord`'s own eta **exactly (200/200 rows,
`eta_diff == 0.0`, max diff 0.0e+00 deg)** — the axis-sign reimplementation
is verified correct, not just plausible. `le_dir` continuity uses the
already-verified-bit-identical `le_repro.py` reproduction. All deltas reuse
`diagnostics.py::circular_delta_deg`; the frame-level association for
`axis_margin`/`out_ref_norm` reuses `real_data_validation.py::_assoc_abs_delta`
unmodified (same convention `margin_count`/`curvature_diff` used in §2); the
transition-level pairing for `le_dir`/chord-axis continuity uses the raw
per-transition `circular_delta_deg` value directly, since a vector "flip"
between two frames is inherently a transition-level event.
`BIG_JUMP_DEG=90°` (§2's own threshold, reused). "Flipped" = `cos_angle <
-0.5` between a vector at frame *t* and frame *t+1* (the task's own
suggested more-discriminating threshold; `cos_angle < 0` reported as a
looser secondary count).

### 4.1 `le_dir` continuity vs `|Δeta|` (198 transitions, both sides pooled)

Of the 87/198 transitions with `|Δeta| > 90°`: only **5 (5.7%)** have
`le_dir` flipping strictly (`cos_angle < -0.5`); 18 (20.7%) cross zero at
all (`cos_angle < 0`). 2x2 contingency (`le_flip_strict` × `big_jump`):

```
big_jump          False  True
le_flip_strict
True                  3      5
False               108     82
```

Standard odds ratio (le_dir flip → big jump) = 2.20, Fisher exact **p =
0.303 (not significant)**. Scatter (`05_le_dir_cos_vs_delta_eta.png`) shows
most big-jump transitions sitting at `cos_angle` **0.8–1.0** (i.e. `le_dir`
barely moved) rather than near -1; the handful of true `le_dir` flips
(`cos_angle` near -1, top-left of the plot) account for only a small
fraction of the big-jump population.

### 4.2 `_oriented_chord_axis` sign flip vs `|Δeta|`

The same "vector cos_angle between adjacent frames" construction, applied
to the reimplemented (verified-correct) oriented chord axis instead of
`le_dir`, tells a very different story. 2x2 contingency
(`axis_flip_strict` × `big_jump`, same 198 transitions):

```
big_jump           False  True
axis_flip_strict
True                   4     65
False                107     22
```

**94.2%** of transitions where the chord axis itself flips (`cos_angle_axis
< -0.5`, 69/198 transitions total) are big jumps, vs **17.1%** of
non-flipping transitions. Standard odds ratio (axis flip → big jump) =
**79.0**, Fisher exact **p = 5.9e-28** — many orders of magnitude more
significant than the `le_dir` result above, and (unlike `le_dir`) `axis
flip` predicts *nearly all* of the axis-flip population being a big jump,
not just a mild enrichment.

**Three-way mechanism classification of the 87 big-jump transitions**
(`le_dir` flip checked first, then chord-axis flip, else unexplained):

| mechanism | count | % of big jumps |
|---|---|---|
| `chord_axis` sign flip (no `le_dir` flip) | 63 | 72.4% |
| unexplained (neither flipped past ±0.5) | 19 | 21.8% |
| `le_dir` flip | 5 | 5.7% |

Per side: L = 33 chord_axis / 12 unexplained / 3 le_dir (of 48 big jumps);
R = 30 chord_axis / 7 unexplained / 2 le_dir (of 39 big jumps) — the same
ordering on both sides. The 19 "unexplained" transitions are not a second
clean cluster: their `cos_angle_axis` spans -0.50 to +0.98 (median +0.13,
9/19 negative but short of the -0.5 cutoff) and `cos_angle_le` spans -0.42
to +0.98 (median -0.04) — i.e. these look like partial/borderline rotations
of both vectors rather than a third distinct flip mechanism, at least at
this threshold.

**`axis_margin` (single-frame sign-call decisiveness) does *not* explain
which frames flip.** Spearman correlation with `|Δeta|` (`_assoc_abs_delta`
convention, n=200): `axis_margin` r=-0.0681 (p=0.338) — weaker than §2's
`margin_count` (r=-0.1958, p=0.0055, recomputed here on the same 200 rows)
and comparable to `|curvature_diff|` (r=0.0172, p=0.809). Cross-checked
against §4.1's classification: `axis_margin` in "big jump & `le_dir` did
NOT flip" frames (n=124, median 0.9534) is not significantly different from
the rest (n=76, median 0.9644; Mann-Whitney U, p=0.107). Scatter
(`06_axis_margin_vs_delta_eta.png`) shows big jumps occurring across nearly
the whole `axis_margin` range (0.5–1.0), including at margins ≥0.9 that
look decisive by this single-frame measure — the axis flip is a
*discontinuity between two single-frame decisions that individually look
confident*, not a case of either frame's own margin being marginal.

### 4.3 `out_ref_norm` (orienting-vector magnitude)

`out_ref_norm = |wing_centroid - body_cm|`, per (frame, side). Mann-Whitney
U, one-sided (smaller in the flagged group):

- eta big jump (`_assoc_abs_delta` convention): True (n=131, median
  1.617e-03 m) vs False (n=69, median 1.632e-03 m) — **p=0.265, not
  significant**.
- `le_dir` flipped (dominant transition): True (n=10, median 1.488e-03 m)
  vs False (n=190, median 1.627e-03 m) — **p=0.101, not significant** (a
  ~9% lower median, trending in the hypothesized direction, but the n=10
  flipped-frame sample is small and the test does not clear significance).

Boxplots (`07_out_ref_norm_boxplot.png`) show heavily overlapping
distributions in both comparisons with no visible separation.

### 4.4 Answers to the section's three questions

1. **Mechanism**: of the 87 big-jump transitions, chord-axis sign flips
   (§4.2's `_oriented_chord_axis` reimplementation, cross-verified
   bit-exact against production eta) account for **72.4%**, an
   `le_dir`-continuity flip accounts for **5.7%**, and **21.8%** cross
   neither threshold cleanly (borderline partial rotations of both
   vectors, §4.2). The axis-flip × big-jump association itself is
   overwhelming (94.2% precision, OR≈79, p=5.9e-28) — far stronger than
   anything found for `le_dir` (p=0.303) or, in §2, for `margin_count`/
   `curvature_diff`.
2. **`axis_margin` vs `margin_count`/`curvature_diff`**: `axis_margin`
   (r=-0.0681) does **not** beat `margin_count` (r=-0.1958) as a predictor
   of `|Δeta|`, and is comparable to `|curvature_diff|` (r=0.0172) —
   despite the axis-flip *event* itself being the strongest single
   predictor found across both rounds of this investigation. A per-frame
   decisiveness score and a per-transition flip event are evidently
   different things here: individual frames' axis calls look confident
   (`axis_margin` mostly 0.85–1.0) right up to the frame where the call
   flips to the other side.
3. **`out_ref_norm`**: not significantly smaller in either the eta-big-jump
   or the `le_dir`-flipped group (p=0.265, p=0.101); the `le_dir`-flip
   comparison trends in the hypothesized direction (~9% lower median) but
   is underpowered (n=10 flipped transitions).

Full numbers: `05_06_07_mechanism_summary.md`; raw per-transition/per-frame
data: `05_le_dir_transitions.csv`, `06_axis_margin_frame_diagnostics.csv`,
`06_chord_axis_transitions.csv`, `05_06_transitions_merged.csv`,
`06_07_merged_per_frame.csv`.


---

## 8. Winner identity (`use_pos`) cross-frame flip vs chord-axis flip / eta big jumps

**Setup**: `winner_flip_diag.py`, joining `03_real_data_merged.csv`'s `count_winner_is_pos` (== `LEDiag.use_pos`, already bit-exact-verified, §0) at each transition's `frame_from`/`frame_to` with `05_06_transitions_merged.csv` (§4's own `le_flip_strict`/`axis_flip_strict`/`big_jump`/`mechanism`, unmodified). No new geometry recomputed. `winner_flip = (use_pos_t != use_pos_{t+1})`, 198 transitions (both sides pooled, 0 dropped for missing lookups).

- `winner_flip=True` on 89/198 transitions (44.9%).

### 8.1 `winner_flip` x `axis_flip_strict`

```
axis_flip_strict  False  True 
winner_flip                   
True                 33     56
False                96     13
```
- Fisher exact, standard odds ratio (`winner_flip` → `axis_flip_strict`): OR=12.53, p=3.257e-14

- For direct scale comparison (same standard-OR convention as §4): `winner_flip → big_jump` OR=9.57, p=5.395e-13 (context 2x2 above); §4.2's `axis_flip_strict → big_jump` OR≈79.0, p=5.9e-28; §4.1's `le_flip_strict → big_jump` OR=2.20, p=0.303. `winner_flip`'s association with both `axis_flip_strict` and `big_jump` is far stronger than `le_dir`'s (orders of magnitude lower p, roughly 4-9x higher OR), though still weaker than `axis_flip_strict`'s own direct association with `big_jump`.

### 8.2 `winner_flip` x `le_flip_strict`

```
le_flip_strict  False  True 
winner_flip                 
True               83      6
False             107      2
```
- Fisher exact, standard odds ratio (`winner_flip` → `le_flip_strict`): OR=3.87, p=0.1433 (not significant; only 6 vs 2 transitions in the `le_flip_strict=True` column, underpowered)
- Of 89 transitions with `winner_flip=True`, 83 (93.3%) do NOT clear the `le_flip_strict` threshold (`cos_angle_le >= -0.5`) -- i.e. `le_dir`'s direction stayed continuous while the underlying pos/neg winner switched.

### 8.3 Three-way reclassification of the 87 big-jump transitions (`winner_flip`-first vs `le_dir`-first)

v1 (original, `le_dir`-first, from `wrap_mechanism_diag.py` §4.2):

- `chord_axis`: 63 (72.4%)
- `unexplained`: 19 (21.8%)
- `le_dir`: 5 (5.7%)

v2 (`winner_flip`-first → `chord_axis` (no `winner_flip`) → `unexplained`):

- `winner_flip`: 64 (73.6%)
- `chord_axis`: 12 (13.8%)
- `unexplained`: 11 (12.6%)

`unexplained` share: 21.8% (v1) → 12.6% (v2).

### 8.4 `winner_flip` rate within the 19 v1-"unexplained" big-jump transitions

- 8/19 (42.1%) of the transitions v1 labeled `unexplained` have `winner_flip=True`.

### 8.5 Time series: `winner_flip` transitions vs eta, both sides

Plots: `08_winner_flip_timeseries_L.png`, `08_winner_flip_timeseries_R.png` (red = `winner_flip` transition start frame; blue dotted, L only = `s6b_real_data_diagnostics_findings.md` item #2's 30 documented wrap-crossing frames).

- Side L: 44 `winner_flip` transitions vs 30 s6b-documented wrap-crossing frames; 14 frames in common (46.7% of the s6b list): `[17, 24, 26, 32, 34, 41, 60, 65, 66, 67, 68, 73, 86, 98]`.

- Side R has no s6b reference list (item #2 only enumerated eta_L); the R plot shows `winner_flip` markers against the eta_R series with no overlay.

