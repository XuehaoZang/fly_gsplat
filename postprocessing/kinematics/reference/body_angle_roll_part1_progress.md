# body_angle_roll_part1 — progress log

## Setup note (before stage 1)

Worktree was missing `postprocessing/kinematics/simulate_gt/` and this brief
itself — both are untracked in the main checkout (`git status` shows `??`)
and git worktrees do not share untracked files. Copied both from the main
checkout (`/home/computer0/fly_project/fly_gsplat/postprocessing/kinematics/simulate_gt/`
and `.../reference/body_angle_roll_part1_brief.md`) into this worktree via
`cp -r` (no git operation involved, so the isolation rule wasn't touched).
Confirmed `git status` in the worktree now shows the same two untracked
paths as the original snapshot.

## Stage 1 — reproduction (2026-08-19)

Ran `python -m postprocessing.kinematics.simulate_gt.run_step2` (100-frame
`scenario_step2_flapping`). Numbers match the brief closely:

- `seg_accuracy`: mean=0.8150, std=0.2443 (brief: mean=0.815, std=0.244) ✓
- `t3_roll_deg`: mean=13.897, max=69.352 (brief: mean=13.90, max=69.35) ✓
- `t4only_roll_deg`: mean=2.426, max=6.533 (brief: mean=2.43, max=6.53) ✓

Conclusion holds: T4 math is accurate given clean labels (~2.4° roll error);
almost all real-world roll error (13.9°) comes from T3 segmentation quality.
No drift from the brief's numbers.

Ran `python -m postprocessing.kinematics.simulate_gt.run_step2_motion`:
`seg_accuracy` mean=0.8095 over the valid 28-frame window [36,63] (brief:
0.810 for the same, matches).

**Redid the frame-aligned kmeans_v2 vs motion comparison myself** (script:
`/tmp/.../scratchpad/aligned_compare.py`, output copied to
`diag/step2_aligned_kmeans_vs_motion.csv`), scoring both methods on the
identical [36,63] 28-frame subset:

- kmeans_v2 on [36,63]: mean=0.8731, std=0.2008
- motion on [36,63]: mean=0.8095, std=0.1178

This reproduces the brief's trap lesson independently: kmeans_v2 is
meaningfully *better* than motion on the aligned subset (0.873 vs 0.810),
the opposite of what the unaligned "0.815 vs 0.810" comparison would
suggest. Confirms **kmeans_v2 is the right baseline to improve**, not
motion — motion doesn't currently beat it even on its own best-case window.

Aggregated confusion matrices on the aligned subset (sum over 28 frames):

kmeans_v2:
```
pred    body  wing_L  wing_R
gt
body    8477     278     401
wing_L   966    3476      66
wing_R   375     170    3571
```
motion:
```
pred    body  wing_L  wing_R
gt
body    9156       0       0
wing_L  1548    2707     253
wing_R  1367     219    2530
```
Both methods: body<->wing confusion dominates over wing_L<->wing_R confusion.
Motion never puts a body point into a wing bucket (all its errors are
wing->body), consistent with its voxel-persistence mechanism picking up
slow-moving wing-root points as "body". kmeans_v2's errors are more
symmetric but still body<->wing dominated.

## Stage 2 — wing-root localization (2026-08-19)

Script: `/tmp/.../scratchpad/localize_wingroot.py`, full 100-frame set,
kmeans_v2, using exact GT `hinge_L`/`hinge_R` positions (never estimated).

- Misclassification kind counts (100 frames): wing_L->body=2711,
  wing_R->body=2651, body->wing_L=2533, body->wing_R=2087,
  wing_L->wing_R=1295, wing_R->wing_L=473. Body<->wing confusion
  (9982 points) dominates over wing_L<->wing_R confusion (1768 points),
  ~85% vs ~15%. Confirms the brief's hypothesis: **the dominant error mode
  is body<->wing confusion, not left/right swap.**
- Spatial concentration near hinge: wrong points have *lower* mean
  distance-to-nearest-hinge than correct points (wrong mean=0.0009mm,
  correct mean=0.0011mm, ratio 1.19). At a tight threshold
  (0.05*body_length ~ 0.0001mm) wrong points are ~3x more likely to fall
  within it than correct points (3.1% vs 1.1%); the effect narrows but
  persists out to 0.3*body_length (48.5% vs 35.8%). This is a real but
  "soft" concentration, not "100% of errors within a tiny radius" — some
  frames (see below) show much more catastrophic, less localized failures.
- Separately noticed: several frames (2,3,5,20,35,36,39,75,82,83,87,98,99)
  have catastrophic seg_accuracy (0.18-0.4) with `status_t3` showing
  `wing_L/R:fit_plane RANSAC` failures downstream in T4's chord fit — this
  is chord.py correctly failing on a garbage wing point set handed to it by
  T3, not a chord.py bug (out of scope per hard constraints, not touched).
  These look like wholesale cluster-assignment failures (not just boundary
  blur), a second, distinct failure mode from the smooth wing-root halo.

**Feature separability check** (script:
`/tmp/.../scratchpad/feature_separability.py`): for every per-point feature
available in `df_unlabeled` but not in kmeans_v2's `FEATURES_V2=[x,y,z,
opacity,R]`, computed Cohen's d of gt=body vs gt=wing, separately for points
near the hinge (<0.15*body_length) vs far from it. Most candidate features
(planarity/scale_ratio/sphericity/linearity/scale_phys_*) are flat or worse
near the hinge than far from it — consistent with kmeans_split.py's existing
"not robust" verdict on planarity et al., not re-litigated.

Two features stood out as *more* separable near the hinge than far from it:
- `local_density`: near-root d=1.10 vs far-root d=0.28 (looks striking, but
  **flagged as an anomaly, not used**: `mock.py::_knn_local_density`'s
  docstring calls it a "placeholder proxy" using `1/mean_knn_dist**3`, while
  real production's `utils/gaussian_features.py` uses `1/mean_knn_dist`
  (no cube). `scene.py::_recompute_whole_cloud_features`'s docstring claims
  it "uses the same formula as gaussian_features.py" but the code actually
  calls `mock._knn_local_density`, which does NOT match — cubing the
  reciprocal distance would exaggerate any density gap. This mismatch means
  the near-root separability of `local_density` in simulate_gt may be a
  synthetic-only artifact, not a transferable real-data signal. Not fixing
  this discrepancy (out of scope, touches shared mock.py used by both step1
  and step2, higher-risk than this task's budget allows) — reporting it as
  a finding per "report anomalies, don't hide them."
- `dist_to_principal_axis`: near-root d=0.98 vs far-root d=0.61 — same
  formula confirmed identical between `scene.py::_recompute_whole_cloud_features`
  and `utils/gaussian_features.py::compute_gaussian_features` (both: distance
  from point to the whole-cloud's 1st PCA axis), so no synthetic-artifact
  risk. This became the stage-3 candidate.

Direction picked: **5.2 (add an existing-but-unused point feature to the T3
KMeans feature set)**, specifically `dist_to_principal_axis`, targeting the
located body<->wing confusion. Considered 5.3 (segment_frame_motion) first
per the brief's instructions but stage 1's aligned comparison already showed
motion doesn't beat kmeans_v2 even on its best window, so investing further
in motion's boundary-frame coverage problem would be building on a weaker
base. Also considered v3's dual-wing seed-guided init (kmeans_split.py
already has `run_kmeans_v3`/`build_seed_init_v3` etc., unused in production)
since it's exactly targeted at "body cluster swallows wing root" — but a
prior memory note (`project_t3_kmeans_body_wing_split.md`) found v3 fixes
only random-seed *instability*, not the hardcut itself, on real data;
skipped re-testing it given the time budget and that prior finding.

## Stage 3 — implementation (2026-08-19)

Added `postprocessing/kinematics/simulate_gt/segment.py::segment_frame_kmeans_v2_axis_dist`
(new function, `kmeans_split.py` and `segment_frame_kmeans_v2` untouched) —
same pipeline as `segment_frame_kmeans_v2` but with one extra standardized
KMeans feature column, `dist_to_principal_axis`, weighted by a new
`axis_dist_weight` param (default candidate under test, see sweep below).
Also added `evaluate.py::evaluate_frame`'s new `segment_fn` param, defaulting
to `segment.segment_frame_kmeans_v2` (unchanged default), so alternative
segmenters can be scored through the same T3/T4-only/GT error-report path
without touching `evaluate_frame`'s existing behavior.

Weight sweep on the full 100-frame set (script:
`/tmp/.../scratchpad/sweep_axis_dist_weight.py`, `seg_accuracy` +
`t3_roll_deg` mean/max per weight): first pass suggested `axis_dist_weight
=0.5` gave a ~30% roll_deg mean reduction (13.90 -> 9.78) at ~unchanged
`seg_accuracy`. **This number was wrong** -- see "self-caught measurement
bug" below. Do not trust it; superseded by the corrected sweep further down.

### Self-caught measurement bug (section-4-style trap, this time in my own script)

Before accepting the promising weight=0.5 result, re-checked it against
`run_step2.py`'s own convention for computing `t3_roll_deg` mean, per this
brief's section 4 rule ("re-verify numbers you produce yourself mid-task,
not just numbers from old CSVs"). Found: my sweep/before-after scripts only
appended `roll_deg` to the running mean when `status_t3 == "ok"`. But
`pipeline._estimate_frame_impl` computes the body frame (yaw/pitch/roll)
*before*, and independently of, the per-wing chord fit that `status_t3`
reports failures for (`wing_L/R:fit_plane RANSAC ...`) -- a wing-only chord
failure does not invalidate an already-computed roll value. Gating on
`status_t3=="ok"` silently dropped the ~13 hardest frames (the ones with
`fit_plane RANSAC` failures, which are exactly the catastrophic-seg-accuracy
frames from stage 2) from the "after" mean but NOT from run_step2.py's own
baseline reporting convention (which does a plain `nanmean` over whatever
`errors_t3` produced, regardless of status) -- an apples-to-oranges
comparison that flattered any change reducing the frame count captured.

Fixed both scripts to nanmean over all frames' `roll_deg` (matching
`run_step2.py`'s own convention) and reran the full sweep:

| axis_dist_weight | seg_acc mean | t3_roll mean | t3_roll max | n_ok_t3 |
|---|---|---|---|---|
| 0 (baseline) | 0.8150 | 13.8971 | 69.3518 | 87 |
| 0.05 | 0.8146 | 13.9730 | 69.3518 | 87 |
| 0.1 | 0.8188 | 14.2122 | 69.3518 | 88 |
| 0.2 | 0.7988 | 13.3562 | 65.7648 | 84 |
| 0.3 | 0.7966 | 13.8177 | 61.7705 | 85 |
| 0.5 | 0.8129 | 15.7293 | 94.9280 | 85 |
| 1.0 | 0.7228 | 24.8189 | 116.4597 | 70 |

Corrected conclusion: **the effect is much weaker and non-monotonic**, not
the clean ~30% win the buggy script suggested. Only `weight=0.2` gives any
improvement (13.90 -> 13.36 mean, ~4%, and max 69.35 -> 65.76); weight=0.5
(the number the bug made look best) is actually *worse* than baseline once
measured correctly. Set `AXIS_DIST_WEIGHT_DEFAULT = 0.2` as the
best-of-what-was-tested candidate, but flagged its docstring as marginal,
not a validated win.

**Further check per brief 5.2's own evaluation criterion** ("judge by
near-root correctness specifically, not overall seg_accuracy"): compared
per-point accuracy in the near-hinge zone (<0.15*body_length) vs far from
it, baseline vs weight=0.2 (script: `/tmp/.../scratchpad/near_root_accuracy.py`):

| | near-root accuracy (n=7433) | far-from-root accuracy (n=56067) |
|---|---|---|
| baseline (v2) | 0.6787 | 0.8330 |
| axis_dist (w=0.2) | 0.6607 | 0.8171 |

**Worse in both zones**, including the exact zone this feature was chosen
to help. The isolated Cohen's d signal (feature separates gt=body vs
gt=wing better near the hinge than far from it, in isolation) did **not**
translate into a better joint 5D/6D KMeans partition -- most likely because
`dist_to_principal_axis` is highly correlated with the `[x,y,z]` block
already in the feature set (both derive from the same point positions via a
global PCA), so adding it doesn't supply much genuinely new separating
information, just reweights/perturbs an already-present signal, and the
~4% `t3_roll_deg` improvement at w=0.2 looks like noise from a handful of
frames rather than a systematic fix.

**Also checked (5.1-adjacent) whether the body seed_mask
(`opacity>=0.98 or R<0.2`) itself preferentially mis-seeds near-hinge wing
points as body** -- a plausible clean mechanism for the body-swallows-root
pattern. Script: `/tmp/.../scratchpad/seed_contamination.py`. Result: 7.63%
of near-hinge wing points get wrongly seeded vs 7.45% far from the hinge --
essentially identical, not a localized effect. Rules out "seed threshold is
biased near the root" as the mechanism; the confusion more likely comes
from the KMeans assignment step itself operating on genuinely overlapping
feature-space regions near the root, not a seeding bias.

**Honest bottom line for stage 3**: the one focused change actually
implemented and validated end-to-end (`segment_frame_kmeans_v2_axis_dist`,
new function, zero change to production defaults) does **not** show a
robust improvement on the roll-error metric this task targets, despite
being motivated by a real, re-verified diagnostic finding (body<->wing
confusion concentrated near the wing hinge) and a plausible feature
candidate. This is reported as a negative/marginal result, not smoothed
into a false "success," per this repo's "report anomalies, don't hide them"
convention. See final report / section 6 for what a next attempt should try
instead.

## Stage 4 — before/after, identical 100-frame set (2026-08-19)

Script: `/tmp/.../scratchpad/before_after.py` (corrected roll accounting),
`scenario_step2_flapping`, `AXIS_DIST_WEIGHT_DEFAULT=0.2`:

| | seg_accuracy mean | t3_roll_deg mean | t3_roll_deg max | n_ok_t3 |
|---|---|---|---|---|
| BEFORE (`segment_frame_kmeans_v2`) | 0.8150 | 13.8971 | 69.3518 | 87/100 |
| AFTER (`segment_frame_kmeans_v2_axis_dist`, w=0.2) | 0.7988 | 13.3562 | 65.7648 | 84/100 |

Confusion matrices (sum over 100 frames):

BEFORE:
```
pred     body  wing_L  wing_R
gt
body    28080    2533    2087
wing_L   2711   12094    1295
wing_R   2651     473   11576
```
AFTER:
```
pred     body  wing_L  wing_R
gt
body    27623    2841    2236
wing_L   2791   11794    1515
wing_R   2907     488   11305
```
Body-recall drops slightly (28080->27623 correct out of 32700 body points),
wing_L/wing_R-recall also drops slightly. All three diagonal cells get
worse; the small roll_deg mean improvement is not backed by a cleaner
confusion matrix, consistent with the "noise, not a systematic fix"
read from the near-root-accuracy check above. Treat this as an essentially
null result, not a validated improvement, despite the positive-looking
top-line roll number.

## Stage 5 — regression check (2026-08-19)

`python -m pytest postprocessing/kinematics/tests/test_s2.py -v`: **9/9
passed**, including `test_clean_scenario_recovers_yaw_pitch_roll` (roll
tolerance 6.0°, yaw/pitch 3.0°) at unmodified tolerances. Expected/required
since `evaluate.py`/`kmeans_split.py`/`segment_frame_kmeans_v2`'s default
behavior was never touched -- all new code is in new, unused-by-default
functions/params.

Real-data sanity check (brief step 5's second half, `calc_kinematics.py` +
`diagnostics.py` on a real subset) was **skipped**: given stage 3/4's
honest finding that `segment_frame_kmeans_v2_axis_dist` is an essentially
null/marginal result on the synthetic ground-truth benchmark itself (worse
confusion matrix, worse near-root accuracy, only a noisy ~4% roll_deg mean
change not backed by the confusion matrix), running it against real data
would not be an informative use of remaining time -- there's no validated
change to sanity-check for regression yet. Left undone; see next-steps.

## What's left / next steps

1. **No validated roll-error fix landed this session.** The localization
   (stages 1-2) is solid and reproducible: body<->wing confusion (not L/R
   swap) dominates, ~85%/15% split, softly concentrated near the wing hinge,
   plus a second distinct catastrophic-failure mode at ~13 specific frames
   (whole-cluster mis-assignment, not just boundary blur) that a next
   attempt should treat as a separate problem from the boundary-halo one.
2. **Ruled out, with numbers, so don't re-try as-is:** adding
   `dist_to_principal_axis` to the KMeans feature set at any weight tested
   (0.05-3.0) does not give a clean win; seed_mask contamination is not
   spatially localized near the hinge (7.6% vs 7.5%), so tightening
   `SEED_OPACITY_THRESH`/`SEED_R_THRESH` is unlikely to be a targeted fix
   either, though it wasn't directly tested.
3. **Best untested lead:** the ~13 catastrophic frames (2,3,5,20,35,36,39,
   40,75,82,83,87,98,99) look like a different mechanism (whole-cluster
   swap) than the boundary halo -- worth separately diagnosing what's
   different about the KMeans clustering result specifically on those
   frames (e.g. does the body cluster's point count/seed count fall below
   normal, does `_wing_merged`'s forced-median-split trigger, is it
   correlated with wingbeat phase/velocity -- an earlier look during this
   session found no clean correlation with `phi`/stroke-reversal timing,
   but that was not rigorously confirmed).
4. **Not tried:** `local_density` genuinely looked like the strongest
   near-root signal (Cohen's d 1.10 vs 0.28 far-root) but is blocked by the
   `mock.py`/`gaussian_features.py` formula mismatch found this session
   (cube vs no-cube). If someone fixes `mock._knn_local_density` to match
   production's `1/mean_knn_dist` (no cube) -- a `mock.py` change, outside
   this task's scope/risk budget -- re-running the same Cohen's d check and
   then the same weight-sweep-with-corrected-roll-accounting methodology
   used here on `local_density` would be the natural next experiment.
5. **Not tried:** combining `segment_frame_motion` with `segment_frame_kmeans_v2`
   (brief 5.3's fallback idea) -- skipped because stage 1 already showed
   motion doesn't outperform kmeans_v2 even on motion's own best window, so
   a fusion's ceiling is unclear without first improving motion's own
   accuracy. Not investigated further this session.
6. **Methodological note for whoever continues:** the section-4-style trap
   bit again this session, just in a script I wrote myself rather than an
   old CSV (see stage 3's "self-caught measurement bug"). Any new
   before/after roll comparison should nanmean over `errors_t3["roll_deg"]`
   for every frame, not gate on `status_t3=="ok"` -- the wing-chord status
   string does not reflect whether the independently-computed body-frame
   roll succeeded.
