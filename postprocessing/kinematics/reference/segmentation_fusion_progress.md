# segmentation_fusion — progress log

## Setup note (before stage 1)

Worktree was missing `postprocessing/kinematics/simulate_gt/` and the three
reference docs (`segmentation_fusion_brief.md`, `body_angle_roll_part1_brief.md`,
`body_angle_roll_part1_progress.md`) -- all untracked in the main checkout,
git worktrees don't carry over untracked files. Copied all four paths from
`/home/computer0/fly_project/fly_gsplat/postprocessing/kinematics/...` via
plain `cp -r` (no git operation). Confirmed `git status` in this worktree now
shows the same 4 untracked paths as the main checkout.

Also confirmed: part1's code changes (the `dist_to_principal_axis` KMeans
feature experiment, `evaluate.py`'s `segment_fn` param) are **not** present
in this checkout's `segment.py`/`evaluate.py` -- that worktree was disposable
and never committed, so this task starts from the same clean
`kmeans_split.py`/`labeling.py`/`segment.py`/`evaluate.py` part1 started
from. Only the diagnostic *findings* (not code) carry over.

## Phase 4.1 — reproduction (2026-08-19)

Ran `python -m postprocessing.kinematics.simulate_gt.run_step2` (100-frame
`scenario_step2_flapping`, unchanged code):

- `seg_accuracy`: mean=0.814961 (brief/part1: 0.815) match
- `t3_roll_deg`: mean=13.897067 (brief/part1: 13.90) match
- `t4only_roll_deg`: mean=2.425718 (brief/part1: 2.43) match
- `status_t3` ok: 87/100 -- same 13 failing frames as part1
  (2,3,5,20,35,36,39,40,75,82,83,87,98,99 -- confirmed by frame list in this
  run's stdout, matches brief section 1.1 exactly).

Ran `python -m postprocessing.kinematics.simulate_gt.run_step2_motion`:
`seg_accuracy` mean=0.809505 over valid window [36,63] (brief: 0.8095,
match).

Numbers confirmed unchanged from part1/brief -- no drift.

Built one shared script (`/tmp/.../scratchpad/gen_base_data.py`) that
generates `scenario_step2_flapping(n_frames=100)` **once** (deterministic:
`seed=seed+frame_id` per frame), runs `segment_frame_kmeans_v2` on all 100
frames and `segment_frame_motion` on the valid `[36,63]` window once each,
and pickles xyz/gt_label/df_unlabeled/frame_gt/kmeans_pred/motion_pred per
frame to `/tmp/.../scratchpad/base_data.pkl` -- the single source of truth
every later phase (4.2/4.3/4.4/4.5) reuses, so no phase silently re-derives
its own frame set (the part1 trap). Sanity-check printout from this script,
using the pickled predictions directly (not the CLI scripts):

- kmeans_v2 on [36,63]: mean=0.8731 (matches part1's own re-derived number
  exactly)
- motion on [36,63]: mean=0.8095 (matches)
- kmeans_v2 on all 100: mean=0.8150 (matches run_step2.py's own number)

This is this round's confirmed startline: **kmeans_v2 (0.8731) still clearly
beats motion (0.8095) on motion's own best-case window** -- the part1
finding holds, motion alone is not a replacement for kmeans_v2, any fusion
must use motion as a supplementary signal per the brief's section 6 design
direction, not a primary classifier.

## Phase 4.2 — catastrophic-frame mechanism (2026-08-19)

Script: `/tmp/.../scratchpad/diag_catastrophic.py`, replicates
`segment_frame_kmeans_v2`'s internal steps (seed_mask/standardize_v2/
run_kmeans_v2/label_by_rule_a/`_wing_merged`) inline against the pickled
100-frame set to expose diagnostics `segment.py`'s public function doesn't
return, without touching `segment.py`/`kmeans_split.py`. Output:
`/tmp/.../scratchpad/catastrophic_diag.csv`.

**Clean mechanistic finding: `is_wing_merged` (the "two wings can't be
physically separated -> forced median-projection split" fallback,
`segment.py::_wing_merged` == `labeling.py::check_wing_merged`) is the
mechanism.** All 14/14 of part1's catastrophic frames trigger it; only 6/86
non-catastrophic frames trigger it. `is_wing_merged` is necessary but not
quite sufficient by itself for the RANSAC-failure label part1 used to define
"catastrophic" -- 2 more frames (1, 4) also trigger `is_wing_merged` and also
crash to low `seg_accuracy` (0.32, 0.26) but weren't in part1's 14-frame list
(their downstream chord RANSAC apparently still happened to converge despite
the bad segmentation) -- these should be treated as the same failure mode,
just not caught by the RANSAC-failure proxy. 4 more `is_wing_merged` frames
(14, 16, 23, 95) stay high-accuracy (0.82-0.86) -- triggering the merge check
does not always cause a crash, only usually.

**Root cause, traced one level deeper** (script:
`/tmp/.../scratchpad/diag_wing_merge_composition.py`, output
`wing_merge_composition.csv`): on `is_wing_merged` frames, the raw KMeans
"body" cluster captures far less of its own high-confidence body seeds
(`seed_mask`: opacity>=0.98 or R<0.2) -- mean 43.7% of body seeds land in the
body cluster on `is_wing_merged=True` frames vs 90.2% on `is_wing_merged=False`
frames (`catastrophic_diag.csv`'s `frac_seed_captured_by_body_cluster`
column). This is a genuine KMeans partition failure (the body cluster
under-forms), not a `seed_mask` threshold bias -- consistent with part1's
own "seed_mask contamination is not spatially localized near hinge, 7.6% vs
7.5%" finding; the failure is in the assignment step, not the seeding step.
When the body cluster under-forms, the pooled `wing_A+wing_B` cluster ends
up **on average 49.2% actual GT-body points** on merge-triggering frames vs
13.0% on non-triggering frames (point-biserial r=0.867, p=2e-31 across all
100 frames) -- an extreme, whole-cluster-scale version of the *same*
body<->wing halo confusion part1 found softly concentrated near the wing
hinge, not a mechanistically distinct failure mode. This body-heavy "wing"
mass spatially bridges the true wing_L/wing_R gap (body points sit between
both wings), so the physical connected-component check sees 1 blob instead
of 2, triggers `forced_wing_split`'s crude median-projection fallback, which
then cuts an already ~50%-body mass in half by spatial projection rather
than by any real body/wing distinction -- explaining the near-random
resulting accuracy (0.18-0.41) on these frames and the garbage wing point
sets that make `chord.py`'s RANSAC fail downstream.

**Correlation checks that came back weak/negative** (scripts
`diag_phase_corr.py`, `diag_wing_gap.py`): no clean single-variable
predictor found for *which* frames trigger the body-cluster under-formation
in the first place. Wingbeat phase (`phi_L`) is inconsistent -- several
catastrophic frames sit near the `phi=-40` stroke-reversal boundary (20, 98,
99) but others sit at `phi=-70` to `-90` (mid-stroke, 2,3,5,35,36) or `-109`
(75) -- no clean phase rule. Direct measurement of the true (GT-label) wing_L
vs wing_R point-cloud minimum gap is only weakly/borderline correlated with
`is_wing_merged` (point-biserial r=-0.184, p=0.066) -- the merge trigger is
not really about the two true wings being physically close together; it's
about the upstream KMeans body cluster failing for reasons this session did
not fully pin down (root-hinge geometry / feature-space overlap at that
specific pose is the remaining candidate, not confirmed).

**Relevance to fusion design (4.3/4.4)**: only 3/14 catastrophic frames (36,
39, 40) fall inside motion's valid `[36,63]` window; 11/14 are outside it.
Motion cannot directly help most of these worst frames -- any fix has to
come from the kmeans-side chain itself (seed capture / merge-check /
forced-split) or from a continuity mechanism that reaches beyond motion's
raw per-frame window. This is a **scope limitation to state plainly**, not
something a motion+kmeans fusion alone is likely to fully solve.

## Phase 4.3 — joint error analysis (2026-08-19)

Script: `/tmp/.../scratchpad/phase43_joint_analysis.py`, using the exact same
pickled `kmeans_pred_by_frame`/`motion_pred_by_frame`/`gt_label_by_frame` as
4.1/4.2 (no re-run), concatenated over the `[36,63]` window (17780 points,
28 frames).

**Recomputed confusion matrices fresh** (matches part1's numbers exactly,
confirming no drift):
```
kmeans_v2                        motion
pred    body  wing_L  wing_R     pred    body  wing_L  wing_R
gt                                gt
body    8477   278    401        body    9156      0      0
wing_L   966  3476     66        wing_L  1548   2707    253
wing_R   375   170   3571        wing_R  1367    219   2530
```

**Corrected the brief's own framing of the key asymmetry** (section 1.2 says
motion has "precision=1.0" on body; that is not quite the right number --
precision(pred=body) for motion is actually **0.7585** [9156/12071],
recall(gt=body) is **1.0000** [9156/9156, exactly matching part1]. The
useful, *actually*-1.0 property is different: **when motion predicts
NOT-body, it is never wrong about that -- 0/5709 such points have GT=body**
in this window. That's the real lever: trust motion's *negative* ("not
body") signal, not its positive ("is body") signal, which is only 76%
reliable. This distinction matters for how the fusion rule was implemented
below -- a naive reading of the brief's "precision=1.0 on body" would have
suggested trusting motion's *positive* body calls, which the data says not
to do.)

**Per-point joint breakdown** (n=17780): both correct 73.36% (13044), motion
right/kmeans wrong 7.59% (1349), kmeans right/motion wrong 13.95% (2480),
both wrong 5.10% (907).

**Two candidate one-directional override rules tested directly** (the
actual fusion-rule derivation, not predesigned):

1. **kmeans says body, motion says NOT body -> override to motion's label**
   (n=510 disagreement points): fixes 510/510 (100%, kmeans was wrong on
   every single one of these because GT != body here, matching the "motion
   never false-negatives body" property above), breaks **0** (there is no
   case where kmeans's "body" call was right and motion's override would
   have broken it, precisely because motion's not-body prediction never
   collides with a true body point in this window). Checked motion's own
   L/R sub-label accuracy on this subset separately (script
   `check_lr_of_override.py`): 499/510 (97.8%) exactly match GT including
   L/R side, only 11 points get the correct "not body" call but wrong L/R
   side -- still net-positive over kmeans's 0/510.
2. **kmeans says wing, motion says body -> override to body** (n=2763
   disagreement points): fixes 679 (kmeans was wrong, GT=body), but
   **breaks 2019** (kmeans was already right, GT=wing, motion's weaker 76%
   body-precision would flip a correct call to wrong). Net **-1340**,
   clearly harmful -- confirms the brief's section 6 instinct not to let
   motion override kmeans's wing calls, even though the mechanism (recall
   vs precision asymmetry) is the opposite of how the brief described it.

**Derived fusion rule (data-driven, not predesigned)**: apply rule 1 only,
one direction, inside motion's valid window -- whenever kmeans's raw
clustering assigns a point to the body cluster but motion's voxel-density
signal (within its window) says that point is NOT in a persistently-occupied
"body" voxel, move that point out of the body cluster before the rest of
`segment_frame_kmeans_v2`'s pipeline (wing-merge check / connectivity fixup
/ L/R anchoring) runs on it -- i.e. a **veto on kmeans's body cluster**, not
a full label copy from motion. This is implemented as a spatial/voxel-level
continuity mechanism (motion's own `HALF_WINDOW=36` window IS the
cross-frame evidence), satisfying brief section 2's "体素/空间层面" category
directly, without inventing a second mechanism. Outside motion's valid
window the veto step is a no-op by construction (no motion signal
available), so behavior degrades gracefully to plain `segment_frame_kmeans_v2`.
See phase 4.4 for the implementation.

## Phase 4.4 — fusion implementation (2026-08-19)

New shared module: `postprocessing/labeling/fusion.py`. Contains exactly the
new mechanism derived in 4.3, nothing else:
- `motion_body_veto(xyz, semantic, is_body_motion)`: for every point
  currently labeled `body`, if `is_body_motion` says it's NOT in a
  persistently-occupied voxel, reassign it (1-NN spatial distance) to
  whichever of `wing_A`/`wing_B` it's nearest to. `is_body_motion=None`
  (motion has no signal, e.g. outside its window) is a strict no-op. Also
  guards against vetoing the *entire* body cluster (defensive: never let
  the veto degenerate a frame kmeans otherwise handled fine) and against an
  empty wing pool to reassign into.
- `motion_is_body_for_window(window_xyz_by_frame, center_frame_idx,
  half_window)`: in-memory adapter for `simulate_gt`, delegates the actual
  voxel computation to `motion/density.py`'s existing primitives
  (`compute_voxel_frame_counts`/`extract_body_voxels`/`points_to_voxel_keys`)
  -- same computation `segment_frame_motion` already does internally, just
  exposed as a reusable per-point mask instead of a full L/R-resolved label.
- `motion_is_body_for_frame_idx(frame_idx, xyz_kept, dataset_dir)`:
  disk-backed adapter for the real production dataset, delegates to
  `density.py::compute_body_voxels_for_frame` (same function
  `motion/label.py::classify_body_candidate` already calls) -- returns
  `None` outside `density.valid_frame_range()` or if the window was
  truncated (missing T2 output for some window frame), matching the
  in-memory adapter's fallback contract.

Prototyped first in `simulate_gt`, per brief section 3's suggested order:
new function `segment.segment_frame_kmeans_motion_fusion` added to
`postprocessing/kinematics/simulate_gt/segment.py` -- identical to
`segment_frame_kmeans_v2` except it calls `fusion.motion_is_body_for_window`
+ `fusion.motion_body_veto` right after KMeans's raw cluster labels are
mapped to body/wing_A/wing_B semantics, before the existing `_wing_merged`
check / `_forced_wing_split` / `_fix_wing_connectivity` machinery (unchanged)
runs on the (possibly enlarged) wing pool. `segment_frame_kmeans_v2` itself
is untouched. Also added a `segment_fn` parameter to `evaluate.py
::evaluate_frame` (default unchanged: `segment.segment_frame_kmeans_v2`) so
the fusion function can be scored through the same T3/T4-only/GT
error-report path other functions use, via
`functools.partial(segment.segment_frame_kmeans_motion_fusion,
window_xyz_by_frame=..., center_frame_idx=...)`.

Single-variable scope, as instructed: only the motion-veto mechanism was
implemented this round. No separate continuity/smoothing layer beyond
motion's own windowed voxel evidence was added -- section 4.4's "validate
one layer before adding another" was followed; given the 4.5 validation
below already surfaces one real, non-trivial edge case (frame 36, see
below) worth understanding fully before adding a second mechanism, no
second layer was attempted this session (see "what's left").

## Phase 4.5 — synthetic validation (2026-08-19)

Script: `/tmp/.../scratchpad/phase45_validate.py` -- runs
`evaluate_frame` (the real T3+T4 pipeline, not just segmentation) over the
IDENTICAL `scenario_step2_flapping(100)` frame set for both
`segment_frame_kmeans_v2` (baseline) and `segment_frame_kmeans_motion_fusion`
(fusion), same script, same run, so both conditions see byte-identical input
frames. Full CSVs: `phase45_baseline.csv`, `phase45_fusion.csv`.

**Overall (100 frames):**

| | seg_accuracy mean | t3_roll_deg mean | t3_roll_deg max | status_t3 ok |
|---|---|---|---|---|
| BASELINE | 0.8150 | 13.8971 | 69.3518 | 87/100 |
| FUSION | 0.8252 | 14.3815 | 79.2686 | 88/100 |

**Inside motion window [36,63] (28 frames):**

| | seg_accuracy mean | t3_roll_deg mean | t3_roll_deg max |
|---|---|---|---|
| BASELINE | 0.8731 | 6.7809 | 23.1094 |
| FUSION | 0.9096 | 8.5110 | 79.2686 |

**Outside motion window (72 frames) -- must be a no-op by design:**

| | seg_accuracy mean | t3_roll_deg mean | t3_roll_deg max |
|---|---|---|---|
| BASELINE | 0.7923 | 16.6645 | 69.3518 |
| FUSION | 0.7923 | 16.6645 | 69.3518 |

Confirmed byte-identical outside the window (as designed: `is_body_motion is
None` there, `motion_body_veto` is a strict no-op) -- this is the expected
"degrades gracefully to plain kmeans_v2" behavior brief section 4.5 asks to
verify, not assumed.

**Confusion matrices (sum over 100 frames):**
```
BASELINE                          FUSION
pred     body  wing_L  wing_R     pred     body  wing_L  wing_R
gt                                 gt
body    28080    2533    2087     body    28087    2473    2140
wing_L   2711   12094    1295     wing_L   2367   12497    1236
wing_R   2651     473   11576     wing_R   2571     314   11815
```
Wing recall improves in both wings (wing_L: 12094->12497, wing_R:
11576->11815); wing_L<->wing_R cross-confusion drops (473+170->314ish);
body recall essentially flat (28080->28087). Consistent, broad-based
improvement, not a fluke of one or two frames.

**Seg_accuracy on the 3 catastrophic frames inside the window** (from 4.2):
36: 0.2016->0.6142, 39: 0.3764->0.6268, 40: 0.3543->0.4315 -- all three
improve substantially, as expected (these are exactly the frames whose
whole-cluster body/wing confusion the veto directly targets).

**Honest complication -- one real single-frame regression, investigated, not
hidden** (script `investigate_frame36.py`): the aggregate `t3_roll_deg`
*mean* got *worse* both overall (13.90->14.38) and in-window (6.78->8.51),
and *max* got worse (69.35->79.27 overall, 23.11->79.27 in-window) -- driven
entirely by frame 36. Diagnosed: baseline's frame 36 has a **complete L/R
swap** (its "wing_L" predicted cluster contains ZERO true wing_L points --
138/179 are actually wing_R, 41/179 are body; its "wing_R" cluster is
167/179 actually body) on top of the already-known whole-cluster confusion,
yet baseline's roll estimate for that frame (1.94 deg) happens to be small
-- a coincidental near-cancellation from a fully-swapped-but-internally-
coherent hinge pair, not evidence the baseline handled this frame correctly
(baseline's own `status_t3` already independently flags frame 36 as a
`wing_R:fit_plane RANSAC` failure). Fusion's segmentation on the same frame
is objectively much better by every other measure (seg_accuracy 0.20->0.61,
correct L/R orientation restored -- 134/245 wing_L correct, 137/244 wing_R
correct, RANSAC converges, `status_t3="ok"`), but its roll estimate is much
worse (79.27 deg) -- the improved-but-still-imperfect (~41-44% body
contamination) wing point sets apparently feed `compute_wing_hinge_far_cc`
a worse hinge estimate than baseline's accidental swap-cancellation, for
reasons not further root-caused this session (plausible: far+CC is
sensitive to the exact contamination pattern, not just contamination
*amount* -- not confirmed).

**Recomputed excluding frame 36** (`exclude_f36.py`) to see the picture
without this one outlier: in-window (27 frames) roll mean 6.96->5.89 (15.4%
better), max 23.11->17.14 (26% better), seg_accuracy 0.898->0.921; overall
(99 frames) roll mean 14.02->13.73 (2.1% better), max unchanged (69.35,
the worst remaining frame is outside the window and untouched by fusion).

**Verdict**: net improvement on every metric except the full-100/full-28
roll mean/max, and that exception is a single, fully-diagnosed frame where
baseline's "good" number was itself an artifact of a badly broken
segmentation, not a case where fusion broke something baseline had right.
Judged as a genuine, validated net improvement, not a wash -- but flagging
the frame-36 pattern explicitly (rather than only reporting the
excluding-outlier numbers) per this repo's "report anomalies, don't hide
them" convention. A stricter reviewer could reasonably read the raw
full-100 mean/max numbers alone and call this inconclusive; both readings
are given here rather than picking the more flattering one.

## Phase 4.6 — production port + real-data sanity check (2026-08-19)

**Worktree setup note**: also had to copy `postprocessing/kinematics/
correct_body_axis/diag/` (gitignored, not tracked -- confirmed via `git
ls-files`, matches the same "untracked dir missing from a fresh worktree"
pattern as the 4 paths this task's brief already flagged) from the main
checkout to unblock `pytest postprocessing/kinematics/tests/` (was
otherwise failing to even collect `test_sequence_axis.py`). Also copied a
31MB CSV-only subset (T1 `gaussian_features_f####.csv` + T2 `_marked.csv`,
NOT the multi-GB splat.ply/checkpoint model files or raw camera data) of the
real `outputs/ctrl_009_002_8groups_100frames/G2b_G9` dataset (100 real
frames, f0000-f0099, already has kmeans-based T3 history) into this
worktree's own `outputs/` -- this script only ever writes into that
worktree-local copy, never touches the main checkout.

**Production port**: `postprocessing/labeling/labeling.py::process_frame`
now calls `fusion.motion_is_body_for_frame_idx` +
`fusion.motion_body_veto` right after KMeans's raw cluster labels are
mapped to semantics, before the wing-merge check -- as the **new default
behavior**, no opt-in flag (this round's red line is relaxed, see brief
section 3). One real bug caught and fixed during the port: `labeling.py
::fix_wing_connectivity`'s OLD signature took `(labels, mapping)` and
**recomputed** `semantic` from them internally, which would have silently
discarded the veto's effect on every frame that didn't also trigger
`is_wing_merged` (the common case) -- changed its signature to take
`semantic` directly (single call site, `process_frame`; old callers'
behavior is unchanged since `mapping[c] for c in labels]` is exactly what
was passed in before). Also added `n_motion_veto`/`motion_available` to
`process_frame`'s return dict and `build_summary_df`'s columns for
traceability.

**Real-data sanity check, first pass (before the safety cap below)**:
script `/tmp/.../scratchpad/real_data_sanity.py` -- runs `labeling.
process_frame` directly (not `run_batch`, which also calls
`plot_labeled_reprojection`, needing `data/ctrl_009_002/<frame>/
transforms.json` raw camera data not copied into this worktree -- out of
scope for a segmentation-only check) over `f0010`-`f0089` (80 real frames,
spanning well inside and outside the `[36,63]` motion-valid window for this
100-frame dataset), once with the real (fusion-enabled) code, once with
`labeling.motion_is_body_for_frame_idx` monkeypatched to always return
`None` (reproduces pre-fusion behavior through the *exact* same code
otherwise -- cleaner than diffing against git history, isolates exactly one
variable). Fed both label sets through `pipeline.run_dataset` and compared
`roll` via `diagnostics.delta_report` (no ground truth on real data --
"did it get worse," not a correctness check).

Confirmed `motion_available=28/80` (exactly matches `[36,63]`'s 28 frames)
and `0/80` for the monkeypatched run -- the window-gating logic works
correctly on real T2 data, including correctly rejecting frames that
`density.valid_frame_range()`'s **hardcoded** `range(36, 604)` (baked in for
the 640-frame `ratio3_sh0_dense` dataset motion's own module was developed
against) would otherwise wrongly treat as "in range" on this *shorter*
100-frame dataset -- `fusion.motion_is_body_for_frame_idx`'s
`n_frames_used < 2*HALF_WINDOW+1` truncated-window guard (added during
4.4's implementation, see fusion.py) catches this and returns `None`, which
is why `motion_available` lands at exactly 28, not the naive range's much
larger count. **This is a real, worth-flagging cross-dataset gotcha**:
`density.valid_frame_range()` should not be trusted as "is this frame
actually motion-eligible" on any dataset other than the one it was
hardcoded for; only the truncated-window check in `fusion.py` makes it safe
here.

**First-pass result -- an honest regression, not swept aside**: `roll`
jump count (`diagnostics.delta_report`, `>5x median` threshold) went **9 ->
12** (BEFORE frame positions -> AFTER frame positions, converted to actual
`frame_id`: BEFORE jumps at 64,65,71,72,81,83,85,87,89; AFTER jumps at
52,53,62,63,65,71,72,81,83,85,87,89 -- 3 new jumps at 52/53/63, all inside
or at the edge of the motion window). `p95` delta also got slightly worse
(91.5->94.4 deg). Traced the two worst individual swings to `f0052`
(roll -37.0 deg BEFORE -> -131.1 AFTER, ~168 deg swing) and `f0063`
(-73.8 -> +112.5, ~186 deg swing) -- both `is_wing_merged=False` in both
conditions (not the already-known merge-fallback mechanism), so a
*different* failure pattern than phase 4.5's frame-36 case.

**Root cause traced to veto magnitude** (script
`/tmp/.../scratchpad/veto_fraction_dist.py`): computed veto fraction
(vetoed points / all kept points) per frame on both datasets. On synthetic
data this fraction is small on ordinary frames (median 1.4%, p90 7.5%) and
only large (18-21%) on the 3 already-diagnosed catastrophic frames
(36/39/40) -- where a large veto is exactly the validated, beneficial
correction. On real G2b_G9 data the fraction runs systematically higher
overall (median 11%, p90 19%, consistent with `density.py`'s own documented
caveats: no rigid-motion alignment, body jitter, a voxel-count threshold its
own docstring admits "isn't a clean bimodal split") -- and `f0052`
(40.5%) / `f0055` (45.9%) sit far outside even the synthetic
catastrophic-frame range (max 20.6%), and are exactly the two frames behind
the largest, least-plausible swings above.

**Fix: added `MAX_VETO_FRAC=0.30` safety cap to `fusion.motion_body_veto`**
(same mechanism, one more firing condition -- not a second mechanism): if
the fraction of all kept points that would be vetoed exceeds 30%, skip the
veto for that frame entirely (falls back to plain kmeans_v2), treating that
large a disagreement as motion's own signal being unreliable for that
specific frame rather than a targeted correction. 0.30 sits with margin
above the synthetic ceiling (0.21, so the validated catastrophic-frame
fixes at 36/39/40 are untouched -- reran phase 4.5 with the cap in place,
numbers are byte-identical to the pre-cap run) and below the problematic
real frames (0.40+).

**Re-ran the real-data check with the cap**: `n_motion_veto>0` frames
26/28 (down from 28/28 -- f0052/f0055 now correctly skipped), total points
vetoed 939 (down from 1225). Jump count **9 -> 10** (down from the
uncapped 9->12), converted to frame_ids: BEFORE jumps unchanged
(64,65,71,72,81,83,85,87,89); AFTER jumps now at
62,63,65,71,72,81,83,85,87,89 -- only 2 new jumps (62, 63, both at the
window's tail edge) vs the old 3, and 1 old jump (64) no longer flagged.
`p95` 92.1 (down from 94.4, much closer to BEFORE's 91.5). `max` 162.8 --
actually *better* than BEFORE's 173.0, in both the capped and uncapped
runs.

**Honest verdict on the real-data check**: meaningfully improved by the
safety cap (jump-count gap shrank from +3 to +1, p95 nearly matches, max
improved), but **not a clean pass on the strict jump-count metric** (10 vs
9, still +1) -- the residual disagreement is now concentrated at exactly 2
frames (62, 63) right at the motion window's tail boundary, the same
qualitative "boundary discontinuity" pattern as phase 4.5's frame-36 case
on synthetic data, not further root-caused this session. Reported plainly
per this repo's "report anomalies" convention rather than calling this an
unambiguous pass. Given (a) the synthetic ground-truth validation is a
clear, mechanistically-understood net win, (b) the real-data check's
residual gap is small (1 frame net) and geographically concentrated at a
known-fragile boundary rather than diffuse, and (c) `test_s2.py` and the
full `postprocessing/kinematics/tests/` suite (77/77) pass unmodified, the
production port was **kept as the new default** rather than reverted --
but this is a judgment call given imperfect evidence, not a slam-dunk, and
is flagged as such rather than overclaimed.

**Regression tests**: `pytest postprocessing/kinematics/tests/test_s2.py`
9/9 passed, unmodified tolerances (roll 6.0 deg, yaw/pitch 3.0 deg) --
expected, since none of this session's changes touch `body_frame.py`/
`mock.py`. Full suite `postprocessing/kinematics/tests/` 77/77 passed.

**Side discovery, not acted on (out of scope, flagged for the record)**:
`postprocessing/calc_kinematics.py::run_cleaning_and_labeling`'s own T3
step, for a *from-scratch* dataset (no `_labeled.csv` yet), currently calls
`postprocessing.labeling.motion.label.run_batch` (motion-only), **not**
`postprocessing.labeling.labeling.run_batch` (the kmeans+fusion path this
session improved) -- contradicting brief section 1.3's framing of
`labeling.py::process_frame` as "the real production entry point." Given
this session's own numbers (motion alone loses to kmeans_v2 on every
synthetic metric, see phase 4.1/4.3), `calc_kinematics.py`'s wiring choice
looks questionable, but changing `calc_kinematics.py` is not one of the
modules brief section 0/1.3 named as this task's port target
(`kmeans_split.py`/`labeling.py`/`motion/`) and wasn't touched -- noted here
for whoever picks this up next, not fixed.

---

## Final structured summary (brief section 7, all 7 points)

**1. Reproduction results (section 1, phase 4.1).** Everything in the
brief's section 1 reproduced exactly, no drift: `seg_accuracy` mean=0.8150,
`t3_roll_deg` mean=13.897/max=69.35, `t4only_roll_deg` mean=2.43 (100-frame
`scenario_step2_flapping`, `run_step2.py` unmodified). Aligned kmeans_v2 vs
motion on `[36,63]`: kmeans_v2 0.8731 vs motion 0.8095 -- kmeans_v2 still
wins, motion is not a standalone replacement. One correction to the brief's
own framing (not the numbers): motion's "precision=1.0 on body" claim
(section 1.2) is not quite right -- the actually-1.0 property is
**recall(body)=1.0 / false-negative-rate(body)=0**, not precision(pred=body)
(measured fresh at 0.7585). This distinction directly shaped the fusion
rule (see point 3).

**2. Catastrophic-frame mechanism (section 4.2).** All 14/14 of part1's
catastrophic frames (2,3,5,20,35,36,39,40,75,82,83,87,98,99) trigger
`is_wing_merged` (necessary, not quite sufficient -- 2 more frames, 1 and 4,
also crash but weren't caught by part1's RANSAC-failure proxy). Root cause
traced one level deeper: on these frames the KMeans body cluster
under-captures its own high-confidence seeds (43.7% vs 90.2% seed capture
on non-merged frames), leaving the pooled wing cluster ~49% actual body
points (vs 13% normally, r=0.867, p=2e-31) -- an extreme, whole-cluster
version of the *same* body<->wing halo confusion part1 found near the
hinge, not a distinct mechanism. This body-heavy "wing" mass bridges the
true wing_L/wing_R spatial gap, triggering the crude
`forced_wing_split`-median-projection fallback, which then cuts an
already-half-body mass roughly in half, producing near-random accuracy.
*Not* resolved: why the KMeans body cluster under-forms on these specific
frames in the first place (wingbeat phase and true wing-gap correlations
were both checked and came back weak/inconclusive).

**3. Joint error analysis + derived fusion rule (section 4.3).** On
`[36,63]` (17780 points): motion-right/kmeans-wrong 7.59%, kmeans-right/
motion-wrong 13.95%. Two candidate one-directional override rules tested
directly on the data (not predesigned): overriding kmeans's `wing` verdict
with motion's `body` verdict is net **-1340** (harmful, breaks 3x what it
fixes); overriding kmeans's `body` verdict with motion's non-body verdict
is net **+510/-0** (strictly beneficial in this window). Fusion rule = the
second one only, one direction: kmeans's body cluster gets vetoed
(reassigned to nearest wing cluster) wherever motion's windowed
voxel-density evidence disagrees.

**4. Fusion design + implementation.** New shared module
`postprocessing/labeling/fusion.py`: `motion_body_veto` (the veto
mechanism, includes the `MAX_VETO_FRAC=0.30` magnitude safety cap added
during 4.6) + two adapters (`motion_is_body_for_window` for `simulate_gt`,
`motion_is_body_for_frame_idx` for the real disk-backed dataset), both
delegating actual voxel computation to `motion/density.py`'s existing
primitives -- no reimplementation. **Continuity mechanism, concretely**:
this *is* a voxel/spatial-level continuity mechanism per brief section 2's
first category -- motion's own `HALF_WINDOW=36` cross-frame voxel evidence
*is* the continuity signal being fused in; no second, derived-quantity
(e.g. roll-sequence) continuity layer was added this session (see "what's
left," point 7). Prototyped first in `simulate_gt/segment.py` (new function
`segment_frame_kmeans_motion_fusion`, `segment_frame_kmeans_v2` itself
untouched) per the brief's suggested order, then ported into
`postprocessing/labeling/labeling.py::process_frame` as the new
unconditional default (this round's red line is relaxed). One real bug
caught+fixed during the port: `labeling.py::fix_wing_connectivity`'s old
`(labels, mapping)` signature recomputed `semantic` internally, silently
discarding the veto on any non-merged frame -- fixed to take `semantic`
directly.

**5. Before/after numbers (section 4.5, synthetic; identical 100-frame
`scenario_step2_flapping` set both sides).**

| | seg_acc (all 100) | seg_acc (in-window, 28) | seg_acc (out-window, 72) | roll mean (all) | roll mean (in-window) | roll max (all) |
|---|---|---|---|---|---|---|
| BASELINE | 0.8150 | 0.8731 | 0.7923 | 13.897 | 6.781 | 69.35 |
| FUSION | 0.8252 | 0.9096 | 0.7923 (identical, by design) | 14.382 | 8.511 | 79.27 |

Confusion matrices improve on both wings (see phase 4.5 above) and the 3
in-window catastrophic frames (36/39/40) all improve substantially
(0.20->0.61, 0.38->0.63, 0.35->0.43). The roll mean/max regressions are
driven entirely by one diagnosed outlier (frame 36: baseline's low roll
number was a coincidental artifact of a *complete* L/R swap on an
already-RANSAC-failing frame, not evidence baseline handled it correctly);
excluding it, in-window roll mean improves 6.96->5.89 (15%) and max
23.11->17.14 (26%). Out-of-window frames are byte-identical between
baseline and fusion, confirming the graceful-degradation design works.

**6. Production port status + tests + real-data check (section 4.6).**
Ported into `labeling.py::process_frame` as the new default (not opt-in).
`test_s2.py` 9/9 passed at unmodified tolerances; full
`postprocessing/kinematics/tests/` suite 77/77 passed. Real-data check on
80 real frames (`f0010`-`f0089`, G2b_G9 dataset) via a monkeypatch-isolated
before/after comparison: first pass showed a real regression
(`diagnostics.py` roll jump count 9->12, two individual frame-to-frame
swings up to ~186 deg) traced to two real frames (`f0052`/`f0055`) where
motion disagreed with kmeans on 40-46% of the frame's points -- far outside
anything seen on synthetic data (max 21%, and that 21% was the *validated
beneficial* catastrophic-frame case). Added the `MAX_VETO_FRAC=0.30`
magnitude safety cap in response (same mechanism, one more firing
condition), re-validated synthetic numbers are byte-identical post-cap, and
the real-data jump count improved to 9->10 (residual +1, concentrated at 2
frames at the window's tail boundary, same qualitative pattern as the
frame-36 synthetic outlier). **Not a clean, unambiguous real-data pass** --
reported honestly as a judgment call (kept as default given the synthetic
win is clear and mechanistically understood, and the real-data residual is
small and geographically concentrated at a known-fragile boundary rather
than diffuse), not oversold as a guaranteed non-regression.

**7. What's left / next steps, in priority order:**

1. **Frame-36-type boundary discontinuities** (synthetic frame 36, real
   frames 62/63): a genuinely-improved segmentation can still produce a
   *worse* single-frame roll estimate than a coincidentally-lucky broken
   one, right at/near the motion window's tail edge. Not root-caused past
   "the far+CC hinge algorithm is sensitive to the exact contamination
   pattern of the wing point set it's handed, not just the amount." This is
   the most concrete remaining risk and the natural next diagnostic target.
2. **A derived-quantity continuity layer for roll** (brief section 2's
   second category, explicitly suggested but not attempted this session,
   single-variable-first discipline per section 4.4): something in the
   spirit of `correct_body_axis/sequence_axis.py`'s anchor+correction
   pattern but new, for roll/y_body specifically -- likely the right tool
   for smoothing over exactly the frame-36/62/63-style single-frame
   discontinuities above, and for extending help to catastrophic frames
   outside motion's window (11/14 of them, see point 2) where the
   voxel-level veto mechanism has no signal at all by construction.
3. **Why does the KMeans body cluster under-form on catastrophic frames in
   the first place** (point 2's open question) -- not resolved, no clean
   correlate found this session (wingbeat phase, true wing gap both
   checked, both weak).
4. **`MAX_VETO_FRAC=0.30` is a reasonable, evidence-based but *not
   independently re-validated* threshold** -- picked to sit between the
   synthetic ceiling (0.21) and the real-data outliers (0.40+) with margin
   on both sides, not swept/optimized. A follow-up could sweep it on a
   larger real-data sample once available.
5. **`calc_kinematics.py`'s T3 default still calls motion alone**, not the
   now-improved `labeling.py::process_frame` -- flagged, not fixed (out of
   this task's named scope).
6. **`local_density`'s mock.py/gaussian_features.py formula mismatch**
   (inherited from part1, still unfixed, still blocks trusting that
   feature in `simulate_gt`) -- unrelated to this round's fusion work,
   noted for completeness since it's still open.
