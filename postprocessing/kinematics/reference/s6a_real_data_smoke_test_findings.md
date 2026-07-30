# S6a — T4 real-data interface smoke test findings

Dataset: `outputs/ctrl_009_002_8groups_100frames/G2b_G9` (100 frames, f0000-f0099)

## 1. File discovery / layout

Real per-frame directory layout matches the documented `f<NNNN>/splatfacto-checkpoint/<ts>/` pattern
exactly (one timestamp subdir per frame in this dataset). But **each frame directory contains three
CSVs**, not one:

```
gaussian_features_f0000.csv           # T1 raw features (21 cols, no if_keep/part_label)
gaussian_features_f0000_marked.csv    # + T2 if_keep (22 cols, NO part_label)
gaussian_features_f0000_labeled.csv   # + T3 part_label, confidence (24 cols) <- the real T3 output
```

Confirmed in `postprocessing/labeling/labeling.py:261-262`: T3 reads `_marked.csv` and writes a
**sibling** `_labeled.csv` ("不改动 _marked.csv" — marked.csv is left untouched). So `_marked.csv`
is T2's output, not T3's, and never gains `part_label`.

`pipeline.py`'s glob (`f*/splatfacto-checkpoint/*/*_marked.csv`) was written against the contract
doc's description ("T2 output adds `if_keep`, suffix `_marked.csv`") which predates T3 landing as a
separate labeling stage with its own output file. The glob technically "matches" (100/100
`_marked.csv` files found) but matches the **wrong file** — one stage behind the real T3 output.

All 100 frames have exactly one `_marked.csv` and one `_labeled.csv`; no duplicate-timestamp
ambiguity in this dataset.

## 2. Single-file inspection (`f0000/.../gaussian_features_f0000_labeled.csv`)

- Shape: 400 rows x 24 columns.
- Columns: exactly the 21 `calc_kinematics.md` §1 columns + `if_keep` + `part_label` + one **extra**
  column, `confidence` (not in the documented contract).
- Dtypes: all numeric columns float64, `color_oob`/`if_keep` bool, `part_label` object (str),
  `confidence` object (str, values `{"high", "low"}`).
- `part_label` values/counts (this frame): `body=222, wing_L=89, wing_R=89` — exactly the
  `{body, wing_L, wing_R}` vocabulary from `io_schema.PART_LABELS`, no typos/case variants.
- `if_keep`: present, proper bool dtype.
- NaNs: zero, in any column.
- Coordinate ranges (this frame, all points): x/y/z spans ~2.8-4.7 mm. Per-part bbox diagonals:
  body ~3.3mm, wing_L ~2.2mm, wing_R ~2.0mm — same order of magnitude as the documented "wing
  length ~2.5-3mm", values stored in meters as required (not mm/cm off-by-1000/10).

Scanned across all 100 `_labeled.csv` files:
- `part_label` vocabulary is exactly `{body, wing_L, wing_R}` everywhere, no NaNs.
- `confidence` vocabulary is exactly `{high, low}` everywhere.
- No frame has fewer than 10 `if_keep`-surviving points in any of body/wing_L/wing_R (pipeline's
  `min_points=10` guard would not reject any real frame on point-count grounds alone).

## 3. `run_dataset` results

**Before fix** (default `frame_glob` = `*_marked.csv`): 100/100 frames discovered, **100/100
`status = "load:missing mandatory column(s) ['part_label']; required = ['x', 'y', 'z',
'part_label']"`**. No hard crash — the per-frame try/except in `run_dataset` caught it as designed —
but the entire dataset produced NaN rows because it was loading T2's file, which structurally cannot
have `part_label`.

**After fix** (`frame_glob = "f*/splatfacto-checkpoint/*/*_labeled.csv"`): **100/100
`status = "ok"`**. First rows show finite yaw/pitch/roll/phi/theta/eta/span/chord_conf values (not
validated for accuracy — out of scope for this task).

## 4. Tagged issues

| # | Issue | Tag | Notes |
|---|---|---|---|
| 1 | `pipeline.py` glob targets `_marked.csv`, but T3's real output file is `_labeled.csv` (a sibling T3 writes alongside, not an in-place amendment of `_marked.csv`) | **(b) T4-side** | Fixed: `frame_glob` is now a `PipelineConfig` field (default unchanged, so mock tests are unaffected); pass `frame_glob="f*/splatfacto-checkpoint/*/*_labeled.csv"` for real data. |
| 2 | `calc_kinematics.md` §1 described only `_marked.csv` (T2) as the T4 input file and didn't mention `_labeled.csv` (T3) at all | **(a) T3-side / doc** — **fixed** | Doc predated the two-file T2/T3 split landing. §1 now spells out the three-file-per-frame layout (T1 raw -> T2 `_marked.csv` adds `if_keep` -> T3 `_labeled.csv` adds `part_label`+`confidence`, written as a sibling of `_marked.csv`) and states explicitly that T4 reads `_labeled.csv`. |
| 3 | Real `_labeled.csv` has an extra `confidence` column (`{high, low}`) not in the documented §1 schema | **(c) benign** | `io_schema.load_frame` only checks mandatory columns and doesn't reject extras; `get_part`/`get_part_columns` select columns by name so the extra column is silently ignored. No action needed. If per-frame label confidence should eventually gate anything in T4, that's a future feature request, not a format bug. |
| 4 | Wing/body bbox sizes run slightly under the doc's "~2.5-3mm" wing-length note (wings ~2.0-2.2mm bbox diagonal here) | **(c) benign** | Same order of magnitude, correct units (meters); bbox diagonal isn't span length anyway (span is root-to-tip along the PCA axis, not the full bbox diagonal). Not a schema/format issue — no action. |

## 5. Remaining T3-side items (blocked, not touched)

- None. Item #2 above (the only T3/doc-side item) was fixed directly in `calc_kinematics.md` §1.

## 6. Conclusion

**The T3 -> T4 interface is now format-clean end-to-end** for this 100-frame real dataset: with the
`frame_glob` fix, `run_dataset` produces `status = "ok"` for all 100 frames, no crashes, no missing
columns, no schema violations found. Angle/chord **accuracy** was explicitly out of scope for this
task and was not evaluated.

Before/after summary:
- Before: 100/100 `status = "load:missing mandatory column(s) ['part_label']..."`
- After: 100/100 `status = "ok"`

Full existing mock test suite (69 tests across `test_s0`-`test_s5`, all standalone-runnable) plus
the updated real-dataset smoke test in `test_s5.py` all pass.
