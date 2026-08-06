"""Steps 5-7: mechanism attribution for eta's ~180 deg wrap crossings --
does the frame-to-frame `le_dir` flip, does the `chord.py::_oriented_chord_axis`
sign call flip, or does neither explain a given big jump?

Continues `correct_wing_pitch/`'s diagnose-only, no-unwrap/no-smoothing
constraints (see `report.md`'s header and this directory's other scripts).
Reuses, rather than redefines: `real_data_validation.py`'s `_assoc_abs_delta`
alignment convention, `BIG_JUMP_DEG` threshold, `_discover_frames`;
`diagnostics.py::circular_delta_deg` for all eta deltas; `chord.py`'s own
`_bin_chords_core`/`_aggregate_chords`/`_eta` (unmodified) for the chord/eta
math *downstream* of the axis sign call; `le_repro.py::estimate_leading_edge_diag`
(already verified bit-identical to `wing_angles.estimate_leading_edge`, see
`00_consistency_check.md`) for the per-frame leading-edge fit.

The one piece of production logic under test here --
`chord.py::_oriented_chord_axis` -- is deliberately *not* imported; see
`_oriented_chord_axis_diag` below, a line-for-line reimplementation
instrumented to expose `raw_projection`/`axis_margin`. Its correctness is
verified indirectly per the task spec: feeding its output axis through the
real (unmodified) `chord.py` binning/aggregation/eta functions and checking
the resulting eta matches `chord.estimate_chord`'s own eta bit-for-bit (a
flipped sign would show up as a ~180 deg eta discrepancy, not a subtle one).

Three per-frame-pair (side, transition) diagnostics are produced:
- Step 5: does `le_dir` itself flip between adjacent frames (cos_angle), and
  does that track the same transition's |Δeta| (from `circular_delta_deg`,
  reused, not `_assoc_abs_delta`'s frame-max reduction -- a flip is
  inherently a transition-level event).
- Step 6: `axis_margin` (how decisive `_oriented_chord_axis`'s sign call
  was) as a per-*frame* diagnostic, paired with |Δeta| via `_assoc_abs_delta`
  (the same convention `real_data_validation.py` used for `margin_count`/
  `curvature_diff`, so the three are directly comparable).
- Step 7: `out_ref_norm` (the orienting reference vector's magnitude), same
  per-frame convention as step 6.

Run: python -m postprocessing.kinematics.correct_wing_pitch.wrap_mechanism_diag
"""
from __future__ import annotations

import pickle
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
from scipy import stats

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import chord as ch  # noqa: E402
from postprocessing.kinematics import diagnostics as diag_mod  # noqa: E402
from postprocessing.kinematics import io_schema, pipeline  # noqa: E402
from postprocessing.kinematics.correct_wing_pitch.le_repro import estimate_leading_edge_diag  # noqa: E402
from postprocessing.kinematics.correct_wing_pitch.real_data_validation import (  # noqa: E402
    BIG_JUMP_DEG,
    FRAME_GLOB,
    REAL_DATASET_ROOT,
    _assoc_abs_delta,
    _discover_frames,
)

DIAG_DIR = Path(__file__).resolve().parent / "diag"
FPS = 16000.0
FLIP_COS_THRESHOLD = -0.5
"""cos_angle below this between two frames' vectors counts as an unambiguous
flip -- the task's own suggested "more discriminating" threshold (plain
`< 0` is also reported as a looser secondary count)."""
_EPS = 1e-12
_SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}


# ---------------------------------------------------------------------------
# Step 6: reimplementation (not import) of chord.py::_oriented_chord_axis
# ---------------------------------------------------------------------------


def _oriented_chord_axis_diag(
    wing_xyz: np.ndarray, plane_normal: np.ndarray, le_dir: np.ndarray, le_inlier_mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Line-for-line mirror of `chord.py::_oriented_chord_axis`'s sign logic,
    additionally returning `raw_projection` (the LE-side mean's projection
    onto the axis *before* the flip-correcting sign check) and `axis_margin`
    (`|raw_projection| / norm(le_side_mean)`, in `[0, 1]`-ish: near 0 means
    the sign call was decided by a nearly-zero-margin projection, near 1
    means it was decisive).
    """
    axis = np.cross(plane_normal, le_dir)
    axis = axis / np.linalg.norm(axis)

    centroid = wing_xyz.mean(axis=0)
    le_side_mean = wing_xyz[le_inlier_mask].mean(axis=0) - centroid
    le_side_norm = float(np.linalg.norm(le_side_mean))
    raw_projection = float(np.dot(le_side_mean, axis))
    axis_margin = abs(raw_projection) / le_side_norm if le_side_norm > _EPS else float("nan")

    if raw_projection > 0.0:
        axis = -axis
    return axis, raw_projection, axis_margin


# ---------------------------------------------------------------------------
# Shared: dominant-neighbor lookup (mirrors _assoc_abs_delta's own max-of-
# neighbors choice, but returns *which* transition it picked instead of the
# value, so a frame's associated jump can be traced to one (frame_from,
# frame_to) transition)
# ---------------------------------------------------------------------------


def _assoc_dominant_transition_idx(eta: np.ndarray) -> np.ndarray:
    n = len(eta)
    cd = np.abs(diag_mod.circular_delta_deg(eta))  # length n-1; cd[t] = transition frame[t]->frame[t+1]
    idx = np.empty(n, dtype=int)
    if n == 1:
        return np.array([-1])
    for i in range(n):
        if i == 0:
            idx[i] = 0
        elif i == n - 1:
            idx[i] = n - 2
        else:
            idx[i] = (i - 1) if cd[i - 1] >= cd[i] else i
    return idx


# ---------------------------------------------------------------------------
# Step 5: le_dir frame-to-frame continuity
# ---------------------------------------------------------------------------


def _run_pipeline_with_debug() -> tuple[pd.DataFrame, dict]:
    debug_pkl = DIAG_DIR / f"kinematics_{REAL_DATASET_ROOT.name}_debug.pkl"
    csv_out = DIAG_DIR / f"kinematics_{REAL_DATASET_ROOT.name}.csv"
    if debug_pkl.exists() and csv_out.exists():
        print(f"reusing existing debug pickle: {debug_pkl}")
        out_df = pd.read_csv(csv_out)
        with open(debug_pkl, "rb") as f:
            debug_by_frame = pickle.load(f)
        return out_df, debug_by_frame

    print(f"running pipeline with write_debug=True on {REAL_DATASET_ROOT} ...")
    config = pipeline.PipelineConfig(min_points=10, output_dir=DIAG_DIR, write_debug=True, frame_glob=FRAME_GLOB)
    out_df = pipeline.run_dataset(REAL_DATASET_ROOT, config)
    with open(debug_pkl, "rb") as f:
        debug_by_frame = pickle.load(f)
    return out_df, debug_by_frame


def _build_le_dir_transitions(ok: pd.DataFrame, debug_by_frame: dict) -> pd.DataFrame:
    rows = []
    for suffix, side in (("L", "wing_L"), ("R", "wing_R")):
        sub = ok[["frame_id", f"eta_{suffix}"]].reset_index(drop=True)
        frame_ids = sub["frame_id"].to_numpy()
        eta = sub[f"eta_{suffix}"].to_numpy()
        cd = diag_mod.circular_delta_deg(eta)  # length n-1
        for i in range(len(frame_ids) - 1):
            f_from, f_to = int(frame_ids[i]), int(frame_ids[i + 1])
            d_from = debug_by_frame.get(f_from) or {}
            d_to = debug_by_frame.get(f_to) or {}
            le_from = d_from.get(f"le_dir_{suffix}")
            le_to = d_to.get(f"le_dir_{suffix}")
            if le_from is None or le_to is None:
                continue
            cos_le = float(np.dot(le_from, le_to) / (np.linalg.norm(le_from) * np.linalg.norm(le_to)))
            abs_delta = float(abs(cd[i]))
            rows.append(dict(
                side=side, suffix=suffix, transition_idx=i, frame_from=f_from, frame_to=f_to,
                cos_angle_le=cos_le, abs_delta_eta_transition=abs_delta,
                big_jump=abs_delta > BIG_JUMP_DEG,
                le_flip_strict=cos_le < FLIP_COS_THRESHOLD, le_flip_loose=cos_le < 0.0,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Steps 6-7: per-frame axis_margin / out_ref_norm (+ eta-match consistency)
# ---------------------------------------------------------------------------


def _collect_axis_margin_diagnostics(frames: list[tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for frame_id, csv_path in frames:
        try:
            df = io_schema.load_frame(csv_path)
            body_xyz = io_schema.get_part(df, "body")
            wingL_xyz = io_schema.get_part(df, "wing_L")
            wingR_xyz = io_schema.get_part(df, "wing_R")
            frame_obj = bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(frame_id=frame_id, side="wing_L", failed=True, fail_reason=str(e)))
            rows.append(dict(frame_id=frame_id, side="wing_R", failed=True, fail_reason=str(e)))
            continue

        for side, wing_xyz in (("wing_L", wingL_xyz), ("wing_R", wingR_xyz)):
            try:
                d = estimate_leading_edge_diag(wing_xyz, frame_obj, side, rng=0)
                axis_diag, raw_proj, axis_margin = _oriented_chord_axis_diag(
                    wing_xyz, d.plane_normal, d.le_dir, d.inlier_mask
                )
                per_bin_chords, _counts, _plan = ch._bin_chords_core(
                    wing_xyz, d.le_dir, axis_diag, n_bins=ch._N_SPAN_BINS, min_bin_points=ch._MIN_BIN_POINTS,
                )
                chord_diag = ch._aggregate_chords(per_bin_chords)
                eta_diag = ch._eta(chord_diag, d.le_dir, frame_obj.n_sp, _SIGN_LEFT[side])

                chord_real = ch.estimate_chord(wing_xyz, frame_obj, side)
                eta_real = float(chord_real.eta)
                eta_diff = float(abs(eta_diag - eta_real))

                wing_centroid = wing_xyz.mean(axis=0)
                out_ref_norm = float(np.linalg.norm(wing_centroid - np.asarray(frame_obj.body_cm, dtype=float)))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(frame_id=frame_id, side=side, failed=True, fail_reason=str(e)))
                continue

            rows.append(dict(
                frame_id=frame_id, side=side, failed=False,
                axis_margin=axis_margin, raw_projection=raw_proj,
                out_ref_norm=out_ref_norm,
                margin_count=d.margin_count, curvature_diff=d.curvature_diff,
                eta_diag=eta_diag, eta_real=eta_real, eta_diff=eta_diff,
                eta_match=bool(eta_diff == 0.0),
                chord_axis_x=float(axis_diag[0]), chord_axis_y=float(axis_diag[1]), chord_axis_z=float(axis_diag[2]),
            ))
    return pd.DataFrame(rows)


def _build_chord_axis_transitions(ok: pd.DataFrame, axis_df: pd.DataFrame) -> pd.DataFrame:
    axis_lookup = {
        (int(r.frame_id), r.side): np.array([r.chord_axis_x, r.chord_axis_y, r.chord_axis_z])
        for r in axis_df.itertuples()
        if not r.failed
    }
    rows = []
    for suffix, side in (("L", "wing_L"), ("R", "wing_R")):
        frame_ids = ok["frame_id"].to_numpy()
        for i in range(len(frame_ids) - 1):
            f_from, f_to = int(frame_ids[i]), int(frame_ids[i + 1])
            a_from = axis_lookup.get((f_from, side))
            a_to = axis_lookup.get((f_to, side))
            if a_from is None or a_to is None:
                continue
            cos_axis = float(np.dot(a_from, a_to) / (np.linalg.norm(a_from) * np.linalg.norm(a_to)))
            rows.append(dict(
                side=side, suffix=suffix, transition_idx=i, frame_from=f_from, frame_to=f_to,
                cos_angle_axis=cos_axis, axis_flip_strict=cos_axis < FLIP_COS_THRESHOLD,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for side, color in (("wing_L", "tab:blue"), ("wing_R", "tab:orange")):
        sub = df[df["side"] == side]
        ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.7, label=side, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("|Δeta| (deg)")
    ax.set_title(title)
    ax.axhline(BIG_JUMP_DEG, color="gray", lw=0.8, ls="--", label=f"{BIG_JUMP_DEG:.0f} deg")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_out_ref_boxplot(merged: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    groups = [
        ("big_jump_dominant", "eta big jump\n(dominant transition)"),
        ("dominant_le_flip", "le_dir flipped\n(dominant transition)"),
    ]
    for ax, (col, label) in zip(axes, groups):
        sub = merged.dropna(subset=[col, "out_ref_norm"])
        data = [sub.loc[~sub[col].astype(bool), "out_ref_norm"], sub.loc[sub[col].astype(bool), "out_ref_norm"]]
        ax.boxplot(data, labels=["False", "True"])
        ax.set_title(label)
        ax.set_ylabel("out_ref_norm (m)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    if not REAL_DATASET_ROOT.exists():
        print(f"ERROR: real dataset root not found: {REAL_DATASET_ROOT}")
        print("Cannot run mechanism diagnostics without it -- refusing to substitute fake data.")
        sys.exit(1)

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    out_df, debug_by_frame = _run_pipeline_with_debug()
    ok = out_df[out_df["status"] == "ok"].reset_index(drop=True)
    print(f"pipeline: {len(out_df)} frames total, {len(ok)} status=ok")

    frames = _discover_frames(REAL_DATASET_ROOT, FRAME_GLOB)
    print(f"discovered {len(frames)} frame CSVs via glob {FRAME_GLOB!r}")

    # --- Step 5: le_dir continuity ---
    le_trans = _build_le_dir_transitions(ok, debug_by_frame)
    le_trans.to_csv(DIAG_DIR / "05_le_dir_transitions.csv", index=False)
    print(f"\nStep 5: {len(le_trans)} le_dir transitions collected")
    _plot_scatter(le_trans, "cos_angle_le", "abs_delta_eta_transition",
                  "cos_angle(le_dir_t, le_dir_t+1)", "le_dir continuity vs |Δeta| (per transition)",
                  DIAG_DIR / "05_le_dir_cos_vs_delta_eta.png")

    big = le_trans["big_jump"]
    print(f"  big_jump (|Δeta|>{BIG_JUMP_DEG:.0f}): {int(big.sum())}/{len(le_trans)} transitions")
    print(f"  of those, le_flip_strict (cos<{FLIP_COS_THRESHOLD}): {int((big & le_trans['le_flip_strict']).sum())}")
    print(f"  of those, le_flip_loose (cos<0): {int((big & le_trans['le_flip_loose']).sum())}")

    table_le = pd.crosstab(le_trans["le_flip_strict"], le_trans["big_jump"])
    table_le = table_le.reindex(index=[True, False], columns=[False, True], fill_value=0)
    odds_le, p_le = stats.fisher_exact(table_le.to_numpy())
    print(f"  2x2 (le_flip_strict x big_jump) Fisher exact: odds_ratio={odds_le:.4f}, p={p_le:.4g}")
    print(table_le)

    # --- Steps 6-7: axis_margin, out_ref_norm ---
    axis_df = _collect_axis_margin_diagnostics(frames)
    axis_df.to_csv(DIAG_DIR / "06_axis_margin_frame_diagnostics.csv", index=False)
    valid_axis = axis_df[~axis_df["failed"].fillna(True).astype(bool)]
    n_eta_match = int(valid_axis["eta_match"].sum())
    max_eta_diff = float(valid_axis["eta_diff"].max()) if len(valid_axis) else float("nan")
    print(f"\nStep 6: {len(axis_df)} (frame,side) axis rows, {len(valid_axis)} valid")
    print(f"  eta-match consistency check: {n_eta_match}/{len(valid_axis)} exact matches "
          f"(eta_diff==0.0), max eta_diff={max_eta_diff:.3e} deg")

    axis_trans = _build_chord_axis_transitions(ok, axis_df)
    axis_trans.to_csv(DIAG_DIR / "06_chord_axis_transitions.csv", index=False)

    # --- per-frame merged table (step 6/7 unit of analysis) ---
    delta_rows = []
    for suffix, side in (("L", "wing_L"), ("R", "wing_R")):
        eta = ok[f"eta_{suffix}"].to_numpy()
        frame_ids = ok["frame_id"].to_numpy()
        abs_delta = _assoc_abs_delta(eta)
        dom_idx = _assoc_dominant_transition_idx(eta)
        delta_rows.append(pd.DataFrame(dict(
            frame_id=frame_ids, side=side, abs_delta_eta_assoc=abs_delta, dominant_transition_idx=dom_idx,
        )))
    delta_df = pd.concat(delta_rows, ignore_index=True)

    le_trans_idx = le_trans.set_index(["side", "transition_idx"])
    axis_trans_idx = axis_trans.set_index(["side", "transition_idx"])

    def _lookup(row, table, col):
        key = (row["side"], row["dominant_transition_idx"])
        return bool(table.loc[key, col]) if key in table.index else np.nan

    delta_df["dominant_le_flip"] = delta_df.apply(lambda r: _lookup(r, le_trans_idx, "le_flip_strict"), axis=1)
    delta_df["dominant_axis_flip"] = delta_df.apply(lambda r: _lookup(r, axis_trans_idx, "axis_flip_strict"), axis=1)
    delta_df["big_jump_dominant"] = delta_df["abs_delta_eta_assoc"] > BIG_JUMP_DEG

    merged = axis_df.merge(delta_df, on=["frame_id", "side"], how="left")
    merged.to_csv(DIAG_DIR / "06_07_merged_per_frame.csv", index=False)

    valid = merged[~merged["failed"].fillna(True).astype(bool) & merged["abs_delta_eta_assoc"].notna()].copy()
    print(f"\nmerged valid per-frame rows: {len(valid)}")

    _plot_scatter(valid, "axis_margin", "abs_delta_eta_assoc", "axis_margin",
                  "axis_margin vs |Δeta| (per frame, _assoc_abs_delta)",
                  DIAG_DIR / "06_axis_margin_vs_delta_eta.png")
    _plot_out_ref_boxplot(valid, DIAG_DIR / "07_out_ref_norm_boxplot.png")

    # --- correlations: axis_margin vs margin_count vs curvature_diff ---
    corr_axis = stats.spearmanr(valid["axis_margin"], valid["abs_delta_eta_assoc"], nan_policy="omit")
    corr_margin = stats.spearmanr(valid["margin_count"], valid["abs_delta_eta_assoc"], nan_policy="omit")
    corr_curv = stats.spearmanr(valid["curvature_diff"].abs(), valid["abs_delta_eta_assoc"], nan_policy="omit")
    print(f"\nSpearman r(axis_margin, |Δeta|)      = {corr_axis.correlation:.4f} (p={corr_axis.pvalue:.4g})")
    print(f"Spearman r(margin_count, |Δeta|)     = {corr_margin.correlation:.4f} (p={corr_margin.pvalue:.4g})")
    print(f"Spearman r(|curvature_diff|, |Δeta|) = {corr_curv.correlation:.4f} (p={corr_curv.pvalue:.4g})")

    # --- cross-validation: axis_margin in "big jump & le_dir NOT flipped" vs rest ---
    mask_no_le = valid["big_jump_dominant"] & (valid["dominant_le_flip"] == False)  # noqa: E712
    mask_rest = ~mask_no_le
    grp_no_le = valid.loc[mask_no_le, "axis_margin"].dropna()
    grp_rest = valid.loc[mask_rest, "axis_margin"].dropna()
    if len(grp_no_le) > 0 and len(grp_rest) > 0:
        mwu_axis = stats.mannwhitneyu(grp_no_le, grp_rest, alternative="less")
        print(f"\nMann-Whitney U axis_margin: big_jump&!le_flip (n={len(grp_no_le)}, median={grp_no_le.median():.4f}) "
              f"vs rest (n={len(grp_rest)}, median={grp_rest.median():.4f}): "
              f"U={mwu_axis.statistic:.1f}, p={mwu_axis.pvalue:.4g}")
    else:
        mwu_axis = None
        print("\nMann-Whitney U axis_margin: insufficient rows in one group")

    # --- out_ref_norm comparisons ---
    def _mwu_group(col: str, label: str):
        g1 = valid.loc[valid[col] == True, "out_ref_norm"].dropna()  # noqa: E712
        g0 = valid.loc[valid[col] == False, "out_ref_norm"].dropna()  # noqa: E712
        if len(g1) == 0 or len(g0) == 0:
            print(f"Mann-Whitney U out_ref_norm ({label}): insufficient rows")
            return None, g0, g1
        res = stats.mannwhitneyu(g1, g0, alternative="less")
        print(f"Mann-Whitney U out_ref_norm ({label}): True (n={len(g1)}, median={g1.median():.4e}) vs "
              f"False (n={len(g0)}, median={g0.median():.4e}): U={res.statistic:.1f}, p={res.pvalue:.4g}")
        return res, g0, g1

    print()
    mwu_outref_bigjump, outref_bigjump_false, outref_bigjump_true = _mwu_group("big_jump_dominant", "eta big jump")
    mwu_outref_leflip, outref_leflip_false, outref_leflip_true = _mwu_group("dominant_le_flip", "le_dir flipped")

    # --- three-way mechanism classification of big-jump transitions ---
    trans = le_trans.merge(
        axis_trans[["side", "transition_idx", "cos_angle_axis", "axis_flip_strict"]],
        on=["side", "transition_idx"], how="left",
    )
    trans["mechanism"] = np.select(
        [trans["le_flip_strict"], trans["axis_flip_strict"] == True],  # noqa: E712
        ["le_dir", "chord_axis"],
        default="unexplained",
    )
    trans.loc[trans["axis_flip_strict"].isna(), "mechanism"] = np.where(
        trans.loc[trans["axis_flip_strict"].isna(), "le_flip_strict"], "le_dir", "unknown_axis_missing"
    )
    trans.to_csv(DIAG_DIR / "05_06_transitions_merged.csv", index=False)

    big_trans = trans[trans["big_jump"]]
    mech_counts = big_trans["mechanism"].value_counts()
    mech_frac = big_trans["mechanism"].value_counts(normalize=True)
    print(f"\nThree-way mechanism classification among {len(big_trans)} big-jump transitions (pooled L+R):")
    for k in mech_counts.index:
        print(f"  {k}: {mech_counts[k]} ({mech_frac[k]:.1%})")

    # --- write machine-readable summary md ---
    lines = ["# Mechanism attribution: le_dir flip vs chord-axis flip vs unexplained\n"]
    lines.append(f"Dataset: `{REAL_DATASET_ROOT}`, {len(ok)}/{len(out_df)} frames status=ok, fps={FPS}.\n")

    lines.append("## Step 5: le_dir continuity vs |Δeta| (per transition)\n")
    lines.append(f"- {len(le_trans)} transitions (both sides pooled); "
                 f"{int(big.sum())} big-jump (|Δeta|>{BIG_JUMP_DEG:.0f} deg)")
    lines.append(f"- of big-jump transitions: le_flip_strict (cos<{FLIP_COS_THRESHOLD})="
                 f"{int((big & le_trans['le_flip_strict']).sum())}, "
                 f"le_flip_loose (cos<0)={int((big & le_trans['le_flip_loose']).sum())}")
    lines.append("- 2x2 contingency (le_flip_strict x big_jump):\n```\n" + table_le.to_string() + "\n```")
    lines.append(f"- Fisher exact: odds_ratio={odds_le:.4f}, p={p_le:.4g}\n")

    lines.append("## Step 6: axis_margin (chord-axis sign decisiveness)\n")
    lines.append(f"- eta-match consistency check (reimplemented axis fed through real chord.py binning/"
                 f"aggregation/eta vs `chord.estimate_chord`'s own eta): {n_eta_match}/{len(valid_axis)} exact "
                 f"matches, max diff={max_eta_diff:.3e} deg")
    lines.append(f"- Spearman r(axis_margin, |Δeta|)      = {corr_axis.correlation:.4f} (p={corr_axis.pvalue:.4g})")
    lines.append(f"- Spearman r(margin_count, |Δeta|)     = {corr_margin.correlation:.4f} (p={corr_margin.pvalue:.4g})")
    lines.append(f"- Spearman r(|curvature_diff|, |Δeta|) = {corr_curv.correlation:.4f} (p={corr_curv.pvalue:.4g})")
    if mwu_axis is not None:
        lines.append(f"- Mann-Whitney U axis_margin, big_jump & !le_flip (n={len(grp_no_le)}, "
                     f"median={grp_no_le.median():.4f}) vs rest (n={len(grp_rest)}, median={grp_rest.median():.4f}): "
                     f"U={mwu_axis.statistic:.1f}, p={mwu_axis.pvalue:.4g}\n")
    else:
        lines.append("- Mann-Whitney U axis_margin: insufficient rows in one group\n")

    lines.append("## Step 7: out_ref_norm\n")
    if mwu_outref_bigjump is not None:
        lines.append(f"- Mann-Whitney U out_ref_norm, eta big jump: True (n={len(outref_bigjump_true)}, "
                     f"median={outref_bigjump_true.median():.4e}) vs False (n={len(outref_bigjump_false)}, "
                     f"median={outref_bigjump_false.median():.4e}): U={mwu_outref_bigjump.statistic:.1f}, "
                     f"p={mwu_outref_bigjump.pvalue:.4g}")
    else:
        lines.append("- Mann-Whitney U out_ref_norm, eta big jump: insufficient rows")
    if mwu_outref_leflip is not None:
        lines.append(f"- Mann-Whitney U out_ref_norm, le_dir flipped: True (n={len(outref_leflip_true)}, "
                     f"median={outref_leflip_true.median():.4e}) vs False (n={len(outref_leflip_false)}, "
                     f"median={outref_leflip_false.median():.4e}): U={mwu_outref_leflip.statistic:.1f}, "
                     f"p={mwu_outref_leflip.pvalue:.4g}\n")
    else:
        lines.append("- Mann-Whitney U out_ref_norm, le_dir flipped: insufficient rows\n")

    lines.append("## Three-way mechanism classification of big-jump transitions (pooled L+R)\n")
    lines.append(f"n={len(big_trans)}\n")
    for k in mech_counts.index:
        lines.append(f"- `{k}`: {mech_counts[k]} ({mech_frac[k]:.1%})")
    lines.append("")

    (DIAG_DIR / "05_06_07_mechanism_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwritten: plots + CSVs + {DIAG_DIR / '05_06_07_mechanism_summary.md'}")


if __name__ == "__main__":
    main()
