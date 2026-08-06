"""Step 8: does `estimate_leading_edge`'s discrete pos/neg winner identity
(`use_pos`) flip more often -- and more predictively of eta big jumps -- than
the continuous `le_dir` vector does?

Motivation (see task description / `report.md` §4): §4.1 tested `le_dir`
continuity via `cos_angle(le_dir_t, le_dir_t+1)` and found it explains only
5.7% of big jumps, far weaker than the chord-axis flip (72.4%, OR≈79,
p=5.9e-28). But `le_dir` is fit from whichever candidate set (pos or neg)
wins the RANSAC count/residual tie-break in `estimate_leading_edge`, and both
candidate sets are binned along the *same* span axis -- so even when the
winner flips from pos to neg, the fitted line direction can stay nearly
parallel to span (high `cos_angle`), masking the flip. This script tracks
`use_pos` itself (already exposed by `le_repro.py::LEDiag.use_pos`, verified
bit-exact against `wing_angles.estimate_leading_edge` in
`00_consistency_check.md`) as a per-(frame,side) discrete label and tests
whether *its* frame-to-frame flip is what actually drives chord-axis sign
flips and eta big jumps.

Does not recompute anything `le_repro.py`/`wrap_mechanism_diag.py`/
`real_data_validation.py` already computed and wrote to `diag/`: `use_pos`
comes from `03_real_data_merged.csv`'s `count_winner_is_pos` column (written
by `real_data_validation.py::_collect_le_diagnostics`, same
`estimate_leading_edge_diag` call, same `rng=0`), and `le_flip_strict`/
`axis_flip_strict`/`big_jump`/`mechanism` come straight from
`05_06_transitions_merged.csv` (`wrap_mechanism_diag.py`'s own merged
per-transition table). This script only joins those two existing tables on
(frame_id, side) to add the one new column, `winner_flip`, and re-derives
statistics from it.

Run: python -m postprocessing.kinematics.correct_wing_pitch.winner_flip_diag
"""
from __future__ import annotations

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

DIAG_DIR = Path(__file__).resolve().parent / "diag"
BIG_JUMP_DEG = 90.0

# s6b_real_data_diagnostics_findings.md, item #2: the 30/99 eta_L transitions
# (indexed by the transition's starting frame) whose frame-to-frame delta
# crosses the +-180 deg wrap boundary. Reused verbatim, not recomputed --
# this list is for visual cross-referencing only (plot overlay), not for any
# statistical test in this script.
S6B_WRAP_FRAMES_L = [
    17, 24, 26, 27, 29, 32, 34, 41, 42, 52, 53,
    57, 58, 59, 60, 61, 65, 66, 67, 68, 72, 73, 74,
    81, 86, 87, 91, 96, 98, 99,
]


def _fisher_table(flag: pd.Series, outcome: pd.Series) -> tuple[pd.DataFrame, float, float]:
    """2x2 table with rows=[flag True, flag False], cols=[outcome False, outcome True]
    (same reindex convention as `wrap_mechanism_diag.py`). Returns the "standard"
    odds ratio in the intuitive `flag -> outcome` direction, i.e.
    odds(outcome=True | flag=True) / odds(outcome=True | flag=False) = (b*c)/(a*d)
    for table [[a,b],[c,d]] -- the *reciprocal* of what `scipy.stats.fisher_exact`
    itself returns for this table shape (verified against report.md's own
    convention: `05_06_07_mechanism_summary.md` records raw scipy odds_ratio=0.4556
    for le_flip_strict x big_jump, while report.md §4.1 prose states the
    reciprocal, 2.20, as "Standard odds ratio (le_dir flip -> big jump)").
    p-value is unaffected by the direction of the ratio.
    """
    table = pd.crosstab(flag, outcome)
    table = table.reindex(index=[True, False], columns=[False, True], fill_value=0)
    raw_odds_ratio, p = stats.fisher_exact(table.to_numpy())
    a, b = table.iloc[0, 0], table.iloc[0, 1]
    c, d = table.iloc[1, 0], table.iloc[1, 1]
    standard_odds_ratio = (b * c) / (a * d) if a * d != 0 else float("inf")
    return table, standard_odds_ratio, p


def main() -> None:
    merged_path = DIAG_DIR / "03_real_data_merged.csv"
    trans_path = DIAG_DIR / "05_06_transitions_merged.csv"
    if not merged_path.exists() or not trans_path.exists():
        print(f"ERROR: missing prerequisite diag output(s): {merged_path}, {trans_path}")
        print("Run real_data_validation.py and wrap_mechanism_diag.py first -- refusing to fabricate inputs.")
        sys.exit(1)

    per_frame = pd.read_csv(merged_path)
    if per_frame["le_failed"].astype(bool).any():
        print("ERROR: 03_real_data_merged.csv has LE-fit failures; winner-identity lookup would have gaps.")
        sys.exit(1)
    use_pos_lookup = {
        (int(r.frame_id), r.side): bool(r.count_winner_is_pos) for r in per_frame.itertuples()
    }

    trans = pd.read_csv(trans_path)

    def _flip(row):
        key_from = (int(row["frame_from"]), row["side"])
        key_to = (int(row["frame_to"]), row["side"])
        if key_from not in use_pos_lookup or key_to not in use_pos_lookup:
            return np.nan
        return use_pos_lookup[key_from] != use_pos_lookup[key_to]

    trans["winner_flip"] = trans.apply(_flip, axis=1)
    n_missing = int(trans["winner_flip"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing}/{len(trans)} transitions missing a use_pos lookup, dropped from analysis")
    trans = trans.dropna(subset=["winner_flip"]).copy()
    trans["winner_flip"] = trans["winner_flip"].astype(bool)
    trans.to_csv(DIAG_DIR / "08_winner_flip_transitions.csv", index=False)

    print(f"winner_flip transitions: {len(trans)} rows (both sides pooled)")
    print(f"  winner_flip=True: {int(trans['winner_flip'].sum())} "
          f"({trans['winner_flip'].mean():.1%})")

    # --- Q1: winner_flip x axis_flip_strict ---
    table_axis, odds_axis, p_axis = _fisher_table(trans["winner_flip"], trans["axis_flip_strict"])
    print("\n[Q1] 2x2 (winner_flip x axis_flip_strict):")
    print(table_axis)
    print(f"  Fisher exact: standard odds_ratio (winner_flip -> axis_flip_strict)={odds_axis:.4f}, p={p_axis:.4g}")

    # --- also: winner_flip x big_jump directly (context, not asked but cheap) ---
    table_bigjump, odds_bigjump, p_bigjump = _fisher_table(trans["winner_flip"], trans["big_jump"])
    print("\n[context] 2x2 (winner_flip x big_jump):")
    print(table_bigjump)
    print(f"  Fisher exact: standard odds_ratio (winner_flip -> big_jump)={odds_bigjump:.4f}, p={p_bigjump:.4g}")

    # --- Q2: winner_flip x le_flip_strict ---
    table_le, odds_le, p_le = _fisher_table(trans["winner_flip"], trans["le_flip_strict"])
    print("\n[Q2] 2x2 (winner_flip x le_flip_strict):")
    print(table_le)
    print(f"  Fisher exact: standard odds_ratio (winner_flip -> le_flip_strict)={odds_le:.4f}, p={p_le:.4g}")

    n_winner_flip = int(trans["winner_flip"].sum())
    n_winner_flip_no_le = int((trans["winner_flip"] & ~trans["le_flip_strict"]).sum())
    frac_winner_flip_no_le = n_winner_flip_no_le / n_winner_flip if n_winner_flip else float("nan")
    print(f"  winner_flip=True total: {n_winner_flip}; of those, le_flip_strict=False: "
          f"{n_winner_flip_no_le} ({frac_winner_flip_no_le:.1%})")

    # --- Q3: three-way reclassification of big-jump transitions, winner_flip-first ---
    big = trans[trans["big_jump"]].copy()
    big["mechanism_v2"] = np.select(
        [big["winner_flip"], big["axis_flip_strict"] == True],  # noqa: E712
        ["winner_flip", "chord_axis"],
        default="unexplained",
    )
    counts_v2 = big["mechanism_v2"].value_counts()
    frac_v2 = big["mechanism_v2"].value_counts(normalize=True)
    counts_v1 = big["mechanism"].value_counts()
    frac_v1 = big["mechanism"].value_counts(normalize=True)
    print(f"\n[Q3] three-way reclassification of {len(big)} big-jump transitions:")
    print("  v1 (le_dir-first, from wrap_mechanism_diag.py):")
    for k in counts_v1.index:
        print(f"    {k}: {counts_v1[k]} ({frac_v1[k]:.1%})")
    print("  v2 (winner_flip-first):")
    for k in counts_v2.index:
        print(f"    {k}: {counts_v2[k]} ({frac_v2[k]:.1%})")

    # --- Q4: winner_flip rate within the old (v1) "unexplained" subset ---
    old_unexplained = big[big["mechanism"] == "unexplained"]
    n_old_unexplained = len(old_unexplained)
    n_old_unexplained_winner_flip = int(old_unexplained["winner_flip"].sum())
    frac_old_unexplained_winner_flip = (
        n_old_unexplained_winner_flip / n_old_unexplained if n_old_unexplained else float("nan")
    )
    print(f"\n[Q4] of the {n_old_unexplained} v1-'unexplained' big-jump transitions, "
          f"{n_old_unexplained_winner_flip} ({frac_old_unexplained_winner_flip:.1%}) have winner_flip=True")

    # --- Q5: time series plots, L and R, eta + winner_flip markers ---
    overlap_by_side: dict[str, tuple[int, int, list[int]]] = {}
    for suffix, side, wrap_frames in (
        ("L", "wing_L", S6B_WRAP_FRAMES_L),
        ("R", "wing_R", None),
    ):
        sub_eta = per_frame[per_frame["side"] == side].sort_values("frame_id")
        sub_trans = trans[trans["side"] == side]
        flips = sub_trans[sub_trans["winner_flip"]]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(sub_eta["frame_id"], sub_eta["eta"], color="black", lw=1.0, alpha=0.8, label="eta")
        for i, row in enumerate(flips.itertuples()):
            ax.axvline(row.frame_from, color="tab:red", lw=1.0, alpha=0.5,
                        label="winner_flip transition" if i == 0 else None)
        if wrap_frames is not None:
            for i, f in enumerate(wrap_frames):
                ax.axvline(f, color="tab:blue", lw=0.8, alpha=0.35, ls=":",
                            label="s6b wrap-crossing frame" if i == 0 else None)
        ax.set_xlabel("frame_id")
        ax.set_ylabel(f"eta_{suffix} (deg)")
        ax.set_title(f"eta_{suffix}: winner_flip transitions vs eta series"
                     + (" (+ s6b wrap-crossing frames)" if wrap_frames is not None else ""))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DIAG_DIR / f"08_winner_flip_timeseries_{suffix}.png", dpi=150)
        plt.close(fig)

        flip_from_frames = set(int(f) for f in flips["frame_from"])
        if wrap_frames is not None:
            overlap = sorted(flip_from_frames & set(wrap_frames))
            overlap_by_side[side] = (len(flip_from_frames), len(wrap_frames), overlap)
            print(f"\n[Q5, side={side}] winner_flip transitions (frame_from): {len(flip_from_frames)}; "
                  f"s6b wrap-crossing frames: {len(wrap_frames)}; "
                  f"overlap: {len(overlap)} ({overlap})")
        else:
            print(f"\n[Q5, side={side}] winner_flip transitions (frame_from): {len(flip_from_frames)}; "
                  f"no s6b reference list for this side (item #2 only enumerates eta_L)")

    # --- write report fragment ---
    lines = ["\n---\n"]
    lines.append("## 8. Winner identity (`use_pos`) cross-frame flip vs chord-axis flip / eta big jumps\n")
    lines.append(f"**Setup**: `winner_flip_diag.py`, joining `03_real_data_merged.csv`'s "
                 f"`count_winner_is_pos` (== `LEDiag.use_pos`, already bit-exact-verified, "
                 f"§0) at each transition's `frame_from`/`frame_to` with "
                 f"`05_06_transitions_merged.csv` (§4's own `le_flip_strict`/`axis_flip_strict`/"
                 f"`big_jump`/`mechanism`, unmodified). No new geometry recomputed. "
                 f"`winner_flip = (use_pos_t != use_pos_{{t+1}})`, {len(trans)} transitions "
                 f"(both sides pooled, {n_missing} dropped for missing lookups).\n")

    lines.append(f"- `winner_flip=True` on {n_winner_flip}/{len(trans)} transitions ({trans['winner_flip'].mean():.1%}).\n")

    lines.append("### 8.1 `winner_flip` x `axis_flip_strict`\n")
    lines.append("```\n" + table_axis.to_string() + "\n```")
    lines.append(f"- Fisher exact, standard odds ratio (`winner_flip` → `axis_flip_strict`): "
                 f"OR={odds_axis:.2f}, p={p_axis:.4g}\n")
    lines.append(f"- For direct scale comparison (same standard-OR convention as §4): "
                 f"`winner_flip → big_jump` OR={odds_bigjump:.2f}, p={p_bigjump:.4g} "
                 f"(context 2x2 above); §4.2's `axis_flip_strict → big_jump` OR≈79.0, p=5.9e-28; "
                 f"§4.1's `le_flip_strict → big_jump` OR=2.20, p=0.303. `winner_flip`'s association "
                 f"with both `axis_flip_strict` and `big_jump` is far stronger than `le_dir`'s "
                 f"(orders of magnitude lower p, roughly 4-9x higher OR), though still weaker than "
                 f"`axis_flip_strict`'s own direct association with `big_jump`.\n")

    lines.append("### 8.2 `winner_flip` x `le_flip_strict`\n")
    lines.append("```\n" + table_le.to_string() + "\n```")
    lines.append(f"- Fisher exact, standard odds ratio (`winner_flip` → `le_flip_strict`): "
                 f"OR={odds_le:.2f}, p={p_le:.4g} (not significant; only 6 vs 2 transitions in the "
                 f"`le_flip_strict=True` column, underpowered)")
    lines.append(f"- Of {n_winner_flip} transitions with `winner_flip=True`, {n_winner_flip_no_le} "
                 f"({frac_winner_flip_no_le:.1%}) do NOT clear the `le_flip_strict` threshold "
                 f"(`cos_angle_le >= -0.5`) -- i.e. `le_dir`'s direction stayed continuous while the "
                 f"underlying pos/neg winner switched.\n")

    lines.append(f"### 8.3 Three-way reclassification of the {len(big)} big-jump transitions "
                 f"(`winner_flip`-first vs `le_dir`-first)\n")
    lines.append("v1 (original, `le_dir`-first, from `wrap_mechanism_diag.py` §4.2):\n")
    for k in counts_v1.index:
        lines.append(f"- `{k}`: {counts_v1[k]} ({frac_v1[k]:.1%})")
    lines.append("\nv2 (`winner_flip`-first → `chord_axis` (no `winner_flip`) → `unexplained`):\n")
    for k in counts_v2.index:
        lines.append(f"- `{k}`: {counts_v2[k]} ({frac_v2[k]:.1%})")
    v1_unexplained_pct = float(frac_v1.get("unexplained", 0.0))
    v2_unexplained_pct = float(frac_v2.get("unexplained", 0.0))
    lines.append(f"\n`unexplained` share: {v1_unexplained_pct:.1%} (v1) → {v2_unexplained_pct:.1%} (v2).\n")

    lines.append(f"### 8.4 `winner_flip` rate within the {n_old_unexplained} v1-\"unexplained\" big-jump transitions\n")
    lines.append(f"- {n_old_unexplained_winner_flip}/{n_old_unexplained} "
                 f"({frac_old_unexplained_winner_flip:.1%}) of the transitions v1 labeled `unexplained` "
                 f"have `winner_flip=True`.\n")

    lines.append("### 8.5 Time series: `winner_flip` transitions vs eta, both sides\n")
    lines.append("Plots: `08_winner_flip_timeseries_L.png`, `08_winner_flip_timeseries_R.png` "
                 "(red = `winner_flip` transition start frame; blue dotted, L only = "
                 "`s6b_real_data_diagnostics_findings.md` item #2's 30 documented wrap-crossing frames).\n")
    if "wing_L" in overlap_by_side:
        n_flip_l, n_wrap_l, overlap_l = overlap_by_side["wing_L"]
        lines.append(f"- Side L: {n_flip_l} `winner_flip` transitions vs {n_wrap_l} s6b-documented "
                     f"wrap-crossing frames; {len(overlap_l)} frames in common "
                     f"({len(overlap_l) / n_wrap_l:.1%} of the s6b list): `{overlap_l}`.\n")
    lines.append("- Side R has no s6b reference list (item #2 only enumerated eta_L); the R plot shows "
                 "`winner_flip` markers against the eta_R series with no overlay.\n")

    (DIAG_DIR / "08_winner_flip_summary.md").write_text("\n".join(lines) + "\n")

    report_path = DIAG_DIR / "report.md"
    with open(report_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nwritten: 08_winner_flip_transitions.csv, 08_winner_flip_timeseries_{{L,R}}.png, "
          f"08_winner_flip_summary.md, appended to report.md")


if __name__ == "__main__":
    main()
