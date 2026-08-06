"""Step 3: real-data validation of the count judge vs. curvature signal,
against the actual eta 180-degree wrap crossings documented in
`reference/s6b_real_data_diagnostics_findings.md` (Top issue #2).

For every (frame, side) with a valid leading-edge fit, records the count
judge's margin (`margin_count`/`margin_ratio`), the curvature difference
(`neg_arc_chord - pos_arc_chord`), and the frame's associated |eta delta|
(the larger of its two neighboring circular deltas -- see `_assoc_abs_delta`
docstring). Reuses `diagnostics.py::circular_delta_deg` for the delta itself
(not re-derived) and `pipeline.py::run_dataset`'s own eta output (not
recomputed by hand) so the eta values analyzed here are exactly the
production pipeline's.

Run: python -m postprocessing.kinematics.correct_wing_pitch.real_data_validation
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
from scipy import stats

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import diagnostics as diag_mod  # noqa: E402
from postprocessing.kinematics import io_schema, pipeline  # noqa: E402
from postprocessing.kinematics.correct_wing_pitch.le_repro import estimate_leading_edge_diag  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"
REAL_DATASET_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
FPS = 16000.0
FRAME_GLOB = "f*/splatfacto-checkpoint/*/*_labeled.csv"
BIG_JUMP_DEG = 90.0
"""Threshold for the 2x2 contingency table's "big jump" bucket (task's own
suggested example threshold)."""

_FRAME_DIR_RE = re.compile(r"^f(\d+)$")


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


def _assoc_abs_delta(eta: np.ndarray) -> np.ndarray:
    """Per-frame |eta delta|, associated with each frame (not each
    transition): frame `i`'s value is the larger of its two neighboring
    circular deltas (`|eta[i]-eta[i-1]|`, `|eta[i]-eta[i+1]|`), or its single
    neighboring delta at the two endpoints. This directly measures "how much
    does this frame's eta disagree with its immediate neighbors" -- the
    quantity relevant to testing whether *this frame's* LE/TE judgment
    diagnostics explain a local eta discontinuity (a single mis-assigned
    frame produces two large flanking deltas, one on each side of it, so
    taking the max of both catches it regardless of which transition index
    convention is used).
    """
    n = len(eta)
    cd = np.abs(diag_mod.circular_delta_deg(eta))  # length n-1
    out = np.empty(n)
    if n == 1:
        return np.array([0.0])
    out[0] = cd[0]
    out[-1] = cd[-1]
    if n > 2:
        out[1:-1] = np.maximum(cd[:-1], cd[1:])
    return out


def _collect_le_diagnostics(frames: list[tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for frame_id, csv_path in frames:
        try:
            df = io_schema.load_frame(csv_path)
            body_xyz = io_schema.get_part(df, "body")
            wingL_xyz = io_schema.get_part(df, "wing_L")
            wingR_xyz = io_schema.get_part(df, "wing_R")
            frame = bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(frame_id=frame_id, side="wing_L", le_failed=True, le_fail_reason=str(e)))
            rows.append(dict(frame_id=frame_id, side="wing_R", le_failed=True, le_fail_reason=str(e)))
            continue

        for side, wing_xyz in (("wing_L", wingL_xyz), ("wing_R", wingR_xyz)):
            try:
                d = estimate_leading_edge_diag(wing_xyz, frame, side, rng=0)
            except Exception as e:  # noqa: BLE001
                rows.append(dict(frame_id=frame_id, side=side, le_failed=True, le_fail_reason=str(e)))
                continue
            rows.append(dict(
                frame_id=frame_id, side=side, le_failed=False,
                margin_count=d.margin_count, margin_ratio=d.margin_ratio,
                pos_count=d.pos_count, neg_count=d.neg_count,
                pos_arc_chord=d.pos_arc_chord, neg_arc_chord=d.neg_arc_chord,
                winner_arc_chord=d.winner_arc_chord, loser_arc_chord=d.loser_arc_chord,
                curvature_diff=d.curvature_diff,
                count_winner_is_pos=bool(d.use_pos),
                curv_winner_is_pos=bool(d.pos_arc_chord < d.neg_arc_chord),
                plane_inlier_frac=d.plane_inlier_frac,
            ))
    return pd.DataFrame(rows)


def _plot_scatter(merged: pd.DataFrame, x_col: str, xlabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for side, color in (("wing_L", "tab:blue"), ("wing_R", "tab:orange")):
        sub = merged[merged["side"] == side]
        ax.scatter(sub[x_col], sub["abs_delta_eta"], s=18, alpha=0.7, label=side, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("associated |Δeta| (deg, per-frame, see docstring)")
    ax.set_title(title)
    ax.axhline(BIG_JUMP_DEG, color="gray", lw=0.8, ls="--", label=f"{BIG_JUMP_DEG:.0f} deg")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_histogram(merged: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(merged["abs_delta_eta"], bins=30, color="tab:purple", alpha=0.8)
    ax.axvspan(160, 180, color="red", alpha=0.15, label="160-180 deg (wrap-boundary band)")
    ax.set_xlabel("associated |Δeta| (deg)")
    ax.set_ylabel("count (frame x side)")
    ax.set_title("Distribution of per-frame |Δeta|")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not REAL_DATASET_ROOT.exists():
        print(f"ERROR: real dataset root not found: {REAL_DATASET_ROOT}")
        print("Cannot run real-data validation without it -- refusing to substitute fake data.")
        sys.exit(1)

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running production pipeline on {REAL_DATASET_ROOT} ...")
    config = pipeline.PipelineConfig(min_points=10, output_dir=DIAG_DIR, write_debug=False, frame_glob=FRAME_GLOB)
    out_df = pipeline.run_dataset(REAL_DATASET_ROOT, config)
    ok = out_df[out_df["status"] == "ok"].reset_index(drop=True)
    print(f"pipeline: {len(out_df)} frames total, {len(ok)} status=ok")
    if len(ok) < 3:
        print("ERROR: fewer than 3 ok frames; cannot compute meaningful deltas")
        sys.exit(1)

    frames = _discover_frames(REAL_DATASET_ROOT, FRAME_GLOB)
    print(f"discovered {len(frames)} frame CSVs via glob {FRAME_GLOB!r}")
    le_df = _collect_le_diagnostics(frames)
    n_le_failed = int(le_df["le_failed"].sum())
    print(f"LE diagnostics: {len(le_df)} (frame,side) rows, {n_le_failed} failed")

    rows = []
    for suffix, side in (("L", "wing_L"), ("R", "wing_R")):
        eta = ok[f"eta_{suffix}"].to_numpy()
        frame_ids = ok["frame_id"].to_numpy()
        abs_delta = _assoc_abs_delta(eta)
        rows.append(pd.DataFrame(dict(frame_id=frame_ids, side=side, eta=eta, abs_delta_eta=abs_delta,
                                       chord_conf=ok[f"chord_conf_{suffix}"].to_numpy())))
    delta_df = pd.concat(rows, ignore_index=True)

    merged = delta_df.merge(le_df, on=["frame_id", "side"], how="left")
    merged.to_csv(DIAG_DIR / "03_real_data_merged.csv", index=False)

    valid = merged[~merged["le_failed"].fillna(True).astype(bool)].copy()
    print(f"merged rows with valid LE diagnostics: {len(valid)} / {len(merged)}")

    # --- scatter plots ---
    _plot_scatter(valid, "margin_count", "margin_count (winner_count - loser_count)",
                  "margin_count vs |Δeta|", DIAG_DIR / "03_margin_count_vs_delta_eta.png")
    _plot_scatter(valid, "curvature_diff", "curvature_diff (neg_arc_chord - pos_arc_chord)",
                  "curvature_diff vs |Δeta|", DIAG_DIR / "03_curvature_diff_vs_delta_eta.png")
    _plot_histogram(valid, DIAG_DIR / "03_delta_eta_histogram.png")

    # --- correlations ---
    corr_margin = stats.spearmanr(valid["margin_count"], valid["abs_delta_eta"])
    corr_curv = stats.spearmanr(valid["curvature_diff"].abs(), valid["abs_delta_eta"])
    print(f"Spearman r(margin_count, |Δeta|) = {corr_margin.correlation:.4f} (p={corr_margin.pvalue:.4g})")
    print(f"Spearman r(|curvature_diff|, |Δeta|) = {corr_curv.correlation:.4f} (p={corr_curv.pvalue:.4g})")

    # --- 2x2 contingency: count vs curvature agreement x big-jump ---
    valid["judges_agree"] = valid["count_winner_is_pos"] == valid["curv_winner_is_pos"]
    valid["big_jump"] = valid["abs_delta_eta"] > BIG_JUMP_DEG
    table = pd.crosstab(valid["judges_agree"], valid["big_jump"])
    table = table.reindex(index=[True, False], columns=[False, True], fill_value=0)
    print("\n2x2 contingency (rows=judges agree, cols=big jump>%.0f deg):" % BIG_JUMP_DEG)
    print(table)
    odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())
    print(f"Fisher exact: odds_ratio={odds_ratio:.4f}, p={fisher_p:.4g}")

    # --- Wilcoxon signed-rank: count-winner curvature vs count-loser curvature ---
    w = valid["winner_arc_chord"].to_numpy()
    l = valid["loser_arc_chord"].to_numpy()
    finite = np.isfinite(w) & np.isfinite(l)
    w, l = w[finite], l[finite]
    diffs = w - l
    n_ties = int(np.sum(diffs == 0))
    wilcoxon_res = stats.wilcoxon(w, l, alternative="less") if len(w) - n_ties > 0 else None
    print(f"\nWilcoxon signed-rank (count-winner arc_chord < count-loser arc_chord), n={len(w)}, ties={n_ties}:")
    if wilcoxon_res is not None:
        print(f"  statistic={wilcoxon_res.statistic:.4f}, p={wilcoxon_res.pvalue:.4g}")
        print(f"  winner median arc_chord={np.median(w):.4f}, loser median={np.median(l):.4f}, "
              f"winner<loser fraction={float(np.mean(diffs < 0)):.4f}")

    # --- write report fragment ---
    lines = ["# Real-data LE/TE judge validation (S6b eta-wrap-crossing frames)\n"]
    lines.append(f"Dataset: `{REAL_DATASET_ROOT}`, {len(out_df)} frames, {len(ok)} status=ok, "
                 f"fps={FPS} (per `diagnostics.py::FPS`).\n")
    lines.append(f"LE diagnostics computed on {len(le_df)} (frame,side) rows, {n_le_failed} LE-fit failures; "
                 f"{len(valid)} rows have valid margin/curvature diagnostics merged with a pipeline eta delta.\n")
    lines.append("## Correlation with |Δeta|\n")
    lines.append(f"- Spearman r(margin_count, |Δeta|) = {corr_margin.correlation:.4f} (p={corr_margin.pvalue:.4g})")
    lines.append(f"- Spearman r(|curvature_diff|, |Δeta|) = {corr_curv.correlation:.4f} (p={corr_curv.pvalue:.4g})\n")
    lines.append(f"## 2x2 contingency: count vs curvature judge agreement x big jump (>{BIG_JUMP_DEG:.0f} deg)\n")
    lines.append("```\n" + table.to_string() + "\n```")
    lines.append(f"\nFisher exact test: odds_ratio={odds_ratio:.4f}, p={fisher_p:.4g}\n")
    lines.append("## Wilcoxon signed-rank: is count-winner's arc-chord systematically < count-loser's?\n")
    if wilcoxon_res is not None:
        lines.append(f"- n={len(w)} (ties excluded: {n_ties}), statistic={wilcoxon_res.statistic:.4f}, "
                     f"p={wilcoxon_res.pvalue:.4g}")
        lines.append(f"- winner median arc_chord={np.median(w):.4f}, loser median arc_chord={np.median(l):.4f}")
        lines.append(f"- fraction of rows where winner_arc_chord < loser_arc_chord: {float(np.mean(diffs < 0)):.4f}\n")
    else:
        lines.append("- insufficient non-tied pairs to run the test\n")
    (DIAG_DIR / "03_real_data_validation_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwritten: plots + {DIAG_DIR / '03_real_data_validation_summary.md'}")


if __name__ == "__main__":
    main()
