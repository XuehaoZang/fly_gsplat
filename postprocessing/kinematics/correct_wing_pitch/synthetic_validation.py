"""Step 2: synthetic ground-truth validation of the count judge (RANSAC
inlier count) vs. a curvature judge (pre-RANSAC arc-chord ratio) for
picking which candidate edge is the leading edge.

Uses `mock.py`'s `default_ground_truth` + `make_wing_points` (clean,
noiseless geometry) as ground truth, then adds positional Gaussian noise of
increasing magnitude (this script's own perturbation, `mock.py` itself is
untouched) to manufacture "boundary" samples where the two candidates'
RANSAC inlier counts are close/tied. Ground truth for "which candidate set
is really the leading edge" is read off the *clean* (pre-noise) point
positions projected onto the true chord direction -- never the noisy
positions -- so it cannot be contaminated by the same noise that makes the
judgment call hard.

Run: python -m postprocessing.kinematics.correct_wing_pitch.synthetic_validation
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

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import mock  # noqa: E402
from postprocessing.kinematics.correct_wing_pitch.le_repro import estimate_leading_edge_diag  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"

NOISE_LEVELS_FRAC_OF_MAX_CHORD = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
"""Positional Gaussian noise std, as a fraction of `mock.WING_MAX_CHORD_M`.
Spans clean (0.0, count judge trivially right, big margin -- see
`chord.py`/`wing_angles.py` module docstrings' own claim that a single
contaminating point can flip the straighter-edge call) to noise comparable
to the whole chord (1.0 -- deliberately extreme, to guarantee the boundary
regime the task asks for is actually reached, not just approached)."""
N_TRIALS_PER_LEVEL = 40
N_WING_POINTS_LEVELS = (120, 400)
"""120: fewer than mock.py's own default (400) -- smaller point clouds give
the per-bin RANSAC more variance, which is what makes ties/near-ties
reachable at moderate noise instead of only at the most extreme levels.
400: mock.py's own default density, run in parallel as a robustness check on
whether the count-vs-curvature comparison is an artifact of point sparsity
(fewer points/bin -> noisier per-bin chordwise-extreme order statistic)."""
NEAR_TIED_MARGIN_RATIO = 1.3
"""margin_ratio <= this counts as "count judge nearly tied" for the
tied-subset accuracy breakdown."""


def _true_body_frame(gt: mock.GroundTruth) -> bf.BodyFrame:
    x_body, y_body, z_body = mock.body_axes(gt)
    n_sp = mock.stroke_plane_normal(gt)
    return bf.BodyFrame(
        x_body=x_body, y_body=y_body, z_body=z_body, n_sp=n_sp,
        yaw=gt.yaw_deg, pitch=gt.pitch_deg, roll=gt.roll_deg,
        hinge_L=gt.wing_L.root, hinge_R=gt.wing_R.root, body_cm=gt.body_center,
    )


def _run_trials() -> pd.DataFrame:
    rows = []
    gt = mock.default_ground_truth()
    frame = _true_body_frame(gt)
    n_sp = mock.stroke_plane_normal(gt)

    for n_wing_points in N_WING_POINTS_LEVELS:
        for side in ("wing_L", "wing_R"):
            wing_gt = getattr(gt, side)
            sign_left = mock._SIGN_LEFT[side]
            chord_dir = mock._chord_dir(wing_gt.span_dir, n_sp, wing_gt.eta_deg, sign_left)

            for level_idx, noise_frac in enumerate(NOISE_LEVELS_FRAC_OF_MAX_CHORD):
                noise_std = noise_frac * mock.WING_MAX_CHORD_M
                for trial in range(N_TRIALS_PER_LEVEL):
                    seed = (
                        10_000 * level_idx + 17 * trial
                        + (0 if side == "wing_L" else 5_000_000)
                        + (0 if n_wing_points == 120 else 50_000_000)
                    )
                    gen_rng = np.random.default_rng(seed)
                    wing_df = mock.make_wing_points(gt, side, n_wing_points, gen_rng)
                    xyz_clean = wing_df[["x", "y", "z"]].to_numpy()
                    clean_offset = (xyz_clean - wing_gt.root) @ chord_dir  # 0-ish at true LE, grows toward true TE

                    noise_rng = np.random.default_rng(seed + 999_983)
                    xyz_noisy = xyz_clean + noise_rng.normal(0.0, noise_std, size=xyz_clean.shape) if noise_std > 0 else xyz_clean

                    try:
                        diag = estimate_leading_edge_diag(xyz_noisy, frame, side, rng=0)
                    except ValueError as e:
                        rows.append(dict(
                            n_wing_points=n_wing_points, side=side, noise_frac=noise_frac, trial=trial,
                            failed=True, fail_reason=str(e),
                        ))
                        continue

                    gt_le_is_pos = float(np.mean(clean_offset[diag.pos_orig_idx])) < float(np.mean(clean_offset[diag.neg_orig_idx]))
                    count_pred_is_pos = diag.use_pos
                    # smaller pre-RANSAC arc-chord ratio = straighter = predicted LE
                    curv_pred_is_pos = diag.pos_arc_chord < diag.neg_arc_chord

                    rows.append(dict(
                        n_wing_points=n_wing_points, side=side, noise_frac=noise_frac, trial=trial, failed=False,
                        margin_count=diag.margin_count, margin_ratio=diag.margin_ratio,
                        pos_count=diag.pos_count, neg_count=diag.neg_count,
                        pos_arc_chord=diag.pos_arc_chord, neg_arc_chord=diag.neg_arc_chord,
                        curvature_diff=diag.curvature_diff,
                        gt_le_is_pos=gt_le_is_pos,
                        count_correct=(count_pred_is_pos == gt_le_is_pos),
                        curv_correct=(curv_pred_is_pos == gt_le_is_pos),
                        near_tied=diag.margin_ratio <= NEAR_TIED_MARGIN_RATIO,
                    ))
    return pd.DataFrame(rows)


_BOOL_COLS = ("failed", "count_correct", "curv_correct", "near_tied", "gt_le_is_pos")


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Non-failed trial rows, with the bool-valued columns forced to real
    bool dtype -- built from a list of per-trial dicts where failed rows omit
    several keys, `pd.DataFrame` otherwise leaves these columns as `object`
    (Python `True`/`False` mixed with `NaN`), and `~` on an object column
    applies Python's bitwise-NOT to each element (`~True == -2`) instead of
    boolean negation -- silently corrupting every downstream `ok[~ok[col]]`
    boolean mask into a column-label lookup.
    """
    for c in _BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    return df[~df["failed"]]


def _accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(df)
    out = []
    for (n_wing_points, noise_frac), g in ok.groupby(["n_wing_points", "noise_frac"]):
        n = len(g)
        n_tied = int(g["near_tied"].sum())
        out.append(dict(
            n_wing_points=n_wing_points,
            noise_frac=noise_frac,
            n_trials=n,
            n_near_tied=n_tied,
            count_acc=float(g["count_correct"].mean()),
            curv_acc=float(g["curv_correct"].mean()),
            count_acc_tied=float(g.loc[g["near_tied"], "count_correct"].mean()) if n_tied else float("nan"),
            curv_acc_tied=float(g.loc[g["near_tied"], "curv_correct"].mean()) if n_tied else float("nan"),
        ))
    return pd.DataFrame(out)


def _rescue_table(df: pd.DataFrame, n_wing_points: int | None = None) -> dict:
    """Among trials where the count judge was WRONG, what fraction did the
    curvature judge get RIGHT ("rescued")? Restricted to `n_wing_points` if given."""
    ok = _ok_rows(df)
    if n_wing_points is not None:
        ok = ok[ok["n_wing_points"] == n_wing_points]
    count_wrong = ok[~ok["count_correct"]]
    count_wrong_tied = count_wrong[count_wrong["near_tied"]]
    return dict(
        n_count_wrong=len(count_wrong),
        n_count_wrong_rescued_by_curv=int(count_wrong["curv_correct"].sum()),
        rescue_rate=float(count_wrong["curv_correct"].mean()) if len(count_wrong) else float("nan"),
        n_count_wrong_near_tied=len(count_wrong_tied),
        n_count_wrong_near_tied_rescued=int(count_wrong_tied["curv_correct"].sum()),
        rescue_rate_near_tied=float(count_wrong_tied["curv_correct"].mean()) if len(count_wrong_tied) else float("nan"),
    )


def _plot_accuracy(acc: pd.DataFrame, out_path: Path) -> None:
    density_levels = sorted(acc["n_wing_points"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)

    for row, n_wing_points in enumerate(density_levels):
        sub = acc[acc["n_wing_points"] == n_wing_points]

        ax = axes[row, 0]
        ax.plot(sub["noise_frac"], sub["count_acc"], marker="o", label="count judge (current algorithm)", color="tab:blue")
        ax.plot(sub["noise_frac"], sub["curv_acc"], marker="s", label="curvature judge (arc/chord ratio)", color="tab:orange")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Overall accuracy (n_wing_points={n_wing_points})")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(sub["noise_frac"], sub["count_acc_tied"], marker="o", label="count judge", color="tab:blue")
        ax.plot(sub["noise_frac"], sub["curv_acc_tied"], marker="s", label="curvature judge", color="tab:orange")
        ax.set_ylabel("accuracy (near-tied)")
        ax.set_title(f"Near-tied trials only (margin_ratio <= {NEAR_TIED_MARGIN_RATIO})")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("positional noise std (fraction of WING_MAX_CHORD_M)")

    fig.suptitle("Synthetic (mock.py) validation: count vs curvature LE/TE judge")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    df = _run_trials()
    df.to_csv(DIAG_DIR / "02_synthetic_trials_raw.csv", index=False)

    n_failed = int(df["failed"].sum())
    print(f"trials: {len(df)} total, {n_failed} failed (ValueError, e.g. degenerate RANSAC)")

    acc = _accuracy_table(df)
    acc.to_csv(DIAG_DIR / "02_synthetic_accuracy_table.csv", index=False)
    print(acc.to_string(index=False))

    _plot_accuracy(acc, DIAG_DIR / "02_synthetic_accuracy.png")

    ok = _ok_rows(df)
    lines = ["# Synthetic (mock.py) LE/TE judge validation\n"]
    lines.append(f"Trials: {len(df)} total ({N_TRIALS_PER_LEVEL} per noise level per side per density), "
                 f"{n_failed} failed with ValueError (excluded from accuracy stats).\n")
    lines.append(f"n_wing_points levels: {N_WING_POINTS_LEVELS}, noise levels (fraction of WING_MAX_CHORD_M="
                 f"{mock.WING_MAX_CHORD_M:.3e} m): {NOISE_LEVELS_FRAC_OF_MAX_CHORD}\n")

    for n_wing_points in N_WING_POINTS_LEVELS:
        ok_d = ok[ok["n_wing_points"] == n_wing_points]
        overall_count_acc = float(ok_d["count_correct"].mean())
        overall_curv_acc = float(ok_d["curv_correct"].mean())
        tied = ok_d[ok_d["near_tied"]]
        tied_count_acc = float(tied["count_correct"].mean()) if len(tied) else float("nan")
        tied_curv_acc = float(tied["curv_correct"].mean()) if len(tied) else float("nan")
        rescue = _rescue_table(df, n_wing_points=n_wing_points)
        print(f"\n[n_wing_points={n_wing_points}] rescue stats:", rescue)

        lines.append(f"## n_wing_points={n_wing_points}\n")
        lines.append("### Overall accuracy (all trials, all noise levels pooled)\n")
        lines.append(f"- count judge (current RANSAC-inlier-count rule): {overall_count_acc:.4f}")
        lines.append(f"- curvature judge (pre-RANSAC arc-chord ratio, smaller=straighter=LE): {overall_curv_acc:.4f}\n")
        lines.append(f"### Accuracy restricted to near-tied trials (margin_ratio <= {NEAR_TIED_MARGIN_RATIO}, n={len(tied)})\n")
        lines.append(f"- count judge: {tied_count_acc:.4f}")
        lines.append(f"- curvature judge: {tied_curv_acc:.4f}\n")
        lines.append("### Per-noise-level accuracy table\n")
        lines.append("```\n" + acc[acc["n_wing_points"] == n_wing_points].to_string(index=False) + "\n```")
        lines.append("\n### Rescue analysis (trials where count judge was WRONG)\n")
        lines.append(f"- n trials where count judge wrong: {rescue['n_count_wrong']}")
        lines.append(f"- of those, curvature judge correct: {rescue['n_count_wrong_rescued_by_curv']} "
                     f"(rescue rate={rescue['rescue_rate']:.4f})")
        lines.append(f"- restricted to near-tied trials: n wrong={rescue['n_count_wrong_near_tied']}, "
                     f"rescued={rescue['n_count_wrong_near_tied_rescued']} "
                     f"(rescue rate={rescue['rescue_rate_near_tied']:.4f})\n")
    (DIAG_DIR / "02_synthetic_validation_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwritten: {DIAG_DIR / '02_synthetic_accuracy.png'}, "
          f"{DIAG_DIR / '02_synthetic_validation_summary.md'}")


if __name__ == "__main__":
    main()
