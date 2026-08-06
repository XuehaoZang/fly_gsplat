"""Step 1 (round 3, "stroke=90 anchor" task): detect per-frame "T-pose" anchor
candidates -- frames where both wings are spread out roughly perpendicular to
the body long axis, which is where the body/wing part-label split is
geometrically cleanest and most trustworthy.

Independent of `wing_angles.py`'s phi/eta stroke-plane formulas on purpose
(task spec: "不依赖wing_angles.py那套phi公式，保持这轮独立、自包含") -- this
is a much cruder, self-contained proxy built only from `part_label in
{"wing_L","wing_R"}` centroids and the *raw* (no continuity correction)
per-frame PCA `x_body` (`sequence_body_axis.csv`'s "before" column / the same
`orient_to_reference(major_axis, UP)` heuristic `body_frame.py` and
`build_sequence.py` already use). Reading the raw estimate rather than the
continuity-corrected one is deliberate: anchors are meant to be an
*independent* absolute reference the bidirectional reconstruction
(`bidirectional.py`) can trust without inheriting any prior frame's error, so
computing them from something that itself depends on prior frames would be
circular.

Proxy definition
-----------------
For each frame with `n_body_points >= build_sequence.MIN_BODY_POINTS` (i.e.
present in `sequence_body_axis.csv` as `failed=False`) and both wing
centroids non-empty:

- `vec_L = wing_L_centroid - body_cm`, `vec_R = wing_R_centroid - body_cm`
  (`body_cm` from `f_body_cm.csv`, the body-point-cloud centroid already
  cached by `f_residual_jitter_attribution.py`).
- `angle_L/R = angle(vec_L/R, x_body_before) in [0, 180]`. This is invariant
  to `x_body_before`'s sign (perpendicular-ness doesn't care which end of the
  body axis you measure from: if the true angle to `+x_body` is 90, the angle
  to `-x_body` is also 90) -- so the proxy sidesteps `x_body_before`'s own
  sign-flip problem entirely, it only uses it as an *axis*, not a direction.
- `angle_dev = mean(|angle_L - 90|, |angle_R - 90|)` -- how far both wings
  are, on average, from perpendicular-to-body ("T" shape).
- `wing_span = |wing_L_centroid - wing_R_centroid|` -- large when both wings
  are extended away from the body, small when folded/overlapping.

Thresholds (both data-driven quantiles of the actual 640-frame distribution,
not hand-picked degrees/lengths -- see task spec "按分布分位数定，不要拍脑袋
写死一个角度数"):
- `ANGLE_DEV_QUANTILE = 0.25`: proxy fires on the closest-to-perpendicular
  quartile of frames (`angle_dev <= that quartile's value`).
- `WING_SPAN_QUANTILE = 0.75`: proxy also requires wing_span in the top
  quartile ("sufficiently extended"), computed on the same population (wing
  span and perpendicularity are somewhat independent signals of "wings
  spread wide"; requiring both quartiles simultaneously, not just one, is
  what actually narrows things down to a genuine T-pose instead of e.g. a
  folded-but-coincidentally-perpendicular wing).
Final anchor = proxy hit AND `eigval_ratio < EIGVAL_RATIO_ANCHOR_THRESHOLD`.

`EIGVAL_RATIO_ANCHOR_THRESHOLD = 0.2` is the one fixed (non-quantile) number
here, per task spec's own suggestion ("eigval_ratio足够低,比如<0.2"). It sits
below `flip_root_cause_check.py`'s measured *normal*-group eigval_ratio
median of 0.230 (see `b_jitter_by_bucket.py` docstring, which built its own
0.4 "high-degenerate" cutoff off the same numbers) -- i.e. anchors are
required to be in the better half of frames that already look "normal", not
merely under the high-degenerate cutoff. This is deliberately stricter than
`b_jitter_by_bucket.py`'s 0.4 because an anchor is meant to be an unquestioned
ground truth for its whole segment, not just "not obviously degenerate".

Only reads `sequence_body_axis.csv` / `f_body_cm.csv` (via
`build_sequence.py` / `f_residual_jitter_attribution.py`, both read-only
dependencies, re-running them if their cache is missing) and `_labeled.csv`
(via `identity_flip_stats.py`'s `discover_labeled_frames`/`load_labeled`/
`frame_wing_centroids`). Does not touch `postprocessing/labeling/` or
`postprocessing/kinematics/`'s existing files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.calc_kinematics import DEFAULT_DATASET_ROOT  # noqa: E402
from postprocessing.kinematics.correct_body_axis.build_sequence import (  # noqa: E402
    SEQUENCE_CSV,
    run as build_sequence_run,
)
from postprocessing.kinematics.correct_body_axis.diag.f_residual_jitter_attribution import (  # noqa: E402
    BODY_CM_CSV,
    compute_body_cm_all,
)
from postprocessing.labeling.motion.diag.identity_flip_stats import (  # noqa: E402
    discover_labeled_frames,
    frame_wing_centroids,
    load_labeled,
)

OUT_DIR = Path(__file__).resolve().parent / "diag"
ANCHORS_CSV = OUT_DIR / "g_anchors.csv"

ANGLE_DEV_QUANTILE = 0.25
WING_SPAN_QUANTILE = 0.75
EIGVAL_RATIO_ANCHOR_THRESHOLD = 0.2

MAX_ANCHOR_GAP = 50
"""Segments between consecutive anchors wider than this (in sequence
*position*, i.e. count of successful frames, not raw frame_id difference --
matches `build_sequence.py`'s existing convention that `angle_to_prev_deg`
etc. are computed over the position-adjacent chain) get no bidirectional
coverage (see `bidirectional.py`), per task spec's suggested "~50帧" ceiling.
Kept here (not in `bidirectional.py`, which is data-agnostic) because it is a
property of *this* anchor detector's expected spacing, not of the merge
algorithm itself.

Checked against the actual 640-frame anchor spacing (`g_real_data_timeseries.py`
prints/uses this): anchors cluster into ~18 bursts (consecutive-frame runs),
and the *inter-burst* gaps are clearly bimodal -- {2,2,2,2,3,3,3,4,31,33,48,
48,51,75,78,79,81} frames -- with a clean break between 51 and 75. 50 sits
exactly in that break, so it is not an arbitrary round number for this
dataset: it is the natural boundary between "another burst of the same
roughly-periodic cycle" (<=51) and "the cycle skipped a beat, T-pose wasn't
detected for an unusually long stretch" (>=75). It was deliberately *not*
retuned upward after seeing that both `g_real_data_timeseries.py`'s named
windows (f513-520, f313-320) fall inside two of those >=75 gaps and therefore
get zero bidirectional coverage under this threshold -- see that script's
module docstring and the round's summary for why that's reported as a
limitation rather than fixed by raising this constant post hoc."""


def load_sequence() -> pd.DataFrame:
    if not SEQUENCE_CSV.exists():
        print(f"[anchor_detect] {SEQUENCE_CSV} 不存在，现算一遍 build_sequence")
        return build_sequence_run()
    return pd.read_csv(SEQUENCE_CSV)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    cos_a = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def compute_wing_geometry(seq_df: pd.DataFrame, cm_df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Per-frame angle_L/angle_R/angle_dev/wing_span, only for frames present
    in both `seq_df` (not failed) and `cm_df` (has a cached body_cm)."""
    frame_paths = discover_labeled_frames(root)
    cm_lookup = cm_df.set_index("frame_id")[["cm_x", "cm_y", "cm_z"]]

    rows = []
    for _, row in seq_df.loc[~seq_df["failed"]].iterrows():
        frame_id = int(row["frame_id"])
        if frame_id not in frame_paths or frame_id not in cm_lookup.index:
            continue
        df = load_labeled(frame_paths[frame_id])
        wing_L_cm, wing_R_cm = frame_wing_centroids(df)
        if wing_L_cm is None or wing_R_cm is None:
            rows.append({"frame_id": frame_id, "has_both_wings": False,
                         "angle_L_deg": np.nan, "angle_R_deg": np.nan,
                         "angle_dev_deg": np.nan, "wing_span": np.nan})
            continue

        body_cm = cm_lookup.loc[frame_id].to_numpy(dtype=float)
        x_body_before = row[["x_before_x", "x_before_y", "x_before_z"]].to_numpy(dtype=float)

        vec_L = wing_L_cm - body_cm
        vec_R = wing_R_cm - body_cm
        angle_L = angle_deg(vec_L, x_body_before)
        angle_R = angle_deg(vec_R, x_body_before)
        angle_dev = float(np.mean([abs(angle_L - 90.0), abs(angle_R - 90.0)]))
        wing_span = float(np.linalg.norm(wing_L_cm - wing_R_cm))

        rows.append({"frame_id": frame_id, "has_both_wings": True,
                     "angle_L_deg": angle_L, "angle_R_deg": angle_R,
                     "angle_dev_deg": angle_dev, "wing_span": wing_span})
    return pd.DataFrame(rows)


def detect_anchors(seq_df: pd.DataFrame, geom_df: pd.DataFrame) -> pd.DataFrame:
    merged = geom_df.merge(seq_df[["frame_id", "eigval_ratio"]], on="frame_id", how="left")
    valid = merged[merged["has_both_wings"]]

    angle_dev_threshold = float(np.quantile(valid["angle_dev_deg"], ANGLE_DEV_QUANTILE))
    wing_span_threshold = float(np.quantile(valid["wing_span"], WING_SPAN_QUANTILE))

    merged["angle_dev_threshold"] = angle_dev_threshold
    merged["wing_span_threshold"] = wing_span_threshold
    merged["proxy_tpose"] = (
        merged["has_both_wings"]
        & (merged["angle_dev_deg"] <= angle_dev_threshold)
        & (merged["wing_span"] >= wing_span_threshold)
    )
    merged["is_anchor"] = merged["proxy_tpose"] & (merged["eigval_ratio"] < EIGVAL_RATIO_ANCHOR_THRESHOLD)

    print(f"[anchor_detect] angle_dev_deg <= {angle_dev_threshold:.3f} deg "
          f"(quantile={ANGLE_DEV_QUANTILE}, n_valid={len(valid)})")
    print(f"[anchor_detect] wing_span >= {wing_span_threshold:.6g} "
          f"(quantile={WING_SPAN_QUANTILE}, n_valid={len(valid)})")
    print(f"[anchor_detect] eigval_ratio < {EIGVAL_RATIO_ANCHOR_THRESHOLD}")
    return merged


def spacing_stats(anchor_frame_ids: list[int]) -> dict:
    if len(anchor_frame_ids) < 2:
        return {"n_anchors": len(anchor_frame_ids), "mean_gap": float("nan"),
                "median_gap": float("nan"), "std_gap": float("nan"),
                "min_gap": float("nan"), "max_gap": float("nan")}
    gaps = np.diff(sorted(anchor_frame_ids))
    return {"n_anchors": len(anchor_frame_ids), "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)), "std_gap": float(np.std(gaps)),
            "min_gap": float(np.min(gaps)), "max_gap": float(np.max(gaps))}


def run(root: Path = DEFAULT_DATASET_ROOT) -> pd.DataFrame:
    seq_df = load_sequence()
    cm_df = compute_body_cm_all(seq_df, root) if not BODY_CM_CSV.exists() else pd.read_csv(BODY_CM_CSV)

    geom_df = compute_wing_geometry(seq_df, cm_df, root)
    result_df = detect_anchors(seq_df, geom_df)

    n_no_wings = int((~result_df["has_both_wings"]).sum())
    n_total = len(result_df)
    print(f"\n{'=' * 78}\ng1) T-pose 锚点检测\n{'=' * 78}")
    print(f"  总帧数(成功帧): {n_total}, 缺任一侧wing质心(跳过proxy判定): {n_no_wings}")

    anchor_frame_ids = result_df.loc[result_df["is_anchor"], "frame_id"].astype(int).tolist()
    stats = spacing_stats(anchor_frame_ids)
    print(f"  锚点帧数: {stats['n_anchors']} / {n_total} "
          f"({100.0 * stats['n_anchors'] / n_total:.1f}%)")
    if stats["n_anchors"] >= 2:
        print(f"  锚点间隔(frame_id差, 按frame_id排序后相邻锚点): "
              f"mean={stats['mean_gap']:.1f} median={stats['median_gap']:.1f} "
              f"std={stats['std_gap']:.1f} min={stats['min_gap']:.0f} max={stats['max_gap']:.0f}")
        n_gap_over_max = int(np.sum(np.diff(sorted(anchor_frame_ids)) > MAX_ANCHOR_GAP))
        print(f"  锚点间隔 > MAX_ANCHOR_GAP({MAX_ANCHOR_GAP}) 的段数: {n_gap_over_max}")
    print(f"  锚点frame_id: {anchor_frame_ids}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(ANCHORS_CSV, index=False)
    print(f"\n  csv -> {ANCHORS_CSV}")

    return result_df


if __name__ == "__main__":
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_ROOT
    run(dataset_root)
