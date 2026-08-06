"""Step 4: parallel (independent of the count-vs-curvature LE/TE comparison
in `real_data_validation.py`) A/B comparison of `chord.py::estimate_chord`'s
S4b contamination guard (`use_gaussian_normals=True, robust=True`) against
the S4a baseline (`pipeline.py`'s current defaults), on the same real
dataset -- does the *already-implemented-but-not-enabled* contamination
guard change eta's wrap-crossing behavior, independent of anything about the
LE/TE straightness judge?

`pipeline.py` never passes `robust=`/`use_gaussian_normals=` (S5 hook not
built yet, per `chord.py`'s own module docstring), so this script drives
`chord.estimate_chord` directly, per frame/side, for both configs, reusing
`io_schema.get_part_columns` for `orientation_*`/`planarity` (present in the
real CSVs -- verified: `gaussian_features_*_labeled.csv` has
`orientation_x/y/z`/`planarity` columns) and
`diagnostics.py::circular_delta_deg`/`delta_report` for the eta delta stats
(not re-derived).

Run: python -m postprocessing.kinematics.correct_wing_pitch.s4b_comparison
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

from postprocessing.kinematics import body_frame as bf  # noqa: E402
from postprocessing.kinematics import chord as ch  # noqa: E402
from postprocessing.kinematics import diagnostics as diag_mod  # noqa: E402
from postprocessing.kinematics import io_schema  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"
REAL_DATASET_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
FRAME_GLOB = "f*/splatfacto-checkpoint/*/*_labeled.csv"
_FRAME_DIR_RE = re.compile(r"^f(\d+)$")
_ORIENT_COLS = ["orientation_x", "orientation_y", "orientation_z"]
FLIP_THRESHOLD_DEG = 150.0
"""|circular_delta| above this is counted as a "wrap-crossing" (eta flipping
by ~180 deg between adjacent frames, the chord-sign-bistability signature
from `reference/s6b_real_data_diagnostics_findings.md` Top issue #2) --
distinct from `diagnostics.delta_report`'s "wrap_frames" (which flags a raw
`np.diff` *artifact* of the atan2 range, not a real bistable jump)."""


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


def _check_orientation_columns_available(csv_path: Path) -> bool:
    df = io_schema.load_frame(csv_path)
    needed = set(_ORIENT_COLS + ["planarity"])
    return needed.issubset(df.columns)


def _run_both_configs(frames: list[tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for frame_id, csv_path in frames:
        try:
            df = io_schema.load_frame(csv_path)
            body_xyz = io_schema.get_part(df, "body")
            wingL_xyz = io_schema.get_part(df, "wing_L")
            wingR_xyz = io_schema.get_part(df, "wing_R")
            for label, xyz in (("body", body_xyz), ("wing_L", wingL_xyz), ("wing_R", wingR_xyz)):
                if xyz.shape[0] < 10:
                    raise ValueError(f"{label}:too_few_points")
            frame = bf.estimate_body_frame(body_xyz, wingL_xyz, wingR_xyz)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(frame_id=frame_id, status=f"body:{e}"))
            continue

        row = dict(frame_id=frame_id, status="ok")
        for side, wing_xyz, suffix in (("wing_L", wingL_xyz, "L"), ("wing_R", wingR_xyz, "R")):
            try:
                orientation = io_schema.get_part_columns(df, side, _ORIENT_COLS)
                planarity = io_schema.get_part_columns(df, side, ["planarity"])[:, 0]

                baseline = ch.estimate_chord(wing_xyz, frame, side)
                enhanced = ch.estimate_chord(
                    wing_xyz, frame, side, robust=True, use_gaussian_normals=True,
                    orientation=orientation, planarity=planarity,
                )
            except Exception as e:  # noqa: BLE001
                row[f"{side}_status"] = str(e)
                row[f"eta_baseline_{suffix}"] = float("nan")
                row[f"eta_enhanced_{suffix}"] = float("nan")
                continue
            row[f"eta_baseline_{suffix}"] = baseline.eta
            row[f"eta_enhanced_{suffix}"] = enhanced.eta
            row[f"chord_conf_baseline_{suffix}"] = baseline.chord_conf
            row[f"chord_conf_enhanced_{suffix}"] = enhanced.chord_conf
            row[f"n_rejected_{suffix}"] = int(enhanced.rejected_mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _wrap_crossings(eta: np.ndarray) -> tuple[int, np.ndarray]:
    cd = np.abs(diag_mod.circular_delta_deg(eta))
    return int(np.sum(cd > FLIP_THRESHOLD_DEG)), cd


def _plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col_idx, suffix in enumerate(("L", "R")):
        ax = axes[0, col_idx]
        ax.plot(df["frame_id"], df[f"eta_baseline_{suffix}"], marker=".", ms=3, lw=1, label="baseline (S4a)", color="tab:blue")
        ax.plot(df["frame_id"], df[f"eta_enhanced_{suffix}"], marker=".", ms=3, lw=1, label="enhanced (S4b)", color="tab:red", alpha=0.7)
        ax.set_title(f"eta_{suffix}: baseline vs S4b-enhanced")
        ax.set_ylabel("eta (deg)")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, col_idx]
        _, cd_base = _wrap_crossings(df[f"eta_baseline_{suffix}"].to_numpy())
        _, cd_enh = _wrap_crossings(df[f"eta_enhanced_{suffix}"].to_numpy())
        ax.plot(df["frame_id"].to_numpy()[1:], cd_base, marker=".", ms=3, lw=1, label="baseline |Δeta|", color="tab:blue")
        ax.plot(df["frame_id"].to_numpy()[1:], cd_enh, marker=".", ms=3, lw=1, label="enhanced |Δeta|", color="tab:red", alpha=0.7)
        ax.axhline(FLIP_THRESHOLD_DEG, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("frame_id")
        ax.set_ylabel("|Δeta| (deg)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("S4b contamination-guard A/B: eta and |Δeta|, baseline vs enhanced")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not REAL_DATASET_ROOT.exists():
        print(f"ERROR: real dataset root not found: {REAL_DATASET_ROOT}")
        sys.exit(1)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    frames = _discover_frames(REAL_DATASET_ROOT, FRAME_GLOB)
    if not frames:
        print("ERROR: no frame CSVs discovered")
        sys.exit(1)

    if not _check_orientation_columns_available(frames[0][1]):
        msg = (f"orientation_*/planarity columns not found in {frames[0][1]}; "
               "S4b's use_gaussian_normals path cannot run -- skipping step 4.")
        print(msg)
        (DIAG_DIR / "04_s4b_comparison_summary.md").write_text(
            "# S4b contamination-guard comparison\n\nSKIPPED: " + msg + "\n"
        )
        return

    print(f"orientation_*/planarity columns present in {frames[0][1].name} -- running S4b A/B comparison")
    df = _run_both_configs(frames)
    df.to_csv(DIAG_DIR / "04_s4b_comparison_raw.csv", index=False)
    ok = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"{len(df)} frames total, {len(ok)} body-frame ok")

    _plot_comparison(ok, DIAG_DIR / "04_s4b_eta_and_delta.png")

    lines = ["# S4b contamination-guard comparison (parallel to the count-vs-curvature LE/TE analysis)\n"]
    lines.append(f"Dataset: `{REAL_DATASET_ROOT}`, {len(df)} frames discovered, {len(ok)} with a valid body frame.\n")
    lines.append("Baseline: `chord.estimate_chord(...)` defaults (`robust=False, use_gaussian_normals=False` -- "
                 "byte-identical to S4a, what `pipeline.py` currently runs). Enhanced: `robust=True, "
                 "use_gaussian_normals=True` with real `orientation_*`/`planarity` columns.\n")

    for suffix in ("L", "R"):
        base = ok[f"eta_baseline_{suffix}"].dropna().to_numpy()
        enh = ok[f"eta_enhanced_{suffix}"].dropna().to_numpy()
        n_flip_base, cd_base = _wrap_crossings(ok[f"eta_baseline_{suffix}"].to_numpy())
        n_flip_enh, cd_enh = _wrap_crossings(ok[f"eta_enhanced_{suffix}"].to_numpy())
        n_rejected_total = int(ok[f"n_rejected_{suffix}"].sum()) if f"n_rejected_{suffix}" in ok else 0
        n_rejected_frames = int((ok[f"n_rejected_{suffix}"] > 0).sum()) if f"n_rejected_{suffix}" in ok else 0

        lines.append(f"## Side {suffix}\n")
        lines.append(f"- wrap-crossings (|Δeta| > {FLIP_THRESHOLD_DEG:.0f} deg): "
                     f"baseline={n_flip_base}/{len(cd_base)} transitions, enhanced={n_flip_enh}/{len(cd_enh)} transitions")
        lines.append(f"- |Δeta| median: baseline={np.median(cd_base):.2f} deg, enhanced={np.median(cd_enh):.2f} deg")
        lines.append(f"- |Δeta| p95: baseline={np.percentile(cd_base, 95):.2f} deg, enhanced={np.percentile(cd_enh, 95):.2f} deg")
        lines.append(f"- |Δeta| max: baseline={np.max(cd_base):.2f} deg, enhanced={np.max(cd_enh):.2f} deg")
        lines.append(f"- frames with >=1 contaminant point rejected: {n_rejected_frames}/{len(ok)}, "
                     f"total points rejected across all frames: {n_rejected_total}")
        lines.append(f"- mean chord_conf: baseline={ok[f'chord_conf_baseline_{suffix}'].mean():.4f}, "
                     f"enhanced={ok[f'chord_conf_enhanced_{suffix}'].mean():.4f}\n")

    (DIAG_DIR / "04_s4b_comparison_summary.md").write_text("\n".join(lines) + "\n")
    print(f"written: {DIAG_DIR / '04_s4b_eta_and_delta.png'}, {DIAG_DIR / '04_s4b_comparison_summary.md'}")


if __name__ == "__main__":
    main()
