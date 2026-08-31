"""Pools real per-part appearance-feature values (Gaussian shape + color +
opacity) from every real `_labeled.csv` under `outputs/`, `if_keep=True`
rows only, into a single reference table `scene.py`'s appearance-resampling
step bootstraps from.

Why this exists: comparing `outputs/.../f0265/.../gaussian_features_f0265_
labeled.csv` against `mock.py`'s synthetic points showed `mock.py`'s
idealized-flat-sheet wing model is far from what a real Gaussian-splat
reconstruction of a flapping wing actually looks like -- e.g. real
planarity is body~0.20 vs wing~0.23 (barely separable) while `mock.py`
produces body~0.18 vs wing~0.89; real `R` is body~0.17 vs wing~0.33-0.36
(the actual discriminating signal `kmeans_split.py`'s v2 seed rule uses)
while `mock.py`'s `R` was on a 60-220 scale instead of `gaussian_features.py`'s
real `[0,1]` range entirely. Rather than hand-picking new target means, this
script pools real per-part *empirical* distributions across many frames so
`scene.py` can bootstrap-sample real rows directly -- automatically correct
on every moment (mean, spread, tail fraction above the `opacity>=0.98` seed
threshold, etc.), not just a first moment match.

Pools ALL real `_labeled.csv` files found under `outputs/` (740 files /
640 distinct frames as of writing, spanning `ratio3_sh0_dense` and
`ctrl_009_002_8groups_100frames/G2b_G9`) -- reading them is cheap (~1s per
60 files) and more frames means the reference distribution is less prone to
one frame's idiosyncrasies (`s2/step2`'s prior write-up used only `f0265`).

Writes `real_appearance_reference.csv` next to this file, columns
`[part_label, lam1, lam2, lam3, opacity, R, G, B]` (`lam1>=lam2>=lam3`,
i.e. `scale_phys_0/1/2` sorted descending -- the same convention
`mock._linearity_planarity_sphericity` and `utils/gaussian_features.py`'s
own `planarity`/`scale_ratio` derivation use, so a bootstrapped triple
dropped straight into `scale_phys_0/1/2` reproduces the real derived
features exactly, not just approximately).

Run: python -m postprocessing.kinematics.simulate_gt.calibrate_appearance
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = Path(__file__).resolve().parent / "real_appearance_reference.csv"
SCALE_COLS = ["scale_phys_0", "scale_phys_1", "scale_phys_2"]
READ_COLS = ["part_label", "if_keep", *SCALE_COLS, "opacity", "R", "G", "B"]
VALID_PARTS = ("body", "wing_L", "wing_R")


def _pool_one(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=READ_COLS)
    df = df[df["if_keep"].astype(bool)]
    scale_sorted = np.sort(df[SCALE_COLS].to_numpy(dtype=float), axis=1)[:, ::-1]
    return pd.DataFrame({
        "part_label": df["part_label"].to_numpy(),
        "lam1": scale_sorted[:, 0], "lam2": scale_sorted[:, 1], "lam3": scale_sorted[:, 2],
        "opacity": df["opacity"].to_numpy(dtype=float),
        "R": df["R"].to_numpy(dtype=float), "G": df["G"].to_numpy(dtype=float), "B": df["B"].to_numpy(dtype=float),
    })


def main() -> None:
    paths = sorted(REPO_ROOT.glob("outputs/**/*_labeled.csv"))
    print(f"found {len(paths)} real _labeled.csv files under outputs/")

    pooled_parts = []
    n_failed = 0
    for p in paths:
        try:
            pooled_parts.append(_pool_one(p))
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            print(f"  skip {p}: {e}")

    pooled = pd.concat(pooled_parts, ignore_index=True)
    pooled = pooled[pooled["part_label"].isin(VALID_PARTS)].reset_index(drop=True)
    pooled.to_csv(OUT_PATH, index=False)

    print(f"pooled {len(pooled)} rows from {len(paths) - n_failed} files ({n_failed} failed to read)")
    print(pooled.groupby("part_label").size().to_string())
    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    main()
