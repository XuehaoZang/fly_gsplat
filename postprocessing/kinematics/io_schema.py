"""Per-point (input) and per-frame (output) data contract for T4 kinematics.

Column names, the mandatory/optional split, and the `part_label` vocabulary
are fixed by `reference/calc_kinematics.md` §1 ("Input / output contract").
All lengths are **meters** (§0). No angle math lives here — this module only
defines and validates the data shapes; see S2+ for body/wing angle
estimation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Input schema (per point) — calc_kinematics.md §1
# ---------------------------------------------------------------------------

INPUT_COLUMNS: list[str] = [
    "x", "y", "z",
    "dist_to_centroid", "dist_to_principal_axis",
    "R", "G", "B", "color_oob",
    "opacity",
    "scale_phys_0", "scale_phys_1", "scale_phys_2", "scale_ratio",
    "linearity", "planarity", "sphericity",
    "orientation_x", "orientation_y", "orientation_z",
    "local_density",
    "if_keep",
    "part_label",
]
"""Full per-point column order, exactly as listed in calc_kinematics.md §1."""

MANDATORY_INPUT_COLUMNS: tuple[str, ...] = ("x", "y", "z", "part_label")
"""§1: "Core fields T4 relies on" — must be present in every input CSV."""

OPTIONAL_INPUT_COLUMNS: tuple[str, ...] = tuple(
    c for c in INPUT_COLUMNS if c not in MANDATORY_INPUT_COLUMNS
)
"""§1: everything else, including `if_keep` (added by T2) — used opportunistically."""

PART_LABELS: frozenset[str] = frozenset({"body", "wing_L", "wing_R"})
"""§1: `part_label` vocabulary, assigned by T3."""

UNITS = "meters"
"""§0: all lengths in this schema are meters (wing length ~= 2.5-3 mm)."""


def load_frame(csv_path: str | Path) -> pd.DataFrame:
    """Load one per-point CSV (T1 features + T2 `if_keep` + T3 `part_label`).

    Only checks that `MANDATORY_INPUT_COLUMNS` are present; the remaining
    `OPTIONAL_INPUT_COLUMNS` may be absent. Rows are never dropped — filtering
    on `if_keep` / `part_label` is the caller's job (see `get_part`).

    Raises:
        ValueError: listing every missing mandatory column, if any.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in MANDATORY_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: missing mandatory column(s) {missing}; "
            f"required = {list(MANDATORY_INPUT_COLUMNS)}"
        )
    return df


def get_part(df: pd.DataFrame, label: str, apply_if_keep: bool = True) -> np.ndarray:
    """Return an (N, 3) xyz array for the rows where `part_label == label`.

    If `apply_if_keep` is True and an `if_keep` column exists, rows with
    `if_keep == False` (T2 floater filtering, §1) are excluded.
    """
    if label not in PART_LABELS:
        raise ValueError(f"unknown part label {label!r}; expected one of {sorted(PART_LABELS)}")
    mask = df["part_label"] == label
    if apply_if_keep and "if_keep" in df.columns:
        mask &= df["if_keep"].astype(bool)
    return df.loc[mask, ["x", "y", "z"]].to_numpy(dtype=float)


def get_part_columns(
    df: pd.DataFrame, label: str, columns: tuple[str, ...] | list[str], apply_if_keep: bool = True
) -> np.ndarray:
    """Return an `(N, len(columns))` float array for arbitrary `columns` at
    `part_label == label` rows, with the exact same row selection/order as
    `get_part` -- so e.g. `get_part(df, "wing_L")` and
    `get_part_columns(df, "wing_L", ["orientation_x", "orientation_y",
    "orientation_z"])` line up row-for-row. Used opportunistically by S4b's
    Gaussian-normal path (`chord.py`) for `orientation_*`/`planarity`, which
    `get_part` itself doesn't carry.
    """
    if label not in PART_LABELS:
        raise ValueError(f"unknown part label {label!r}; expected one of {sorted(PART_LABELS)}")
    mask = df["part_label"] == label
    if apply_if_keep and "if_keep" in df.columns:
        mask &= df["if_keep"].astype(bool)
    return df.loc[mask, list(columns)].to_numpy(dtype=float)


def body_xyz(df: pd.DataFrame, apply_if_keep: bool = True) -> np.ndarray:
    """`get_part(df, "body", ...)`."""
    return get_part(df, "body", apply_if_keep)


def wingL_xyz(df: pd.DataFrame, apply_if_keep: bool = True) -> np.ndarray:
    """`get_part(df, "wing_L", ...)`."""
    return get_part(df, "wing_L", apply_if_keep)


def wingR_xyz(df: pd.DataFrame, apply_if_keep: bool = True) -> np.ndarray:
    """`get_part(df, "wing_R", ...)`."""
    return get_part(df, "wing_R", apply_if_keep)


# ---------------------------------------------------------------------------
# Output schema (one row per frame) — calc_kinematics.md §1 / §6
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: list[str] = [
    "frame_id",
    "yaw", "pitch", "roll",
    "phi_L", "theta_L", "eta_L",
    "phi_R", "theta_R", "eta_R",
    "sp_normal_x", "sp_normal_y", "sp_normal_z",
    "chord_conf_L", "chord_conf_R",
]
"""Per-frame output column order (§1: "one row per frame")."""


@dataclass
class OutputRow:
    """One frame of T4 output: body pose + per-wing stroke angles (§1, §6).

    `yaw/pitch/roll` (§3) and `phi_*/theta_*/eta_*` (§4, §5) are degrees, in
    the stroke-plane frame per §0 ("Angle output frame"). `sp_normal_*` is the
    unit stroke-plane normal `n_sp` (§2, step 4). `chord_conf_*` is the S4
    chord-fit confidence in [0, 1] (§5), NaN when not (yet) computed.
    """

    frame_id: int
    yaw: float
    pitch: float
    roll: float
    phi_L: float
    theta_L: float
    eta_L: float
    phi_R: float
    theta_R: float
    eta_R: float
    sp_normal_x: float
    sp_normal_y: float
    sp_normal_z: float
    chord_conf_L: float
    chord_conf_R: float

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def empty_output_row(frame_id: int = -1) -> OutputRow:
    """`OutputRow` for `frame_id` with every angle/normal/confidence as NaN."""
    nan = float("nan")
    return OutputRow(
        frame_id=frame_id,
        yaw=nan, pitch=nan, roll=nan,
        phi_L=nan, theta_L=nan, eta_L=nan,
        phi_R=nan, theta_R=nan, eta_R=nan,
        sp_normal_x=nan, sp_normal_y=nan, sp_normal_z=nan,
        chord_conf_L=nan, chord_conf_R=nan,
    )
