"""第 5 步的输入输出：发现逐帧 `_labeled.csv`，读出 body / wing_L / wing_R 点。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABELED_GLOB = "f*/*/*/gaussian_features_f*_labeled.csv"
"""<param_set>/f####/<method>/<时间戳>/gaussian_features_f####_labeled.csv"""
_FRAME_RE = re.compile(r"gaussian_features_f(\d+)_labeled\.csv$")

OUTPUT_COLUMNS: list[str] = [
    "frame_id",
    "yaw", "pitch", "roll",
    "phi_L", "theta_L", "eta_L",
    "phi_R", "theta_R", "eta_R",
    "sp_normal_x", "sp_normal_y", "sp_normal_z",
    "chord_conf_L", "chord_conf_R",
    "span_L_x", "span_L_y", "span_L_z",
    "span_R_x", "span_R_y", "span_R_z",
    "status",
]
"""angles.csv 的列。角度单位为度；sp_normal 是拍动平面法向；span 是 φ/θ 所用的翅展单位向量。"""


@dataclass
class FrameParts:
    """一帧中 if_keep=1 的各部分点（米，(N,3)）。"""
    body: np.ndarray
    wing_L: np.ndarray
    wing_R: np.ndarray


def discover_frames(param_dir: Path) -> dict[int, Path]:
    """frame_id -> `_labeled.csv`。同一帧有多个时间戳目录时取最新的。"""
    by_frame: dict[int, list[Path]] = {}
    for p in Path(param_dir).glob(LABELED_GLOB):
        m = _FRAME_RE.search(p.name)
        if m:
            by_frame.setdefault(int(m.group(1)), []).append(p)
    return {fid: sorted(paths)[-1] for fid, paths in sorted(by_frame.items())}


def load_parts(csv_path: Path) -> FrameParts:
    df = pd.read_csv(csv_path, usecols=lambda c: c in ("x", "y", "z", "part_label", "if_keep"))
    kept = df[df["if_keep"].astype(bool)] if "if_keep" in df.columns else df

    def xyz(label: str) -> np.ndarray:
        return kept.loc[kept["part_label"] == label, ["x", "y", "z"]].to_numpy(dtype=float)

    return FrameParts(body=xyz("body"), wing_L=xyz("wing_L"), wing_R=xyz("wing_R"))
