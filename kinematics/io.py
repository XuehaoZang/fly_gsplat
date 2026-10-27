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
    "chord_L_x", "chord_L_y", "chord_L_z",
    "chord_R_x", "chord_R_y", "chord_R_z",
    "status",
]
"""angles.csv 的列。角度单位为度，在实验室坐标系（+z 朝上）里定义；sp_normal 是拍动平面法向，span 是 φ/θ 所用的
翅展单位向量，chord 是 η 所用的定号弦向，这些向量写回输入点云的坐标系，方便叠加显示。"""

DATA_TO_LAB = {
    "+z": np.eye(3),
    "-z": np.diag([1.0, -1.0, -1.0]),
}
"""输入点云坐标系 -> 实验室坐标系（+z 朝上）。真实 4 相机的 easyWand 坐标系里 -z 才朝上（绕 x 轴转 180°）；
合成数据是 +z 朝上。两种矩阵都是自己的逆。"""


@dataclass
class Gaussians:
    """一个部分的完整高斯（实验室坐标系，米）。"""
    xyz: np.ndarray
    """(N,3) 中心。"""
    cov: np.ndarray
    """(N,3,3) 1σ 协方差 Σ = R S² Rᵀ。"""
    opacity: np.ndarray
    """(N,) sigmoid 后的不透明度。"""


@dataclass
class FrameParts:
    """一帧中 if_keep=1 的各部分点（米，(N,3)）。gauss 只在 load_parts(with_cov=True) 时给出。"""
    body: np.ndarray
    wing_L: np.ndarray
    wing_R: np.ndarray
    gauss: dict[str, Gaussians] | None = None
    low_conf: bool = False
    """segment 的 confidence 列里有 low（两翅粘连、被强行切开）。"""


def discover_frames(param_dir: Path) -> dict[int, Path]:
    """frame_id -> `_labeled.csv`。同一帧有多个时间戳目录时取最新的。"""
    by_frame: dict[int, list[Path]] = {}
    for p in Path(param_dir).glob(LABELED_GLOB):
        m = _FRAME_RE.search(p.name)
        if m:
            by_frame.setdefault(int(m.group(1)), []).append(p)
    return {fid: sorted(paths)[-1] for fid, paths in sorted(by_frame.items())}


SPLAT_XYZ_MATCH_TOL = 1e-6
"""splat.ply 换算后的中心与 CSV 的 x,y,z 逐行最大允许偏差（米），用来确认两者逐行对应。"""


def load_covariances(csv_dir: Path, xyz_csv: np.ndarray) -> np.ndarray:
    """同目录 splat.ply + dataparser_transforms.json -> 每行高斯的 1σ 协方差（标定坐标系，(N,3,3)）。

    第 3 步逐行写 CSV、不删点，所以 ply 和 CSV 行号一一对应；这里用中心坐标再核对一遍。
    """
    from plyfile import PlyData

    from gaussians.extract_features import load_dataparser_transform, quat_to_rotmat

    v = PlyData.read(str(csv_dir / "splat.ply"))["vertex"]
    R_ns, t_ns, s_ns = load_dataparser_transform(csv_dir / "dataparser_transforms.json")
    xyz_ns = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    xyz = (R_ns.T @ (xyz_ns.T / s_ns - t_ns[:, None])).T
    if xyz.shape != xyz_csv.shape or np.abs(xyz - xyz_csv).max() > SPLAT_XYZ_MATCH_TOL:
        raise ValueError(f"{csv_dir}: splat.ply rows do not match the CSV rows")

    scale = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float64)) / s_ns
    quat = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float64)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    axes = R_ns.T @ quat_to_rotmat(quat)  # 列为各局部轴，已转到标定坐标系
    return np.einsum("nik,nk,njk->nij", axes, scale**2, axes)


def load_parts(csv_path: Path, to_lab: np.ndarray | None = None, with_cov: bool = False) -> FrameParts:
    """读一帧的点并转到实验室坐标系（to_lab 为 None 时不转）。with_cov 时另外读出各部分的完整高斯。"""
    cols = ("x", "y", "z", "part_label", "if_keep", "confidence") + (("opacity",) if with_cov else ())
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    keep = df["if_keep"].astype(bool).to_numpy() if "if_keep" in df.columns else np.ones(len(df), bool)
    A = np.eye(3) if to_lab is None else to_lab
    xyz_all = df[["x", "y", "z"]].to_numpy(dtype=float)
    cov_all = load_covariances(Path(csv_path).parent, xyz_all) if with_cov else None

    def mask(label: str) -> np.ndarray:
        return keep & (df["part_label"] == label).to_numpy()

    def xyz(label: str) -> np.ndarray:
        return xyz_all[mask(label)] @ A.T

    gauss = None
    if with_cov:
        gauss = {
            label: Gaussians(
                xyz=xyz(label),
                cov=A @ cov_all[mask(label)] @ A.T,
                opacity=df.loc[mask(label), "opacity"].to_numpy(dtype=float),
            )
            for label in ("body", "wing_L", "wing_R")
        }
    low_conf = bool((df["confidence"] == "low").any()) if "confidence" in df.columns else False
    return FrameParts(body=xyz("body"), wing_L=xyz("wing_L"), wing_R=xyz("wing_R"), gauss=gauss, low_conf=low_conf)
