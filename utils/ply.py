import subprocess
from pathlib import Path
import open3d as o3d
import numpy as np


def export_splat(splat_dir: Path) -> None:
    config_path = splat_dir / "config.yml"
    if not (splat_dir / "splat.ply").exists():
        subprocess.run([
            "ns-export", "gaussian-splat",
            "--load-config", str(config_path),
            "--output-dir", str(splat_dir)
        ], check=True)


def load_ply(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


def print_stats(label: str, pts: np.ndarray) -> None:
    if len(pts) == 0:
        print(f"[{label}] EMPTY")
        return
    print(f"[{label}]")
    print(f"  n       = {len(pts)}")
    print(f"  center  = {pts.mean(axis=0)}")
    print(f"  min     = {pts.min(axis=0)}")
    print(f"  max     = {pts.max(axis=0)}")
    print(f"  extent  = {pts.max(axis=0) - pts.min(axis=0)}")

def unrescale(pts: np.ndarray, R_ns: np.ndarray, t_ns: np.ndarray, scale: float) -> np.ndarray:
    """把 dataparser 归一化坐标变换回原始物理坐标（transforms.json 空间）。"""
    return (R_ns.T @ (pts.T / scale - t_ns[:, None])).T

def characterize_sphere(pts: np.ndarray, expected_radius: float) -> None:
    center = pts.mean(axis=0)
    r = np.linalg.norm(pts - center, axis=1)
    print(f"[sphere check] n={len(pts)}")
    print(f"  radius: mean={r.mean():.6f}  std={r.std():.6f}  expected={expected_radius:.6f}")
    print(f"  radius min/max = {r.min():.6f} / {r.max():.6f}")

    cov = np.cov((pts - center).T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    axis_len = np.sqrt(eigvals) * 2  # 近似轴长
    print(f"  PCA axis lengths (long->short) = {axis_len}")
    print(f"  anisotropy ratio (max/min) = {axis_len[0]/axis_len[2]:.3f}")

    hist, edges = np.histogram(r, bins=10, range=(0, expected_radius*1.2))
    print(f"  radius histogram (10 bins, 0~{expected_radius*1.2:.4f}):")
    for h, e in zip(hist, edges):
        print(f"    {e:.5f}: ({h})")