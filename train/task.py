"""单帧训练任务：ns-train splatfacto -> ns-export splat.ply -> 诊断指标 -> 删 checkpoint / tfevents。

输出目录遵循 nerfstudio 的约定：<out_root>/<name>/<param_set>/f####/splatfacto/<时间戳>/
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from plyfile import PlyData

REPO = Path(__file__).resolve().parent.parent
METHOD = "splatfacto"
PARAM_SETS = json.loads((Path(__file__).resolve().parent / "param_sets.json").read_text())


def frame_dir(cfg: dict, frame: int) -> Path:
    return (REPO / cfg["data_dir"]).resolve() / f"f{frame:04d}"


def experiment_name(cfg: dict, frame: int) -> str:
    return f"{cfg['name']}/{cfg['param_set']}/f{frame:04d}"


def latest_splat_dir(cfg: dict, frame: int) -> Path | None:
    d = (REPO / cfg["out_root"]).resolve() / experiment_name(cfg, frame) / METHOD
    subdirs = sorted(d.iterdir()) if d.is_dir() else []
    return subdirs[-1] if subdirs else None


def train_cmd(cfg: dict, frame: int) -> list[str]:
    return [
        "ns-train", METHOD,
        "--data", str(frame_dir(cfg, frame)),
        "--vis", "tensorboard",
        "--max-num-iterations", str(cfg["max_iters"]),
        "--pipeline.model.background-color", "white",
        "--experiment-name", experiment_name(cfg, frame),
        "--output-dir", str((REPO / cfg["out_root"]).resolve()),
    ] + PARAM_SETS[cfg["param_set"]] + [
        "nerfstudio-data", "--eval-mode", "all",
    ]


def export_cmd(splat_dir: Path) -> list[str]:
    return ["ns-export", "gaussian-splat", "--load-config", str(splat_dir / "config.yml"),
            "--output-dir", str(splat_dir)]


def scale_ratio_median(ply_path: Path) -> tuple[int, float]:
    """(高斯数, 每个高斯 max(scale)/min(scale) 的中位数)；比值越大越"尖刺"。"""
    v = PlyData.read(str(ply_path))["vertex"]
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1))
    ratios = scales.max(axis=-1) / np.clip(scales.min(axis=-1), 1e-12, None)
    return len(ratios), float(np.median(ratios))


def is_done(cfg: dict, frame: int) -> bool:
    d = latest_splat_dir(cfg, frame)
    if d is None or not (d / "splat.ply").exists():
        return False
    try:
        scale_ratio_median(d / "splat.ply")
    except Exception:  # noqa: BLE001  导出不完整的 ply 视为未完成
        return False
    return True


def run(cfg: dict, frame: int, env: dict) -> dict:
    """训练 + 导出一帧；失败时抛异常。返回诊断指标。"""
    proc = subprocess.run(train_cmd(cfg, frame), cwd=str(REPO), env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"ns-train exit={proc.returncode} for {experiment_name(cfg, frame)}")
    splat_dir = latest_splat_dir(cfg, frame)
    if splat_dir is None:
        raise RuntimeError(f"no {METHOD} output dir for {experiment_name(cfg, frame)}")

    subprocess.run(export_cmd(splat_dir), cwd=str(REPO), env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not (splat_dir / "splat.ply").exists():
        raise RuntimeError(f"ns-export did not produce splat.ply for {experiment_name(cfg, frame)}")
    n_gaussians, ratio_median = scale_ratio_median(splat_dir / "splat.ply")

    for tf_event in splat_dir.glob("events.out.tfevents.*"):
        tf_event.unlink()
    shutil.rmtree(splat_dir / "nerfstudio_models", ignore_errors=True)
    return {"n_gaussians": n_gaussians, "scale_ratio_median": ratio_median, "splat_dir": str(splat_dir)}
