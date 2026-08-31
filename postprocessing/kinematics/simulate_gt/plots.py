"""Comparison plots for `simulate_gt`: body/wing angle time series (T3+T4
full pipeline vs exact-segmentation T4-only vs ground truth, three lines per
subplot, styled like `diagnostics.py`'s own `plot_body_angles`/
`plot_lr_overlay`), plus a one-viewing-angle point-cloud scatter showing the
cloud's shape and how ground-truth vs T3-predicted segmentation split it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics.simulate_gt.evaluate import FrameEvalResult  # noqa: E402
from postprocessing.viz._colors import PART_COLORS  # noqa: E402

_LABELS = {
    "t3": "T3-predicted seg -> T4 (full pipeline)",
    "t4_only": "exact seg -> T4 only",
    "gt": "ground truth",
}
_COLORS = {"t3": "tab:blue", "t4_only": "tab:orange", "gt": "tab:green"}
_UNASSIGNED_COLOR = (0.85, 0.0, 0.85)  # magenta -- part_label values outside PART_COLORS (e.g. wing_unassigned)


def _plot_three_lines(ax, frame_ids: list[int], results: list[FrameEvalResult], key: str) -> None:
    ax.plot(frame_ids, [r.row_t3[key] for r in results], marker=".", ms=4, lw=1,
            label=_LABELS["t3"], color=_COLORS["t3"])
    ax.plot(frame_ids, [r.row_t4_only[key] for r in results], marker=".", ms=4, lw=1,
            label=_LABELS["t4_only"], color=_COLORS["t4_only"])
    ax.plot(frame_ids, [r.row_gt[key] for r in results], marker=".", ms=4, lw=1, ls="--",
            label=_LABELS["gt"], color=_COLORS["gt"])
    ax.set_ylabel(f"{key} (deg)")
    ax.grid(alpha=0.3)


def plot_body_angles_compare(results: list[FrameEvalResult], out_path: Path) -> None:
    """3x1: yaw/pitch/roll vs frame_id, each with the 3 comparison lines."""
    frame_ids = [r.frame_id for r in results]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, col in zip(axes, ("yaw", "pitch", "roll")):
        _plot_three_lines(ax, frame_ids, results, col)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("frame_id")
    fig.suptitle("Body angles: T3+T4 vs exact-seg T4-only vs ground truth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wing_angles_compare(results: list[FrameEvalResult], out_path: Path) -> None:
    """3x2: (phi, theta, eta) x (L, R) vs frame_id, each with the 3
    comparison lines -- kept as 6 separate subplots (not phi_L/phi_R
    overlaid in one) so every subplot has exactly the 3 comparison lines."""
    frame_ids = [r.frame_id for r in results]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for row_idx, metric in enumerate(("phi", "theta", "eta")):
        for col_idx, side in enumerate(("L", "R")):
            _plot_three_lines(axes[row_idx, col_idx], frame_ids, results, f"{metric}_{side}")
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("frame_id")
    fig.suptitle("Wing angles: T3+T4 vs exact-seg T4-only vs ground truth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_point_cloud_segmentation(
    xyz: np.ndarray,
    gt_label: np.ndarray,
    pred_label: np.ndarray,
    frame_id: int,
    out_path: Path,
    elev: float = 20.0,
    azim: float = -60.0,
) -> None:
    """One viewing angle (fixed `elev`/`azim`, no real camera model -- this
    is synthetic data with no calibration to reproject through), two panels
    side by side: the same points colored by ground-truth `part_label` vs
    by `segment.segment_frame`'s predicted `part_label`, so a mismatch is
    visible directly on the cloud's own shape (`PART_COLORS`, same palette
    `labeling.py`/`reprojection_viewer.py` use; any label outside that
    palette, e.g. `wing_unassigned`, gets `_UNASSIGNED_COLOR`).
    """
    fig = plt.figure(figsize=(13, 6.5))
    for panel_idx, (title, labels) in enumerate((("ground truth", gt_label), ("T3-lite predicted", pred_label))):
        ax = fig.add_subplot(1, 2, panel_idx + 1, projection="3d")
        for part in sorted(set(labels)):
            mask = labels == part
            color = PART_COLORS.get(part, _UNASSIGNED_COLOR)
            ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], s=4, color=color, label=part, depthshade=False)
        ax.set_title(f"frame {frame_id} -- {title}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_MULTIVIEW_ANGLES = (
    ("front", dict(elev=0, azim=-90)),
    ("top", dict(elev=90, azim=-90)),
    ("side", dict(elev=0, azim=0)),
    ("3/4", dict(elev=20, azim=-60)),
)
"""Same front/top pairing `kmeans_split.plot_kmeans_clusters` uses, plus a
pure side view and the 3/4 view the single-angle plot already used -- enough
angles that a wing plane edge-on to one view is still readable from another
(see the earlier single-view "why does wing_R look like a line" false alarm,
which was exactly this -- a plane edge-on to one fixed camera)."""


def plot_point_cloud_segmentation_multiview(
    xyz: np.ndarray,
    gt_label: np.ndarray,
    pred_label: np.ndarray,
    frame_id: int,
    out_path: Path,
    angles: tuple = _MULTIVIEW_ANGLES,
) -> None:
    """Like `plot_point_cloud_segmentation` but a `len(angles) x 2` grid
    (rows = viewing angle, columns = ground truth vs predicted) instead of a
    single fixed angle -- for visually auditing one frame's segmentation
    without a single unlucky viewing angle (a flat wing edge-on to the
    camera, see module docstring) reading as a generation/segmentation bug.
    """
    n_rows = len(angles)
    fig = plt.figure(figsize=(11, 5.2 * n_rows))
    for row_idx, (view_name, view_kw) in enumerate(angles):
        for col_idx, (title, labels) in enumerate((("ground truth", gt_label), ("predicted", pred_label))):
            ax = fig.add_subplot(n_rows, 2, row_idx * 2 + col_idx + 1, projection="3d")
            for part in sorted(set(labels)):
                mask = labels == part
                color = PART_COLORS.get(part, _UNASSIGNED_COLOR)
                ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], s=4, color=color, label=part, depthshade=False)
            ax.set_title(f"frame {frame_id} -- {view_name} -- {title}", fontsize=9)
            ax.view_init(**view_kw)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
