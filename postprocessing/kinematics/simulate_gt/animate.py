"""Ground-truth labeled point-cloud animation for a `simulate_gt` sequence,
written out as an actual video (`.mp4`, via `cv2.VideoWriter` -- no system
`ffmpeg` binary is installed in this environment, and matplotlib's own
animation writers are limited to `pillow`/`html` without it, see chat
history) -- purpose is a plain eyeball check of whether the generated
flapping motion *looks* like a plausible wingbeat before trusting any
downstream segmentation/T4 number, not a diagnostic plot.

Colors by `FrameGroundTruth.part_label` only (never a segmentation
prediction) -- this is about auditing `scene.py`'s own forward kinematics,
not `segment.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics.simulate_gt.scene import FrameGroundTruth  # noqa: E402
from postprocessing.viz._colors import PART_COLORS  # noqa: E402

_UNASSIGNED_COLOR = (0.85, 0.0, 0.85)

VIEWS = (
    ("front", dict(elev=0, azim=-90)),
    ("top", dict(elev=90, azim=-90)),
)
"""Two panels per frame (same front/top pairing `kmeans_split.plot_kmeans_
clusters` uses): front shows the stroke-plane/pitch profile, top shows the
phi sweep -- between the two, a wing edge-on in one panel is normally
face-on in the other (see the earlier single-fixed-angle false alarm where
a wing looked like a 1D line from one view only)."""


def _render_panel(ax, xyz: np.ndarray, part_label: np.ndarray, elev: float, azim: float,
                   title: str, lims: tuple) -> None:
    for part in sorted(set(part_label)):
        mask = part_label == part
        color = PART_COLORS.get(part, _UNASSIGNED_COLOR)
        ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], s=5, color=color, depthshade=False)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def _render_frame_image(
    xyz: np.ndarray, frame_gt: FrameGroundTruth, lims: tuple, views: tuple, figsize: tuple,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize)
    for j, (name, view_kw) in enumerate(views):
        ax = fig.add_subplot(1, len(views), j + 1, projection="3d")
        _render_panel(ax, xyz, frame_gt.part_label, view_kw["elev"], view_kw["azim"],
                      f"{name} (frame {frame_gt.frame_id})", lims)
    fig.suptitle(
        f"yaw={frame_gt.yaw_deg:.1f} pitch={frame_gt.pitch_deg:.1f} roll={frame_gt.roll_deg:.1f}   "
        f"phi_L={frame_gt.phi_L_deg:.1f} theta_L={frame_gt.theta_L_deg:.1f} eta_L={frame_gt.eta_L_deg:.1f}   "
        f"phi_R={frame_gt.phi_R_deg:.1f} theta_R={frame_gt.theta_R_deg:.1f} eta_R={frame_gt.eta_R_deg:.1f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def render_ground_truth_video(
    frames: list, out_path: str | Path, fps: int = 15, views: tuple = VIEWS, figsize: tuple = (11, 6),
) -> Path:
    """Render `frames` (a `scene.py` `list[(df_unlabeled, FrameGroundTruth)]`
    sequence, e.g. from `scene.scenario_step2_flapping`) to an `.mp4` at
    `out_path`, one video frame per sequence frame, colored by ground-truth
    `part_label`. Axis limits are fixed to the *whole sequence's* combined
    xyz bounding box (computed once up front) so the camera doesn't
    rescale/jitter frame to frame -- only the points themselves should
    appear to move.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_xyz = np.concatenate([df[["x", "y", "z"]].to_numpy() for df, _ in frames], axis=0)
    pad = 0.05 * (all_xyz.max(axis=0) - all_xyz.min(axis=0))
    lims = tuple((all_xyz[:, i].min() - pad[i], all_xyz[:, i].max() + pad[i]) for i in range(3))

    writer = None
    for df, frame_gt in frames:
        xyz = df[["x", "y", "z"]].to_numpy()
        img = _render_frame_image(xyz, frame_gt, lims, views, figsize)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(bgr)
    if writer is not None:
        writer.release()

    return out_path
