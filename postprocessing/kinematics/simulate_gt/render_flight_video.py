"""Render `scene.scenario_step2_flapping`'s ground-truth-labeled point cloud
as a video, for a plain eyeball check of whether the generated flapping
motion looks like a plausible wingbeat -- see `animate.py`'s module
docstring.

Run: python -m postprocessing.kinematics.simulate_gt.render_flight_video
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocessing.kinematics.simulate_gt.animate import render_ground_truth_video  # noqa: E402
from postprocessing.kinematics.simulate_gt.scene import scenario_step2_flapping  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent / "diag"


def main() -> None:
    frames = scenario_step2_flapping(n_frames=100)
    out_path = render_ground_truth_video(frames, DIAG_DIR / "step2_ground_truth_flight.mp4", fps=15)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
