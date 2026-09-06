"""synthfly: render synthetic multi-camera fruit-fly videos with ground truth.

Minimal-dependency renderer (numpy, mujoco, h5py, imageio) for the synthetic
fly data of the fly_gsplat project. It replays recorded simulator states of the
flybody fruit fly (rollouts made with the fly_mimic policy, shipped as small
.npz files) through the real four-camera calibration and writes images in
fly_gsplat's own dataset layout, plus per-frame ground truth.

No policy, no physics and no training run here: the model is only posed
(mj_forward) and rendered. New rollouts are produced by the gs_recon project
(Physical_AI/gs_recon, `python -m gs_recon.synth.pipeline`), which is also the
source of this package (`scripts/export_synth_package.py` regenerates it).
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_XML = PACKAGE_ROOT / "model" / "fruitfly_synth.xml"

CONTROL_DT = 2e-4  # s, one policy step = one recorded state (5 kHz)
# centre of mass in the thorax frame, cm (flybody / fly_mimic COM_OFFSET)
COM_OFFSET_CM = (-0.03697732, 0.00029205, -0.0142447)
PHYSICS_DT = 5e-5
NATIVE_FPS = int(round(1.0 / CONTROL_DT))

# Six wing hinges in qpos order; MuJoCo joint angle <-> recorded wing angle:
#   yaw = stroke - 90 deg, roll = -deviation, pitch = rotation - 45 deg.
WING_JOINTS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
               "wing_yaw_right", "wing_roll_right", "wing_pitch_right")
WING_ANGLE_NAMES = ("stroke_left", "deviation_left", "rotation_left",
                    "stroke_right", "deviation_right", "rotation_right")
HEAD_JOINTS = ("head_abduct", "head_twist", "head")
HALTERE_JOINTS = ("haltere_left", "haltere_right")
ABDOMEN_JOINTS = tuple(
    name for i in range(1, 8)
    for name in ((f"abdomen_abduct_{i}", f"abdomen_{i}") if i > 1 else ("abdomen_abduct", "abdomen"))
)

# Body-part labels. 0 is the background in part images.
PART_NAMES = ("background", "thorax", "head", "antenna", "abdomen", "wing_left", "wing_right",
              "haltere_left", "haltere_right", "leg_left", "leg_right")
PART_IDS = {name: i for i, name in enumerate(PART_NAMES)}
LEG_PATTERNS = ("_T1_", "_T2_", "_T3_")


def part_of_body(body_name: str) -> str:
    """Map a flybody body name to a part label."""
    name = body_name.split("/", 1)[-1]
    if any(p in name for p in LEG_PATTERNS):
        return "leg_left" if name.endswith("_left") else "leg_right"
    if name == "thorax":
        return "thorax"
    if name.startswith("antenna"):
        return "antenna"
    if name in ("head", "rostrum", "haustellum") or name.startswith("labrum"):
        return "head"
    if name.startswith("wing_left"):
        return "wing_left"
    if name.startswith("wing_right"):
        return "wing_right"
    if name.startswith("abdomen"):
        return "abdomen"
    if name.startswith("haltere_left"):
        return "haltere_left"
    if name.startswith("haltere_right"):
        return "haltere_right"
    raise KeyError(f"no part label for body {body_name!r}")


def wing_joint_to_angles_deg(wing_qpos_rad):
    """(..., 6) MuJoCo wing joint angles -> (..., 6) stroke/deviation/rotation in degrees."""
    import numpy as np

    q = np.degrees(np.asarray(wing_qpos_rad, dtype=np.float64))
    out = np.empty_like(q)
    for side in (0, 3):
        out[..., side + 0] = q[..., side + 0] + 90.0
        out[..., side + 1] = -q[..., side + 1]
        out[..., side + 2] = q[..., side + 2] + 45.0
    return out
