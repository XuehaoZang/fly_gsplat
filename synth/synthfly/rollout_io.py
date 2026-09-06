"""Reader for the rollout .npz files produced by gs_recon (recorded simulator states)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REQUIRED = ("qpos", "qvel", "joint_names", "jnt_qposadr", "jnt_dofadr", "jnt_width",
            # scalar metadata the renderer dereferences
            "trial_id", "split", "arm", "checkpoint", "checkpoint_sha256", "seed", "control_dt",
            "reference_length", "n_valid_frames", "ended_by", "root_qposadr")


def load_rollout(path: Path) -> dict:
    """Return the record saved by gs_recon.synth.rollout.save_rollout.

    Arrays: qpos (F, 43), qvel (F, 42), ctrl_step (F,), action (F-1, 7), carrier_phase (F,),
    ctrl_freq_hz (F,), wing_command (F, 6), ref_root (L, 7), ref_com (L, 3), ref_quat (L, 4),
    ref_wing_qpos (L, 6), joint layout arrays. Scalars (from __meta__): trial_id, body_trial,
    template_trial, split, arm, checkpoint, checkpoint_sha256, seed, control_dt, reference_length,
    n_valid_frames, ended_by, base_freq_hz, root_qposadr, tracking."""
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files if k != "__meta__"}
        rec.update(json.loads(str(z["__meta__"])))
    rec["joint_names"] = [str(s) for s in rec["joint_names"]]
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        raise ValueError(f"{path}: not a gs_recon rollout, missing {missing}")
    return rec


def rollout_summary(rec: dict) -> dict:
    tr = rec.get("tracking") or {}
    return {
        "trial_id": rec.get("trial_id"),
        "body_trial": rec.get("body_trial"),
        "template_trial": rec.get("template_trial"),
        "valid_frames": int(rec.get("n_valid_frames", len(rec["qpos"]))),
        "control_dt_s": float(rec.get("control_dt", 2e-4)),
        "duration_s": float(int(rec.get("n_valid_frames", len(rec["qpos"]))) * float(rec.get("control_dt", 2e-4))),
        "ended_by": rec.get("ended_by"),
        "base_freq_hz": rec.get("base_freq_hz"),
        "com_rms_cm": tr.get("com_rms_cm"),
        "orientation_rms_deg": tr.get("orientation_rms_deg"),
        "checkpoint_sha256": rec.get("checkpoint_sha256"),
    }
