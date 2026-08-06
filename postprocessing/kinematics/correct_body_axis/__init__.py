"""Cross-frame continuity fix for body-axis (`x_body`) 180-degree flips.

Standalone experiment, not wired into `postprocessing.kinematics.pipeline` /
`body_frame.py`. See `continuity.py` for the core per-frame update and
`build_sequence.py` for the 640-frame batch run this was validated on.
"""
