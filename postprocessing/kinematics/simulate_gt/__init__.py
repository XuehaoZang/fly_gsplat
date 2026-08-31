"""End-to-end synthetic ground-truth validation for the T3+T4 kinematics
pipeline (segmentation quality + body angles + wing angles), as opposed to
`mock.py` and the various `correct_*/synthetic*.py` scripts, which each
validate one sub-algorithm in isolation against a hand-fed already-labeled
point cloud. See `scene.py`, `segment.py`, `evaluate.py`.
"""
