"""共享: diag/下所有3D散点图用的"前视/俯视"角度，避免matplotlib 3D的一个已知问题——
当elev恰好=0或90时视线方向跟某根坐标轴完全平行，那根轴的刻度/标签在投影后全部挤到
同一点附近，糊成一团(旧voxel_threshold_tuning.py的front视角就是这样，z/y轴标签重叠
看不清)。用略微偏离0/90的角度保留"近似前视/近似俯视"的直觉，同时避开这个退化情形。
"""
from __future__ import annotations

FRONT_VIEW = dict(elev=10, azim=-90)
TOP_VIEW = dict(elev=70, azim=-90)
VIEWS = [("front", FRONT_VIEW), ("top", TOP_VIEW)]


def view_title(name: str, view_kw: dict) -> str:
    return f"{name} (elev={view_kw['elev']}, azim={view_kw['azim']})"
