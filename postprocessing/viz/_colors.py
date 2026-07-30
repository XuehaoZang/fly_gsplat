"""viz 配色单一来源，被 postprocessing/labeling/labeling.py、postprocessing/viz/splat_viewer.py、
postprocessing/viz/reprojection_viewer.py 共用，避免几处画法各定一份颜色导致后续不一致。
"""
import numpy as np
from matplotlib.colors import to_rgb

BODY_COLOR = to_rgb("#888888")
WING_L_COLOR = to_rgb("#1f77b4")
WING_R_COLOR = to_rgb("#d62728")
PART_COLORS = {"body": BODY_COLOR, "wing_L": WING_L_COLOR, "wing_R": WING_R_COLOR}

SINGLE_COLOR = to_rgb("#2ca02c")  # 没有part_label列(T1/T2阶段)时的退化单色模式

# viser 场景用的 uint8 RGB，跟上面 matplotlib 用的 0~1 float tuple 分开表示
RGB_BG = np.array([255, 255, 255], dtype=np.float64)   # rgb 按 opacity 与该背景色混合，模拟透明度
HULL_COLOR = np.array([50, 200, 50], dtype=np.uint8)    # splat_viewer 的 hull 模式统一颜色
DROP_COLOR = np.array([220, 30, 30], dtype=np.uint8)    # if_keep=False 高亮色，同T3重投影图的红叉惯例
