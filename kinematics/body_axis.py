"""整段的体长轴 x_body（带序列连续性修正）。

单帧 PCA 主轴在体点云接近圆盘时方向不稳，头尾符号也只能靠"头朝上"猜。这里改为整段处理：
1. 锚点帧：两翅展开、近似 T 字形，且体点云足够细长的帧。锚点帧的 PCA 主轴（朝上）是可信的。
2. 引导链：逐帧用上一帧的轴作为引导，取体点云两端的端点质心（far+CC），用头减尾得到本帧轴。
   遇到断帧时重新起链：优先用最近锚点的轴作为引导，没有锚点时用本帧 PCA 主轴（朝上）。
3. 锚点校验：在每个锚点处若链的方向与锚点相反，就把上一个检查点到该锚点之间的一段整体翻转。
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from .body import UP, end_centroids
from .io import FrameParts

MIN_BODY_POINTS = 10
ANGLE_DEV_QUANTILE = 0.25
"""锚点条件：两翅质心相对体轴偏离 90° 的平均值 <= 全段该分位数。"""
WING_SPAN_QUANTILE = 0.75
"""锚点条件：两翅质心间距 >= 全段该分位数。"""
EIGVAL_RATIO_ANCHOR_THRESHOLD = 0.2
"""锚点条件：体点云 PCA 第二/第一特征值之比 < 该值（足够细长）。"""

_RESET_METHODS = ("anchor_guide", "pca_up_fallback")


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(float(np.clip(np.dot(a_n, b_n), -1.0, 1.0)))))


def _pca_axis_up(body_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """本帧 PCA 主轴（朝上）和第二/第一特征值之比。"""
    eigvals, eigvecs, _ = geo.weighted_pca(body_xyz)
    eigval_ratio = float(eigvals[-2] / eigvals[-1]) if eigvals[-1] > 0 else float("nan")
    return geo.orient_to_reference(eigvecs[:, -1], UP), eigval_ratio


def detect_anchors(frames: dict[int, FrameParts], frame_ids: list[int]) -> dict[int, np.ndarray]:
    """锚点帧 -> 该帧 PCA 主轴（朝上）。"""
    pca_axis, eigval_ratio, angle_dev, wing_span = {}, {}, {}, {}
    for fid in frame_ids:
        parts = frames[fid]
        pca_axis[fid], eigval_ratio[fid] = _pca_axis_up(parts.body)
        if len(parts.wing_L) == 0 or len(parts.wing_R) == 0:
            continue
        body_cm = parts.body.mean(axis=0)
        wing_L_cm, wing_R_cm = parts.wing_L.mean(axis=0), parts.wing_R.mean(axis=0)
        angle_L = _angle_deg(wing_L_cm - body_cm, pca_axis[fid])
        angle_R = _angle_deg(wing_R_cm - body_cm, pca_axis[fid])
        angle_dev[fid] = float(np.mean([abs(angle_L - 90.0), abs(angle_R - 90.0)]))
        wing_span[fid] = float(np.linalg.norm(wing_L_cm - wing_R_cm))

    if not angle_dev:
        return {}
    angle_dev_thr = float(np.quantile(list(angle_dev.values()), ANGLE_DEV_QUANTILE))
    wing_span_thr = float(np.quantile(list(wing_span.values()), WING_SPAN_QUANTILE))
    return {
        fid: pca_axis[fid] for fid in angle_dev
        if angle_dev[fid] <= angle_dev_thr
        and wing_span[fid] >= wing_span_thr
        and eigval_ratio[fid] < EIGVAL_RATIO_ANCHOR_THRESHOLD
    }


def _robust_x_body(
    body_xyz: np.ndarray, prev_axis: np.ndarray | None, anchor_axis: np.ndarray | None
) -> tuple[np.ndarray, str]:
    w = np.ones(body_xyz.shape[0])
    body_cm = (body_xyz * w[:, None]).sum(axis=0) / w.sum()
    if prev_axis is not None:
        guide_axis, method = geo.unit(prev_axis), "prev_frame_guide"
    elif anchor_axis is not None:
        guide_axis, method = geo.unit(anchor_axis), "anchor_guide"
    else:
        _, eigvecs, _ = geo.weighted_pca(body_xyz, w)
        guide_axis, method = geo.orient_to_reference(eigvecs[:, -1], UP), "pca_up_fallback"
    head_cm, tail_cm = end_centroids(body_xyz, body_cm, guide_axis, w)
    return geo.unit(head_cm - tail_cm), method


def _guide_chain(
    frames: dict[int, FrameParts], frame_ids: list[int], anchor_axes: dict[int, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    axes: dict[int, np.ndarray] = {}
    methods: dict[int, str] = {}
    prev_axis: np.ndarray | None = None
    prev_fid: int | None = None
    for fid in frame_ids:
        if prev_fid is None or fid - prev_fid > 1:
            prev_axis = None
        anchor_axis = None
        if prev_axis is None and anchor_axes:
            anchor_axis = anchor_axes[min(anchor_axes, key=lambda a: abs(a - fid))]
        try:
            axes[fid], methods[fid] = _robust_x_body(frames[fid].body, prev_axis, anchor_axis)
            prev_axis = axes[fid]
        except Exception as e:  # noqa: BLE001
            axes[fid], methods[fid] = np.full(3, np.nan), f"failed:{e}"
            prev_axis = None
        prev_fid = fid
    return axes, methods


def _flip_by_anchors(
    frame_ids: list[int],
    axes: dict[int, np.ndarray],
    methods: dict[int, str],
    anchor_axes: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """检查点 = 锚点帧（与锚点方向相反时翻转）和链重启帧（不翻转）。
    每个检查点决定它和上一个检查点之间那一段是否翻转；最后一个检查点的决定延伸到段尾。"""
    corrected = dict(axes)

    def valid(fid: int) -> bool:
        return fid in corrected and not np.any(np.isnan(corrected[fid]))

    checkpoints: list[tuple[int, bool]] = []
    for pos, fid in enumerate(frame_ids):
        if not valid(fid):
            continue
        if fid in anchor_axes:
            checkpoints.append((pos, float(np.dot(axes[fid], geo.unit(anchor_axes[fid]))) < 0.0))
        elif methods.get(fid, "") in _RESET_METHODS:
            checkpoints.append((pos, False))
    if not checkpoints:
        return corrected

    def flip_segment(start: int, end: int, should_flip: bool) -> None:
        if should_flip:
            for p in range(start, end + 1):
                if valid(frame_ids[p]):
                    corrected[frame_ids[p]] = -corrected[frame_ids[p]]

    prev_pos = -1
    for pos, should_flip in checkpoints:
        flip_segment(prev_pos + 1, pos, should_flip)
        prev_pos = pos
    last_pos, last_flip = checkpoints[-1]
    flip_segment(last_pos + 1, len(frame_ids) - 1, last_flip)
    return corrected


def sequence_x_body(frames: dict[int, FrameParts]) -> dict[int, np.ndarray]:
    """frame_id -> 单位 x_body。体点数不足或链失败的帧不在表里（由 body.py 退回单帧 PCA）。"""
    frame_ids = sorted(fid for fid, p in frames.items() if len(p.body) >= MIN_BODY_POINTS)
    anchor_axes = detect_anchors(frames, frame_ids)
    axes, methods = _guide_chain(frames, frame_ids, anchor_axes)
    corrected = _flip_by_anchors(frame_ids, axes, methods, anchor_axes)
    return {fid: a for fid, a in corrected.items() if not np.any(np.isnan(a))}
