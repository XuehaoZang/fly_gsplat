"""单帧单翅角度：翅展 -> φ（stroke）/ θ（deviation）；弦向 -> η（pitch）。

翅展：翅平面（RANSAC）内点的 PCA 主轴，令其指向翅质心远离身体质心的方向（对应 MATLAB 的 spanHat）。
弦向：沿用 MATLAB find_chords_quad.m 的对角线法。取翅展中段附近的一条带，用其中离翅质心最远的点
      求最远点对，得到主对角线；再取垂直于主对角线所在平面的方向上的两个极值点，得到备选对角线。
      备选对角线长度 >= 1.3 倍主对角线时改用备选，否则用主对角线。
      弦向的正负号由逐帧连续性决定（与上一帧弦向点积 >= 0）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from utils import geometry as geo
from .body import BodyFrame

SIGN_LEFT = {"wing_L": -1.0, "wing_R": 1.0}

SPAN_ORIENT_MIN_COS_MARGIN = 0.1
"""|cos(翅展轴, 翅质心 - 体质心)| 小于该值时，翅展朝向无法判定，该翅本帧报错（不猜）。"""

CHORD_FRACTION = 0.33
"""中段带内按离翅质心距离取最远的这部分点，用来求主对角线（MATLAB chordFraction）。"""
DIAG_LENGTH_RATIO_DECISIVE = 1.3
MIN_BAND_POINTS = 5
BAND_FRACTIONS = (0.12, 0.36)
"""中段带半宽占翅展跨度的比例；第一档点数不够时用第二档。"""


def estimate_span(wing_xyz: np.ndarray, body_frame: BodyFrame) -> np.ndarray:
    """翅展单位向量（翅根 -> 翅尖）。"""
    n = wing_xyz.shape[0]
    nn_dist, _ = cKDTree(wing_xyz).query(wing_xyz, k=min(2, n))
    scale = float(np.median(nn_dist[:, -1])) if n > 1 else 0.0

    plane_mask = geo.ransac_plane_inliers(wing_xyz, threshold=2.0 * scale, rng=0)
    pts_plane = wing_xyz[np.nonzero(plane_mask)[0]]
    _, eigvecs, _ = geo.weighted_pca(pts_plane, np.ones(pts_plane.shape[0]))
    span_axis = eigvecs[:, -1]

    out_ref = wing_xyz.mean(axis=0) - np.asarray(body_frame.body_cm, dtype=float)
    out_ref_norm = float(np.linalg.norm(out_ref))
    cos_margin = abs(float(np.dot(span_axis, out_ref)) / out_ref_norm) if out_ref_norm > 0.0 else 0.0
    if cos_margin < SPAN_ORIENT_MIN_COS_MARGIN:
        raise ValueError(
            "estimate_span: outward-reference direction (wing_centroid - body_cm) is "
            f"too close to perpendicular to the wing's PCA axis to reliably orient it "
            f"(|cos|={cos_margin:.4f} < {SPAN_ORIENT_MIN_COS_MARGIN}, |out_ref|={out_ref_norm:.3g}) "
            "-- span sign for this frame is ambiguous, not silently guessed"
        )
    return geo.orient_to_reference(span_axis, out_ref)


def phi_theta(span: np.ndarray, body_frame: BodyFrame, side: str) -> tuple[float, float]:
    """φ：翅展在拍动平面内相对 x_body 的方位角；θ：翅展偏离拍动平面的角度。单位为度。"""
    n_sp = body_frame.n_sp
    x_sp = geo.project_onto_plane(body_frame.x_body, n_sp)
    y_sp = geo.project_onto_plane(body_frame.y_body, n_sp)
    span_sp = geo.project_onto_plane(span, n_sp)

    phi = math.degrees(math.atan2(SIGN_LEFT[side] * float(np.dot(span_sp, y_sp)), float(np.dot(span_sp, x_sp))))
    theta = 90.0 - math.degrees(math.acos(np.clip(np.dot(n_sp, span), -1.0, 1.0)))
    return phi, theta


def _eta(chord: np.ndarray, span: np.ndarray, n_sp: np.ndarray, sign_left: float) -> float:
    """η：弦向在垂直翅展的平面内，相对 (翅展 × 拍动平面法向) 的角度。单位为度。"""
    le_spn = np.cross(n_sp, span) if sign_left > 0 else np.cross(span, n_sp)
    le_spn = le_spn / np.linalg.norm(le_spn)
    sp_chord = np.cross(span, le_spn)
    sp_chord = sp_chord / np.linalg.norm(sp_chord)
    x = float(np.dot(chord, le_spn))
    y = float(np.dot(chord, sp_chord))
    return math.degrees(math.atan2(sign_left * y, x))


def _diagonals(wing_xyz: np.ndarray, span: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """(主对角线方向, 备选对角线方向, 主长度, 备选长度)，方向已去掉翅展分量并归一化。退化时返回 None。"""
    centroid = wing_xyz.mean(axis=0)
    t = (wing_xyz - centroid) @ span
    span_extent = t.max() - t.min()
    if span_extent < 1e-12:
        return None

    band_idx = None
    for frac in BAND_FRACTIONS:
        idx = np.nonzero(np.abs(t) < frac * span_extent)[0]
        if idx.size >= MIN_BAND_POINTS:
            band_idx = idx
            break
    if band_idx is None:
        return None

    band_pts = wing_xyz[band_idx]
    order = np.argsort(np.linalg.norm(band_pts - centroid, axis=1))[::-1]
    n_select = max(2, int(np.ceil(band_idx.size * CHORD_FRACTION)))
    selected = band_pts[order[:n_select]]
    if selected.shape[0] < 2:
        return None

    dmat = squareform(pdist(selected))
    i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
    raw_main = selected[i] - selected[j]
    chord_hat = raw_main - span * np.dot(span, raw_main)
    diag1 = float(np.linalg.norm(chord_hat))
    if diag1 < 1e-12:
        return None
    chord_hat = chord_hat / diag1

    wing_norm = np.cross(span, chord_hat)
    wn_norm = np.linalg.norm(wing_norm)
    if wn_norm < 1e-12:
        return None
    wing_norm = wing_norm / wn_norm

    proj = (band_pts - centroid) @ wing_norm
    i_max, i_min = int(np.argmax(proj)), int(np.argmin(proj))
    if proj[i_max] <= 0 or proj[i_min] >= 0:
        return None
    raw_alt = band_pts[i_max] - band_pts[i_min]
    chord_alt_hat = raw_alt - span * np.dot(span, raw_alt)
    diag2 = float(np.linalg.norm(chord_alt_hat))
    if diag2 < 1e-12:
        return None
    return chord_hat, chord_alt_hat / diag2, diag1, diag2


@dataclass
class ChordResult:
    eta: float
    chord: np.ndarray
    """单位弦向，已按连续性定号；下一帧作为 prev_chord 传入。"""
    chord_conf: float
    """clip(长对角线/短对角线 / 1.3, 0, 1)：长度判据的确定程度。"""


def estimate_chord(
    wing_xyz: np.ndarray,
    body_frame: BodyFrame,
    side: str,
    span: np.ndarray,
    prev_chord: np.ndarray | None,
) -> ChordResult:
    diags = _diagonals(wing_xyz, span)
    if diags is None:
        raise ValueError("estimate_chord_matlab: degenerate mid-span band/plane, cannot form a chord")
    chord_hat, chord_alt_hat, diag1, diag2 = diags

    chosen = chord_alt_hat if (diag2 / diag1) >= DIAG_LENGTH_RATIO_DECISIVE else chord_hat
    if prev_chord is not None and float(np.dot(chosen, prev_chord)) < 0.0:
        chosen = -chosen

    ratio = max(diag1, diag2) / min(diag1, diag2)
    return ChordResult(
        eta=_eta(chosen, span, body_frame.n_sp, SIGN_LEFT[side]),
        chord=chosen,
        chord_conf=float(np.clip(ratio / DIAG_LENGTH_RATIO_DECISIVE, 0.0, 1.0)),
    )


def plate_span(fit, body_cm: np.ndarray) -> np.ndarray:
    """平板翅展（翅根 -> 翅尖）：平板主轴，令其指向平板质心远离身体质心的方向。"""
    out_ref = fit.centroid - np.asarray(body_cm, dtype=float)
    out_ref_norm = float(np.linalg.norm(out_ref))
    cos_margin = abs(float(np.dot(fit.span_axis, out_ref)) / out_ref_norm) if out_ref_norm > 0.0 else 0.0
    if cos_margin < SPAN_ORIENT_MIN_COS_MARGIN:
        raise ValueError(f"plate_span: span sign ambiguous (|cos|={cos_margin:.4f} < {SPAN_ORIENT_MIN_COS_MARGIN})")
    return geo.orient_to_reference(fit.span_axis, out_ref)



LINK_MIN_COS = 0.5
"""相邻两帧弦线 |cos| 不小于该值（夹角 <= 60°）才连成一条连续链；否则断链。"""


def plate_chord_line(fit, span: np.ndarray) -> np.ndarray:
    """未定号的弦线 = 平板法向 × 翅展。"""
    return geo.unit(np.cross(fit.normal, span))


def resolve_chord_signs(
    frame_ids: list[int], span: dict[int, np.ndarray], chord_line: dict[int, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """一只翅整段弦线定号：前缘朝运动方向 + 逐帧连续链。

    1. 连续链：相邻帧都有效且 |c_t·c_{t+1}| >= LINK_MIN_COS 时相连，链内按点积 >= 0 逐帧对齐符号。
       翻转（换向）时弦向转得快但逐帧仍连续，链会穿过换向，所以一条链同时拿到上、下拍两侧的证据。
    2. 运动证据：翅展中心差分 v = s_{t+1} − s_{t−1}，去掉沿翅展的分量后与弦线点积 e_t = c_t·v⊥。
       e_t 的大小正比于翅展角速度，换向附近角速度趋于零，证据自然趋于零，不会被硬判。
    3. 每条链取 sign(Σ e_t) 作为整体符号（前缘 = 运动方向）。返回 (frame_id -> 定号弦向, frame_id -> 所在链的
       一致度 |Σe|/Σ|e|)。整条链没有任何运动证据时沿用上一条链末帧的方向。
    """
    valid = [f for f in frame_ids if f in chord_line]
    chains: list[list[int]] = []
    aligned: dict[int, np.ndarray] = {}
    for f in valid:
        c = chord_line[f]
        prev = f - 1
        if chains and chains[-1][-1] == prev and abs(float(np.dot(aligned[prev], c))) >= LINK_MIN_COS:
            aligned[f] = c if float(np.dot(aligned[prev], c)) >= 0.0 else -c
            chains[-1].append(f)
        else:
            aligned[f] = c
            chains.append([f])

    out: dict[int, np.ndarray] = {}
    conf: dict[int, float] = {}
    last: np.ndarray | None = None
    for chain in chains:
        e = []
        for f in chain:
            if (f - 1) in span and (f + 1) in span:
                v = span[f + 1] - span[f - 1]
                v = v - span[f] * float(np.dot(v, span[f]))
                e.append(float(np.dot(aligned[f], v)))
        total, mag = float(np.sum(e)), float(np.sum(np.abs(e)))
        if mag > 0.0:
            sign = 1.0 if total >= 0.0 else -1.0
            c_val = abs(total) / mag
        else:
            sign = 1.0 if last is None or float(np.dot(last, aligned[chain[0]])) >= 0.0 else -1.0
            c_val = 0.0
        for f in chain:
            out[f] = sign * aligned[f]
            conf[f] = c_val
        last = out[chain[-1]]
    return out, conf


def eta_from_chord(chord: np.ndarray, span: np.ndarray, body_frame: BodyFrame, side: str) -> float:
    return _eta(chord, span, body_frame.n_sp, SIGN_LEFT[side])
