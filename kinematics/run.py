"""第 5 步 kinematics：一段连续帧的 `_labeled.csv` -> angles.csv。

    python -m kinematics.run --run <recon_dir> --up=-z --fps 16000 [--param-set ratio3_sh0_dense]
        [--wing-method {plate,diag}] [--roll {keyframe,hinge}] [--out-name angles.csv]

--up：输入点云坐标系里哪个方向朝上（必填）。真实 4 相机（easyWand 标定）是 -z，合成数据是 +z。
计算前先转到 +z 朝上的实验室坐标系；头的方向默认取朝上的一端（锚点帧和链重启时），见 body_axis.py。
--fps：拍摄帧率（必填，真实 4 相机 16000，合成数据 5000）。用于平滑的截止频率/窗口，以及换算 segment 的累积窗口。

输入：<recon_dir>/<param_set>/f####/<method>/<时间戳>/gaussian_features_f####_labeled.csv
     plate 方法另读同目录的 splat.ply + dataparser_transforms.json（每个高斯的完整协方差，逐行对应 CSV）。
输出：<recon_dir>/kinematics/<out-name>，每帧一行（列见 io.OUTPUT_COLUMNS、PLATE_COLUMNS 和下面的派生列）。
     有任一部分失败的帧，对应列为 NaN，原因写在 status 列；成功的帧 status 为 "ok"。

方法（默认 --wing-method plate --roll keyframe；diag / hinge 是旧方法，保留作对照）：
1. 翅平面（plate，wing_plate.py）：每只翅取 3σ 连通的最大块，按"中心距离 + 沿法向厚度"投票出平板，
   法向 = 平板 2σ 椭球并集的最小主轴，翅展 = 连通块 3σ 并集的主轴（投影到平板平面）。
2. 体坐标系：x_body 由 body_axis.sequence_x_body 给出；roll 只在关键帧上求（每个拍动周期取两翅展夹角最大、
   两翅平板内点质量 >= KEY_MIN_MASS 的一帧，y_body = span_L − span_R），对关键帧 roll 拟合 GCV 平滑样条
   （对应 MATLAB calcBodyRoll.m 的 spaps）。segment 累积窗口不完整的段首段尾帧不当关键帧。
3. 身体平滑：yaw/pitch/roll -> *_smooth（smooth.py，移植 MATLAB smoothBodyAngles.m），段首段尾帧不参与、取最近值。
4. 左右翻转纠正：平滑体坐标系里左翅展指向右侧、右翅展指向左侧时交换两翅（wing_swapped）。
5. 弦向：平板法向 × 翅展；符号 = 前缘朝运动方向 + 逐帧连续链（wing.resolve_chord_signs）。
6. φ/θ/η 在平滑后的体坐标系里计算（逐帧体坐标系下的值保留为 *_rawbody）；φ、η 以整段圆均值为中心表示
   （smooth.unwrap_centered，不在 ±180° 处折回）。
7. 翅角平滑：phi/theta/eta_{L,R}_smooth（smooth.smooth_wing_angles，移植 MATLAB smoothWingAngles.m）。
   wing_masked 帧（segment 低置信即两翅粘连，或平板内点质量低）和段首段尾帧当作缺失、由 PCHIP 补；
   η 判反段（与迭代参考相差 > 90°）只在计算平滑曲线时翻转 180°（eta_flipfix），原始 η 和弦向量不改。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

from utils import geometry as geo
from .body import (
    STROKE_PLANE_PITCH_DEG, BodyFrame, _calculate_roll, estimate_body_frame, y_body_from_roll, zero_roll_basis,
)
from segment.motion import half_window_for_fps
from .body_axis import sequence_x_body
from .io import DATA_TO_LAB, OUTPUT_COLUMNS, FrameParts, discover_frames, load_parts
from .wing import (
    eta_from_chord, estimate_chord, estimate_span, phi_theta, plate_chord_line, plate_span, resolve_chord_signs,
)
from .smooth import smooth_body_angles, smooth_wing_angles, unwrap_centered
from .wing_plate import fit_wing_plate

PLATE_COLUMNS = [
    "plate_flat_L", "plate_flat_R", "plate_mass_L", "plate_mass_R", "plate_n_L", "plate_n_R", "roll_keyframe",
    "segment_low_conf", "wing_swapped", "wing_masked_L", "wing_masked_R",
]
"""--wing-method plate 时追加的诊断列：并集 λ3/λ2、内点质量占比、并集高斯数；roll_keyframe 标记关键帧；
segment_low_conf：segment 把粘连的两翅强行切开的帧；wing_swapped：本帧左右翅已交换（翻转纠正）；
wing_masked_L/R：翅角平滑时当作缺失、由样条补的帧（segment_low_conf 或该翅平板内点质量 < KEY_MIN_MASS）。
plate 时 chord_conf 是弦向所在连续链的运动证据一致度 |Σe|/Σ|e|（见 wing.resolve_chord_signs）。"""

MIN_POINTS = 10
"""body / wing_L / wing_R 各自至少需要的点数。"""


def _all_finite(*values) -> bool:
    return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)


def _plate_fits(frames: dict[int, FrameParts], span_source: str = "comp3") -> dict[int, dict[str, object]]:
    """frame_id -> {side: PlateFit 或 失败原因 str}。"""
    out: dict[int, dict[str, object]] = {}
    for fid, parts in frames.items():
        out[fid] = {}
        for side in ("wing_L", "wing_R"):
            try:
                out[fid][side] = fit_wing_plate(parts.gauss[side], span_source=span_source)
            except Exception as e:  # noqa: BLE001
                out[fid][side] = str(e)
    return out


KEY_MIN_MASS = 0.4
"""关键帧候选要求两翅平板内点质量占比都不低于该值（更低的帧多是 segment 出错）。"""


def wingbeat_period(alpha: dict[int, float]) -> int:
    """两翅展夹角序列的自相关：越过第一个负值之后的第一个峰的滞后（帧）。"""
    ids = np.arange(min(alpha), max(alpha) + 1)
    known = np.array(sorted(alpha))
    x = np.interp(ids, known, [alpha[k] for k in known])
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    neg = np.nonzero(ac < 0)[0]
    if neg.size == 0:
        raise ValueError("wingbeat_period: autocorrelation never goes negative")
    lag = int(neg[0] + np.argmax(ac[neg[0]: x.size // 2]))
    if lag <= neg[0]:
        raise ValueError("wingbeat_period: no autocorrelation peak")
    return lag


MIN_SPLINE_KEYS = 5
"""关键帧少于该数时退回分段线性插值（平滑样条需要至少 5 个点）。"""


def full_window_frames(frame_ids: list[int], fps: float) -> set[int]:
    """segment 累积窗口完整的帧：fid ± half_window 内每一帧都在。"""
    hw = half_window_for_fps(fps)
    have = set(frame_ids)
    return {f for f in frame_ids if all((f + d) in have for d in range(-hw, hw + 1))}


def smooth_roll(keys: np.ndarray, vals_deg: np.ndarray, all_ids: np.ndarray) -> np.ndarray:
    """展开后的关键帧 roll -> 每帧 roll（度，未折回）。GCV 平滑样条，首尾关键帧之外取端点值。"""
    if keys.size < MIN_SPLINE_KEYS:
        return np.interp(all_ids, keys, vals_deg)
    spline = make_smoothing_spline(keys.astype(float), vals_deg)
    return spline(np.clip(all_ids, keys[0], keys[-1]).astype(float))


def keyframe_roll(
    frames: dict[int, FrameParts],
    x_body_table: dict[int, np.ndarray],
    fits: dict[int, dict[str, object]],
    spans: dict[str, dict[int, np.ndarray]],
    fps: float,
) -> tuple[dict[int, float], set[int], int]:
    """每个拍动周期取两翅展夹角最大的一帧，y_body = span_L − span_R 求 roll，帧间分段线性插值。
    返回 (frame_id -> roll 度, 关键帧集合, 周期帧数)。"""
    alpha: dict[int, float] = {}
    trusted = full_window_frames(sorted(frames), fps)
    for fid in sorted(frames):
        if fid not in trusted:
            continue
        if fid not in spans["wing_L"] or fid not in spans["wing_R"] or fid not in x_body_table:
            continue
        if min(fits[fid]["wing_L"].inlier_mass_frac, fits[fid]["wing_R"].inlier_mass_frac) < KEY_MIN_MASS:
            continue
        alpha[fid] = float(np.degrees(np.arccos(np.clip(np.dot(spans["wing_L"][fid], spans["wing_R"][fid]), -1, 1))))
    if len(alpha) < 3:
        return {}, set(), 0
    period = wingbeat_period(alpha)

    key_roll: dict[int, float] = {}
    first = min(frames)
    windows: dict[int, list[int]] = {}
    for fid in alpha:
        windows.setdefault((fid - first) // period, []).append(fid)
    for members in windows.values():
        fid = max(members, key=lambda f: alpha[f])
        x_body = x_body_table[fid]
        y_body = geo.project_onto_plane(spans["wing_L"][fid] - spans["wing_R"][fid], x_body)
        if not np.all(np.isfinite(y_body)):
            continue
        yaw, pitch, _, _ = zero_roll_basis(x_body)
        key_roll[fid] = float(np.degrees(_calculate_roll(yaw, pitch, y_body)))
    if not key_roll:
        return {}, set(), period
    # 左右翻转的关键帧会让 span_L − span_R 反向（roll 差 180°）：与关键帧 roll 的圆中位数相差 > 90° 的翻回来
    ref = np.degrees(np.angle(np.median(np.cos(np.radians(list(key_roll.values()))))
                              + 1j * np.median(np.sin(np.radians(list(key_roll.values()))))))
    for fid, v in key_roll.items():
        if abs((v - ref + 180.0) % 360.0 - 180.0) > 90.0:
            key_roll[fid] = (v + 360.0) % 360.0 - 180.0
    keys = np.array(sorted(key_roll))
    vals = np.degrees(np.unwrap(np.radians([key_roll[k] for k in keys])))
    all_ids = np.array(sorted(frames))
    interp = smooth_roll(keys, vals, all_ids)
    return {int(f): float((r + 180.0) % 360.0 - 180.0) for f, r in zip(all_ids, interp)}, set(key_roll), period


def _compute_angles_diag(frames: dict[int, FrameParts]) -> pd.DataFrame:
    x_body_table = sequence_x_body(frames)
    prev_chord: dict[str, np.ndarray | None] = {"wing_L": None, "wing_R": None}

    rows = []
    for fid in sorted(frames):
        parts = frames[fid]
        row = {c: np.nan for c in OUTPUT_COLUMNS}
        row["frame_id"] = fid
        rows.append(row)

        too_few = [label for label in ("body", "wing_L", "wing_R") if len(getattr(parts, label)) < MIN_POINTS]
        if too_few:
            row["status"] = f"{too_few[0]}:too_few_points"
            continue

        try:
            frame = estimate_body_frame(parts.body, parts.wing_L, parts.wing_R, x_body=x_body_table.get(fid))
            if not _all_finite(frame.x_body, frame.y_body, frame.z_body, frame.n_sp,
                               frame.yaw, frame.pitch, frame.roll):
                raise ValueError("non-finite result (degenerate hinge/PCA geometry)")
        except Exception as e:  # noqa: BLE001
            row["status"] = f"body:{e}"
            continue
        row["yaw"], row["pitch"], row["roll"] = frame.yaw, frame.pitch, frame.roll
        row["sp_normal_x"], row["sp_normal_y"], row["sp_normal_z"] = (float(v) for v in frame.n_sp)

        wing_errors = []
        for side, s in (("wing_L", "L"), ("wing_R", "R")):
            wing_xyz = getattr(parts, side)
            try:
                span = estimate_span(wing_xyz, frame)
                phi, theta = phi_theta(span, frame, side)
                chord = estimate_chord(wing_xyz, frame, side, span, prev_chord[side])
                if not _all_finite(phi, theta, chord.eta, chord.chord_conf, span):
                    raise ValueError("non-finite result")
            except Exception as e:  # noqa: BLE001
                wing_errors.append(f"{side}:{e}")
                continue
            row[f"phi_{s}"], row[f"theta_{s}"], row[f"eta_{s}"] = phi, theta, chord.eta
            row[f"chord_conf_{s}"] = chord.chord_conf
            row[f"span_{s}_x"], row[f"span_{s}_y"], row[f"span_{s}_z"] = (float(v) for v in span)
            row[f"chord_{s}_x"], row[f"chord_{s}_y"], row[f"chord_{s}_z"] = (float(v) for v in chord.chord)
            prev_chord[side] = chord.chord.copy()

        row["status"] = ";".join(wing_errors) if wing_errors else "ok"

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _compute_angles_plate(
    frames: dict[int, FrameParts], roll_method: str, span_source: str, fps: float
) -> pd.DataFrame:
    """平板法：先整段算翅展、体坐标系和未定号弦线，再整段定号弦向，最后算 η。"""
    frame_ids = sorted(frames)
    x_body_table = sequence_x_body(frames)
    fits = _plate_fits(frames, span_source)

    # 翅展只依赖平板和体质心，与 roll 无关，先整段算出
    spans: dict[str, dict[int, np.ndarray]] = {"wing_L": {}, "wing_R": {}}
    span_err: dict[tuple[int, str], str] = {}
    for fid in frame_ids:
        body_cm = frames[fid].body.mean(axis=0) if len(frames[fid].body) else None
        for side in ("wing_L", "wing_R"):
            fit = fits[fid][side]
            try:
                if isinstance(fit, str):
                    raise ValueError(fit)
                if body_cm is None:
                    raise ValueError("no body points")
                spans[side][fid] = plate_span(fit, body_cm)
            except Exception as e:  # noqa: BLE001
                span_err[(fid, side)] = str(e)

    roll_table, key_frames, period = ({}, set(), 0)
    if roll_method == "keyframe":
        roll_table, key_frames, period = keyframe_roll(frames, x_body_table, fits, spans, fps)
        print(f"[kinematics] wingbeat period {period} frames, {len(key_frames)} roll keyframes")

    columns = OUTPUT_COLUMNS + PLATE_COLUMNS
    rows: dict[int, dict] = {}
    body_frames = {}
    chord_line: dict[str, dict[int, np.ndarray]] = {"wing_L": {}, "wing_R": {}}
    wing_errors: dict[int, list[str]] = {}
    for fid in frame_ids:
        parts = frames[fid]
        row = {c: np.nan for c in columns}
        row["frame_id"] = fid
        row["roll_keyframe"] = fid in key_frames
        rows[fid] = row
        row["segment_low_conf"] = parts.low_conf
        if len(parts.body) < MIN_POINTS:
            row["status"] = "body:too_few_points"
            continue
        try:
            y_body = None
            if roll_method == "keyframe":
                x_body = x_body_table.get(fid)
                if fid not in roll_table or x_body is None:
                    raise ValueError("no keyframe roll / x_body for this frame")
                y_body = y_body_from_roll(x_body, roll_table[fid])
            frame = estimate_body_frame(parts.body, parts.wing_L, parts.wing_R,
                                        x_body=x_body_table.get(fid), y_body=y_body)
            if not _all_finite(frame.x_body, frame.y_body, frame.z_body, frame.n_sp,
                               frame.yaw, frame.pitch, frame.roll):
                raise ValueError("non-finite result (degenerate hinge/PCA geometry)")
        except Exception as e:  # noqa: BLE001
            row["status"] = f"body:{e}"
            continue
        body_frames[fid] = frame
        row["yaw"], row["pitch"], row["roll"] = frame.yaw, frame.pitch, frame.roll
        row["sp_normal_x"], row["sp_normal_y"], row["sp_normal_z"] = (float(v) for v in frame.n_sp)

    # 左右翻转纠正：在平滑后的体坐标系里，左翅展指向右侧且右翅展指向左侧 -> 交换两翅的平板结果
    trusted = np.array([f in full_window_frames(frame_ids, fps) for f in frame_ids])
    sm = smooth_body_angles(*(np.array([rows[f][a] for f in frame_ids], dtype=float) for a in ("yaw", "pitch", "roll")),
                            fps, trusted)
    for k, fid in enumerate(frame_ids):
        rows[fid]["wing_swapped"] = False
        if fid not in spans["wing_L"] or fid not in spans["wing_R"] or not _all_finite(sm[0][k], sm[1][k], sm[2][k]):
            continue
        y_s = y_body_from_roll(x_body_from_angles(sm[0][k], sm[1][k]), sm[2][k])
        if np.dot(spans["wing_L"][fid], y_s) < 0.0 < np.dot(spans["wing_R"][fid], y_s):
            spans["wing_L"][fid], spans["wing_R"][fid] = spans["wing_R"][fid], spans["wing_L"][fid]
            fits[fid]["wing_L"], fits[fid]["wing_R"] = fits[fid]["wing_R"], fits[fid]["wing_L"]
            rows[fid]["wing_swapped"] = True

    for fid in frame_ids:
        if fid not in body_frames:
            continue
        frame, row = body_frames[fid], rows[fid]
        wing_errors[fid] = []
        for side, s in (("wing_L", "L"), ("wing_R", "R")):
            if (fid, side) in span_err:
                wing_errors[fid].append(f"{side}:{span_err[(fid, side)]}")
                continue
            span = spans[side][fid]
            phi, theta = phi_theta(span, frame, side)
            row[f"phi_{s}"], row[f"theta_{s}"] = phi, theta
            row[f"span_{s}_x"], row[f"span_{s}_y"], row[f"span_{s}_z"] = (float(v) for v in span)
            chord_line[side][fid] = plate_chord_line(fits[fid][side], span)

    for side, s in (("wing_L", "L"), ("wing_R", "R")):
        chords, conf = resolve_chord_signs(frame_ids, spans[side], chord_line[side])
        for fid, chord in chords.items():
            row = rows[fid]
            row[f"eta_{s}"] = eta_from_chord(chord, spans[side][fid], body_frames[fid], side)
            row[f"chord_conf_{s}"] = conf[fid]
            row[f"chord_{s}_x"], row[f"chord_{s}_y"], row[f"chord_{s}_z"] = (float(v) for v in chord)

    for fid in frame_ids:
        row = rows[fid]
        for side, s in (("wing_L", "L"), ("wing_R", "R")):
            f = fits[fid][side]
            if not isinstance(f, str):
                row[f"plate_flat_{s}"], row[f"plate_mass_{s}"] = f.flatness, f.inlier_mass_frac
                row[f"plate_n_{s}"] = int(f.inlier.sum())
            mass = row[f"plate_mass_{s}"]
            row[f"wing_masked_{s}"] = bool(row["segment_low_conf"] or not np.isfinite(mass) or mass < KEY_MIN_MASS)
        if fid in wing_errors:
            row["status"] = ";".join(wing_errors[fid]) if wing_errors[fid] else "ok"
    return pd.DataFrame([rows[f] for f in frame_ids], columns=columns)


def compute_angles(
    frames: dict[int, FrameParts], fps: float, wing_method: str = "plate", roll_method: str = "keyframe",
    span_source: str = "comp3",
) -> pd.DataFrame:
    """frame_id -> FrameParts（必须是一段连续帧）-> 每帧一行的角度表（含平滑后的身体角度列）。"""
    if roll_method == "keyframe" and wing_method != "plate":
        raise ValueError("--roll keyframe needs --wing-method plate")
    if wing_method == "plate":
        df = _compute_angles_plate(frames, roll_method, span_source, fps)
    else:
        df = _compute_angles_diag(frames)
    trusted = df["frame_id"].isin(full_window_frames(sorted(frames), fps)).to_numpy()
    df["yaw_smooth"], df["pitch_smooth"], df["roll_smooth"] = smooth_body_angles(
        df["yaw"].to_numpy(), df["pitch"].to_numpy(), df["roll"].to_numpy(), fps, trusted)
    df = wing_angles_in_smoothed_frame(df)
    for s in ("L", "R"):
        for a in ("phi", "eta"):  # 原始 φ/η 也用与平滑列相同的连续表示
            df[f"{a}_{s}"] = unwrap_centered(df[f"{a}_{s}"].to_numpy())
        ignore = ~trusted
        if f"wing_masked_{s}" in df:
            ignore = ignore | df[f"wing_masked_{s}"].astype(bool).to_numpy()
        res = smooth_wing_angles(df[f"phi_{s}"].to_numpy(), df[f"theta_{s}"].to_numpy(), df[f"eta_{s}"].to_numpy(),
                                 fps, ignore)
        # 弦向符号判反段：参考曲线迭代求（每轮把偏离参考 > ETA_REF_OUTLIER_DEG 的点当缺失、重新平滑，避免判反段
        # 本身把参考拉偏），原始 η 与参考相差 > ETA_FLIP_DEG 的帧翻转 180° 后再平滑一遍。只影响 *_smooth 列：
        # 合成数据（5000 fps）上该判据会误翻快速转动的正确帧，所以不改写原始 η 和弦向量。
        eta = df[f"eta_{s}"].to_numpy()
        ref = res[2]
        for _ in range(ETA_REF_ITERS):
            out = ignore | ~np.isfinite(eta) | (np.abs(eta - ref) > ETA_REF_OUTLIER_DEG)
            ref = smooth_wing_angles(df[f"phi_{s}"].to_numpy(), df[f"theta_{s}"].to_numpy(), eta, fps, out)[2]
        diff = eta - ref
        flip = np.isfinite(diff) & (np.abs(diff) > ETA_FLIP_DEG)
        df[f"eta_flipfix_{s}"] = flip
        if flip.any():  # 翻转只用于计算平滑曲线；原始 η 列和弦向量保持测量值
            res = smooth_wing_angles(df[f"phi_{s}"].to_numpy(), df[f"theta_{s}"].to_numpy(),
                                     np.where(flip, eta - 180.0 * np.sign(diff), eta), fps, ignore)
        df[f"phi_{s}_smooth"], df[f"theta_{s}_smooth"], df[f"eta_{s}_smooth"] = res
    return df


ETA_REF_ITERS = 3
ETA_REF_OUTLIER_DEG = 60.0
ETA_FLIP_DEG = 90.0
"""原始 η 与第一遍平滑相差超过该值的帧视为弦向符号判反（真实的 η 不会在一帧里偏离平滑趋势这么多）。"""


RAW_FRAME_SUFFIX = "_rawbody"
"""翅角改在平滑体坐标系里计算后，原逐帧体坐标系下的 φ/θ/η 保留为 <列名>_rawbody（诊断用）。"""


def x_body_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y_r, p_r = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([np.cos(p_r) * np.cos(y_r), np.cos(p_r) * np.sin(y_r), np.sin(p_r)])


def wing_angles_in_smoothed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """由 yaw/pitch/roll_smooth 重建每帧体坐标系（x_body、y_body、拍动平面法向），用各翅的翅展、定号弦向
    重算 φ/θ/η 并覆盖原列；sp_normal 也换成平滑后的。翅展/弦向缺失的翅保持 NaN。"""
    for s in ("L", "R"):
        for a in ("phi", "theta", "eta"):
            df[f"{a}_{s}{RAW_FRAME_SUFFIX}"] = df[f"{a}_{s}"]
            df[f"{a}_{s}"] = np.nan
    for i, row in df.iterrows():
        yaw, pitch, roll = row["yaw_smooth"], row["pitch_smooth"], row["roll_smooth"]
        if not _all_finite(yaw, pitch, roll):
            continue
        x_body = x_body_from_angles(yaw, pitch)
        y_body = y_body_from_roll(x_body, roll)
        n_sp = geo.rodrigues_rotate(x_body, y_body, np.radians(-STROKE_PLANE_PITCH_DEG))
        frame = BodyFrame(x_body=x_body, y_body=y_body, z_body=np.cross(x_body, y_body), n_sp=n_sp,
                          yaw=yaw, pitch=pitch, roll=roll, body_cm=np.full(3, np.nan))
        df.loc[i, ["sp_normal_x", "sp_normal_y", "sp_normal_z"]] = n_sp
        for side, s in (("wing_L", "L"), ("wing_R", "R")):
            span = row[[f"span_{s}_x", f"span_{s}_y", f"span_{s}_z"]].to_numpy(dtype=float)
            chord = row[[f"chord_{s}_x", f"chord_{s}_y", f"chord_{s}_z"]].to_numpy(dtype=float)
            if not _all_finite(span, chord):
                continue
            df.loc[i, f"phi_{s}"], df.loc[i, f"theta_{s}"] = phi_theta(span, frame, side)
            df.loc[i, f"eta_{s}"] = eta_from_chord(chord, span, frame, side)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="重建结果目录，例如 outputs/ctrl_009/<run>")
    ap.add_argument("--up", required=True, choices=sorted(DATA_TO_LAB), help="真实 4 相机 -z，合成数据 +z")
    ap.add_argument("--fps", type=float, required=True, help="拍摄帧率：真实 4 相机 16000，合成数据 5000")
    ap.add_argument("--param-set", default="ratio3_sh0_dense")
    ap.add_argument("--wing-method", default="plate", choices=("plate", "diag"), help="diag：旧的对角线法（对照）")
    ap.add_argument("--roll", default="keyframe", choices=("keyframe", "hinge"), help="hinge：旧的逐帧翅根 roll（对照）")
    ap.add_argument("--out-name", default="angles.csv")
    args = ap.parse_args()

    paths = discover_frames(args.run / args.param_set)
    if not paths:
        raise SystemExit(f"no *_labeled.csv under {args.run / args.param_set}")
    print(f"[kinematics] {len(paths)} frames f{min(paths):04d}..f{max(paths):04d}")
    to_lab = DATA_TO_LAB[args.up]
    frames = {fid: load_parts(p, to_lab, with_cov=args.wing_method == "plate") for fid, p in paths.items()}

    df = compute_angles(frames, args.fps, args.wing_method, args.roll)
    for vec in ("sp_normal", "span_L", "span_R", "chord_L", "chord_R"):  # 向量写回输入坐标系（to_lab 是自己的逆）
        cols = [f"{vec}_x", f"{vec}_y", f"{vec}_z"]
        df[cols] = df[cols].to_numpy() @ to_lab
    out = args.run / "kinematics" / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[kinematics] status=ok {int((df['status'] == 'ok').sum())}/{len(df)} -> {out}")


if __name__ == "__main__":
    main()
