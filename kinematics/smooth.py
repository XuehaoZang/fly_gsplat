"""身体角度的整段平滑，移植自 MATLAB `smoothBodyAngles.m` / `filterEulerAngle.m`（旧库 gsplat-sim
postprocessing/kinematics/smoothing.py 的同名函数，参数保持 MATLAB `setSmoothingParams.m` 默认值）。

每个角度：Hampel 去尖峰（每侧 3 个样本、3σ）-> 零相位 3 阶 Butterworth 低通（filtfilt）。
截止频率以 Hz 给出：yaw / pitch 100 Hz，roll 200 Hz，按调用方传入的 fps 换算。
yaw 先按前 20 帧中位数重定分支后再滤波（不是全局 unwrap）。
NaN 帧不参与 Hampel；滤波时线性插值穿过。不可信帧（trusted=False，即 segment 累积窗口不完整的段首段尾）
在滤波前置为 NaN，不影响其他帧；最后所有 NaN 位置都填上：内部线性插值，两端取最近有效帧的值。

翅角：smooth_wing_angles，移植自 MATLAB `smoothWingAngles.m`（旧库同名函数，scale_windows_to_fps=True、
unwrap_phi_theta=True 两项在旧库里用合成数据 GT 验证过）：
  φ、θ 整段展开；η 展开（MATLAB unwrap）；θ、η 各做 Hampel（每侧 7 个样本、2.5σ）标出离群帧；
  φ 在 (NaN | ignore | θ 离群 | η 离群 | φ 自身离群) 处置空（φ 自身离群是本库加的：MATLAB 不查 φ，
  屏蔽段旁边一个没被屏蔽的离群帧会成为插值锚点，把曲线拉出几十度的过冲），η 在 (ignore | η 离群) 处置空，θ 只在 ignore 处置空
  （MATLAB 源码原样：θ 自己的离群帧不置空，旧库标为疑似 bug，这里照搬）；
  空缺用三次样条补；再做 Savitzky-Golay（φ 21 点 7 阶，θ/η 11 点 5 阶）。
  窗口按 8000 fps 调好，这里按 fps 换算成相同时长的奇数样本数。
  展开（偏离 MATLAB）：MATLAB/np.unwrap 逐帧和前一帧比，个别接近 180° 的离群帧（如弦向符号判反）会让其后
  整段平移 360°，平滑跨过这个台阶会造出假的大摆动。φ、η 是有界振荡、不会真的转整圈，所以改用
  unwrap_centered：以整段圆均值为中心折回，切口远离数据，没有累积漂移。输出（原始列和平滑列）都用这个表示。
  补空缺（偏离 MATLAB 的平滑样条）：内部空缺用 PCHIP（保形，不会冲出两侧数据的范围；三次插值样条跨过
  十几帧的屏蔽段会过冲几十度），两端的空缺取最近有效值。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, savgol_filter

YAW_FILT_HZ = 100.0
PITCH_FILT_HZ = 100.0
ROLL_FILT_HZ = 200.0
HAMPEL_WINDOW = 3
HAMPEL_N_SIGMA = 3.0
BUTTER_ORDER = 3
YAW_INIT_WINDOW = 20


def hampel_filter(x: np.ndarray, window: int = HAMPEL_WINDOW, n_sigma: float = HAMPEL_N_SIGMA) -> np.ndarray:
    """MATLAB `hampel`：偏离局部中位数超过 n_sigma × 1.4826·MAD 的样本替换为局部中位数（窗口在两端收缩）。"""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    for i in range(x.size):
        if not np.isfinite(x[i]):
            continue
        seg = x[max(0, i - window): i + window + 1]
        seg = seg[np.isfinite(seg)]
        med = np.median(seg)
        mad = 1.4826 * np.median(np.abs(seg - med))
        if mad > 0 and abs(x[i] - med) > n_sigma * mad:
            out[i] = med
    return out


def butter_lowpass_filtfilt(x: np.ndarray, cutoff_hz: float, fps: float, order: int = BUTTER_ORDER) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nan_mask = ~np.isfinite(x)
    if nan_mask.all():
        return x.copy()
    if nan_mask.any():
        idx = np.arange(x.size)
        x = x.copy()
        x[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    wn = cutoff_hz / (0.5 * fps)
    if not 0 < wn < 1:
        raise ValueError(f"cutoff {cutoff_hz} Hz is not below nyquist for fps={fps}")
    b, a = butter(order, wn, btype="low")
    y = filtfilt(b, a, x) if x.size > 3 * max(len(a), len(b)) else x.copy()
    y[nan_mask] = np.nan
    return y


def filter_euler_angle(x: np.ndarray, cutoff_hz: float, fps: float) -> np.ndarray:
    return butter_lowpass_filtfilt(hampel_filter(x), cutoff_hz, fps)


def recenter_yaw(yaw_deg: np.ndarray, init_window: int = YAW_INIT_WINDOW) -> np.ndarray:
    """以前 init_window 个有效帧的中位数为中心，把 yaw 折回 ±180° 内（MATLAB 的 while 循环；
    段首帧被置为 NaN 时跳过它们，开头没有 NaN 时与 MATLAB 相同）。"""
    x = np.asarray(yaw_deg, dtype=float)
    head = x[np.isfinite(x)][:init_window]
    center = float(np.median(head)) if head.size else 0.0
    return (x - center + 180.0) % 360.0 - 180.0 + center


def _fill(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return x
    idx = np.arange(x.size)
    return np.interp(idx, idx[ok], x[ok])


def smooth_body_angles(
    yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray, fps: float, trusted: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yaw_smooth, pitch_smooth, roll_smooth)，单位度，每帧都有值（见模块说明）。"""
    def prep(x):
        x = np.asarray(x, dtype=float).copy()
        if trusted is not None:
            x[~np.asarray(trusted, dtype=bool)] = np.nan
        return x
    return (
        _fill(filter_euler_angle(recenter_yaw(prep(yaw)), YAW_FILT_HZ, fps)),
        _fill(filter_euler_angle(prep(pitch), PITCH_FILT_HZ, fps)),
        _fill(filter_euler_angle(prep(roll), ROLL_FILT_HZ, fps)),
    )


WING_HAMPEL_K = 7
WING_HAMPEL_N_SIGMA = 2.5
PHI_SGOLAY = (21, 7)
THETA_SGOLAY = (11, 5)
ETA_SGOLAY = (11, 5)
MATLAB_TUNING_FPS = 8000.0


def _scale_window(window: int, fps: float) -> int:
    w = max(int(round(window * fps / MATLAB_TUNING_FPS)), 3)
    return w + 1 if w % 2 == 0 else w


def _hampel_flags(x: np.ndarray, window: int, n_sigma: float) -> np.ndarray:
    flags = np.zeros(x.size, dtype=bool)
    for i in range(x.size):
        if not np.isfinite(x[i]):
            continue
        seg = x[max(0, i - window): i + window + 1]
        seg = seg[np.isfinite(seg)]
        med = np.median(seg)
        mad = 1.4826 * np.median(np.abs(seg - med))
        flags[i] = mad > 0 and abs(x[i] - med) > n_sigma * mad
    return flags


def unwrap_centered(x_deg: np.ndarray) -> np.ndarray:
    """把有界振荡的角度（φ、η）表示成连续序列：以整段圆均值为中心，每帧折回到 [中心 − 180°, 中心 + 180°)。
    折回切口落在离数据最远的地方（η 约在 −90°/270°，φ 约在 −280°/80°），每帧独立、不会像逐帧 unwrap 那样
    因个别离群帧累积出整圈漂移。NaN 保持 NaN。"""
    x = np.asarray(x_deg, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return x.copy()
    r = np.radians(x[ok])
    center = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return center + (x - center + 180.0) % 360.0 - 180.0


def _spline_fill(y: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(y)
    if not bad.any() or (~bad).sum() < 2:
        return y
    t = np.arange(y.size, dtype=float)
    good = np.nonzero(~bad)[0]
    y = y.copy()
    inner = bad & (t > good[0]) & (t < good[-1])
    y[inner] = PchipInterpolator(t[good], y[good])(t[inner])
    y[: good[0]] = y[good[0]]
    y[good[-1] + 1:] = y[good[-1]]
    return y


def _sgolay(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    w = min(window, y.size if y.size % 2 == 1 else y.size - 1)
    if w <= polyorder:
        w = polyorder + 1 + (polyorder % 2 == 0)
    if w > y.size or not np.all(np.isfinite(y)):
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=polyorder, mode="interp")


def smooth_wing_angles(
    phi: np.ndarray, theta: np.ndarray, eta: np.ndarray, fps: float, ignore: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单只翅的 (phi_smooth, theta_smooth, eta_smooth)，单位度，连续（不折回，见模块说明）。"""
    phi = unwrap_centered(phi)
    theta = np.asarray(theta, dtype=float).copy()  # θ ∈ [−90°, 90°]，不会碰到折回切口
    eta = unwrap_centered(eta)
    ign = np.zeros(phi.size, dtype=bool) if ignore is None else np.asarray(ignore, dtype=bool)
    k = max((_scale_window(2 * WING_HAMPEL_K + 1, fps) - 1) // 2, 1)
    theta_out = _hampel_flags(theta, k, WING_HAMPEL_N_SIGMA)
    eta_out = _hampel_flags(eta, k, WING_HAMPEL_N_SIGMA)
    phi_out = _hampel_flags(phi, k, WING_HAMPEL_N_SIGMA)

    phi[np.isnan(phi) | ign | theta_out | eta_out | phi_out] = np.nan
    theta[ign] = np.nan
    eta[ign | eta_out] = np.nan

    return (
        _sgolay(_spline_fill(phi), _scale_window(PHI_SGOLAY[0], fps), PHI_SGOLAY[1]),
        _sgolay(_spline_fill(theta), _scale_window(THETA_SGOLAY[0], fps), THETA_SGOLAY[1]),
        _sgolay(_spline_fill(eta), _scale_window(ETA_SGOLAY[0], fps), ETA_SGOLAY[1]),
    )
