import warnings
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
)
from scipy.ndimage import label as cc_label
from skimage.filters import threshold_otsu
from skimage.morphology import disk, remove_small_objects

# --- 参数表（seg2d_spec.md 第7节，v4）---
BODY_MIN_SIZE = 100
DEFAULT_OPEN_RADIUS = 5
BODY_DILATE_RADIUS = 7
WING_CLOSE_RADIUS = 7
MAX_WING_COMPONENTS = 2
MIN_FOREGROUND_PX = 50
DEGENERATE_STD_TH = 1.0  # 前景灰度值方差小于此认为退化（Otsu在近乎常数分布上不可靠）

DEFAULT_DELTA = 36
DEFAULT_FIT_SCOPE = 100
DEFAULT_CM_POLY_DEGREE = 2
DEFAULT_BODY_TH_RATIO = 0.85
BODY_PX_DEGENERATE_MIN = 20
BODY_PX_DEGENERATE_MAX_RATIO = 0.9


def load_sparse_frame(sparse_path: Path, frame_idx: int,
                       frame_size: Tuple[int, int] = (800, 1280)) -> np.ndarray:
    """从 Camera{cam}_sparse.mat (h5py, v7.3) 读出某一帧，返回 (H,W) uint8 密集灰度图，背景=0。"""
    im = np.zeros(frame_size, dtype=np.uint8)
    with h5py.File(str(sparse_path), "r") as f:
        refs = f["/frames/indIm"][0]
        indIm = f[refs[frame_idx]][:]
        if indIm.shape[0] == 3:
            indIm = indIm.T

        if indIm.size > 0:
            rows = indIm[:, 0].astype(int) - 1
            cols = indIm[:, 1].astype(int) - 1
            vals = indIm[:, 2].astype(np.uint8)

            valid = (rows >= 0) & (rows < frame_size[0]) & (cols >= 0) & (cols < frame_size[1])
            im[rows[valid], cols[valid]] = vals[valid]

    return im


def _read_coords(f: h5py.File, refs, frame_idx: int,
                  frame_size: Tuple[int, int]) -> np.ndarray:
    """从已打开的h5py.File里读一帧坐标，(N,2) 0-based [row,col]。"""
    indIm = f[refs[frame_idx]][:]
    if indIm.shape[0] == 3:
        indIm = indIm.T
    if indIm.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    rows = indIm[:, 0].astype(np.int64) - 1
    cols = indIm[:, 1].astype(np.int64) - 1
    valid = (rows >= 0) & (rows < frame_size[0]) & (cols >= 0) & (cols < frame_size[1])
    return np.stack([rows[valid], cols[valid]], axis=1)


def load_sparse_coords(sparse_path: Path, frame_idx: int,
                        frame_size: Tuple[int, int] = (800, 1280)) -> np.ndarray:
    """只读某一帧的前景坐标，(N,2) 0-based [row,col]，不建立800x1280密集图（motion对齐用）。"""
    with h5py.File(str(sparse_path), "r") as f:
        refs = f["/frames/indIm"][0]
        return _read_coords(f, refs, frame_idx, frame_size)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留面积最大的连通域，其余置0（全False输入原样返回）。"""
    labels, num = cc_label(mask)
    if num == 0:
        return mask
    areas = np.bincount(labels.ravel())
    areas[0] = 0  # 排除背景label
    largest_id = int(np.argmax(areas))
    return labels == largest_id


def _build_mask_from_coords(coords: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(frame_size, dtype=bool)
    if coords.shape[0] == 0:
        return mask
    rows, cols = coords[:, 0], coords[:, 1]
    valid = (rows >= 0) & (rows < frame_size[0]) & (cols >= 0) & (cols < frame_size[1])
    mask[rows[valid], cols[valid]] = True
    return mask


def compute_motion_counts(sparse_path: Path, frame_idx: int,
                           delta: int = DEFAULT_DELTA,
                           fit_scope: int = DEFAULT_FIT_SCOPE,
                           cm_poly_degree: int = DEFAULT_CM_POLY_DEGREE,
                           frame_size: Tuple[int, int] = (800, 1280),
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, dict]:
    """Step A/B/C（seg2d_spec.md 3.1节）：算出对齐后每个像素坐标的重复计数，
    不做任何body_th_ratio阈值判定。

    这是整个motion算法里最贵的部分（要读±(delta+fit_scope)范围的帧），
    而body_th_ratio需要经验调参、大概率要试好几个值——所以拆成这个函数
    单独返回(uniq_coords, counts)，配合threshold_counts()复用，sweep
    body_th_ratio时不用重跑Step A/B/C。

    返回 (uniq_coords, counts, ok, info)。ok=False时（Step B的CM拟合本身
    失败，跟body_th_ratio无关）uniq_coords/counts为None。
    info 里带 window_size/target_fg，供threshold_counts()和退化检测复用。

    数据集边界（frame_idx附近没有足够的±(delta+fit_scope)帧）时，窗口会
    被截断为实际可读到的帧数，而不是硬编码假设的 2*delta+1 —— 这是对
    spec伪代码（假设两侧都有足够padding）的必要工程化处理，不影响窗口
    完整时的行为。
    """
    with h5py.File(str(sparse_path), "r") as f:
        refs = f["/frames/indIm"][0]
        n_frames = len(refs)

        coord_cache: dict = {}

        def get_coords(fr: int) -> Optional[np.ndarray]:
            if fr < 0 or fr >= n_frames:
                return None
            if fr not in coord_cache:
                coord_cache[fr] = _read_coords(f, refs, fr, frame_size)
            return coord_cache[fr]

        # --- Step A: 粗估CM轨迹（不对齐，靠窗口内"全程不动"的像素近似body位置）---
        fit_offsets = list(range(frame_idx - fit_scope, frame_idx + fit_scope + 1))
        cm_row = np.full(len(fit_offsets), np.nan)
        cm_col = np.full(len(fit_offsets), np.nan)
        for k, fr_i in enumerate(fit_offsets):
            window_frames = [fr for fr in range(fr_i - delta, fr_i + delta + 1)
                              if get_coords(fr) is not None]
            if not window_frames:
                continue
            window_coords = [get_coords(fr) for fr in window_frames]
            all_coords = np.concatenate(window_coords, axis=0)
            if all_coords.shape[0] == 0:
                continue
            uniq, counts = np.unique(all_coords, axis=0, return_counts=True)
            body_candidate = uniq[counts == len(window_frames)]
            if body_candidate.shape[0] > 0:
                cm_row[k] = body_candidate[:, 0].mean()
                cm_col[k] = body_candidate[:, 1].mean()

        # --- Step B: 多项式平滑CM轨迹，算出每个邻近帧对齐所需的平移量 ---
        valid = ~np.isnan(cm_row)
        n_valid = int(valid.sum())
        if n_valid < fit_scope // 2:
            return None, None, False, {"reason": "too_few_valid_cm_frames", "n_valid": n_valid}

        valid_frames = np.array(fit_offsets)[valid]
        p_row = np.polyfit(valid_frames, cm_row[valid], cm_poly_degree)
        p_col = np.polyfit(valid_frames, cm_col[valid], cm_poly_degree)

        def cm_smooth(fr: int) -> Tuple[float, float]:
            return float(np.polyval(p_row, fr)), float(np.polyval(p_col, fr))

        cm_target = cm_smooth(frame_idx)

        # --- Step C: 对齐后重复像素投票，得到(uniq_coords, counts) ---
        loop_frames = [fr for fr in range(frame_idx - delta, frame_idx + delta + 1)
                        if get_coords(fr) is not None]
        aligned_list = []
        for fr_loop in loop_frames:
            r0, c0 = cm_smooth(fr_loop)
            dr = int(round(r0 - cm_target[0]))
            dc = int(round(c0 - cm_target[1]))
            coords = get_coords(fr_loop)
            if coords.shape[0] == 0:
                continue
            aligned_list.append(coords - np.array([dr, dc]))

        window_size = len(loop_frames)
        if window_size == 0 or not aligned_list:
            return None, None, False, {"reason": "no_aligned_frames", "n_valid": n_valid}

        aligned_all = np.concatenate(aligned_list, axis=0)
        uniq, counts = np.unique(aligned_all, axis=0, return_counts=True)

        target_coords = get_coords(frame_idx)
        target_fg = int(target_coords.shape[0]) if target_coords is not None else 0

    return uniq, counts, True, {
        "n_valid_cm_frames": n_valid,
        "window_size": window_size,
        "target_fg": target_fg,
    }


def threshold_counts(uniq_coords: np.ndarray, counts: np.ndarray, window_size: int,
                      body_th_ratio: float, frame_size: Tuple[int, int] = (800, 1280),
                      ) -> np.ndarray:
    """只做 counts > body_th_ratio*window_size 判定 + 建mask，不重跑Step A/B/C。"""
    body_th = body_th_ratio * window_size
    body_coords = uniq_coords[counts > body_th]
    return _build_mask_from_coords(body_coords, frame_size)


def segment_body_motion(sparse_path: Path, frame_idx: int,
                         delta: int = DEFAULT_DELTA,
                         fit_scope: int = DEFAULT_FIT_SCOPE,
                         cm_poly_degree: int = DEFAULT_CM_POLY_DEGREE,
                         body_th_ratio: float = DEFAULT_BODY_TH_RATIO,
                         frame_size: Tuple[int, int] = (800, 1280),
                         ) -> Tuple[Optional[np.ndarray], bool, dict]:
    """motion-based body分割主算法（seg2d_spec.md 第3.1节）：
    compute_motion_counts + threshold_counts + 退化检测的薄封装。

    body是刚性躯干，运动轨迹可用低阶多项式平滑拟合；wing独立扑动不跟随
    身体整体平移。把±delta帧按body的平滑CM轨迹对齐到目标帧后，body像素
    应在几乎所有对齐帧里重复出现，wing像素对不齐、重复次数低。
    """
    uniq, counts, ok, info = compute_motion_counts(
        sparse_path, frame_idx, delta, fit_scope, cm_poly_degree, frame_size,
    )
    if not ok:
        return None, False, info

    window_size = info["window_size"]
    target_fg = info["target_fg"]
    body_mask_motion = threshold_counts(uniq, counts, window_size, body_th_ratio, frame_size)

    body_px = int(body_mask_motion.sum())
    if body_px < BODY_PX_DEGENERATE_MIN or (
        target_fg > 0 and body_px > BODY_PX_DEGENERATE_MAX_RATIO * target_fg
    ):
        return body_mask_motion, False, {**info, "reason": "degenerate_body_size", "body_px": body_px}

    return body_mask_motion, True, {**info, "body_px": body_px}


def segment_body_intensity(gray: np.ndarray) -> np.ndarray:
    """intensity兜底（seg2d_spec.md 第3.2节），逻辑同v2/v3。

    只做阈值分割，不做形态学清理——清理统一放在segment_body里，
    motion/intensity两条路径共用同一套后处理。
    """
    silhouette = gray > 0
    if silhouette.sum() < MIN_FOREGROUND_PX:
        warnings.warn(
            f"segment_body_intensity: 前景像素数({int(silhouette.sum())}) < "
            f"MIN_FOREGROUND_PX({MIN_FOREGROUND_PX})，视为空/坏帧，返回全False"
        )
        return np.zeros(gray.shape, dtype=bool)

    values = gray[silhouette]
    std = float(values.std())
    if std < DEGENERATE_STD_TH:
        th = float(np.median(values))
        warnings.warn(
            f"segment_body_intensity: 前景像素灰度方差极小(std={std:.3f})，"
            f"Otsu在此退化，改用前景中位数 TH={th:.1f}"
        )
    else:
        th = float(threshold_otsu(values))

    return (gray <= th) & silhouette


def cleanup_body_mask(body_bin: np.ndarray, open_radius: int = DEFAULT_OPEN_RADIUS) -> np.ndarray:
    """形态学清理（v3结论）：body-wing根部连接过渡，开运算是断开的关键步骤，
    不管body_bin来自motion还是intensity都要过这一步。单独提出来是为了
    body_th_ratio sweep时能复用同一套清理逻辑，不用在调用方重复写。
    """
    body_bin = remove_small_objects(body_bin, min_size=BODY_MIN_SIZE)
    body_bin = binary_opening(body_bin, structure=disk(open_radius))
    body_bin = keep_largest_component(body_bin)
    body_bin = binary_fill_holes(body_bin)
    return body_bin


def segment_body(sparse_path: Path, gray: np.ndarray, frame_idx: int,
                  delta: int = DEFAULT_DELTA,
                  fit_scope: int = DEFAULT_FIT_SCOPE,
                  cm_poly_degree: int = DEFAULT_CM_POLY_DEGREE,
                  body_th_ratio: float = DEFAULT_BODY_TH_RATIO,
                  open_radius: int = DEFAULT_OPEN_RADIUS,
                  ) -> Tuple[np.ndarray, str, dict]:
    """汇总+形态学清理（seg2d_spec.md 第3.3节）。motion优先，失败/退化时落到intensity兜底。"""
    body_mask_motion, ok, info = segment_body_motion(
        sparse_path, frame_idx, delta, fit_scope, cm_poly_degree, body_th_ratio,
        frame_size=gray.shape,
    )
    if ok:
        body_bin, source = body_mask_motion, "motion"
    else:
        body_bin = segment_body_intensity(gray)
        source = "intensity_fallback"
        warnings.warn(
            f"segment_body: motion-based失败/退化(reason={info.get('reason')})，"
            f"回退到intensity兜底"
        )

    body_bin = cleanup_body_mask(body_bin, open_radius)
    return body_bin, source, info


def segment_wing(gray: np.ndarray, body_mask: np.ndarray, leg_th: int = 100) -> np.ndarray:
    """在扣除body之后的前景区域里，找面积最大的最多2个连通域作为wing。"""
    silhouette = gray > 0
    body_dilated = binary_dilation(body_mask, structure=disk(BODY_DILATE_RADIUS))
    diff = silhouette & ~body_dilated
    diff = binary_closing(diff, structure=disk(WING_CLOSE_RADIUS))

    labels, num = cc_label(diff)
    if num == 0:
        return np.zeros(gray.shape, dtype=bool)

    areas = np.bincount(labels.ravel())
    areas[0] = 0  # 排除背景label

    component_ids = np.argsort(areas)[::-1]
    valid_ids = [cid for cid in component_ids if areas[cid] >= leg_th][:MAX_WING_COMPONENTS]

    if not valid_ids:
        largest_id = int(component_ids[0])
        warnings.warn(
            f"segment_wing: 没有连通域满足 area>={leg_th}，"
            f"保留面积最大的一个 (area={areas[largest_id]}) 作为fallback"
        )
        valid_ids = [largest_id]

    wing_mask = np.isin(labels, valid_ids)
    return wing_mask


def segment_body_wing(sparse_path: Path, cam: int, frame_idx: int,
                       delta: int = DEFAULT_DELTA,
                       fit_scope: int = DEFAULT_FIT_SCOPE,
                       cm_poly_degree: int = DEFAULT_CM_POLY_DEGREE,
                       body_th_ratio: float = DEFAULT_BODY_TH_RATIO,
                       open_radius: int = DEFAULT_OPEN_RADIUS,
                       leg_th: int = 100) -> Tuple[np.ndarray, dict]:
    """汇总，生成 (label_map, meta)（seg2d_spec.md 第5节）。

    label约定: 0=背景, 1=body, 2=wing(不分左右), 3=前景未分类。
    meta: {'cam', 'body_source'("motion"/"intensity_fallback"), 'body_info'}，
    body_info是segment_body_motion返回的info dict，成功时含n_valid_cm_frames/
    body_px/window_size，失败时含reason，用于统计motion法实际成功率。
    """
    gray = load_sparse_frame(sparse_path, frame_idx)
    body_mask, source, info = segment_body(
        sparse_path, gray, frame_idx, delta, fit_scope, cm_poly_degree,
        body_th_ratio, open_radius,
    )
    wing_mask = segment_wing(gray, body_mask, leg_th)

    label_map = np.zeros(gray.shape, dtype=np.uint8)
    label_map[gray > 0] = 3
    label_map[body_mask] = 1
    label_map[wing_mask] = 2  # wing与body理论上不重叠；若重叠，wing覆盖body

    meta = {"cam": cam, "body_source": source, "body_info": info}
    return label_map, meta
