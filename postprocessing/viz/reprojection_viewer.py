"""T5: 重投影叠加图 —— 把某帧点云投影回4个相机原图，叠加显示，用于诊断
T2(if_keep)/T3(part_label, postprocessing/labeling/labeling.py)的清理/聚类结果。

通用版: 只需要csv里有x/y/z列，part_label/if_keep列都可选。没有part_label列
(T1/T2阶段)时退化成单色全点模式；没有if_keep列时不画x叉号。跟
postprocessing/labeling/labeling.py::plot_labeled_reprojection是同一份重投影/
画序/配色逻辑的通用重构版，labeling.py侧改为直接调用这里的plot_reprojection_overlay。

用法:
    python -m postprocessing.viz.reprojection_viewer --frame f0061
    python -m postprocessing.viz.reprojection_viewer --start 0 --end 5
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import normalize_frame_name  # noqa: E402
from postprocessing.viz._colors import PART_COLORS, SINGLE_COLOR  # noqa: E402
from postprocessing.viz._io import load_stage_csv  # noqa: E402
from postprocessing.viz.splat_viewer import DATASET_DIR, RAW_DATA_DIR  # noqa: E402
from utils.camera import CameraConfig  # noqa: E402
from utils.reproject import project_points  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "postprocessing" / "viz" / "eda_outputs" / "reprojection"

PART_DRAW_ORDER = ("body", "wing_L", "wing_R")  # body先画(背景，点数多)，两翼后画(前景)盖上

CROP_PAD_FRAC = 0.6
"""自动裁剪视野的padding，占点云bbox宽/高的比例，见_crop_to_points。"""
CROP_MIN_PAD_PX = 40
"""bbox很小(比如只有几个点挤在一起)时的最小padding(像素)，避免裁剪框比散点marker还小。"""


def _crop_to_points(ax, us: np.ndarray, vs: np.ndarray, img_shape: tuple[int, int],
                     pad_frac: float = CROP_PAD_FRAC, min_pad_px: float = CROP_MIN_PAD_PX) -> None:
    """把视野裁剪到散点bbox(留padding)附近，而不是整张原图—— 果蝇在整帧里往往只占
    很小一块(如1280x800里~80x80px)，四相机原图默认铺满画布时，密集的半透明散点会
    互相叠加到接近不透明，把底下的果蝇轮廓完全盖住(看着像没画背景图)。只改视野范围
    (xlim/ylim)，不改imshow的数据本身，所以不影响散点位置/颜色/alpha或图片分辨率。"""
    h, w = img_shape
    x0, x1 = float(us.min()), float(us.max())
    y0, y1 = float(vs.min()), float(vs.max())
    pad_x = max((x1 - x0) * pad_frac, min_pad_px)
    pad_y = max((y1 - y0) * pad_frac, min_pad_px)
    ax.set_xlim(max(0.0, x0 - pad_x), min(float(w), x1 + pad_x))
    ax.set_ylim(min(float(h), y1 + pad_y), max(0.0, y0 - pad_y))  # 图像坐标y向下，反转ylim


def plot_reprojection_overlay(frame: str, df: pd.DataFrame, out_path: Path,
                               raw_data_dir: Path = RAW_DATA_DIR, title_suffix: str = "") -> None:
    """2x2四相机重投影叠加图。背景相机原图不透明画在最底层，散点(alpha=0.3)叠加在上面，
    视野自动裁剪到该相机散点bbox附近(见_crop_to_points，没有有效点则退回整张原图)。
    有part_label列时按PART_DRAW_ORDER分层画(body先画、wing_L/wing_R后画盖上)；没有
    part_label列(T1/T2阶段)则退化成单色全点模式。有if_keep列时if_keep=False的点直接
    不画（从散点里排除，不是画出来再标记）；没有该列则全部点都画。"""
    frame_data_dir = raw_data_dir / frame
    with open(frame_data_dir / "transforms.json") as f:
        cam_frames = json.load(f)["frames"]

    xyz = df[["x", "y", "z"]].to_numpy()
    has_label = "part_label" in df.columns
    part_label = df["part_label"].to_numpy() if has_label else None
    has_keep = "if_keep" in df.columns
    is_dropped = ~df["if_keep"].astype(bool).to_numpy() if has_keep else np.zeros(len(df), dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, cam_frame in enumerate(cam_frames[:4]):
        ax = axes[idx]
        cam = CameraConfig.from_opengl(cam_frame)
        cam.cam_idx = idx + 1
        img_path = frame_data_dir / cam_frame["file_path"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            ax.set_title(f"CAM {cam.cam_idx}: missing image")
            ax.axis("off")
            continue

        uv, depth = project_points(xyz, cam)
        us, vs = uv[:, 0], uv[:, 1]
        valid = (depth > 0) & ~is_dropped
        ax.imshow(img, cmap="gray")

        if has_label:
            for lab in PART_DRAW_ORDER:
                mask = valid & (part_label == lab)
                ax.scatter(us[mask], vs[mask], s=10, c=[PART_COLORS[lab]], marker="o", alpha=0.3)
        else:
            ax.scatter(us[valid], vs[valid], s=10, c=[SINGLE_COLOR], marker="o", alpha=0.3)

        if valid.any():
            _crop_to_points(ax, us[valid], vs[valid], img.shape)

        ax.set_title(f"CAM {cam.cam_idx}: {img_path.name}")
        ax.axis("off")

    color_note = "  gray=body  blue=wing_L  red=wing_R" if has_label else "  (no part_label, single color)"
    drop_note = "  if_keep=False points hidden" if has_keep else ""
    fig.suptitle(f"{frame}{title_suffix}{color_note}{drop_note}", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(frame: str, data_root: Path = DATASET_DIR, raw_data_dir: Path = RAW_DATA_DIR,
        out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    frame = normalize_frame_name(frame)
    csv_path, df = load_stage_csv(frame, data_root)

    confidence = df["confidence"].iloc[0] if "confidence" in df.columns else None
    suffix = f"  [confidence={confidence}]" if confidence is not None else ""

    out_path = out_dir / f"reproj_{frame}.png"
    plot_reprojection_overlay(frame, df, out_path, raw_data_dir=raw_data_dir, title_suffix=suffix)
    print(f"[{frame}] csv -> {csv_path}")
    print(f"[{frame}] reprojection -> {out_path}")
    return out_path


def run_batch(frames: list[str], data_root: Path = DATASET_DIR, raw_data_dir: Path = RAW_DATA_DIR,
              out_dir: Path = DEFAULT_OUT_DIR) -> list[dict]:
    """逐帧处理，单帧异常catch住、跳过、记录帧号，不中断整批(同
    postprocessing/labeling/labeling.py::run_batch的容错约定)。"""
    failures = []
    for frame in frames:
        try:
            run(frame, data_root, raw_data_dir, out_dir)
        except Exception as e:
            failures.append({"frame": frame, "error": f"{type(e).__name__}: {e}"})
            print(f"[{frame}] FAILED: {type(e).__name__}: {e}")
    print(f"\n{'=' * 60}\n{len(frames)}帧请求汇总: 成功 {len(frames) - len(failures)}  失败 {len(failures)}")
    if failures:
        print(f"失败帧: {[f['frame'] for f in failures]}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=str, default=None, help="单帧模式，如 f0061 或 61")
    parser.add_argument("--start", type=int, default=None, help="批处理模式起始帧号(含)")
    parser.add_argument("--end", type=int, default=None, help="批处理模式结束帧号(含)")
    parser.add_argument("--data-root", type=str, default=str(DATASET_DIR),
                         help="存放各帧gaussian_features_*.csv的数据集根目录")
    parser.add_argument("--raw-data-dir", type=str, default=str(RAW_DATA_DIR),
                         help="存放各帧transforms.json+images的原始数据目录")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="图像输出目录")
    args = parser.parse_args()

    data_root, raw_data_dir, out_dir = Path(args.data_root), Path(args.raw_data_dir), Path(args.out_dir)

    if args.frame is not None:
        run(args.frame, data_root, raw_data_dir, out_dir)
    elif args.start is not None and args.end is not None:
        frames = [f"f{i:04d}" for i in range(args.start, args.end + 1)]
        run_batch(frames, data_root, raw_data_dir, out_dir)
    else:
        parser.error("需要指定 --frame，或者 --start/--end")


if __name__ == "__main__":
    main()
