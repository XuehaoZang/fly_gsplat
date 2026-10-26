"""第 1 步 dataset (a)：稀疏录像 -> 逐帧 PNG。

    python -m dataset.sparse_to_images --sparse /mnt/x/.../Expr_009_mov_003 --calib <calibration_easyWandData.mat> \
        --out data/ctrl_009/003 --frames 425-905

输入：<sparse>/Camera{c}*_sparse.mat（MATLAB v7.3 / HDF5；/frames/indIm 的第 f 项是第 f 帧的 (行, 列, 灰度)，
     行列从 1 开始计），按文件名排序依次对应相机 1..N。
输出：<out>/f####/images/P{f}CAM{c}.png，白底（255）8 位灰度、整幅分辨率、无损。
--frames a-b 含 a 不含 b（与 train 配置的 frames 一致）。图像尺寸取自 easyWand 标定。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from .make_transforms import load_easywand


def parse_frames(spec: str) -> list[int]:
    a, b = (int(x) for x in spec.split("-"))
    return list(range(a, b))


def sparse_files(sparse_dir: Path) -> list[Path]:
    files = sorted(Path(sparse_dir).glob("Camera*_sparse.mat"))
    if not files:
        raise SystemExit(f"no Camera*_sparse.mat under {sparse_dir}")
    return files


def frame_image(sp: h5py.File, frame: int, w: int, h: int) -> np.ndarray:
    """一台相机第 frame 帧的整幅图（白底，前景为录到的灰度值）。"""
    ind = sp[sp["/frames/indIm"][0][frame]][:]
    if ind.shape[0] == 3:
        ind = ind.T
    im = np.full((h, w), 255, dtype=np.uint8)
    if ind.size > 0:
        rows = ind[:, 0].astype(int) - 1
        cols = ind[:, 1].astype(int) - 1
        vals = ind[:, 2].astype(float)
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        im[rows[ok], cols[ok]] = vals[ok].astype(np.uint8)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparse", type=Path, required=True, help="含 Camera*_sparse.mat 的目录（Linux 路径，如 /mnt/x/...）")
    ap.add_argument("--calib", type=Path, required=True, help="calibration_easyWandData.mat（取图像尺寸）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", required=True, help="a-b，含 a 不含 b")
    args = ap.parse_args()

    ew = load_easywand(args.calib)
    w = int(ew.imageWidth[0] if isinstance(ew.imageWidth, np.ndarray) else 1280)
    h = int(ew.imageHeight[0] if isinstance(ew.imageHeight, np.ndarray) else 800)
    files = sparse_files(args.sparse)
    if len(files) != ew.nCams:
        raise SystemExit(f"{len(files)} sparse files but calibration has {ew.nCams} cameras")
    frames = parse_frames(args.frames)

    handles = [h5py.File(f, "r") for f in files]
    try:
        for f in frames:
            img_dir = args.out / f"f{f:04d}" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for c, sp in enumerate(handles, start=1):
                cv2.imwrite(str(img_dir / f"P{f}CAM{c}.png"), frame_image(sp, f, w, h))
    finally:
        for sp in handles:
            sp.close()
    print(f"[sparse_to_images] {len(frames)} frames x {len(files)} cams -> {args.out}")


if __name__ == "__main__":
    main()
