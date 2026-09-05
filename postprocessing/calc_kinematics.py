"""一键跑通某数据集从raw splat.ply到可视化的全流程验收，落盘到
`<dataset_root父目录>/kinematics/`。

自适应检测数据集根目录(命令行第一个参数，不传则用`DEFAULT_DATASET_ROOT`)当前跑到了
哪一步，只补跑缺的部分，除数据来源外不支持任何其它参数:

  - 已有kinematics结果(`kinematics/kinematics_<dataset>.csv`存在) ->
    只做可视化: 角度图 + 重投影叠加图 + viser。
  - 没有kinematics结果，但已有T3标注(`_labeled.csv`) -> 跳过T1/T2/T3，
    只跑T4(kinematics pipeline) + 可视化。
  - 都没有，只有训练后的原始每帧splat.ply -> 从T1开始跑满T1-T4:
    T1 `utils.gaussian_features.compute_gaussian_features` 算逐点特征表 ->
    T2 `postprocessing.cleaning.mark_floaters.run_batch` 清洗(标if_keep) ->
    T3 `postprocessing.labeling.motion.label.run_batch` 跨帧运动累加密度分割得
    body/wing_L/wing_R(`_labeled.csv`) -> T4。

T4 = kinematics最终csv(`postprocessing.kinematics.pipeline.run_dataset`，输出到
`kinematics/kinematics_<dataset>.csv`+debug pkl) + 身体角度(yaw/pitch/roll)、
翅膀角度(phi/theta/eta, L/R)关于frame_id的图 + 重投影叠加图(从已跑通T3的帧里
等距挑最多`N_FRAMES`帧，调`postprocessing.viz.reprojection_viewer.run_batch`画)。

点云查看走已有的`postprocessing.viz.splat_viewer`(阻塞的viser web server)，画完图
后自动拉起，Ctrl+C退出。

用法:
    python -m postprocessing.calc_kinematics [dataset_root] [--half-window N]
    python -m postprocessing.calc_kinematics outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense

不传`dataset_root`则用本文件里的`DEFAULT_DATASET_ROOT`。`--half-window`只在从raw
splat.ply现跑T1-T3时生效(已有`_labeled.csv`或kinematics结果时T3不会重跑，这个参数
不起作用)，默认按`DEFAULT_DATASET_ROOT`(ctrl_009_002, 16000fps)锁定=36帧，见
`postprocessing.labeling.motion.density.HALF_WINDOW`的docstring——别的拍摄fps数据集
必须显式按比例换算传入(例如8000fps的3相机数据集应传`--half-window 18`)，不能依赖默认值。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning import mark_floaters  # noqa: E402
from postprocessing.kinematics import pipeline  # noqa: E402
from postprocessing.kinematics.diagnostics import plot_body_angles  # noqa: E402
from postprocessing.labeling.motion import label as labeling  # noqa: E402
from postprocessing.labeling.motion import density as motion_density  # noqa: E402
from postprocessing.viz import reprojection_viewer  # noqa: E402
from postprocessing.viz.reprojection_viewer import RAW_DATA_DIR  # noqa: E402
from utils.gaussian_features import compute_gaussian_features  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_ratio3_sh0_dense" / "ratio3_sh0_dense"
LABELED_FRAME_GLOB = "f*/*/*/*_labeled.csv"
"""不写死splatfacto/splatfacto-checkpoint(不同sweep的method目录名不同，见
postprocessing.cleaning.mark_floaters.latest_checkpoint_dir同样两选一的处理)，用
f<帧>/<method目录>/<时间戳>/*_labeled.csv这个统一深度匹配两种命名。T3是否已跑过、
以及T4 pipeline发现逐帧csv都用这个glob。"""

N_FRAMES = 5
"""重投影叠加图最多画几帧。"""
PORT = 8080
"""viser web server端口。"""

WING_ANGLE_ROWS = (
    ("phi_L", "phi_R", "phi (deg)"),
    ("theta_L", "theta_R", "theta (deg)"),
    ("eta_L", "eta_R", "eta (deg)"),
)


def plot_wing_angles(df, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (col_l, col_r, ylabel) in zip(axes, WING_ANGLE_ROWS):
        ax.plot(df["frame_id"], df[col_l], marker=".", ms=6, lw=1, label=col_l, color="tab:blue")
        ax.plot(df["frame_id"], df[col_r], marker=".", ms=6, lw=1, label=col_r, color="tab:orange")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame_id")
    fig.suptitle("Wing angles vs frame (raw, single-frame estimates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pick_reprojection_frames(frame_ids: list[int], n: int) -> list[str]:
    """从已算出kinematics的帧号里等距选最多n帧(不足n就全选)，返回'f0000'形式。"""
    frame_ids = sorted(frame_ids)
    if len(frame_ids) <= n:
        chosen = frame_ids
    else:
        idx = np.linspace(0, len(frame_ids) - 1, n).round().astype(int)
        chosen = sorted({frame_ids[i] for i in idx})
    return [f"f{fid:04d}" for fid in chosen]


def discover_frame_range(dataset_root: Path) -> tuple[int, int]:
    """扫描dataset-root下的f####帧目录，推断要处理的[min, max]帧号范围
    (处理该目录下的全部帧)。"""
    frame_dirs = sorted(p for p in dataset_root.glob("f[0-9][0-9][0-9][0-9]") if p.is_dir())
    if not frame_dirs:
        raise FileNotFoundError(f"未在 {dataset_root} 下找到 f#### 帧目录")
    indices = [int(p.name[1:]) for p in frame_dirs]
    return min(indices), max(indices)


def ensure_gaussian_features(dataset_root: Path, start: int, end: int) -> None:
    """T1: 对[start, end]里还没有 gaussian_features_f####.csv 的帧，从该帧最新训练的
    splat.ply(见mark_floaters.latest_checkpoint_dir同样的splatfacto-checkpoint/splatfacto
    两选一约定)现算并落盘；已存在的帧直接跳过复用，不重算。"""
    for frame_idx in range(start, end + 1):
        frame_name = f"f{frame_idx:04d}"
        frame_dir = dataset_root / frame_name
        if not frame_dir.is_dir():
            continue
        try:
            splat_dir = mark_floaters.latest_checkpoint_dir(frame_dir)
        except FileNotFoundError as e:
            print(f"  [T1][{frame_name}] SKIP: {e}")
            continue
        csv_path = splat_dir / f"gaussian_features_{frame_name}.csv"
        if csv_path.exists():
            continue
        df = compute_gaussian_features(splat_dir / "splat.ply", splat_dir / "dataparser_transforms.json")
        df.to_csv(csv_path, index=False)
        print(f"  [T1][{frame_name}] computed -> {csv_path}")


def find_splat_config(dataset_root: Path) -> Path | None:
    """splat_viewer的splat类display mode(Points/Ellipsoids/Gaussians/Hull)需要一份
    gpu/schedule/schedule.py的sweep config——按约定`outputs/<sweep_name>/<group>/`就是
    dataset_root，配套config固定落在`gpu/schedule/configs/<sweep_name>.json`。找不到就
    返回None，调用方据此只启用processed类(T1/T2/T3)、splat类checkbox置灰(不报错)。"""
    config_path = REPO_ROOT / "gpu" / "schedule" / "configs" / f"{dataset_root.parent.name}.json"
    return config_path if config_path.exists() else None


def detect_splat_method(dataset_root: Path) -> str:
    """自动探测这批帧训练用的是splatfacto还是splatfacto-checkpoint(sweep config本身不记录
    这个，取决于跑的时候有没有加--debug-checkpoint)，跟mark_floaters.latest_checkpoint_dir
    同样"splatfacto-checkpoint优先，否则splatfacto"的两选一约定。"""
    for frame_dir in sorted(dataset_root.glob("f[0-9][0-9][0-9][0-9]")):
        for dirname in ("splatfacto-checkpoint", "splatfacto"):
            if (frame_dir / dirname).is_dir():
                return dirname
    return "splatfacto"


def run_cleaning_and_labeling(dataset_root: Path, start: int, end: int,
                               half_window: int = motion_density.HALF_WINDOW) -> None:
    """把raw splat.ply变成T4 pipeline能读的`_labeled.csv`:
      T1 逐点特征表(ensure_gaussian_features，已存在的帧跳过) ->
      T2 `mark_floaters.run_batch` 清洗(标if_keep，产出`_marked.csv`) ->
      T3 `labeling.run_batch` kmeans聚类得body/wing_L/wing_R(产出`_labeled.csv`)。
    T2/T3两步任何一帧失败都只跳过该帧、打印警告，不中断整个批处理(沿用两个模块自己
    run_batch里"单帧异常catch住"的约定)。

    half_window透传给T3的`labeling.run_batch`，默认=motion_density.HALF_WINDOW(36帧，
    按ctrl_009_002数据集16000fps锁定)——别的拍摄fps数据集(比如8000fps的3相机数据集)
    调用方必须显式按比例传入换算后的值(见density.compute_body_voxels_for_frame
    docstring)，不要依赖这个默认值。"""
    print("  [T1] gaussian_features (跳过已算过的帧) ...")
    ensure_gaussian_features(dataset_root, start, end)

    print(f"\n  [T2] mark_floaters (cleaning) ...")
    _, t2_failures = mark_floaters.run_batch(dataset_root, start, end)
    if t2_failures:
        print(f"  [WARN][T2] 失败帧: {[f['frame'] for f in t2_failures]}")

    print(f"\n  [T3] motion accumulation labeling (half_window={half_window}) ...")
    frames = [f"f{i:04d}" for i in range(start, end + 1)]
    _, t3_failures = labeling.run_batch(frames, data_root=dataset_root, save_reprojection=False,
                                         half_window=half_window)
    if t3_failures:
        print(f"  [WARN][T3] 失败帧: {[f['frame'] for f in t3_failures]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path, nargs="?", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--half-window", type=int, default=motion_density.HALF_WINDOW,
                     help="T3 motion累加窗口半宽(帧)，只在从raw splat.ply现跑T1-T3时生效"
                          "(已有_labeled.csv或kinematics结果时不起作用)。默认按16000fps"
                          "锁定=36，别的拍摄fps数据集需要按比例换算显式传入，例如8000fps"
                          "的3相机数据集应传18，见density.HALF_WINDOW的docstring。")
    args = ap.parse_args()
    dataset_root = args.dataset_root
    raw_data_dir = RAW_DATA_DIR
    out_dir = dataset_root.parent / "kinematics"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_root.name
    csv_path = out_dir / f"kinematics_{dataset_name}.csv"

    # 自适应检测当前处理进度 -------------------------------------------------
    if csv_path.exists():
        print(f"== 检测到已有kinematics结果 {csv_path}，跳过T1-T4，只做可视化 ==")
        df = pd.read_csv(csv_path)
    else:
        if any(dataset_root.glob(LABELED_FRAME_GLOB)):
            print("== 检测到T3标注(_labeled.csv)，跳过T1-T3，只跑T4(kinematics+可视化) ==")
        else:
            print("== 未检测到T3标注，从raw splat.ply开始跑T1-T3 ==")
            start, end = discover_frame_range(dataset_root)
            run_cleaning_and_labeling(dataset_root, start, end, half_window=args.half_window)

        print(f"\n== T4 kinematics pipeline: {dataset_root} (frame_glob={LABELED_FRAME_GLOB!r}) ==")
        config = pipeline.PipelineConfig(output_dir=out_dir, write_debug=True, frame_glob=LABELED_FRAME_GLOB)
        # 用连续性链条 + 锚点校验(必做，见 correct_body_axis/sequence_axis.py)算出的
        # x_body 表，而不是逐帧独立PCA猜符号，并在写CSV前对eta_L/eta_R做整段
        # unwrap(圆域中值滤波去野值 + 180度翻转纠正 + unwrap，见
        # pipeline.run_dataset_with_eta_unwrap / eta_unwrap.py 的模块级说明)。
        df = pipeline.run_dataset_with_eta_unwrap(dataset_root, config)
        print(f"  -> {csv_path}  ({len(df)} frame(s))")

    ok = df[df["status"] == "ok"].reset_index(drop=True)
    bad = df[df["status"] != "ok"]
    print(f"  status=ok: {len(ok)}/{len(df)}")
    if len(bad):
        print(f"  非ok帧(已保留在csv里，画图/角度图会跳过): "
              f"{list(zip(bad['frame_id'].tolist(), bad['status'].tolist()))}")
    if ok.empty:
        print("  [WARN] 没有status=ok的帧，跳过角度图和重投影图。")
    else:
        for col in ("yaw", "pitch", "roll", "phi_L", "phi_R", "theta_L", "theta_R", "eta_L", "eta_R"):
            print(f"  {col}: min={ok[col].min():.2f}  max={ok[col].max():.2f}")

    # 身体角度 / 翅膀角度 vs frame -------------------------------------------
    print(f"\n== angle plots -> {out_dir} ==")
    if not ok.empty:
        body_path = out_dir / "body_angles.png"
        wing_path = out_dir / "wing_angles.png"
        plot_body_angles(ok, body_path)
        plot_wing_angles(ok, wing_path)
        print(f"  -> {body_path}")
        print(f"  -> {wing_path}")

    # 重投影叠加图(挑最多N_FRAMES帧) ------------------------------------------
    print(f"\n== reprojection overlays (<= {N_FRAMES} frames) ==")
    reproj_dir = out_dir / "reprojection"
    if ok.empty:
        print("  跳过(没有status=ok的帧)")
    else:
        frames = pick_reprojection_frames(ok["frame_id"].tolist(), N_FRAMES)
        print(f"  frames: {frames}")
        reprojection_viewer.run_batch(frames, dataset_root, raw_data_dir, reproj_dir)

    print(f"\n验收产物全部在: {out_dir}")

    frame_ids = ok["frame_id"].tolist() if not ok.empty else df["frame_id"].tolist()
    if not frame_ids:
        print("[viser] 没有可显示的帧，跳过。")
        return
    start, end = min(frame_ids), max(frame_ids)
    viewer_cmd = [
        sys.executable, "-m", "postprocessing.viz.splat_viewer",
        "--data-root", str(dataset_root),
        "--raw-data-dir", str(raw_data_dir),
        "--start", str(start), "--end", str(end),
        "--port", str(PORT),
    ]
    config_path = find_splat_config(dataset_root)
    if config_path is not None:
        method = detect_splat_method(dataset_root)
        viewer_cmd += ["--config", str(config_path), "--group", dataset_root.name, "--method", method]
        print(f"  [viser] 找到splat config {config_path}，method={method}，一并显示Points/Ellipsoids/Gaussians/Hull")
    else:
        print(f"  [viser] 未找到 {REPO_ROOT / 'gpu' / 'schedule' / 'configs' / (dataset_root.parent.name + '.json')}，"
              f"splat类display mode(Points/Ellipsoids/Gaussians/Hull)会置灰，只显示T1/T2/T3")
    print(f"\n[viser] 拉起点云查看器(Ctrl+C退出): {' '.join(viewer_cmd)}")
    subprocess.run(viewer_cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
