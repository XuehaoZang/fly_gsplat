"""
prepare_round1_5.py

Round 1.5（sweep_hyper_params.md）预处理层面消融的数据准备 + config生成。
P7(mask二值化阈值)/P9(hull采样点数)只需要重新生成init_points.ply，images/直接软链接
原frame目录，不物理复制(镜像目录写法沿用run/serial/batch_densify_6groups_100frames.py
的prepare_dense_frame先例)。P8(去腿)images本身要变，必须重新跑一遍generate_dataset
(remove_appendages=True)，不能软链接。

每个变体各建一份镜像base_name(data/ctrl_119_3cam_<variant>/<mov>/)，ply统一命名
init_points.ply、transforms.json的ply_file_path也统一指向它 -- 这样
gpu/schedule/schedule.py的Phase A幂等检查(只认"transforms.json"+"init_points.ply"
是否都在)会直接判定"已就绪"而跳过，不会用它内置的默认参数把这里精心生成的变体覆盖掉。

不改动 gpu/schedule/schedule.py / common.py / worker.py 任何代码。

用法:
    python gpu/schedule/prepare_round1_5.py            # 准备数据 + 写config
    python gpu/schedule/prepare_round1_5.py --configs-only   # 只重写config(数据已就绪时)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
CONFIG_OUT_DIR = REPO / "gpu" / "schedule" / "configs" / "round1_5"

VIDEOS = {
    "004": {
        "orig_base": "data/ctrl_119_3cam/004",
        "sparse_dir": "data/tests/ctrl_3cam_test/Sparse/Expr_119_mov_004",
        "frames": (730, 880),  # [start, end) -- widened 100->150 for Round 2 (kinematics needs
                                # >=150 contiguous frames, see sweep_hyper_params.md Round 2 notes)
    },
    "010": {
        "orig_base": "data/ctrl_119_3cam/010",
        "sparse_dir": "data/tests/ctrl_3cam_test/Sparse/Expr_119_mov_010",
        "frames": (373, 523),
    },
}

MAX_ITERS = 2000
BASE_PARAM_SET = {
    "P_baseline": [
        "--pipeline.model.use-scale-regularization", "True",
        "--pipeline.model.max-gauss-ratio", "3.0",
        "--pipeline.model.sh-degree", "0",
        "--pipeline.model.warmup-length", "50",
        "--pipeline.model.stop-split-at", "1800",
        "--pipeline.model.densify-grad-thresh", "0.0004",
        "--pipeline.model.refine-every", "50",
    ]
}

# variant name -> kind + generate_hull/generate_dataset kwargs
VARIANTS = {
    "p7_thresh20": {"kind": "hull_only", "hull_kwargs": {"mask_threshold": 20}},
    "p7_thresh50": {"kind": "hull_only", "hull_kwargs": {"mask_threshold": 50}},
    "p8_leg_erosion": {"kind": "full_regen",
                        "dataset_kwargs": {"remove_appendages": True},
                        "hull_kwargs": {"remove_appendages": True}},
    "p9_hull30k": {"kind": "hull_only", "hull_kwargs": {"n_samples": 30_000}},
    "p9_hull100k": {"kind": "hull_only", "hull_kwargs": {"n_samples": 100_000}},
}


def _link_calibration(orig_base_dir: Path, mirror_base_dir: Path) -> None:
    mirror_base_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("calibration_easyWandData.mat", "camera_KRX0.mat"):
        src = orig_base_dir / fname
        dst = mirror_base_dir / fname
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())


def prepare_hull_only_frame(orig_frame_dir: Path, mirror_frame_dir: Path, hull_kwargs: dict) -> None:
    from generate_hull import generate_hull

    if (mirror_frame_dir / "transforms.json").exists() and (mirror_frame_dir / "init_points.ply").exists():
        return
    mirror_frame_dir.mkdir(parents=True, exist_ok=True)

    images_link = mirror_frame_dir / "images"
    if not images_link.exists():
        images_link.symlink_to((orig_frame_dir / "images").resolve())

    transforms = json.loads((orig_frame_dir / "transforms.json").read_text())
    transforms["ply_file_path"] = "init_points.ply"  # already default, kept explicit
    (mirror_frame_dir / "transforms.json").write_text(json.dumps(transforms, indent=4))

    generate_hull(str(mirror_frame_dir), if_viser=False, out_name="init_points.ply", **hull_kwargs)


def prepare_full_regen_frame(mirror_frame_dir: Path, sparse_dir: str, frame_idx: int,
                              mirror_base_dir: Path, dataset_kwargs: dict, hull_kwargs: dict) -> None:
    from generate_dataset import generate_dataset
    from generate_hull import generate_hull

    if (mirror_frame_dir / "transforms.json").exists() and (mirror_frame_dir / "init_points.ply").exists():
        return
    mirror_frame_dir.mkdir(parents=True, exist_ok=True)

    generate_dataset(str(mirror_frame_dir), sparse_dir, target_frame=frame_idx,
                      if_crop=False, white_bg=True, if_mask=False,
                      calib_dir=str(mirror_base_dir), **dataset_kwargs)
    generate_hull(str(mirror_frame_dir), if_viser=False, out_name="init_points.ply", **hull_kwargs)


def prepare_variant(variant_name: str, variant_spec: dict) -> str:
    """returns the (single, shared) base_name PREFIX for this variant -- actual
    base_name per video is f'{prefix}/{mov}'."""
    prefix = f"data/ctrl_119_3cam_{variant_name}"
    for mov, info in VIDEOS.items():
        orig_base_dir = REPO / info["orig_base"]
        mirror_base_dir = REPO / prefix / mov
        _link_calibration(orig_base_dir, mirror_base_dir)

        start, end = info["frames"]
        for frame_idx in range(start, end):
            orig_frame_dir = orig_base_dir / f"f{frame_idx:04d}"
            mirror_frame_dir = mirror_base_dir / f"f{frame_idx:04d}"

            if variant_spec["kind"] == "hull_only":
                prepare_hull_only_frame(orig_frame_dir, mirror_frame_dir, variant_spec["hull_kwargs"])
            else:
                prepare_full_regen_frame(mirror_frame_dir, info["sparse_dir"], frame_idx,
                                          mirror_base_dir, variant_spec["dataset_kwargs"],
                                          variant_spec["hull_kwargs"])
        print(f"  [{variant_name}] {mov}: frames {start}-{end-1} ready under {prefix}/{mov}")
    return prefix


def write_configs(variant_name: str, prefix: str) -> None:
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mov, info in VIDEOS.items():
        start, end = info["frames"]
        cfg = {
            "name": f"round1_5/ctrl_119_{mov}_{variant_name}",
            "sparse_dir": info["sparse_dir"],
            "base_name": f"{prefix}/{mov}",
            "max_iters": MAX_ITERS,
            "param_sets": BASE_PARAM_SET,
            "frames": {"start": start, "end": end},
        }
        out_path = CONFIG_OUT_DIR / f"ctrl_119_{mov}_{variant_name}.json"
        out_path.write_text(json.dumps(cfg, indent=2))
        print(f"  wrote {out_path.relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-only", action="store_true",
                     help="跳过数据准备，只(重新)生成config json(数据已就绪时用)")
    ap.add_argument("--variant", type=str, default=None,
                     help="只处理这一个variant(默认全部5个)")
    args = ap.parse_args()

    variants = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS

    for variant_name, spec in variants.items():
        print(f"=== {variant_name} ===")
        prefix = f"data/ctrl_119_3cam_{variant_name}"
        if not args.configs_only:
            prefix = prepare_variant(variant_name, spec)
        write_configs(variant_name, prefix)


if __name__ == "__main__":
    main()
