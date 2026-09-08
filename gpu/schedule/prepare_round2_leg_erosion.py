"""
prepare_round2_leg_erosion.py

Round 2 去腿erosion参数修正版数据准备。Round1.5的p8_leg_erosion(kernel_size=9,无安全网)
经`gpu/schedule/analysis/round1_diag_reproj.py`目视复查发现f0730的CAM3几乎被腐蚀掉整只
苍蝇(body本身被吃掉，不只是腿)——单一kernel_size在body在某个相机视角下因透视缩短显得
更窄时会连body一起吃掉。`utils/image.py::erode_appendages`已加`min_area_ratio`安全网参数
(单张图erosion后前景面积低于原图比例阈值时，整张图退回不erosion的原mask，代价是那一帧
那个相机视角保留更多腿部像素，好过大面积吃掉body)，本脚本准备用它的多组erosion变体：

  p8b_k5        kernel_size=5，无安全网(比round1.5的k9更温和，直接验证减小kernel能否
                避免body被吃掉)
  p8b_k7        kernel_size=7，无安全网
  p8b_k9_safe   kernel_size=9(round1.5原始激进核)+min_area_ratio=0.65(安全网兜底)
  p8b_k7_safe   kernel_size=7+min_area_ratio=0.65(双重保险)

复用prepare_round1_5.py里的_link_calibration/prepare_full_regen_frame(images+hull都要
重新生成，不能像p7/p9那样只重开hull)，只是换一套VARIANTS参数、换一个镜像目录前缀
(data/ctrl_119_3cam_<variant>/)，不改prepare_round1_5.py本身。

用法:
    python gpu/schedule/prepare_round2_leg_erosion.py            # 准备全部4个erosion变体
    python gpu/schedule/prepare_round2_leg_erosion.py --variant p8b_k9_safe
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_round1_5 import VIDEOS, _link_calibration, prepare_full_regen_frame  # noqa: E402

VARIANTS = {
    "p8b_k5": {"dataset_kwargs": {"remove_appendages": True, "appendage_kernel_size": 5},
               "hull_kwargs": {"remove_appendages": True, "appendage_kernel_size": 5}},
    "p8b_k7": {"dataset_kwargs": {"remove_appendages": True, "appendage_kernel_size": 7},
               "hull_kwargs": {"remove_appendages": True, "appendage_kernel_size": 7}},
    "p8b_k9_safe": {"dataset_kwargs": {"remove_appendages": True, "appendage_kernel_size": 9,
                                        "appendage_min_area_ratio": 0.65},
                     "hull_kwargs": {"remove_appendages": True, "appendage_kernel_size": 9,
                                      "appendage_min_area_ratio": 0.65}},
    "p8b_k7_safe": {"dataset_kwargs": {"remove_appendages": True, "appendage_kernel_size": 7,
                                        "appendage_min_area_ratio": 0.65},
                     "hull_kwargs": {"remove_appendages": True, "appendage_kernel_size": 7,
                                      "appendage_min_area_ratio": 0.65}},
}


def prepare_variant(variant_name: str, spec: dict) -> str:
    prefix = f"data/ctrl_119_3cam_{variant_name}"
    for mov, info in VIDEOS.items():
        orig_base_dir = REPO / info["orig_base"]
        mirror_base_dir = REPO / prefix / mov
        _link_calibration(orig_base_dir, mirror_base_dir)

        start, end = info["frames"]
        for frame_idx in range(start, end):
            mirror_frame_dir = mirror_base_dir / f"f{frame_idx:04d}"
            prepare_full_regen_frame(mirror_frame_dir, info["sparse_dir"], frame_idx,
                                      mirror_base_dir, spec["dataset_kwargs"], spec["hull_kwargs"])
        print(f"  [{variant_name}] {mov}: frames {start}-{end-1} ready under {prefix}/{mov}")
    return prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, default=None, help="只处理这一个variant(默认全部4个)")
    args = ap.parse_args()
    variants = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS

    for variant_name, spec in variants.items():
        print(f"=== {variant_name} ===")
        prepare_variant(variant_name, spec)


if __name__ == "__main__":
    main()
