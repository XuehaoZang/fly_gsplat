"""
generate_round1_configs.py

Round 1（sweep_hyper_params.md）单变量消融的config生成器。围绕当前生产配置
ratio3_sh0_dense（= 已被densify_6groups_100frames验证过的H6组合，不是未调优基线）
逐一改动单个参数，8个param_sets（P1a/P1b/P1c/P2/P3/P4a/P4b/P6）打包进一个"core"config，
另外3个max_iters变体（P5，1000/1500/3000）因为schedule.py的max_iters是per-config顶层字段
不是per-param_set，只能各开一个独立config。每个视频各出4个config文件（1个core + 3个iters）。

不改动 gpu/schedule/schedule.py / common.py / worker.py 任何代码，只生成它们能直接消费的
config json——Round1用的dev帧(f0730-0830 / f0373-0473)在原有480帧全量sweep里已经生成过
transforms.json+init_points.ply，schedule.py的Phase A会直接跳过数据准备，本轮只有ns-train
本身在跑。

用法:
    python gpu/schedule/generate_round1_configs.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "gpu" / "schedule" / "configs" / "round1"

VIDEOS = {
    "ctrl_119_004": {
        "sparse_dir": "data/tests/ctrl_3cam_test/Sparse/Expr_119_mov_004",
        "base_name": "data/ctrl_119_3cam/004",
        "frames": {"start": 730, "end": 830},  # 100 frames, >=1 full wingbeat cycle
    },
    "ctrl_119_010": {
        "sparse_dir": "data/tests/ctrl_3cam_test/Sparse/Expr_119_mov_010",
        "base_name": "data/ctrl_119_3cam/010",
        "frames": {"start": 373, "end": 473},  # 100 frames, >=1 full wingbeat cycle
    },
}

MAX_ITERS = 2000

# production baseline (ratio3_sh0_dense) -- already == G2b_G9 + H6(grad_thresh_low+refine_fast)
BASE = [
    "--pipeline.model.use-scale-regularization", "True",
    "--pipeline.model.max-gauss-ratio", "3.0",
    "--pipeline.model.sh-degree", "0",
    "--pipeline.model.warmup-length", "50",
    "--pipeline.model.stop-split-at", "1800",
    "--pipeline.model.densify-grad-thresh", "0.0004",
    "--pipeline.model.refine-every", "50",
]


def _override(base: list, **overrides) -> list:
    """base里已有的flag替换成新值，不在base里的追加。overrides的key用下划线，
    转成--pipeline.model.xxx-yyy形式。"""
    args = list(base)
    for key, value in overrides.items():
        flag = "--pipeline.model." + key.replace("_", "-")
        if flag in args:
            idx = args.index(flag)
            args[idx + 1] = str(value)
        else:
            args += [flag, str(value)]
    return args


# ---- Round 1 core: 8 param_sets, all at MAX_ITERS=2000 ----
CORE_PARAM_SETS = {
    # P1: densify-grad-thresh/refine-every 往两个方向探测(生产值已是H6激进组合)
    "P1a_grad_conservative": _override(BASE, densify_grad_thresh="0.0008"),  # nerfstudio默认，比生产更保守
    "P1b_grad_aggressive":   _override(BASE, densify_grad_thresh="0.0002"),  # 比生产更激进，验证是否让漂移更差
    "P1c_refine_slow":       _override(BASE, refine_every="100"),
    # P2: 复用G6(warmup200/stop-split-at1200)，更早冻结几何生长
    "P2_freeze_early":       _override(BASE, warmup_length="200", stop_split_at="1200"),
    # P3: 开启位姿优化，吸收3cam下冗余度更低的标定残差
    "P3_camera_optimizer":   BASE + ["--pipeline.model.camera-optimizer.mode", "SO3xR3"],
    # P4: cull阈值收紧，复用G7 + 一组更激进变体
    "P4a_cull_strict":       BASE + ["--pipeline.model.cull-alpha-thresh", "0.2",
                                      "--pipeline.model.cull-scale-thresh", "0.3",
                                      "--pipeline.model.cull-screen-size", "0.10"],
    "P4b_cull_stricter":     BASE + ["--pipeline.model.cull-alpha-thresh", "0.3",
                                      "--pipeline.model.cull-scale-thresh", "0.15",
                                      "--pipeline.model.cull-screen-size", "0.08"],
    # P6: SH degree confirmatory(预期无geometry收益，输入图像近乎二值剪影)
    "P6_sh1":                _override(BASE, sh_degree="1"),
}

# ---- Round 1 P5: max_iters变体，每个iters一个独立config(只含baseline参数) ----
ITERS_VARIANTS = [1000, 1500, 3000]


def write_config(path: Path, name: str, sparse_dir: str, base_name: str,
                  max_iters: int, param_sets: dict, frames: dict) -> None:
    cfg = {
        "name": name,
        "sparse_dir": sparse_dir,
        "base_name": base_name,
        "max_iters": max_iters,
        "param_sets": param_sets,
        "frames": frames,
    }
    path.write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {path.relative_to(REPO)}  ({len(param_sets)} param_set(s), "
          f"{frames['end']-frames['start']} frames, max_iters={max_iters})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for video, info in VIDEOS.items():
        core_name = f"round1/{video}_core"
        write_config(OUT_DIR / f"{video}_core.json", core_name,
                     info["sparse_dir"], info["base_name"], MAX_ITERS,
                     CORE_PARAM_SETS, info["frames"])
        written.append(f"{video}_core.json")

        for iters in ITERS_VARIANTS:
            name = f"round1/{video}_iters{iters}"
            write_config(OUT_DIR / f"{video}_iters{iters}.json", name,
                         info["sparse_dir"], info["base_name"], iters,
                         {"P5_baseline": list(BASE)}, info["frames"])
            written.append(f"{video}_iters{iters}.json")

    print(f"\n[round1 configs] wrote {len(written)} config(s) -> {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
