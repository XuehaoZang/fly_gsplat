"""
generate_round2_configs.py

Round 2（sweep_hyper_params.md）联合网格config生成器。基于Round1/1.5的结果：
- D1(iters1000)是最强单变量杠杆(点数在iters1000时已接近2000收敛值，多训的1000~2000
  iters只贡献漂移不贡献密度)，本轮把D1当"骨架"，绝大多数新组合都在max_iters=1000上叠加
  别的杠杆(cull/ratio/densify阈值/refine频率/camera-optimizer/去腿数据变体)。
- D2(freeze_early)/D3(grad_conservative)是另外两个独立生效但要用max_iters=2000的机制，
  各自单开一个config(schedule.py的max_iters是per-config顶层字段，不能跟D1混进同一个
  config——见generate_round1_configs.py同样的限制)。
- 用户在round1基础上追加的新方向：(a)放开/收紧max-gauss-ratio看是否更稠密，
  (b)去腿erosion参数改用utils/image.py新加的min_area_ratio安全网(kernel_size 5/7/9，
  9带安全网)重新验证，不能再用round1.5那组把CAM3几乎腐蚀掉整只苍蝇的k9-no-safety结果，
  (c)几组"更稠密"专门设计的组合(grad_thresh从0.0004小幅松到0.0003、refine-every缩到25，
  都叠加在iters1000骨架上，靠早停规避P1b那种"更激进densify=更差漂移"的失败模式)。

去腿/mask阈值/hull采样数这几个"数据变体"(base_name不同)必须已经用
prepare_round2_leg_erosion.py / prepare_round1_5.py(--variant p7_thresh50/p9_hull100k/
p8_leg_erosion，注意先把prepare_round1_5.py的VIDEOS帧范围改宽到本轮150帧)准备好，
本脚本只负责把它们接到param_sets里的base_name override(见schedule.py本轮新加的
_param_set_base_name支持)，不在这里做数据准备。

Dev帧窗口本轮从round1的100帧/视频拓宽到150帧/视频(004: 730-880, 010: 373-523)——
T3 motion累加(±18帧窗口，8000fps修正值)的安全下限经验证是150帧连续窗口，round1的
100帧不足以支撑"每个pipeline都要跑kinematics"这个新要求，所以round2所有组都在150帧
上重跑，不复用round1的100帧结果(即使param相同，如D1本身)。

用法:
    python gpu/schedule/generate_round2_configs.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "gpu" / "schedule" / "configs" / "round2"

VIDEOS = {
    "ctrl_119_004": {
        "sparse_dir": "data/tests/ctrl_3cam_test/Sparse/Expr_119_mov_004",
        "base_name": "data/ctrl_119_3cam/004",
        "frames": {"start": 730, "end": 880},  # 150 frames (widened from round1's 100)
    },
    "ctrl_119_010": {
        "sparse_dir": "data/tests/ctrl_3cam_test/Sparse/Expr_119_mov_010",
        "base_name": "data/ctrl_119_3cam/010",
        "frames": {"start": 373, "end": 523},  # 150 frames
    },
}

BASE = [
    "--pipeline.model.use-scale-regularization", "True",
    "--pipeline.model.max-gauss-ratio", "3.0",
    "--pipeline.model.sh-degree", "0",
    "--pipeline.model.warmup-length", "50",
    "--pipeline.model.stop-split-at", "1800",
    "--pipeline.model.densify-grad-thresh", "0.0004",
    "--pipeline.model.refine-every", "50",
]

CULL_B = ["--pipeline.model.cull-alpha-thresh", "0.3",
          "--pipeline.model.cull-scale-thresh", "0.15",
          "--pipeline.model.cull-screen-size", "0.08"]


def _override(base: list, **overrides) -> list:
    args = list(base)
    for key, value in overrides.items():
        flag = "--pipeline.model." + key.replace("_", "-")
        if flag in args:
            idx = args.index(flag)
            args[idx + 1] = str(value)
        else:
            args += [flag, str(value)]
    return args


def _variant_base_name(video_key: str, mov: str, variant_dir: str) -> str:
    return f"data/ctrl_119_3cam_{variant_dir}/{mov}"


def build_d1_param_sets(mov: str) -> dict:
    """max_iters=1000骨架：cull/ratio/densify阈值/refine频率/camera-optimizer/
    去腿数据变体，共25组。纯训练参数变体用list(共用config默认base_name)；数据变体
    (hull采样数/mask阈值/去腿)用{"extra_args":..., "base_name":...}覆盖。"""
    ps = {
        "D1_baseline": list(BASE),
        "D1_cullB": BASE + CULL_B,
        "D1_ratio2": _override(BASE, max_gauss_ratio="2.0"),
        "D1_ratio2p5": _override(BASE, max_gauss_ratio="2.5"),
        "D1_ratio4": _override(BASE, max_gauss_ratio="4.0"),
        "D1_ratio6": _override(BASE, max_gauss_ratio="6.0"),
        "D1_ratio10": _override(BASE, max_gauss_ratio="10.0"),
        "D1_gradconserv": _override(BASE, densify_grad_thresh="0.0008"),
        "D1_gradmid": _override(BASE, densify_grad_thresh="0.0003"),
        "D1_refine25": _override(BASE, refine_every="25"),
        "D1_camopt": BASE + ["--pipeline.model.camera-optimizer.mode", "SO3xR3"],
        "D1_cullB_camopt": BASE + CULL_B + ["--pipeline.model.camera-optimizer.mode", "SO3xR3"],
        "D1_cullB_gradmid": _override(BASE, densify_grad_thresh="0.0003") + CULL_B,
        "D1_cullB_refine25": _override(BASE, refine_every="25") + CULL_B,
        "D1_cullB_ratio6": _override(BASE, max_gauss_ratio="6.0") + CULL_B,
        "D1_cullB_ratio2": _override(BASE, max_gauss_ratio="2.0") + CULL_B,
        "D1_hull100k": {"extra_args": list(BASE),
                         "base_name": _variant_base_name("d1", mov, "p9_hull100k")},
        "D1_thresh50": {"extra_args": list(BASE),
                         "base_name": _variant_base_name("d1", mov, "p7_thresh50")},
        "D1_legerosion_orig": {"extra_args": list(BASE),
                                "base_name": _variant_base_name("d1", mov, "p8_leg_erosion")},
        "D1_legerosion_k5": {"extra_args": list(BASE),
                              "base_name": _variant_base_name("d1", mov, "p8b_k5")},
        "D1_legerosion_k7": {"extra_args": list(BASE),
                              "base_name": _variant_base_name("d1", mov, "p8b_k7")},
        "D1_legerosion_k9safe": {"extra_args": list(BASE),
                                  "base_name": _variant_base_name("d1", mov, "p8b_k9_safe")},
        "D1_legerosion_k7safe": {"extra_args": list(BASE),
                                  "base_name": _variant_base_name("d1", mov, "p8b_k7_safe")},
        "D1_cullB_legerosion_k9safe": {"extra_args": BASE + CULL_B,
                                        "base_name": _variant_base_name("d1", mov, "p8b_k9_safe")},
        "D1_cullB_legerosion_k7safe": {"extra_args": BASE + CULL_B,
                                        "base_name": _variant_base_name("d1", mov, "p8b_k7_safe")},
    }
    return ps


def build_d2_param_sets() -> dict:
    freeze = _override(BASE, warmup_length="200", stop_split_at="1200")
    return {
        "D2_freeze_early": freeze,
        "D2_cullB": freeze + CULL_B,
    }


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
        mov = video.rsplit("_", 1)[-1]

        d1_name = f"round2/{video}_d1"
        write_config(OUT_DIR / f"{video}_d1.json", d1_name, info["sparse_dir"], info["base_name"],
                     1000, build_d1_param_sets(mov), info["frames"])
        written.append(f"{video}_d1.json")

        iters750_name = f"round2/{video}_iters750"
        write_config(OUT_DIR / f"{video}_iters750.json", iters750_name, info["sparse_dir"],
                     info["base_name"], 750, {"iters750_baseline": list(BASE)}, info["frames"])
        written.append(f"{video}_iters750.json")

        d2_name = f"round2/{video}_d2"
        write_config(OUT_DIR / f"{video}_d2.json", d2_name, info["sparse_dir"], info["base_name"],
                     2000, build_d2_param_sets(), info["frames"])
        written.append(f"{video}_d2.json")

    n_param_sets = len(build_d1_param_sets("004")) + 1 + len(build_d2_param_sets())
    n_tasks = n_param_sets * 150 * len(VIDEOS)
    print(f"\n[round2 configs] wrote {len(written)} config(s) -> {OUT_DIR.relative_to(REPO)}")
    print(f"[round2 configs] {n_param_sets} param_set(s)/video x 150 frames x {len(VIDEOS)} videos "
          f"= {n_tasks} task(s) total")


if __name__ == "__main__":
    main()
