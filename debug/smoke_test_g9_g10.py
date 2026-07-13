"""
冒烟测试：G9(sh-degree=0) / G10(use-absgrad=False)，两个全新flag，
frame 0/1/2，完整2000步，验证训练不报错 + stats.json 数值量级合理。
不加任何checkpoint相关flag（stats-every/save-points），走splatfacto-checkpoint默认
（save_stats=True, stats_every=1000），final_step(=max_iters-1=1999)必定强制dump一次，
配合默认的step 0 / step 1000，足够看出G9颜色相关是否异常（scale/opacity/n_gaussians
量级是否和G3 baseline接近）。
"""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_NAME = "ctrl_009_002"
FRAMES = [0, 1, 2]
MAX_ITERS = 2000

GROUPS = {
    "G9_sh_degree_0": [
        "--pipeline.model.sh-degree", "0",
        "--pipeline.model.warmup-length", "50",
        "--pipeline.model.stop-split-at", "1800",
    ],
    "G10_use_absgrad_false": [
        "--pipeline.model.use-absgrad", "False",
        "--pipeline.model.warmup-length", "50",
        "--pipeline.model.stop-split-at", "1800",
    ],
}


def run_group_frame(group_name: str, extra_args: list, frame_idx: int, log_path: Path) -> dict:
    data_dir = REPO / "data" / BASE_NAME / f"f{frame_idx:04d}"
    exp_name = f"{BASE_NAME}_smoke_g9_g10/{group_name}/f{frame_idx:04d}"
    cmd = [
        "ns-train", "splatfacto-checkpoint",
        "--data", str(data_dir),
        "--vis", "tensorboard",
        "--max-num-iterations", str(MAX_ITERS),
        "--pipeline.model.background-color", "white",
        "--experiment-name", exp_name,
    ] + extra_args + [
        "nerfstudio-data", "--eval-mode", "all",
    ]
    print(f"\n{'=' * 20} {group_name} / frame {frame_idx} {'=' * 20}")
    print(" ".join(cmd))

    with open(log_path, "a") as logf:
        logf.write(f"\n{'=' * 20} {group_name} / frame {frame_idx} {'=' * 20}\n")
        logf.write(" ".join(cmd) + "\n")
        logf.flush()
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        return {"group": group_name, "frame": frame_idx, "status": "failed",
                "error": f"returncode={result.returncode}, see {log_path}"}

    splat_dir = sorted((REPO / "outputs" / exp_name / "splatfacto-checkpoint").iterdir())[-1]
    stats_files = sorted((splat_dir / "debug_checkpoints" / "stats").glob("step_*_stats.json"))
    if not stats_files:
        return {"group": group_name, "frame": frame_idx, "status": "failed",
                "error": "no stats.json produced"}
    final = json.loads(stats_files[-1].read_text())

    return {
        "group": group_name, "frame": frame_idx, "status": "ok",
        "final_step": final.get("step"),
        "n_gaussians": final.get("n_gaussians"),
        "scale_ratio_median": final.get("scale_ratio", {}).get("median"),
        "scale_ratio_p95": final.get("scale_ratio", {}).get("p95"),
        "opacity_median": final.get("opacity", {}).get("median"),
        "opacity_p10": final.get("opacity", {}).get("p10"),
        "bbox_extent_max": max(final["bbox_extent"]) if "bbox_extent" in final else None,
        "splat_dir": str(splat_dir),
    }


def main():
    out_dir = REPO / "outputs" / f"{BASE_NAME}_smoke_g9_g10"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_stdout.log"

    records = []
    for group_name, extra_args in GROUPS.items():
        for frame_idx in FRAMES:
            records.append(run_group_frame(group_name, extra_args, frame_idx, log_path))

    with open(out_dir / "smoke_results.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[Saved] {out_dir / 'smoke_results.json'}")

    header = (f"{'group':<24}{'frame':>6}{'status':>8}{'step':>6}{'n_gauss':>9}"
              f"{'ratio_med':>11}{'ratio_p95':>11}{'op_med':>8}{'op_p10':>8}{'extent':>10}")
    print("\n" + header)
    for r in records:
        if r["status"] != "ok":
            print(f"{r['group']:<24}{r['frame']:>6}{'FAIL':>8}  {r.get('error')}")
            continue
        print(f"{r['group']:<24}{r['frame']:>6}{'ok':>8}{r['final_step']:>6}{r['n_gaussians']:>9}"
              f"{r['scale_ratio_median']:>11.3f}{r['scale_ratio_p95']:>11.3f}"
              f"{r['opacity_median']:>8.3f}{r['opacity_p10']:>8.3f}{r['bbox_extent_max']:>10.5f}")

    n_ok = sum(r["status"] == "ok" for r in records)
    print(f"\n{n_ok}/{len(records)} runs ok. Full stdout log: {log_path}")


if __name__ == "__main__":
    main()
