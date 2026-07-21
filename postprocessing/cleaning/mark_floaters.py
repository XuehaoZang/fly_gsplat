"""
T2: 在T1(utils/gaussian_features.py)输出的逐点特征表基础上标记明显孤立的floater点，
新增一列 if_keep(bool)，不删行不删列。

判据: 点所在的 k-近邻连通分量大小(见 utils.ply.connected_component_sizes)
<= min_patch_size 视为floater。参数 k=10, dist_percentile=75, min_patch_size=10
已验证锁定，不开放成CLI可调参数。

详细背景(判据依据、已验证范围、已知问题/TODO)见 README.md。
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from utils.ply import connected_component_sizes

MIN_PATCH_SIZE = 10
K_NEIGHBORS = 10
DIST_PERCENTILE = 75.0

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "postprocessing" / "cleaning" / "eda_outputs"


def mark_floaters(df: pd.DataFrame, min_patch_size: int = MIN_PATCH_SIZE,
                   k: int = K_NEIGHBORS, dist_percentile: float = DIST_PERCENTILE) -> pd.DataFrame:
    xyz = df[["x", "y", "z"]].to_numpy()
    patch_size = connected_component_sizes(xyz, k=k, dist_percentile=dist_percentile)
    out = df.copy()
    out["if_keep"] = patch_size > min_patch_size
    return out


# ---------------------------------------------------------------------------
# 批处理 (阶段E第一步)
# ---------------------------------------------------------------------------

def latest_checkpoint_dir(frame_dir: Path) -> Path:
    ckpt_root = frame_dir / "splatfacto-checkpoint"
    return sorted(ckpt_root.iterdir())[-1]


def find_features_csv(dataset_dir: Path, frame_idx: int) -> Path:
    frame_name = f"f{frame_idx:04d}"
    splat_dir = latest_checkpoint_dir(dataset_dir / frame_name)
    csv_path = splat_dir / f"gaussian_features_{frame_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no gaussian_features csv found for {frame_name} at {csv_path}")
    return csv_path


def assert_marked_consistency(df_in: pd.DataFrame, df_out: pd.DataFrame) -> None:
    if len(df_out) != len(df_in):
        raise AssertionError(f"行数不一致: 输入{len(df_in)} 输出{len(df_out)}")
    if len(df_out.columns) != len(df_in.columns) + 1:
        raise AssertionError(f"列数不一致: 输入{len(df_in.columns)}列 输出{len(df_out.columns)}列 (应为输入+1)")
    if list(df_out.columns[:-1]) != list(df_in.columns) or df_out.columns[-1] != "if_keep":
        raise AssertionError(f"列名/顺序不一致或新增列不是if_keep: 输出列={list(df_out.columns)}")
    pd.testing.assert_frame_equal(df_out[df_in.columns], df_in)


def component_multiset(patch_size: np.ndarray) -> np.ndarray:
    """把逐点的 patch_size 数组还原成分量大小的多重集(每个分量恰好出现一次)。

    与 debug/floater_census_100frames.py 同一份技巧: size=s 的值出现次数 c 必是 s 的
    整数倍(同分量点值相同)，c/s 即该大小分量的个数。
    """
    sizes, counts = np.unique(patch_size, return_counts=True)
    n_comp_per_size = counts // sizes
    comp_sizes = np.repeat(sizes, n_comp_per_size)
    return np.sort(comp_sizes)[::-1]


def process_frame(dataset_dir: Path, frame_idx: int) -> dict:
    frame_name = f"f{frame_idx:04d}"
    csv_path = find_features_csv(dataset_dir, frame_idx)
    df = pd.read_csv(csv_path)
    marked = mark_floaters(df)
    assert_marked_consistency(df, marked)

    out_path = csv_path.with_name(csv_path.stem + "_marked.csv")
    marked.to_csv(out_path, index=False)

    n_total = len(marked)
    n_floater = int((~marked["if_keep"]).sum())

    xyz = df[["x", "y", "z"]].to_numpy()
    patch_size = connected_component_sizes(xyz, k=K_NEIGHBORS, dist_percentile=DIST_PERCENTILE)
    comp_sizes = component_multiset(patch_size)
    mid_mask = (comp_sizes > 10) & (comp_sizes < 17)
    mid_sizes = sorted(comp_sizes[mid_mask].tolist())

    return {
        "frame": frame_name,
        "status": "ok",
        "n_total": n_total,
        "n_floater": n_floater,
        "floater_ratio_pct": 100 * n_floater / n_total,
        "mid_10_17_sizes": mid_sizes,
        "out_path": str(out_path),
    }


def run_batch(dataset_dir: Path, start: int, end: int) -> tuple[list[dict], list[dict]]:
    results = []
    failures = []
    for frame_idx in range(start, end + 1):
        frame_name = f"f{frame_idx:04d}"
        try:
            r = process_frame(dataset_dir, frame_idx)
            results.append(r)
            print(f"[{frame_name}] n_total={r['n_total']} n_floater={r['n_floater']} "
                  f"({r['floater_ratio_pct']:.1f}%) mid_10_17={r['mid_10_17_sizes']} "
                  f"saved -> {r['out_path']}")
        except Exception as e:
            failures.append({"frame": frame_name, "error": f"{type(e).__name__}: {e}"})
            print(f"[{frame_name}] FAILED: {type(e).__name__}: {e}")
    return results, failures


def build_summary_text(dataset_dir: Path, group_name: str, start: int, end: int,
                        results: list[dict], failures: list[dict]) -> str:
    n_requested = end - start + 1
    n_ok = len(results)
    n_failed = len(failures)

    lines = [
        f"# mark_floaters 批处理汇总 ({group_name}, f{start:04d}~f{end:04d})",
        "",
        f"数据集目录: `{dataset_dir}`",
        f"判据参数(锁定): k={K_NEIGHBORS}, dist_percentile={DIST_PERCENTILE}, min_patch_size={MIN_PATCH_SIZE}",
        "",
        "## 处理结果",
        "",
        f"- 请求处理帧数: {n_requested}",
        f"- 成功: {n_ok}",
        f"- 失败: {n_failed}",
    ]

    if failures:
        lines.append("")
        lines.append("### 失败帧列表")
        lines.append("")
        for f in failures:
            lines.append(f"- {f['frame']}: {f['error']}")

    if results:
        ratios = np.array([r["floater_ratio_pct"] for r in results])
        lines += [
            "",
            "## floater占比统计 (%, 仅成功帧)",
            "",
            f"- min: {ratios.min():.3f}",
            f"- max: {ratios.max():.3f}",
            f"- mean: {ratios.mean():.3f}",
            f"- median: {np.median(ratios):.3f}",
            f"- std: {ratios.std():.3f}",
        ]

        mid_frames = [r["frame"] for r in results if len(r["mid_10_17_sizes"]) > 0]
        lines += [
            "",
            f"## 10~17区间(不含端点)非空的帧号列表 (共{len(mid_frames)}帧)",
            "",
            ", ".join(mid_frames) if mid_frames else "(无)",
        ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                         help="单帧模式: T1输出的逐点特征表路径 (gaussian_features_*.csv)。"
                              "指定后仅处理这一个文件，忽略下面的批处理参数。")
    parser.add_argument("--out", type=str, default=None, help="单帧模式输出路径，默认在原文件名后加 _marked")

    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                         help="批处理模式: 数据集根目录，默认 outputs/ctrl_009_002_8groups_100frames/G2b_G9")
    parser.add_argument("--start", type=int, default=0, help="批处理起始帧号(含)")
    parser.add_argument("--end", type=int, default=99, help="批处理结束帧号(含)")
    parser.add_argument("--summary-dir", type=str, default=str(DEFAULT_SUMMARY_DIR),
                         help="批处理汇总文件输出目录")
    args = parser.parse_args()

    if args.csv is not None:
        csv_path = Path(args.csv)
        df = pd.read_csv(csv_path)
        marked = mark_floaters(df)

        out_path = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_marked.csv")
        marked.to_csv(out_path, index=False)

        n_total = len(marked)
        n_floater = int((~marked["if_keep"]).sum())
        print(f"[{csv_path.name}] n_total={n_total}  n_floater={n_floater} "
              f"({100 * n_floater / n_total:.1f}%)  saved -> {out_path}")
        return

    dataset_dir = Path(args.data_root)
    group_name = dataset_dir.name
    results, failures = run_batch(dataset_dir, args.start, args.end)

    summary = build_summary_text(dataset_dir, group_name, args.start, args.end, results, failures)
    print("\n" + summary)

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"batch_mark_floaters_summary_{group_name}_{date.today().isoformat()}.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[Saved summary] {summary_path}")


if __name__ == "__main__":
    main()
