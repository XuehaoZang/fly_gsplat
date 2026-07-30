"""按frame加载该帧最完整csv的共享逻辑，被splat_viewer.py和
reprojection_viewer.py共用。同一帧只有一份csv逐阶段累加列(不删行)：
gaussian_features_{frame}.csv -> _marked.csv(+if_keep) -> _labeled.csv(+part_label)，
见postprocessing/cleaning/mark_floaters.assert_marked_consistency。
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.mark_floaters import latest_checkpoint_dir  # noqa: E402


def load_stage_csv(frame: str, data_root: Path) -> tuple[Path, pd.DataFrame]:
    """按frame找该帧最完整的csv并读入: 优先_labeled.csv，其次_marked.csv，最后原始csv。
    返回(csv_path, df)。"""
    splat_dir = latest_checkpoint_dir(data_root / frame)
    base = splat_dir / f"gaussian_features_{frame}"
    for suffix in ("_labeled.csv", "_marked.csv", ".csv"):
        p = base.with_name(base.name + suffix)
        if p.exists():
            return p, pd.read_csv(p)
    raise FileNotFoundError(f"no gaussian_features csv found for {frame} under {data_root}")
