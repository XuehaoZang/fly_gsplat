"""一个训练 worker：反复从队列认领任务（pending/ -> claimed/ 的原子重命名），直到队列为空。

由 train.run 启动，不单独调用。CUDA_VISIBLE_DEVICES 和线程数由启动方通过环境变量设好。
每个任务结束后往 <progress_dir>/<tag>.jsonl 追加一行：task_id, frame, status, wall_s, n_gaussians, scale_ratio_median。
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from . import task


def claim_next(pending: Path, claimed: Path) -> Path | None:
    for name in sorted(os.listdir(pending)):
        try:
            os.rename(pending / name, claimed / name)
            return claimed / name
        except OSError:
            continue  # 被别的 worker 先认领了
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--queue-dir", type=Path, required=True)
    ap.add_argument("--progress-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    pending, claimed = args.queue_dir / "pending", args.queue_dir / "claimed"
    progress = args.progress_dir / f"{args.tag}.jsonl"
    while (claimed_path := claim_next(pending, claimed)) is not None:
        frame = json.loads(claimed_path.read_text())["frame"]
        t0 = time.perf_counter()
        try:
            record = {"status": "ok", **task.run(cfg, frame, os.environ.copy())}
        except Exception as e:  # noqa: BLE001
            record = {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        record = {"task_id": claimed_path.stem, "frame": frame, "worker": args.tag,
                  "wall_s": time.perf_counter() - t0, **record}
        with open(progress, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{args.tag}] f{frame:04d} -> {record['status']} ({record['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
