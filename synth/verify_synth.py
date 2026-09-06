"""Acceptance checks for a synthetic dataset folder (fly_gsplat layout).

    python verify_synth.py out/<clip>/dataset [--reference ../data/ctrl_009_002/f0200/transforms.json] [--max-frames N]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from synthfly.verify import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
