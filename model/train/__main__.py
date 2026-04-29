from __future__ import annotations

import os
import sys
from pathlib import Path


def _restart_with_conda_libstdcxx() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    libstdcxx = Path(conda_prefix) / "lib" / "libstdc++.so.6"
    if not libstdcxx.exists():
        return

    current = os.environ.get("LD_PRELOAD", "")
    if str(libstdcxx) in current:
        return

    os.environ["LD_PRELOAD"] = (
        f"{libstdcxx}:{current}" if current else str(libstdcxx)
    )

    os.execv(sys.executable, [sys.executable, "-m", "model.train", *sys.argv[1:]])


_restart_with_conda_libstdcxx()

from model.train.runner import run_train


if __name__ == "__main__":
    run_train()