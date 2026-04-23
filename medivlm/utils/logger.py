"""logger wrapper to help module logs consistently."""
from __future__ import annotations

import logging
import sys
from typing import Optional

_INITIALISED = False


def _init_root(level: int = logging.INFO) -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger("medivlm")
    root.setLevel(level)
    # avoid duplicating handlers if re-imported
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _INITIALISED = True


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    _init_root(level)
    return logging.getLogger(f"medivlm.{name}" if name else "medivlm")
