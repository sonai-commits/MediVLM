"""Checkpoint save / load helpers (handles the DDP unwrapping and partial loading)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module  # DataParallel / DDP
    return model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": _unwrap(model).state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    if config is not None:
        state["config"] = config
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    if model is not None and "model" in state:
        missing, unexpected = _unwrap(model).load_state_dict(state["model"], strict=strict)
        state["missing"] = list(missing)
        state["unexpected"] = list(unexpected)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state
