"""Access to configs/experiments.yaml (the registry of every experiment in the paper)."""
from __future__ import annotations

from pathlib import Path

import yaml

from src import common

_FRAME = (common.FRAME_W, common.FRAME_H)


def load_registry() -> dict[str, dict]:
    """Flat ``id -> spec`` mapping (first group wins for ids that appear in several groups)."""
    raw = yaml.safe_load((common.CONFIGS / "experiments.yaml").read_text())
    flat: dict[str, dict] = {}
    for group, items in raw.items():
        for exp_id, spec in items.items():
            flat.setdefault(exp_id, dict(spec, group=group))
    return flat


def get(exp_id: str) -> dict:
    reg = load_registry()
    if exp_id not in reg:
        raise KeyError(f"unknown experiment '{exp_id}'. Known: {sorted(reg)}")
    return reg[exp_id]


def tracker_yaml(name: str) -> Path:
    return common.CONFIGS / "trackers" / f"{name}.yaml"


def gm_parameters(tracker_name: str, gamma: float | None = None) -> dict:
    """GM-cost parameters from the tracker YAML (``gm_cost`` block); ``gamma`` overrides the YAML value."""
    cfg = yaml.safe_load(tracker_yaml(tracker_name).read_text())
    gm = dict(cfg.get("gm_cost", {"gamma": 0.55, "sigma": 0.15}))
    if gamma is not None:
        gm["gamma"] = float(gamma)
    return gm


def sahi_parameters(tile: int) -> dict:
    cfg = yaml.safe_load((common.CONFIGS / "sahi.yaml").read_text())
    t = cfg["tiles"][int(tile)]
    return dict(slice=int(t["slice"]), overlap=float(t["overlap"]), conf=float(cfg["confidence_threshold"]),
                match_threshold=float(cfg["postprocess_match_threshold"]))
