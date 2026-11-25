"""YAML config loading with single-level inheritance and CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config.

    A top-level ``base: other.yaml`` key (path relative to the config file)
    is loaded first and the current file is merged over it.
    ``overrides`` are ``dotted.key=value`` strings; values are parsed as YAML.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.subkey=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            # A key set to null (a check that is off by default) becomes a section when an override goes into it.
            if part in node and node[part] is None:
                node[part] = {}
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg
