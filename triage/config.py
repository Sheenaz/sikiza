"""Config loading. Paths in the YAML are resolved relative to the project root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("TRIAGE_CONFIG", ROOT / "configs" / "config.yaml"))
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


CATEGORIES: list[str] = load_config()["labels"]["categories"]
CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
