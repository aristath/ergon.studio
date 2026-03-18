from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_GLOBAL_CONFIG: dict[str, Any] = {
    "providers": {},
    "role_assignments": {},
    "approvals": {},
    "ui": {},
}


def load_or_create_global_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        save_global_config(config_path, DEFAULT_GLOBAL_CONFIG)
        return deepcopy(DEFAULT_GLOBAL_CONFIG)

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config.json must contain a JSON object")
    return loaded


def save_global_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
