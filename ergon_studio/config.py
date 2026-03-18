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

    return parse_global_config_text(config_path.read_text(encoding="utf-8"))


def save_global_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_global_config_text(config_path: Path, text: str) -> dict[str, Any]:
    config = parse_global_config_text(text)
    save_global_config(config_path, config)
    return config


def parse_global_config_text(text: str) -> dict[str, Any]:
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("config.json must contain a JSON object")
    return loaded
