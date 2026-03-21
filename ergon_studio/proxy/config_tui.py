from __future__ import annotations

from pathlib import Path

from ergon_studio.app_config import ProxyAppConfig


def run_config_tui(
    *,
    app_dir: Path,
    definitions_dir: Path,
    initial_config: ProxyAppConfig,
) -> int:
    del app_dir, definitions_dir, initial_config
    raise NotImplementedError("configuration TUI is not implemented yet")
