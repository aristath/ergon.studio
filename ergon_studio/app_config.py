from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from ergon_studio.file_ops import atomic_write_text


@dataclass(frozen=True)
class ProxyAppConfig:
    upstream_base_url: str = ""
    upstream_api_key: str = ""
    host: str = "127.0.0.1"
    port: int = 4000
    instruction_role: str = "system"
    disable_tool_calling: bool = False


def validate_proxy_host(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("proxy host must be non-empty")
    return stripped


def validate_proxy_port(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("proxy port must be an integer")
    if value <= 0 or value > 65535:
        raise ValueError("proxy port must be between 1 and 65535")
    return value


def default_app_dir() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "ergon"
    return Path.home() / ".config" / "ergon"


def config_path(app_dir: Path) -> Path:
    return app_dir / "config.json"


def definitions_dir(app_dir: Path) -> Path:
    return app_dir / "definitions"


def load_app_config(path: Path) -> ProxyAppConfig:
    if not path.exists():
        return ProxyAppConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return ProxyAppConfig(
        upstream_base_url=_optional_str(payload.get("upstream_base_url")),
        upstream_api_key=_optional_str(payload.get("upstream_api_key")),
        host=validate_proxy_host(_optional_str(payload.get("host")) or "127.0.0.1"),
        port=validate_proxy_port(_optional_int(payload.get("port")) or 4000),
        instruction_role=_optional_str(payload.get("instruction_role")) or "system",
        disable_tool_calling=(
            _optional_bool(payload.get("disable_tool_calling")) or False
        ),
    )


def save_app_config(path: Path, config: ProxyAppConfig) -> None:
    atomic_write_text(
        path,
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
    )


def _optional_str(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("config string values must be strings")
    return value.strip()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("config numeric values must be integers")
    return value


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError("config boolean values must be bools")
    return value
