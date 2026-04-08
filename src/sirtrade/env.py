from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_ENV_FILE_LOADED = False


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    global _ENV_FILE_LOADED

    path = Path(env_path) if env_path is not None else _DEFAULT_ENV_PATH
    if env_path is None and _ENV_FILE_LOADED:
        return {}
    if not path.exists():
        if env_path is None:
            _ENV_FILE_LOADED = True
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        normalized_value = _strip_wrapping_quotes(value.strip())
        if normalized_key not in os.environ:
            os.environ[normalized_key] = normalized_value
            loaded[normalized_key] = normalized_value

    if env_path is None:
        _ENV_FILE_LOADED = True
    return loaded