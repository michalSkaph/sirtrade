from __future__ import annotations

import os
import subprocess
from pathlib import Path


def get_last_commit_info() -> str:
    env_val = os.getenv("SIRTRADE_LAST_COMMIT")
    if env_val:
        return env_val

    for file_name in ("LAST_COMMIT", ".last_commit"):
        path = Path(file_name)
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if content:
            return content

    try:
        if Path(".git").exists():
            output = subprocess.check_output(
                ["git", "log", "-1", "--format=%h %cI %s"],
                stderr=subprocess.DEVNULL,
            )
            value = output.decode().strip()
            if value:
                return value
    except Exception:
        pass

    return "unknown"