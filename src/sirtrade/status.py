from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_FILE = Path("data/automation_status.json")


def write_automation_status(payload: dict[str, Any], status_file: Path = STATUS_FILE) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with status_file.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def read_automation_status(status_file: Path = STATUS_FILE) -> dict[str, Any] | None:
    if not status_file.exists():
        return None
    with status_file.open("r", encoding="utf-8") as f:
        return json.load(f)
