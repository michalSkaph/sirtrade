from __future__ import annotations

import json
import os
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


UI_STATE_FILE = Path("data/ui_last_run.json")
UI_RUNTIME_FILE = Path("data/ui_runtime_state.json")
UI_SEGMENT_STATE_FILE = Path("data/ui_segment_runs.json")


def _write_json_atomic(file_path: Path, payload: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_name(
        f"{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        last_error: OSError | None = None
        for attempt in range(6):
            try:
                os.replace(temp_path, file_path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))

        if last_error is not None:
            raise last_error
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def _load_json_resilient(file_path: Path, default: Any) -> Any:
    if not file_path.exists():
        return default

    try:
        raw = file_path.read_text(encoding="utf-8")
    except Exception:
        return default

    if not raw.strip():
        return default

    try:
        payload = json.loads(raw)
        return payload
    except json.JSONDecodeError:
        pass

    # Recover the first valid JSON value and ignore trailing garbage.
    try:
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(raw.lstrip())
        return payload
    except json.JSONDecodeError:
        return default


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key)


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _df_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    frame = df.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = frame.index.astype(str)
    return {
        "index": list(frame.index),
        "columns": list(frame.columns),
        "rows": frame.to_dict(orient="records"),
    }


def _df_from_payload(payload: dict[str, Any], parse_datetime_index: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(payload.get("rows", []), columns=payload.get("columns", []))
    index_values = payload.get("index", [])
    if index_values:
        frame.index = index_values
    if parse_datetime_index:
        frame.index = pd.to_datetime(frame.index)
    return frame


def _serialize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    model_markets = summary.get("model_markets", {})
    payload: dict[str, Any] = {
        "segment": summary.get("segment"),
        "week": summary.get("week"),
        "generation": summary.get("generation"),
        "portfolio_vol_annual": summary.get("portfolio_vol_annual"),
        "market_source": summary.get("market_source"),
        "symbol": summary.get("symbol"),
        "interval": summary.get("interval", "1d"),
        "champion": summary.get("champion", {}),
        "research": [
            {
                "title": _value(item, "title"),
                "year": _value(item, "year"),
                "evidence_strength": _value(item, "evidence_strength"),
                "limitations": _value(item, "limitations"),
                "overfit_risk": _value(item, "overfit_risk"),
                "proposal": _value(item, "proposal"),
            }
            for item in summary.get("research", [])
        ],
        "proposed_orders": summary.get("proposed_orders", []),
        "model_trades": summary.get("model_trades", {}),
        "final_positions": summary.get("final_positions", {}),
        "final_open_slots": summary.get("final_open_slots", {}),
        "model_open_positions": summary.get("model_open_positions", {}),
        "model_selected_symbols": summary.get("model_selected_symbols", {}),
        "candidate_symbols": summary.get("candidate_symbols", []),
        "latest_prices": summary.get("latest_prices", {}),
        "results": _df_to_payload(summary.get("results", pd.DataFrame())),
        "long_tail": _df_to_payload(summary.get("long_tail", pd.DataFrame())),
        "market": _df_to_payload(summary.get("market", pd.DataFrame())),
        "model_markets": {
            str(model_id): _df_to_payload(frame)
            for model_id, frame in model_markets.items()
            if isinstance(frame, pd.DataFrame)
        },
    }
    return _sanitize_json(payload)


def _deserialize_summary(payload: dict[str, Any]) -> dict[str, Any]:
    model_markets_payload = payload.get("model_markets", {})
    return {
        "segment": payload.get("segment"),
        "week": payload.get("week"),
        "generation": payload.get("generation"),
        "portfolio_vol_annual": payload.get("portfolio_vol_annual"),
        "market_source": payload.get("market_source"),
        "symbol": payload.get("symbol"),
        "interval": payload.get("interval", "1d"),
        "champion": payload.get("champion", {}),
        "research": payload.get("research", []),
        "proposed_orders": payload.get("proposed_orders", []),
        "model_trades": payload.get("model_trades", {}),
        "final_positions": payload.get("final_positions", {}),
        "final_open_slots": payload.get("final_open_slots", {}),
        "model_open_positions": payload.get("model_open_positions", {}),
        "model_selected_symbols": payload.get("model_selected_symbols", {}),
        "candidate_symbols": payload.get("candidate_symbols", []),
        "latest_prices": payload.get("latest_prices", {}),
        "results": _df_from_payload(payload.get("results", {}), parse_datetime_index=False),
        "long_tail": _df_from_payload(payload.get("long_tail", {}), parse_datetime_index=False),
        "market": _df_from_payload(payload.get("market", {}), parse_datetime_index=True),
        "model_markets": {
            str(model_id): _df_from_payload(frame_payload, parse_datetime_index=True)
            for model_id, frame_payload in model_markets_payload.items()
            if isinstance(frame_payload, dict)
        },
    }


def save_last_ui_run(summary: dict[str, Any], file_path: Path = UI_STATE_FILE) -> None:
    payload = _serialize_summary(summary)
    _write_json_atomic(file_path, payload)


def load_last_ui_run(file_path: Path = UI_STATE_FILE) -> dict[str, Any] | None:
    payload = _load_json_resilient(file_path, None)
    if not isinstance(payload, dict):
        return None

    return _deserialize_summary(payload)


def save_segment_runs(segment_runs: dict[str, dict[str, Any]], file_path: Path = UI_SEGMENT_STATE_FILE) -> None:
    payload = {
        str(segment): _serialize_summary(summary)
        for segment, summary in segment_runs.items()
        if isinstance(summary, dict)
    }
    _write_json_atomic(file_path, payload)


def load_segment_runs(file_path: Path = UI_SEGMENT_STATE_FILE) -> dict[str, dict[str, Any]]:
    payload = _load_json_resilient(file_path, {})

    if not isinstance(payload, dict):
        return {}

    return {
        str(segment): _deserialize_summary(summary)
        for segment, summary in payload.items()
        if isinstance(summary, dict)
    }


def clear_last_ui_run(file_path: Path = UI_STATE_FILE) -> None:
    if file_path.exists():
        file_path.unlink()


def clear_segment_runs(file_path: Path = UI_SEGMENT_STATE_FILE) -> None:
    if file_path.exists():
        file_path.unlink()


def save_runtime_state(state: dict[str, Any], file_path: Path = UI_RUNTIME_FILE) -> None:
    _write_json_atomic(file_path, state)


def load_runtime_state(file_path: Path = UI_RUNTIME_FILE) -> dict[str, Any]:
    data = _load_json_resilient(file_path, {})
    return data if isinstance(data, dict) else {}


def clear_runtime_state(file_path: Path = UI_RUNTIME_FILE) -> None:
    if file_path.exists():
        file_path.unlink()
