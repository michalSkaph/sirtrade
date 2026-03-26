from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import websocket
except Exception:  # pragma: no cover - optional runtime dependency fallback
    websocket = None


BINANCE_WS_BASE_URL = "wss://stream.binance.com:9443/ws"
SUPPORTED_STREAM_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def streams_enabled() -> bool:
    return websocket is not None and _env_flag("SIRTRADE_ENABLE_BINANCE_STREAM", True)


@dataclass
class _StreamState:
    symbol: str
    interval: str
    started: bool = False
    connected: bool = False
    last_message_at: float = 0.0
    last_error: str = ""
    latest_kline: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class _BinanceKlineStreamManager:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], _StreamState] = {}
        self._lock = threading.Lock()

    def ensure_stream(self, symbol: str, interval: str) -> None:
        if not streams_enabled() or interval not in SUPPORTED_STREAM_INTERVALS:
            return
        key = (symbol.upper(), interval)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _StreamState(symbol=key[0], interval=key[1])
                self._states[key] = state
            if state.started:
                return
            state.started = True
            thread = threading.Thread(target=self._run_stream, args=(state,), daemon=True, name=f"binance-kline-{key[0]}-{key[1]}")
            thread.start()

    def latest_kline(self, symbol: str, interval: str) -> dict[str, Any] | None:
        state = self._states.get((symbol.upper(), interval))
        if state is None:
            return None
        with state.lock:
            if state.latest_kline is None:
                return None
            return dict(state.latest_kline)

    def diagnostics(self) -> dict[str, Any]:
        stream_rows = []
        for (symbol, interval), state in list(self._states.items()):
            with state.lock:
                stream_rows.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "connected": state.connected,
                        "last_message_at": pd.Timestamp(state.last_message_at, unit="s", tz="UTC").isoformat() if state.last_message_at else None,
                        "last_error": state.last_error or None,
                    }
                )
        return {
            "enabled": streams_enabled(),
            "available": websocket is not None,
            "stream_count": len(stream_rows),
            "streams": stream_rows,
        }

    def _run_stream(self, state: _StreamState) -> None:
        assert websocket is not None
        url = f"{BINANCE_WS_BASE_URL}/{state.symbol.lower()}@kline_{state.interval}"
        reconnect_delay = 2.0

        while True:
            try:
                ws_app = websocket.WebSocketApp(
                    url,
                    on_open=lambda _: self._on_open(state),
                    on_message=lambda _, message: self._on_message(state, message),
                    on_error=lambda _, error: self._on_error(state, error),
                    on_close=lambda *_args: self._on_close(state),
                )
                ws_app.run_forever(ping_interval=20, ping_timeout=10, reconnect=5)
            except Exception as exc:
                self._on_error(state, exc)
            time.sleep(reconnect_delay)

    def _on_open(self, state: _StreamState) -> None:
        with state.lock:
            state.connected = True
            state.last_error = ""

    def _on_close(self, state: _StreamState) -> None:
        with state.lock:
            state.connected = False

    def _on_error(self, state: _StreamState, error: object) -> None:
        with state.lock:
            state.connected = False
            state.last_error = str(error)

    def _on_message(self, state: _StreamState, message: str) -> None:
        payload = json.loads(message)
        kline = payload.get("k")
        if not isinstance(kline, dict):
            return
        snapshot = {
            "open_time": int(kline.get("t", 0)),
            "close_time": int(kline.get("T", 0)),
            "open": float(kline.get("o", 0.0)),
            "high": float(kline.get("h", 0.0)),
            "low": float(kline.get("l", 0.0)),
            "close": float(kline.get("c", 0.0)),
            "is_closed": bool(kline.get("x", False)),
        }
        with state.lock:
            state.connected = True
            state.last_message_at = time.time()
            state.latest_kline = snapshot


_STREAM_MANAGER = _BinanceKlineStreamManager()


def _recompute_market_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_index()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    out["ret"] = close.pct_change().fillna(0.0)
    out["sentiment"] = out["ret"].rolling(5).mean().fillna(0.0).clip(-0.3, 0.3) * 10
    out["onchain"] = out["ret"].rolling(10).mean().fillna(0.0).clip(-0.25, 0.25) * 12
    out["regime"] = np.where(out["ret"].abs().rolling(20).mean().fillna(out["ret"].abs().mean()) > 0.03, 2, 0)
    return out


def apply_stream_kline_to_market(frame: pd.DataFrame, kline: dict[str, Any] | None) -> pd.DataFrame:
    if frame.empty or not kline:
        return frame

    ts = pd.Timestamp(int(kline["open_time"]), unit="ms", tz="UTC")
    row = {
        "open": float(kline["open"]),
        "high": float(kline["high"]),
        "low": float(kline["low"]),
        "close": float(kline["close"]),
    }

    out = frame.copy()
    if ts in out.index:
        for key, value in row.items():
            out.loc[ts, key] = value
    elif ts > out.index.max():
        new_row = pd.DataFrame([row], index=pd.DatetimeIndex([ts]))
        out = pd.concat([out, new_row])
    else:
        return frame

    return _recompute_market_indicators(out)


def merge_rest_market_with_stream(frame: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    if frame.empty or interval not in SUPPORTED_STREAM_INTERVALS:
        return frame
    _STREAM_MANAGER.ensure_stream(symbol, interval)
    return apply_stream_kline_to_market(frame, _STREAM_MANAGER.latest_kline(symbol, interval))


def get_stream_diagnostics() -> dict[str, Any]:
    return _STREAM_MANAGER.diagnostics()