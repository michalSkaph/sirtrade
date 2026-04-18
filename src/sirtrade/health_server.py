from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .build_info import get_last_commit_info
from .copy_trading import get_copy_trading_status
from .data import fetch_binance_market
from .market_stream import get_stream_diagnostics
from .ui_state import load_runtime_state
from .ui_state import load_worker_status


DEFAULT_HEALTH_PORT = 8080
DEFAULT_WORKER_STALE_SECONDS = 90

_server_lock = threading.Lock()
_server_started = False


def _json_default(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_market_chart_payload(symbol: str, interval: str, limit: int) -> dict[str, object]:
    frame = fetch_binance_market(symbol=symbol, interval=interval, limit=limit)
    if frame.empty:
        raise ValueError("No chart data available.")

    normalized = frame.copy().sort_index()
    for col_name in ["open", "high", "low", "close"]:
        normalized[col_name] = pd.to_numeric(normalized[col_name], errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    if normalized.empty:
        raise ValueError("Chart data is empty after normalization.")

    candles: list[dict[str, float | int]] = []
    for timestamp, row in normalized.iterrows():
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        candles.append(
            {
                "time": int(ts.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )

    last_bar = candles[-1]
    prev_bar = candles[-2] if len(candles) > 1 else None
    change_pct = None
    if prev_bar and float(prev_bar["close"]) != 0.0:
        change_pct = ((float(last_bar["close"]) - float(prev_bar["close"])) / float(prev_bar["close"])) * 100.0

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "candles": candles,
        "last_price": float(last_bar["close"]),
        "change_pct": change_pct,
        "updated_at": pd.Timestamp.utcnow(),
    }


def _build_worker_health_payload() -> tuple[bool, dict[str, object]]:
    status = load_worker_status()
    heartbeat = pd.to_datetime(status.get("heartbeat_at"), utc=True, errors="coerce")
    stale_after = int(os.getenv("SIRTRADE_WORKER_STALE_SECONDS", str(DEFAULT_WORKER_STALE_SECONDS)))
    stream_status = status.get("market_stream") if isinstance(status.get("market_stream"), dict) else get_stream_diagnostics()
    deployed_commit = get_last_commit_info()

    if pd.isna(heartbeat):
        return False, {
            "deployed_commit": deployed_commit,
            "worker": {
                "status": "missing",
                "fresh": False,
                "detail": "No worker heartbeat found",
            }
            , "market_stream": stream_status
        }

    age_seconds = max(0.0, (pd.Timestamp.now(tz="UTC") - heartbeat).total_seconds())
    fresh = age_seconds <= stale_after
    payload = {
        "deployed_commit": deployed_commit,
        "worker": {
            **status,
            "fresh": fresh,
            "heartbeat_age_seconds": round(age_seconds, 1),
            "stale_after_seconds": stale_after,
        },
        "market_stream": stream_status,
    }
    return fresh, payload


def _build_status_payload() -> dict[str, object]:
    runtime_state = load_runtime_state()
    worker_status = load_worker_status()
    return {
        "deployed_commit": get_last_commit_info(),
        "runtime_state": runtime_state,
        "worker": worker_status,
        "market_stream": worker_status.get("market_stream", get_stream_diagnostics()),
        "copy_trading": get_copy_trading_status(),
        "updated_at": pd.Timestamp.utcnow(),
    }


class HealthHandler(BaseHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def _send_json(self, code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            worker_fresh, worker_payload = _build_worker_health_payload()
            code = 200 if worker_fresh else 503
            self._send_json(code, {"status": "ok" if worker_fresh else "degraded", **worker_payload})
            return

        if parsed.path == "/status":
            self._send_json(200, _build_status_payload())
            return

        if parsed.path == "/market-chart":
            query = parse_qs(parsed.query)
            symbol = str(query.get("symbol", ["BTCUSDT"])[0]).strip().upper() or "BTCUSDT"
            interval = str(query.get("interval", ["1h"])[0]).strip() or "1h"
            try:
                limit = max(2, min(1000, int(query.get("limit", ["300"])[0])))
            except (TypeError, ValueError):
                limit = 300

            try:
                payload = _build_market_chart_payload(symbol=symbol, interval=interval, limit=limit)
            except Exception as exc:
                self._send_json(500, {"error": "market_chart_failed", "detail": str(exc)})
                return

            self._send_json(200, payload)
            return

        self._send_json(404, {"error": "Not found"})

    def log_message(self, format: str, *args) -> None:
        return


def serve_health_server(port: int | None = None) -> None:
    bind_port = int(port or os.getenv("SIRTRADE_HEALTH_PORT", str(DEFAULT_HEALTH_PORT)))
    server = ThreadingHTTPServer(("0.0.0.0", bind_port), HealthHandler)
    print(f"Health server listening on 0.0.0.0:{bind_port}")
    server.serve_forever()


def _serve_health_server_background(port: int | None = None) -> None:
    try:
        serve_health_server(port=port)
    except OSError as exc:
        if "Address already in use" in str(exc) or "Only one usage of each socket address" in str(exc):
            print("Health server already running on requested port; reusing existing instance.")
            return
        raise


def ensure_health_server_started(port: int | None = None) -> None:
    global _server_started
    with _server_lock:
        if _server_started:
            return
        thread = threading.Thread(target=_serve_health_server_background, kwargs={"port": port}, name="sirtrade-health-server", daemon=True)
        thread.start()
        _server_started = True