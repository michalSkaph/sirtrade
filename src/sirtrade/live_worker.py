from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

from .automation import run_segment_cycle
from .config import DEFAULT_CONFIG, INITIAL_PAPER_WALLET_CZK, PAPER_TRADE_SIZE_CZK
from .engine import TradingEngine
from .execution import build_dry_run_orders
from .market_stream import get_stream_diagnostics
from .reporting import export_weekly_report
from .storage import init_db, save_closed_positions, save_open_positions, save_week_result
from .ui_state import (
    load_runtime_state,
    load_segment_runs,
    save_last_ui_run,
    save_segment_runs,
    save_worker_status,
)


SEGMENT_DEFAULTS = {
    "Scalp": {"interval": "1m", "sim_days": 1, "namespace": "SC", "binance_decision_seconds": 5},
    "Intraday": {"interval": "15m", "sim_days": 7, "namespace": "ID", "binance_decision_seconds": 10},
    "Swing": {"interval": "4h", "sim_days": 30, "namespace": "SW", "binance_decision_seconds": 30},
}

BINANCE_DECISION_SECONDS = 30
WORKER_SLEEP_SECONDS = 1.0
WORKER_HEARTBEAT_SECONDS = 10.0
WORKER_LOCK_FILE = Path("data/live_worker.lock")

_worker_lock = threading.Lock()
_worker_started = False
_worker_process_lock: "_WorkerProcessLock | None" = None


class _WorkerProcessLock:
    def __init__(self, lock_path: Path = WORKER_LOCK_FILE) -> None:
        self.lock_path = Path(lock_path)
        self._handle = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        try:
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            if os.name == "nt":
                if msvcrt is None:
                    raise OSError("msvcrt is unavailable")
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                if fcntl is None:
                    raise OSError("fcntl is unavailable")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            handle.seek(0)
            handle.truncate()
            payload = f"pid={os.getpid()} acquired_at={pd.Timestamp.utcnow().isoformat()}\n"
            handle.write(payload.encode("utf-8"))
            handle.flush()
            self._handle = handle
            return True
        except OSError:
            handle.close()
            return False

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                if msvcrt is not None:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


def _acquire_worker_process_lock() -> bool:
    global _worker_process_lock
    if _worker_process_lock is not None:
        return True

    process_lock = _WorkerProcessLock()
    if not process_lock.acquire():
        return False

    _worker_process_lock = process_lock
    return True


def _release_worker_process_lock() -> None:
    global _worker_process_lock
    if _worker_process_lock is None:
        return
    _worker_process_lock.release()
    _worker_process_lock = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def should_start_embedded_worker() -> bool:
    return _env_flag("SIRTRADE_ENABLE_EMBEDDED_WORKER", True)


def _write_worker_status(**fields: Any) -> None:
    payload = {
        "worker_enabled": True,
        "heartbeat_at": pd.Timestamp.utcnow(),
        "market_stream": get_stream_diagnostics(),
        **fields,
    }
    save_worker_status(payload)


def _extract_slot_delta(action: str, entry: bool) -> int:
    pattern = r"\(\+(\d+)\)" if entry else r"\(-?(\d+)\)"
    match = re.search(pattern, str(action))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _normalize_side(value: Any) -> str:
    side = str(value).strip().upper()
    if side in {"BUY", "LONG", "1", "+1"}:
        return "LONG"
    if side in {"SELL", "SHORT", "-1"}:
        return "SHORT"
    return side


def _coerce_cutoff_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, "", 0, "0"):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _filter_events_since(events: Any, cutoff_ts: pd.Timestamp | None) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    if cutoff_ts is None:
        return [event for event in events if isinstance(event, dict)]

    filtered: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_ts = pd.to_datetime(event.get("executed_at", event.get("timestamp")), utc=True, errors="coerce")
        if pd.isna(event_ts) or event_ts < cutoff_ts:
            continue
        filtered.append(event)
    return filtered


def _build_open_position_states(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: dict[tuple[str, str], dict[str, Any]] = {}

    sorted_events = sorted(
        events,
        key=lambda item: pd.to_datetime(item.get("executed_at", item.get("timestamp")), utc=True, errors="coerce"),
    )

    for event in sorted_events:
        action = str(event.get("akce", ""))
        side = _normalize_side(event.get("strana", ""))
        event_symbol = str(event.get("symbol", "")).upper()
        if not event_symbol:
            continue
        try:
            price = float(event.get("cena", 0.0))
        except Exception:
            price = 0.0
        timestamp = str(event.get("executed_at", event.get("timestamp", "")))

        if "Vstup" in action and side in {"LONG", "SHORT"}:
            state_key = (event_symbol, side)
            state = states.get(
                state_key,
                {
                    "open_slots": 1,
                    "side": side,
                    "symbol": event_symbol,
                    "opened_at": timestamp,
                    "entry_price": price,
                },
            )
            state["open_slots"] = 1
            if float(state.get("entry_price", 0.0) or 0.0) <= 0.0:
                state["entry_price"] = price
            state["opened_at"] = state.get("opened_at") or timestamp
            state["symbol"] = event_symbol
            state["side"] = side
            states[state_key] = state
            continue

        if "Výstup" in action and side in {"LONG", "SHORT"}:
            state_key = (event_symbol, side)
            state = states.get(state_key)
            if state is None or int(state.get("open_slots", 0)) <= 0:
                continue
            states.pop(state_key, None)

    return [state for state in states.values() if int(state.get("open_slots", 0)) > 0]


def _apply_trade_cutoff(summary: dict[str, Any], cutoff_value: Any) -> dict[str, Any]:
    cutoff_ts = _coerce_cutoff_timestamp(cutoff_value)
    if cutoff_ts is None:
        return summary

    filtered_summary = dict(summary)
    model_trades = summary.get("model_trades", {})
    trade_events_delta = summary.get("trade_events_delta", {})
    results_df = summary.get("results")
    if not isinstance(model_trades, dict) or not isinstance(results_df, pd.DataFrame):
        return filtered_summary

    filtered_trades = {
        str(model_id): _filter_events_since(events, cutoff_ts)
        for model_id, events in model_trades.items()
    }
    filtered_trade_events_delta = {
        str(model_id): _filter_events_since(events, cutoff_ts)
        for model_id, events in trade_events_delta.items()
    } if isinstance(trade_events_delta, dict) else {}
    slot_size = PAPER_TRADE_SIZE_CZK / INITIAL_PAPER_WALLET_CZK
    final_positions: dict[str, float] = {}
    final_open_slots: dict[str, int] = {}
    model_open_positions: dict[str, list[dict[str, Any]]] = {}

    for _, row in results_df.reset_index(drop=True).iterrows():
        model_id = str(row["model_id"])
        model_name = str(row["name"])
        states = _build_open_position_states(filtered_trades.get(model_id, []))
        open_slots = sum(int(state.get("open_slots", 0)) for state in states)
        signed_slot_count = sum(
            (1 if state.get("side") == "LONG" else -1 if state.get("side") == "SHORT" else 0)
            * int(state.get("open_slots", 0))
            for state in states
        )
        final_open_slots[model_id] = open_slots
        final_positions[model_id] = float(signed_slot_count) * slot_size

        if open_slots > 0 and states:
            positions: list[dict[str, Any]] = []
            for state in states:
                state_side = str(state.get("side", ""))
                state_symbol = str(state.get("symbol", "")).upper()
                state_slots = int(state.get("open_slots", 0))
                if state_side not in {"LONG", "SHORT"} or not state_symbol:
                    continue
                positions.append(
                    {
                        "symbol": state_symbol,
                        "side": state_side,
                        "slots": state_slots,
                        "model_id": model_id,
                        "model_name": model_name,
                        "opened_at": state.get("opened_at"),
                        "entry_price": state.get("entry_price"),
                    }
                )
            model_open_positions[model_id] = positions
        else:
            model_open_positions[model_id] = []

    filtered_summary["model_trades"] = filtered_trades
    filtered_summary["trade_events_delta"] = filtered_trade_events_delta
    champion_model_id = str(summary.get("champion", {}).get("model_id", ""))
    filtered_summary["champion_trades"] = filtered_trades.get(champion_model_id, [])
    filtered_summary["final_positions"] = final_positions
    filtered_summary["final_open_slots"] = final_open_slots
    filtered_summary["model_open_positions"] = model_open_positions

    order_source = results_df.copy()
    order_source["model_open_positions"] = order_source["model_id"].map(model_open_positions)
    filtered_summary["proposed_orders"] = [
        order.__dict__
            for order in build_dry_run_orders(
                order_source,
                symbol=str(summary.get("symbol", "BTCUSDT")),
                trade_size_czk=PAPER_TRADE_SIZE_CZK,
            )
    ]
    return filtered_summary


def _build_engines() -> dict[str, TradingEngine]:
    return {
        segment: TradingEngine(
            DEFAULT_CONFIG,
            model_namespace=cfg["namespace"],
            model_label_prefix=segment,
        )
        for segment, cfg in SEGMENT_DEFAULTS.items()
    }


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_binance_cadence_seconds(segment: str) -> int:
    cfg = SEGMENT_DEFAULTS.get(segment, {})
    default_seconds = max(1, _coerce_int(cfg.get("binance_decision_seconds"), BINANCE_DECISION_SECONDS))
    env_key = f"SIRTRADE_BINANCE_DECISION_SECONDS_{segment.upper()}"
    return max(
        1,
        _coerce_int(
            os.getenv(env_key, os.getenv("SIRTRADE_BINANCE_DECISION_SECONDS", str(default_seconds))),
            default_seconds,
        ),
    )


def _hydrate_engines_from_saved_runs(
    engines: dict[str, TradingEngine],
    segment_runs: dict[str, dict[str, Any]],
) -> None:
    for segment, engine in engines.items():
        summary = segment_runs.get(segment)
        if not isinstance(summary, dict):
            continue
        week = max(0, _coerce_int(summary.get("week"), 0))
        generation = max(1, _coerce_int(summary.get("generation"), 1))
        engine.week = max(engine.week, week)
        engine.generation = max(engine.generation, generation)
        for model in engine.models:
            model.generation = engine.generation


def _choose_segments(
    runnable_segments: list[str],
    active_segment: str,
    worker_state: dict[str, Any],
    data_source: str,
) -> list[str]:
    if not runnable_segments:
        return []
    if data_source in {"binance", "binance_copy"}:
        return runnable_segments

    cursor = _coerce_int(worker_state.get("segment_cursor"), 0)
    if cursor < 0:
        cursor = 0
    cursor %= len(runnable_segments)
    selected_segment = runnable_segments[cursor]
    worker_state["segment_cursor"] = (cursor + 1) % len(runnable_segments)
    return [selected_segment]


def _run_worker_loop() -> None:
    init_db()
    engines = _build_engines()
    worker_state: dict[str, Any] = {"last_run_by_segment": {}, "segment_cursor": 0, "reset_token": None}
    last_heartbeat_write = 0.0

    _write_worker_status(status="starting", message="Worker booting")

    while True:
        now = time.time()
        if (now - last_heartbeat_write) >= WORKER_HEARTBEAT_SECONDS:
            _write_worker_status(status="idle", message="Worker heartbeat")
            last_heartbeat_write = now

        runtime_state = load_runtime_state()
        segment_runs = load_segment_runs()
        _hydrate_engines_from_saved_runs(engines, segment_runs)
        reset_token = runtime_state.get("reset_token")
        if reset_token != worker_state.get("reset_token"):
            worker_state["last_run_by_segment"] = {}
            worker_state["reset_token"] = reset_token

        simulation_running_by_segment = runtime_state.get("simulation_running_by_segment", {})
        active_segment = str(runtime_state.get("active_segment", "Swing"))
        data_source = str(runtime_state.get("data_source", "binance"))
        symbol = str(runtime_state.get("symbol", "BTCUSDT")).upper()
        paper_trade_cutoff_ts = runtime_state.get("paper_trade_cutoff_ts")
        default_simulation_seconds = max(1, _coerce_int(runtime_state.get("simulation_cycle_seconds"), 10))

        runnable_segments = [
            segment
            for segment in SEGMENT_DEFAULTS.keys()
            if bool(simulation_running_by_segment.get(segment, False))
        ]

        selected_segments = _choose_segments(runnable_segments, active_segment, worker_state, data_source)
        updated_segment_runs = dict(segment_runs)

        for segment in selected_segments:
            cadence_seconds = (
                _get_binance_cadence_seconds(segment)
                if data_source in {"binance", "binance_copy"}
                else default_simulation_seconds
            )
            last_run = float(worker_state["last_run_by_segment"].get(segment, 0.0))
            if (now - last_run) < cadence_seconds:
                continue

            cfg = SEGMENT_DEFAULTS[segment]
            try:
                _write_worker_status(
                    status="running",
                    active_segment=segment,
                    market_source=data_source,
                    symbol=symbol,
                    interval=str(cfg["interval"]),
                    message=f"Running segment {segment}",
                )
                result = run_segment_cycle(
                    engine=engines[segment],
                    segment=segment,
                    market_source=data_source,
                    symbol=symbol,
                    days=int(cfg["sim_days"]),
                    interval=str(cfg["interval"]),
                    previous_summary=segment_runs.get(segment),
                )
                result = _apply_trade_cutoff(result, paper_trade_cutoff_ts)
                latest_runtime_state = load_runtime_state()
                latest_running = latest_runtime_state.get("simulation_running_by_segment", {})
                if (
                    latest_runtime_state.get("reset_token") != reset_token
                    or not bool(latest_running.get(segment, False))
                ):
                    continue
                updated_segment_runs[segment] = result
                save_week_result(result)
                save_open_positions(result)
                save_closed_positions(result)
                export_weekly_report(result, DEFAULT_CONFIG)
                save_segment_runs(updated_segment_runs)
                if segment == active_segment:
                    save_last_ui_run(result)

                worker_state["last_run_by_segment"][segment] = now
                _write_worker_status(
                    status="ok",
                    active_segment=segment,
                    market_source=data_source,
                    symbol=symbol,
                    interval=str(cfg["interval"]),
                    last_success_at=pd.Timestamp.utcnow(),
                    last_completed_week=result.get("week"),
                    message=f"Completed segment {segment}",
                )
                last_heartbeat_write = time.time()
            except Exception as exc:
                _write_worker_status(
                    status="error",
                    active_segment=segment,
                    market_source=data_source,
                    symbol=symbol,
                    interval=str(cfg["interval"]),
                    last_error_at=pd.Timestamp.utcnow(),
                    message=str(exc),
                )
                print(f"[WORKER][ERROR] segment={segment}: {exc}")

        time.sleep(WORKER_SLEEP_SECONDS)


def _run_worker_loop_forever() -> None:
    global _worker_started
    try:
        _run_worker_loop()
    finally:
        with _worker_lock:
            _worker_started = False
        _release_worker_process_lock()


def serve_live_worker() -> None:
    if not _acquire_worker_process_lock():
        print("[WORKER] Another SirTrade worker process is already running; skipping duplicate worker start.")
        return
    _run_worker_loop_forever()


def is_live_worker_started() -> bool:
    return _worker_started


def ensure_live_worker_started() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        if not _acquire_worker_process_lock():
            print("[WORKER] Another SirTrade worker process already holds the worker lock; embedded worker will not start.")
            return
        thread = threading.Thread(target=_run_worker_loop_forever, name="sirtrade-live-worker", daemon=True)
        thread.start()
        _worker_started = True