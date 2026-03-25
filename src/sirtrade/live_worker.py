from __future__ import annotations

import re
import threading
import time
from typing import Any

import pandas as pd

from .automation import run_segment_cycle
from .config import DEFAULT_CONFIG
from .engine import TradingEngine
from .execution import build_dry_run_orders
from .reporting import export_weekly_report
from .status import write_automation_status
from .storage import init_db, save_closed_positions, save_open_positions, save_week_result
from .ui_state import load_runtime_state, load_segment_runs, save_last_ui_run, save_segment_runs


SEGMENT_DEFAULTS = {
    "Scalp": {"interval": "5m", "sim_days": 3, "namespace": "SC"},
    "Intraday": {"interval": "15m", "sim_days": 7, "namespace": "ID"},
    "Swing": {"interval": "4h", "sim_days": 30, "namespace": "SW"},
}

BINANCE_DECISION_SECONDS = 30
WORKER_SLEEP_SECONDS = 1.0

_worker_lock = threading.Lock()
_worker_started = False


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
        event_ts = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(event_ts) or event_ts < cutoff_ts:
            continue
        filtered.append(event)
    return filtered


def _build_open_position_states(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: dict[tuple[str, str], dict[str, Any]] = {}

    sorted_events = sorted(
        events,
        key=lambda item: pd.to_datetime(item.get("timestamp"), utc=True, errors="coerce"),
    )

    for event in sorted_events:
        action = str(event.get("akce", ""))
        side = _normalize_side(event.get("strana", ""))
        qty = _extract_slot_delta(action, entry="Vstup" in action)
        event_symbol = str(event.get("symbol", symbol)).upper()
        try:
            price = float(event.get("cena", 0.0))
        except Exception:
            price = 0.0
        timestamp = str(event.get("timestamp", ""))

        if "Vstup" in action and side in {"LONG", "SHORT"}:
            if qty <= 0:
                qty = 1
            state_key = (event_symbol, side)
            state = states.get(
                state_key,
                {
                    "open_slots": 0,
                    "side": side,
                    "symbol": event_symbol,
                    "opened_at": timestamp,
                    "entry_price": price,
                },
            )
            total_cost = (float(state["entry_price"]) * int(state["open_slots"])) + (price * qty)
            total_slots = int(state["open_slots"]) + qty
            state["open_slots"] = total_slots
            state["entry_price"] = total_cost / max(total_slots, 1)
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
            if qty <= 0:
                qty = int(state.get("open_slots", 0))
            state["open_slots"] = max(0, int(state.get("open_slots", 0)) - qty)
            if int(state["open_slots"]) <= 0:
                states.pop(state_key, None)
            else:
                states[state_key] = state

    return [state for state in states.values() if int(state.get("open_slots", 0)) > 0]


def _apply_trade_cutoff(summary: dict[str, Any], cutoff_value: Any) -> dict[str, Any]:
    cutoff_ts = _coerce_cutoff_timestamp(cutoff_value)
    if cutoff_ts is None:
        return summary

    filtered_summary = dict(summary)
    model_trades = summary.get("model_trades", {})
    results_df = summary.get("results")
    if not isinstance(model_trades, dict) or not isinstance(results_df, pd.DataFrame):
        return filtered_summary

    filtered_trades = {
        str(model_id): _filter_events_since(events, cutoff_ts)
        for model_id, events in model_trades.items()
    }
    slot_size = DEFAULT_CONFIG.risk.max_asset_exposure / 5
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
            slot_cursor = 1
            for state in states:
                state_side = str(state.get("side", ""))
                state_symbol = str(state.get("symbol", "")).upper()
                if state_side not in {"LONG", "SHORT"} or not state_symbol:
                    continue
                for _ in range(int(state.get("open_slots", 0))):
                    positions.append(
                        {
                            "slot": slot_cursor,
                            "symbol": state_symbol,
                            "side": state_side,
                            "model_id": model_id,
                            "model_name": model_name,
                            "opened_at": state.get("opened_at"),
                            "entry_price": state.get("entry_price"),
                        }
                    )
                    slot_cursor += 1
            model_open_positions[model_id] = positions
        else:
            model_open_positions[model_id] = []

    filtered_summary["model_trades"] = filtered_trades
    champion_model_id = str(summary.get("champion", {}).get("model_id", ""))
    filtered_summary["champion_trades"] = filtered_trades.get(champion_model_id, [])
    filtered_summary["final_positions"] = final_positions
    filtered_summary["final_open_slots"] = final_open_slots
    filtered_summary["model_open_positions"] = model_open_positions

    order_source = results_df.copy()
    order_source["model_open_positions"] = order_source["model_id"].map(model_open_positions)
    filtered_summary["proposed_orders"] = [
        order.__dict__
        for order in build_dry_run_orders(order_source, symbol=str(summary.get("symbol", "BTCUSDT")), nav_usd=1000.0)
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
    if data_source not in {"binance", "binance_copy"}:
        return runnable_segments
    if active_segment in runnable_segments:
        return [active_segment]

    cursor = int(worker_state.get("segment_cursor", 0)) % len(runnable_segments)
    worker_state["segment_cursor"] = (cursor + 1) % len(runnable_segments)
    return [runnable_segments[cursor]]


def _run_worker_loop() -> None:
    init_db()
    engines = _build_engines()
    worker_state: dict[str, Any] = {"last_run_by_segment": {}, "segment_cursor": 0, "reset_token": None}

    while True:
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
        cadence_seconds = (
            BINANCE_DECISION_SECONDS
            if data_source in {"binance", "binance_copy"}
            else max(1, _coerce_int(runtime_state.get("simulation_cycle_seconds"), 10))
        )

        runnable_segments = [
            segment
            for segment in SEGMENT_DEFAULTS.keys()
            if bool(simulation_running_by_segment.get(segment, False))
        ]

        now = time.time()
        selected_segments = _choose_segments(runnable_segments, active_segment, worker_state, data_source)
        updated_segment_runs = dict(segment_runs)

        for segment in selected_segments:
            last_run = float(worker_state["last_run_by_segment"].get(segment, 0.0))
            if (now - last_run) < cadence_seconds:
                continue

            cfg = SEGMENT_DEFAULTS[segment]
            try:
                result = run_segment_cycle(
                    engine=engines[segment],
                    segment=segment,
                    market_source=data_source,
                    symbol=symbol,
                    days=int(cfg["sim_days"]),
                    interval=str(cfg["interval"]),
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

                champion = result.get("champion", {})
                write_automation_status(
                    {
                        "ok": True,
                        "segment": segment,
                        "result": {
                            "week": result.get("week"),
                            "generation": result.get("generation"),
                            "market_source": result.get("market_source"),
                            "symbol": result.get("symbol"),
                            "champion": {
                                "model_id": champion.get("model_id"),
                                "name": champion.get("name"),
                                "score": champion.get("score"),
                            },
                        },
                    }
                )
                worker_state["last_run_by_segment"][segment] = now
            except Exception as exc:
                write_automation_status(
                    {
                        "ok": False,
                        "segment": segment,
                        "error": str(exc),
                        "source": data_source,
                        "symbol": symbol,
                    }
                )

        time.sleep(WORKER_SLEEP_SECONDS)


def ensure_live_worker_started() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_run_worker_loop, name="sirtrade-live-worker", daemon=True)
        thread.start()
        _worker_started = True