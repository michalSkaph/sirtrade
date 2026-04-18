from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_CONFIG, PAPER_TRADE_SIZE_CZK


def empty_trade_analytics() -> dict[str, Any]:
    return {
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_pnl_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "expectancy_pct": 0.0,
        "expectancy_czk": 0.0,
        "profit_factor": 0.0,
        "avg_holding_minutes": 0.0,
        "exit_reasons": [],
    }


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_trade_side(value: object) -> str:
    side = str(value).strip().upper()
    if side in {"BUY", "LONG", "+1", "1"}:
        return "LONG"
    if side in {"SELL", "SHORT", "-1"}:
        return "SHORT"
    return side


def _extract_event_slot_count(event: dict[str, Any], entry: bool) -> int:
    direct_quantity = event.get("quantity_slots")
    try:
        if direct_quantity is not None:
            return max(0, int(abs(float(direct_quantity))))
    except Exception:
        pass

    direct_slots = event.get("sloty")
    try:
        if direct_slots is not None and float(direct_slots) > 0:
            return max(0, int(abs(float(direct_slots))))
    except Exception:
        pass

    action = str(event.get("akce", ""))
    pattern = r"\(\+(\d+)\)" if entry else r"\(-?(\d+)\)"
    match = re.search(pattern, action)
    if not match:
        return 0
    try:
        return max(0, int(match.group(1)))
    except Exception:
        return 0


def _build_closed_trade_records(model_trades: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    closed_records: list[dict[str, Any]] = []
    open_states: dict[tuple[str, str, str], dict[str, Any]] = {}

    for model_id, events in model_trades.items():
        if not isinstance(events, list):
            continue

        sorted_events = sorted(
            [event for event in events if isinstance(event, dict)],
            key=lambda item: pd.to_datetime(item.get("executed_at", item.get("timestamp")), utc=True, errors="coerce"),
        )
        for event in sorted_events:
            action = str(event.get("akce", ""))
            side = _normalize_trade_side(event.get("strana", ""))
            symbol = str(event.get("symbol", "")).upper()
            if side not in {"LONG", "SHORT"} or not symbol:
                continue

            state_key = (str(model_id), symbol, side)
            if "Vstup" in action:
                open_slots = _extract_event_slot_count(event, entry=True)
                if open_slots <= 0:
                    continue
                entry_price = _coerce_float(event.get("entry_price", event.get("cena", 0.0)))
                open_states[state_key] = {
                    "opened_at": event.get("opened_at", event.get("executed_at", event.get("timestamp"))),
                    "entry_price": entry_price,
                    "quantity_slots": open_slots,
                    "model_id": str(model_id),
                    "model_name": str(event.get("model_name", model_id)),
                    "symbol": symbol,
                    "side": side,
                }
                continue

            if "Výstup" not in action:
                continue

            exit_price = _coerce_float(event.get("cena", 0.0))
            quantity_slots = _extract_event_slot_count(event, entry=False)
            opened_at = event.get("opened_at")
            entry_price = event.get("entry_price")
            open_state = open_states.get(state_key)

            if (opened_at in (None, "") or not entry_price) and open_state is not None:
                opened_at = open_state.get("opened_at")
                entry_price = open_state.get("entry_price")
                if quantity_slots <= 0:
                    quantity_slots = int(open_state.get("quantity_slots", 0) or 0)

            entry_price_value = _coerce_float(entry_price)
            if entry_price_value <= 0 or exit_price <= 0 or quantity_slots <= 0:
                continue

            pnl_pct = ((exit_price - entry_price_value) / entry_price_value) * 100.0
            if side == "SHORT":
                pnl_pct = -pnl_pct
            pnl_czk = float(quantity_slots) * float(PAPER_TRADE_SIZE_CZK) * (pnl_pct / 100.0)

            opened_ts = pd.to_datetime(opened_at, utc=True, errors="coerce")
            closed_ts = pd.to_datetime(event.get("executed_at", event.get("timestamp")), utc=True, errors="coerce")
            holding_minutes = None
            if not pd.isna(opened_ts) and not pd.isna(closed_ts):
                holding_minutes = max(0.0, float((closed_ts - opened_ts).total_seconds() / 60.0))

            closed_records.append(
                {
                    "model_id": str(model_id),
                    "model_name": str(event.get("model_name", open_state.get("model_name") if open_state else model_id)),
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price_value,
                    "exit_price": exit_price,
                    "quantity_slots": int(quantity_slots),
                    "pnl_pct": float(pnl_pct),
                    "pnl_czk": float(pnl_czk),
                    "exit_reason": str(event.get("duvod_vystupu", "NEURČENO")),
                    "holding_minutes": holding_minutes,
                }
            )
            open_states.pop(state_key, None)

    return closed_records


def _summarize_trade_records(closed_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not closed_records:
        return empty_trade_analytics()

    closed_frame = pd.DataFrame(closed_records)
    pnl_pct = pd.to_numeric(closed_frame["pnl_pct"], errors="coerce").fillna(0.0)
    pnl_czk = pd.to_numeric(closed_frame["pnl_czk"], errors="coerce").fillna(0.0)
    holding_minutes = pd.to_numeric(closed_frame.get("holding_minutes"), errors="coerce")

    wins = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct < 0]
    win_count = int(len(wins))
    loss_count = int(len(losses))
    trade_count = int(len(pnl_pct))
    win_rate = (win_count / trade_count) if trade_count > 0 else 0.0
    avg_win_pct = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_pct = float(losses.mean()) if not losses.empty else 0.0
    expectancy_pct = (win_rate * avg_win_pct) + ((1.0 - win_rate) * avg_loss_pct)
    gross_profit = float(pnl_czk[pnl_czk > 0].sum())
    gross_loss = float(abs(pnl_czk[pnl_czk < 0].sum()))
    exit_breakdown = (
        closed_frame["exit_reason"].fillna("NEURČENO").astype(str).value_counts().reset_index().values.tolist()
    )

    return {
        "closed_trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": float(win_rate * 100.0),
        "avg_pnl_pct": float(pnl_pct.mean()),
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "expectancy_pct": float(expectancy_pct),
        "expectancy_czk": float(pnl_czk.mean()) if len(pnl_czk) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit > 0),
        "avg_holding_minutes": float(holding_minutes.dropna().mean())
        if holding_minutes is not None and not holding_minutes.dropna().empty
        else 0.0,
        "exit_reasons": [
            {"reason": str(reason), "count": int(count)}
            for reason, count in exit_breakdown
        ],
    }


def champion_trade_metrics_from_summary(summary: dict[str, Any], trade_size_czk: float = PAPER_TRADE_SIZE_CZK) -> dict[str, Any]:
    champion = summary.get("champion", {}) if isinstance(summary, dict) else {}
    champion_model_id = str(champion.get("model_id", "")).strip()
    has_champion_metrics = any(
        key in champion
        for key in ("closed_trades", "win_rate", "profit_factor", "pnl_czk")
    )
    if champion_model_id and has_champion_metrics:
        return {
            "champion_model_id": champion_model_id,
            "champion_closed_trades": _coerce_int(champion.get("closed_trades"), 0),
            "champion_win_rate": _coerce_float(champion.get("win_rate"), 0.0),
            "champion_profit_factor": _coerce_float(champion.get("profit_factor"), 0.0),
            "champion_pnl_czk": _coerce_float(champion.get("pnl_czk"), 0.0),
        }

    model_trades = summary.get("model_trades", {}) if isinstance(summary, dict) else {}
    if not champion_model_id or not isinstance(model_trades, dict):
        return {
            "champion_model_id": champion_model_id or None,
            "champion_closed_trades": 0,
            "champion_win_rate": 0.0,
            "champion_profit_factor": 0.0,
            "champion_pnl_czk": 0.0,
        }

    closed_records = _build_closed_trade_records({champion_model_id: list(model_trades.get(champion_model_id, []))})
    analytics = _summarize_trade_records(closed_records)
    expectancy_czk = _coerce_float(analytics.get("expectancy_czk"), 0.0)
    closed_trades = _coerce_int(analytics.get("closed_trades"), 0)
    total_pnl_czk = expectancy_czk * float(closed_trades)
    return {
        "champion_model_id": champion_model_id,
        "champion_closed_trades": closed_trades,
        "champion_win_rate": _coerce_float(analytics.get("win_rate"), 0.0),
        "champion_profit_factor": _coerce_float(analytics.get("profit_factor"), 0.0),
        "champion_pnl_czk": float(total_pnl_czk),
    }


def reset_summary_for_new_cycle(summary: dict[str, Any] | None, *, generation: int = 1) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None

    reset_summary = dict(summary)
    reset_summary["week"] = 0
    reset_summary["generation"] = int(max(1, generation))
    reset_summary["portfolio_vol_annual"] = 0.0
    reset_summary["research"] = []
    reset_summary["proposed_orders"] = []
    reset_summary["champion_trades"] = []
    reset_summary["trade_analytics"] = empty_trade_analytics()

    results = summary.get("results")
    if isinstance(results, pd.DataFrame):
        normalized_results = results.copy()
        if "generation" in normalized_results.columns:
            normalized_results["generation"] = int(max(1, generation))
        for column in ("sortino", "calmar", "cvar95", "max_dd", "cost", "turnover", "score", "profit_factor", "pnl_czk"):
            if column in normalized_results.columns:
                normalized_results[column] = 0.0
        if "passed" in normalized_results.columns:
            normalized_results["passed"] = False
        if "closed_trades" in normalized_results.columns:
            normalized_results["closed_trades"] = 0
        if "win_rate" in normalized_results.columns:
            normalized_results["win_rate"] = 0.0
        reset_summary["results"] = normalized_results

    champion = summary.get("champion") if isinstance(summary.get("champion"), dict) else {}
    reset_champion = dict(champion)
    reset_champion["generation"] = int(max(1, generation))
    for key in ("sortino", "calmar", "cvar95", "max_dd", "cost", "turnover", "score", "reward_usd", "profit_factor", "pnl_czk"):
        if key in reset_champion:
            reset_champion[key] = 0.0
    reset_champion["passed"] = False
    reset_summary["champion"] = reset_champion

    results_frame = reset_summary.get("results")
    model_ids: list[str] = []
    if isinstance(results_frame, pd.DataFrame) and "model_id" in results_frame.columns:
        model_ids = [str(value) for value in results_frame["model_id"].tolist()]

    reset_summary["model_trades"] = {model_id: [] for model_id in model_ids}
    reset_summary["trade_events_delta"] = {model_id: [] for model_id in model_ids}
    reset_summary["final_positions"] = {model_id: 0.0 for model_id in model_ids}
    reset_summary["final_open_slots"] = {model_id: 0 for model_id in model_ids}
    reset_summary["model_open_positions"] = {model_id: [] for model_id in model_ids}
    reset_summary["live_model_state"] = {model_id: {"entry_armed": True, "positions": []} for model_id in model_ids}

    return reset_summary


def hydrate_engine_from_summary(engine: Any, summary: dict[str, Any] | None, *, reset_counters: bool = False) -> None:
    if not isinstance(summary, dict):
        return

    target_week = 0 if reset_counters else max(0, _coerce_int(summary.get("week"), 0))
    target_generation = 1 if reset_counters else max(1, _coerce_int(summary.get("generation"), 1))

    if reset_counters:
        engine.week = target_week
        engine.generation = target_generation
    else:
        engine.week = max(getattr(engine, "week", 0), target_week)
        engine.generation = max(getattr(engine, "generation", 1), target_generation)
        target_generation = int(engine.generation)

    model_by_id = {str(getattr(model, "model_id", "")): model for model in getattr(engine, "models", [])}
    results = summary.get("results")
    if isinstance(results, pd.DataFrame) and {"model_id", "name"}.issubset(results.columns):
        for _, row in results.iterrows():
            model = model_by_id.get(str(row.get("model_id", "")))
            if model is None:
                continue
            model.name = str(row.get("name", model.name))
            model.generation = target_generation

    for model in getattr(engine, "models", []):
        model.generation = target_generation


def report_json_path(*, week: object, generation: object, segment: object, symbol: object, market_source: object, reports_dir: Path | str = "reports") -> Path:
    return Path(reports_dir) / (
        f"week_{_coerce_int(week, 0):03d}_gen_{_coerce_int(generation, 0):02d}_"
        f"{str(segment or 'unknown').lower()}_{str(symbol or 'BTCUSDT')}_{str(market_source or 'simulation')}.json"
    )


def _load_report_payload_from_run_row(row: pd.Series, reports_dir: Path | str = "reports") -> dict[str, Any]:
    json_path = report_json_path(
        week=row.get("week"),
        generation=row.get("generation"),
        segment=row.get("segment") or row.get("segment_inferred"),
        symbol=row.get("symbol"),
        market_source=row.get("market_source"),
        reports_dir=reports_dir,
    )
    if not json_path.exists():
        return {}

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _champion_model_id_from_run_row(
    row: pd.Series,
    report_payload: dict[str, Any] | None = None,
    reports_dir: Path | str = "reports",
) -> str | None:
    stored_id = str(row.get("champion_model_id", "") or "").strip()
    if stored_id:
        return stored_id

    payload = report_payload if isinstance(report_payload, dict) else _load_report_payload_from_run_row(row, reports_dir=reports_dir)

    champion = payload.get("champion", {}) if isinstance(payload, dict) else {}
    model_id = str(champion.get("model_id", "") or "").strip()
    return model_id or None


def _report_champion_payload(report_payload: dict[str, Any] | None) -> dict[str, Any]:
    champion = report_payload.get("champion", {}) if isinstance(report_payload, dict) else {}
    return champion if isinstance(champion, dict) else {}


def _is_generation_resolution_row(
    row: pd.Series,
    champion_payload: dict[str, Any],
    *,
    generation_horizon_weeks: int,
) -> tuple[bool, int]:
    stored_generation = max(0, _coerce_int(row.get("generation"), 0))
    previous_generation = max(0, _coerce_int(row.get("previous_generation"), 0))
    champion_generation = max(0, _coerce_int(champion_payload.get("generation"), 0))
    has_rollover = bool(champion_generation > 0 and stored_generation > 0 and champion_generation != stored_generation)

    if champion_generation > 0:
        if not has_rollover:
            return False, stored_generation
        return True, champion_generation

    if previous_generation > 0 and stored_generation > previous_generation:
        return True, previous_generation
    return False, stored_generation


def _resolve_history_metrics(
    row: pd.Series,
    champion_payload: dict[str, Any],
    positions: pd.DataFrame,
    *,
    champion_model_id: str | None,
    champion_generation: int,
    trade_size_czk: float,
) -> dict[str, int | float | None]:
    stored_model_id = str(row.get("champion_model_id") or "").strip()
    stored_metrics_available = bool(stored_model_id) and all(
        column in row.index and pd.notna(row.get(column))
        for column in (
            "champion_closed_trades",
            "champion_win_rate",
            "champion_profit_factor",
            "champion_pnl_czk",
        )
    )
    if stored_metrics_available:
        return {
            "closed_trades": _coerce_int(row.get("champion_closed_trades"), 0),
            "win_rate": _coerce_float(row.get("champion_win_rate"), 0.0),
            "profit_factor": _coerce_float(row.get("champion_profit_factor"), 0.0),
            "pnl_czk": _coerce_float(row.get("champion_pnl_czk"), 0.0),
        }

    metrics: dict[str, int | float | None] = {
        "closed_trades": _coerce_optional_int(champion_payload.get("closed_trades")),
        "win_rate": _coerce_optional_float(champion_payload.get("win_rate")),
        "profit_factor": _coerce_optional_float(champion_payload.get("profit_factor")),
        "pnl_czk": _coerce_optional_float(champion_payload.get("pnl_czk")),
    }

    if not champion_model_id or positions.empty or "model_id" not in positions.columns:
        return metrics

    champion_positions = positions[positions["model_id"] == str(champion_model_id)].copy()
    if champion_positions.empty:
        return metrics

    if champion_generation > 0 and "generation" in champion_positions.columns:
        generation_values = pd.to_numeric(champion_positions["generation"], errors="coerce")
        champion_positions = champion_positions[generation_values == float(champion_generation)].copy()
        if champion_positions.empty:
            return metrics

    created_at = row.get("created_at")
    if not pd.isna(created_at):
        champion_positions = champion_positions[
            champion_positions["closed_at"].notna() & (champion_positions["closed_at"] <= created_at)
        ].copy()
    if champion_positions.empty:
        return metrics

    derived_metrics = summarize_closed_positions_frame(champion_positions, trade_size_czk=trade_size_czk)
    for key in ("closed_trades", "win_rate", "profit_factor", "pnl_czk"):
        if metrics.get(key) is None:
            metrics[key] = derived_metrics.get(key)
    return metrics


def summarize_closed_positions_frame(frame: pd.DataFrame, trade_size_czk: float = PAPER_TRADE_SIZE_CZK) -> dict[str, Any]:
    if frame.empty:
        return {
            "closed_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "pnl_czk": 0.0,
        }

    pnl_pct = pd.to_numeric(frame.get("pnl_pct"), errors="coerce").dropna()
    if pnl_pct.empty:
        return {
            "closed_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "pnl_czk": 0.0,
        }

    aligned = frame.loc[pnl_pct.index].copy()
    slots = pd.to_numeric(aligned.get("quantity_slots"), errors="coerce").fillna(0.0).abs()
    pnl_czk = (slots * float(trade_size_czk) * (pnl_pct / 100.0)).astype(float)
    wins = int((pnl_pct > 0).sum())
    losses = int((pnl_pct < 0).sum())
    decided = max(1, wins + losses)
    gross_profit = float(pnl_czk[pnl_czk > 0].sum())
    gross_loss = float(abs(pnl_czk[pnl_czk < 0].sum()))
    return {
        "closed_trades": int(len(pnl_pct)),
        "win_rate": float((wins / decided) * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit > 0),
        "pnl_czk": float(pnl_czk.sum()),
    }


def build_segment_winner_history_table(
    weekly_runs: pd.DataFrame,
    closed_positions: pd.DataFrame,
    *,
    reports_dir: Path | str = "reports",
    trade_size_czk: float = PAPER_TRADE_SIZE_CZK,
    generation_horizon_weeks: int = DEFAULT_CONFIG.generation_horizon_weeks,
) -> pd.DataFrame:
    if weekly_runs.empty:
        return pd.DataFrame()

    runs = weekly_runs.copy()
    runs["created_at"] = pd.to_datetime(runs.get("created_at"), errors="coerce", utc=True)
    runs = runs.sort_values(by=["created_at", "id"], ascending=[True, True], na_position="last")
    runs["previous_generation"] = pd.to_numeric(runs.get("generation"), errors="coerce").shift(1)

    positions = closed_positions.copy()
    if not positions.empty:
        positions["closed_at"] = pd.to_datetime(positions.get("closed_at"), errors="coerce", utc=True)
        if "model_id" in positions.columns:
            positions["model_id"] = positions["model_id"].astype(str)

    overview_rows: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        report_payload = _load_report_payload_from_run_row(row, reports_dir=reports_dir)
        champion_payload = _report_champion_payload(report_payload)
        is_generation_resolution, champion_generation = _is_generation_resolution_row(
            row,
            champion_payload,
            generation_horizon_weeks=int(max(1, generation_horizon_weeks)),
        )
        if not is_generation_resolution:
            continue

        champion_model_id = _champion_model_id_from_run_row(row, report_payload=report_payload, reports_dir=reports_dir)
        metrics = _resolve_history_metrics(
            row,
            champion_payload,
            positions,
            champion_model_id=champion_model_id,
            champion_generation=champion_generation,
            trade_size_czk=trade_size_czk,
        )
        model_name = str(champion_payload.get("name") or row.get("champion_model") or "Neznámý model")

        overview_rows.append(
            {
                "Vítězství": row.get("created_at"),
                "Týden": _coerce_int(row.get("week"), 0),
                "Generace": champion_generation,
                "Model": model_name,
                "Model ID": champion_model_id or "",
                "Uzavřené obchody před výhrou": metrics.get("closed_trades"),
                "Profit factor": metrics.get("profit_factor"),
                "PnL CZK": metrics.get("pnl_czk"),
                "Win rate": metrics.get("win_rate"),
                "Zdroj dat": str(row.get("market_source") or "simulation"),
                "Symbol": str(row.get("symbol") or "BTCUSDT"),
            }
        )

    if not overview_rows:
        return pd.DataFrame()

    history_frame = pd.DataFrame(overview_rows)
    history_frame = history_frame.sort_values(by=["Vítězství", "Týden"], ascending=[False, False], na_position="last")
    return history_frame.reset_index(drop=True)