from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_DB_PATH = Path("data/sirtrade.db")


def _normalize_side(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized in {"BUY", "LONG", "1", "+1"}:
        return "LONG"
    if normalized in {"SELL", "SHORT", "-1"}:
        return "SHORT"
    return normalized


def _extract_slot_delta(action: str, entry: bool) -> float:
    pattern = r"\(\+(\d+)\)" if entry else r"\(-?(\d+)\)"
    match = re.search(pattern, str(action))
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except Exception:
        return 0.0


def _build_closed_positions_rows(summary: dict) -> list[tuple]:
    model_trades = summary.get("model_trades", {})
    results_df = summary.get("results")
    symbol = str(summary.get("symbol", "BTCUSDT"))
    market_source = str(summary.get("market_source", "simulation"))
    week = int(summary.get("week", 0))
    generation = int(summary.get("generation", 0))

    if not isinstance(model_trades, dict):
        return []

    model_names: dict[str, str] = {}
    if isinstance(results_df, pd.DataFrame) and {"model_id", "name"}.issubset(results_df.columns):
        model_names = {
            str(row["model_id"]): str(row["name"])
            for _, row in results_df[["model_id", "name"]].iterrows()
        }

    rows: list[tuple] = []
    for model_id, events in model_trades.items():
        events_df = pd.DataFrame(events)
        if events_df.empty or "akce" not in events_df.columns:
            continue

        if "timestamp" in events_df.columns:
            events_df = events_df.sort_values("timestamp")

        open_qty = 0.0
        avg_entry_price = 0.0
        opened_at: str | None = None
        current_side: str | None = None

        for _, event in events_df.iterrows():
            action = str(event.get("akce", ""))
            side = _normalize_side(str(event.get("strana", "")))
            try:
                price = float(event.get("cena", 0.0))
            except Exception:
                continue
            timestamp = str(event.get("timestamp", ""))

            if "Vstup" in action:
                qty = _extract_slot_delta(action, entry=True)
                if qty <= 0:
                    qty = 1.0
                if open_qty <= 0 or current_side != side:
                    open_qty = 0.0
                    avg_entry_price = 0.0
                    opened_at = timestamp
                    current_side = side
                total_cost = (avg_entry_price * open_qty) + (price * qty)
                open_qty += qty
                avg_entry_price = total_cost / open_qty if open_qty > 0 else price
                if not opened_at:
                    opened_at = timestamp
                continue

            if "Výstup" in action and open_qty > 0 and current_side == side:
                qty = _extract_slot_delta(action, entry=False)
                if qty <= 0:
                    qty = open_qty
                qty = min(qty, open_qty)

                if avg_entry_price <= 0:
                    pnl_pct = 0.0
                elif side == "LONG":
                    pnl_pct = ((price - avg_entry_price) / avg_entry_price) * 100
                else:
                    pnl_pct = ((avg_entry_price - price) / avg_entry_price) * 100

                pnl_status = "ZISK" if pnl_pct > 0 else ("ZTRÁTA" if pnl_pct < 0 else "NULA")
                rows.append(
                    (
                        timestamp,
                        opened_at or timestamp,
                        str(model_id),
                        model_names.get(str(model_id), str(model_id)),
                        symbol,
                        side,
                        float(avg_entry_price),
                        float(price),
                        float(qty),
                        float(pnl_pct),
                        pnl_status,
                        market_source,
                        week,
                        generation,
                    )
                )

                open_qty -= qty
                if open_qty <= 1e-9:
                    open_qty = 0.0
                    avg_entry_price = 0.0
                    opened_at = None
                    current_side = None

    return rows


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                week INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                market_source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                champion_model TEXT NOT NULL,
                champion_score REAL NOT NULL,
                champion_sortino REAL NOT NULL,
                champion_calmar REAL NOT NULL,
                champion_max_dd REAL NOT NULL,
                champion_cvar95 REAL NOT NULL,
                reward_usd REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS open_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                position_size REAL NOT NULL,
                market_source TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS closed_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                closed_at TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity_slots REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                pnl_status TEXT NOT NULL,
                market_source TEXT NOT NULL,
                week INTEGER NOT NULL,
                generation INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            UPDATE open_positions
            SET side = CASE UPPER(TRIM(side))
                WHEN 'BUY' THEN 'LONG'
                WHEN 'SELL' THEN 'SHORT'
                ELSE UPPER(TRIM(side))
            END
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_week_result(summary: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    champion = summary["champion"]
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO weekly_runs (
                week, generation, market_source, symbol,
                champion_model, champion_score, champion_sortino,
                champion_calmar, champion_max_dd, champion_cvar95, reward_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(summary["week"]),
                int(summary["generation"]),
                str(summary.get("market_source", "simulation")),
                str(summary.get("symbol", "BTCUSDT")),
                str(champion.get("name", "unknown")),
                float(champion.get("score", 0.0)),
                float(champion.get("sortino", 0.0)),
                float(champion.get("calmar", 0.0)),
                float(champion.get("max_dd", 0.0)),
                float(champion.get("cvar95", 0.0)),
                float(champion.get("reward_usd", 0.0)),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_recent_runs(limit: int = 50, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT * FROM weekly_runs ORDER BY id DESC LIMIT ?"
        frame = pd.read_sql_query(query, conn, params=(limit,))
        return frame
    finally:
        conn.close()


def save_open_positions(summary: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    final_positions = summary.get("final_positions", {})
    model_open_positions = summary.get("model_open_positions", {})
    results_df = summary.get("results")
    symbol = str(summary.get("symbol", "BTCUSDT"))
    market_source = str(summary.get("market_source", "simulation"))

    if results_df is None or not isinstance(results_df, pd.DataFrame):
        return

    model_names = {
        str(row["model_id"]): str(row["name"])
        for _, row in results_df[["model_id", "name"]].iterrows()
    }

    rows_to_insert = []
    if isinstance(model_open_positions, dict) and model_open_positions:
        for model_id, positions in model_open_positions.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                side = _normalize_side(position.get("side", ""))
                if side not in {"LONG", "SHORT"}:
                    continue
                symbol_value = str(position.get("symbol", symbol)).upper()
                rows_to_insert.append(
                    (
                        str(model_id),
                        model_names.get(str(model_id), str(position.get("model_name", model_id))),
                        symbol_value,
                        side,
                        1.0,
                        market_source,
                    )
                )
    else:
        for model_id, size in final_positions.items():
            size_val = float(size)
            if abs(size_val) < 1e-9:
                continue
            side = _normalize_side("LONG" if size_val > 0 else "SHORT")
            rows_to_insert.append(
                (
                    str(model_id),
                    model_names.get(str(model_id), str(model_id)),
                    symbol,
                    side,
                    abs(size_val),
                    market_source,
                )
            )

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM open_positions")
        if rows_to_insert:
            conn.executemany(
                """
                INSERT INTO open_positions (
                    model_id, model_name, symbol, side, position_size, market_source
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
        conn.commit()
    finally:
        conn.close()


def load_open_positions(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        frame = pd.read_sql_query(
            "SELECT * FROM open_positions ORDER BY updated_at DESC, model_id ASC",
            conn,
        )
        if "side" in frame.columns:
            frame["side"] = frame["side"].astype(str).map(_normalize_side)
        return frame
    finally:
        conn.close()


def save_closed_positions(summary: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    rows_to_insert = _build_closed_positions_rows(summary)
    if not rows_to_insert:
        return

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            """
            INSERT INTO closed_positions (
                closed_at, opened_at, model_id, model_name, symbol, side,
                entry_price, exit_price, quantity_slots, pnl_pct, pnl_status,
                market_source, week, generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()
    finally:
        conn.close()


def load_closed_positions(limit: int = 2000, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            "SELECT * FROM closed_positions ORDER BY closed_at DESC, id DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )
    finally:
        conn.close()
