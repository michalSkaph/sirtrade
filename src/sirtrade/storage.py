from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

from .run_summary import champion_trade_metrics_from_summary


DEFAULT_DB_PATH = Path("data/sirtrade.db")
SQLITE_TIMEOUT_SECONDS = 30.0
SQLITE_BUSY_TIMEOUT_MS = 30_000

_init_db_lock = threading.Lock()
_initialized_dbs: set[str] = set()


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


def _infer_segment_from_model_id(model_id: object) -> str | None:
    normalized = str(model_id).strip().upper()
    if normalized.startswith("SC_"):
        return "Scalp"
    if normalized.startswith("ID_"):
        return "Intraday"
    if normalized.startswith("SW_"):
        return "Swing"
    return None


def _closed_trade_identity(
    *,
    opened_at: object,
    segment: object,
    model_id: object,
    symbol: object,
    side: object,
    entry_price: object,
    quantity_slots: object,
    market_source: object,
    week: object,
    generation: object,
) -> tuple[object, ...]:
    try:
        entry_price_value = round(float(entry_price), 12)
    except Exception:
        entry_price_value = 0.0
    try:
        quantity_value = round(float(quantity_slots), 8)
    except Exception:
        quantity_value = 0.0

    return (
        str(opened_at),
        str(segment or ""),
        str(model_id),
        str(symbol).upper(),
        _normalize_side(str(side)),
        entry_price_value,
        quantity_value,
        str(market_source),
        int(week or 0),
        int(generation or 0),
    )


def _closed_trade_identity_from_row(row: tuple) -> tuple[object, ...]:
    return _closed_trade_identity(
        opened_at=row[1],
        segment=row[13],
        model_id=row[2],
        symbol=row[4],
        side=row[5],
        entry_price=row[6],
        quantity_slots=row[8],
        market_source=row[12],
        week=row[14],
        generation=row[15],
    )


def _closed_trade_already_saved(conn: sqlite3.Connection, row: tuple) -> bool:
    identity = _closed_trade_identity_from_row(row)
    existing = conn.execute(
        """
        SELECT 1
        FROM closed_positions
        WHERE hidden = 0
          AND opened_at = ?
          AND COALESCE(segment, '') = ?
          AND model_id = ?
          AND symbol = ?
          AND side = ?
          AND ABS(entry_price - ?) <= 1e-12
          AND ABS(quantity_slots - ?) <= 1e-8
          AND market_source = ?
          AND week = ?
          AND generation = ?
        LIMIT 1
        """,
        identity,
    ).fetchone()
    return existing is not None


def _dedupe_closed_positions(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        opened_at,
                        COALESCE(segment, ''),
                        model_id,
                        symbol,
                        side,
                        ROUND(entry_price, 12),
                        ROUND(quantity_slots, 8),
                        market_source,
                        week,
                        generation
                    ORDER BY closed_at ASC, id ASC
                ) AS row_num
            FROM closed_positions
            WHERE hidden = 0
        )
        UPDATE closed_positions
        SET hidden = 1
        WHERE id IN (
            SELECT id
            FROM ranked
            WHERE row_num > 1
        )
        """
    )


def _build_closed_positions_rows(summary: dict) -> list[tuple]:
    model_trades = summary.get("trade_events_delta", summary.get("model_trades", {}))
    results_df = summary.get("results")
    default_symbol = str(summary.get("symbol", "BTCUSDT"))
    market_source = str(summary.get("market_source", "simulation"))
    segment = str(summary.get("segment", "")).strip() or None
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

        open_states: dict[tuple[str, str], dict[str, float | str | None]] = {}

        for _, event in events_df.iterrows():
            action = str(event.get("akce", ""))
            side = _normalize_side(str(event.get("strana", "")))
            try:
                price = float(event.get("cena", 0.0))
            except Exception:
                continue
            timestamp = str(event.get("executed_at", event.get("timestamp", "")))
            symbol = str(event.get("symbol", default_symbol)).upper()

            if "Vstup" in action:
                qty = _extract_slot_delta(action, entry=True)
                if qty <= 0:
                    qty = 1.0
                state_key = (symbol, side)
                state = open_states.get(
                    state_key,
                    {
                        "open_qty": 0.0,
                        "avg_entry_price": 0.0,
                        "opened_at": timestamp,
                    },
                )
                existing_qty = float(state.get("open_qty", 0.0) or 0.0)
                existing_avg = float(state.get("avg_entry_price", 0.0) or 0.0)
                total_cost = (existing_avg * existing_qty) + (price * qty)
                state["open_qty"] = existing_qty + qty
                state["avg_entry_price"] = total_cost / float(state["open_qty"]) if float(state["open_qty"]) > 0 else price
                state["opened_at"] = state.get("opened_at") or timestamp
                open_states[state_key] = state
                continue

            direct_opened_at = event.get("opened_at")
            direct_entry_price = event.get("entry_price")
            direct_qty = event.get("quantity_slots")
            if "Výstup" in action and direct_opened_at is not None and direct_entry_price is not None:
                try:
                    entry_price = float(direct_entry_price)
                except Exception:
                    entry_price = 0.0
                try:
                    qty = float(direct_qty if direct_qty is not None else _extract_slot_delta(action, entry=False))
                except Exception:
                    qty = 0.0
                if qty <= 0:
                    qty = 1.0

                if entry_price <= 0:
                    pnl_pct = 0.0
                elif side == "LONG":
                    pnl_pct = ((price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - price) / entry_price) * 100

                pnl_status = "ZISK" if pnl_pct > 0 else ("ZTRÁTA" if pnl_pct < 0 else "NULA")
                exit_reason = str(event.get("duvod_vystupu", "NEURČENO"))
                rows.append(
                    (
                        timestamp,
                        str(direct_opened_at),
                        str(model_id),
                        model_names.get(str(model_id), str(model_id)),
                        symbol,
                        side,
                        float(entry_price),
                        float(price),
                        float(qty),
                        float(pnl_pct),
                        pnl_status,
                        exit_reason,
                        market_source,
                        segment,
                        week,
                        generation,
                    )
                )
                state_key = (symbol, side)
                state = open_states.get(state_key)
                if state is not None:
                    existing_qty = float(state.get("open_qty", 0.0) or 0.0)
                    remaining_qty = max(0.0, existing_qty - qty)
                    if remaining_qty <= 1e-9:
                        open_states.pop(state_key, None)
                    else:
                        state["open_qty"] = remaining_qty
                        open_states[state_key] = state
                continue

            state_key = (symbol, side)
            state = open_states.get(state_key)
            open_qty = float(state.get("open_qty", 0.0) or 0.0) if state else 0.0
            avg_entry_price = float(state.get("avg_entry_price", 0.0) or 0.0) if state else 0.0
            opened_at = str(state.get("opened_at") or timestamp) if state else timestamp

            if "Výstup" in action and open_qty > 0:
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
                exit_reason = str(event.get("duvod_vystupu", "NEURČENO"))
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
                        exit_reason,
                        market_source,
                        segment,
                        week,
                        generation,
                    )
                )

                open_qty -= qty
                if open_qty <= 1e-9:
                    open_states.pop(state_key, None)
                else:
                    state["open_qty"] = open_qty
                    state["avg_entry_price"] = avg_entry_price
                    state["opened_at"] = opened_at
                    open_states[state_key] = state
                continue

            if "Výstup" in action and open_qty <= 0:
                continue

    return rows


def _db_key(db_path: Path) -> str:
    return str(Path(db_path).resolve())


def _connect_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _normalize_open_position_sides(conn: sqlite3.Connection) -> None:
    has_legacy_values = conn.execute(
        """
        SELECT 1
        FROM open_positions
        WHERE UPPER(TRIM(side)) IN ('BUY', 'SELL')
        LIMIT 1
        """
    ).fetchone()
    if not has_legacy_values:
        return

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


def _backfill_segments(conn: sqlite3.Connection, table_name: str) -> None:
    rows = conn.execute(
        f"SELECT id, model_id FROM {table_name} WHERE segment IS NULL OR TRIM(segment) = ''"
    ).fetchall()
    for row_id, model_id in rows:
        segment = _infer_segment_from_model_id(model_id)
        if segment is None:
            continue
        conn.execute(
            f"UPDATE {table_name} SET segment = ? WHERE id = ?",
            (segment, int(row_id)),
        )


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    key = _db_key(db_path)
    if key in _initialized_dbs:
        return

    with _init_db_lock:
        if key in _initialized_dbs:
            return

        last_error: sqlite3.OperationalError | None = None
        for attempt in range(3):
            conn = _connect_db(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS weekly_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        segment TEXT,
                        week INTEGER NOT NULL,
                        generation INTEGER NOT NULL,
                        market_source TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        interval TEXT,
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
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN segment TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN interval TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN champion_model_id TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN champion_closed_trades INTEGER NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN champion_win_rate REAL NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN champion_profit_factor REAL NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN champion_pnl_czk REAL NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS open_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        segment TEXT,
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
                        segment TEXT,
                        model_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        quantity_slots REAL NOT NULL,
                        pnl_pct REAL NOT NULL,
                        pnl_status TEXT NOT NULL,
                        exit_reason TEXT NOT NULL DEFAULT 'NEURČENO',
                        market_source TEXT NOT NULL,
                        week INTEGER NOT NULL,
                        generation INTEGER NOT NULL
                    )
                    """
                )
                try:
                    conn.execute("ALTER TABLE open_positions ADD COLUMN segment TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE closed_positions ADD COLUMN segment TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE closed_positions ADD COLUMN exit_reason TEXT NOT NULL DEFAULT 'NEURČENO'")
                except sqlite3.OperationalError:
                    pass

                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_open_positions_updated_model ON open_positions(updated_at DESC, model_id ASC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_open_positions_segment_updated ON open_positions(segment, updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_closed_positions_closed_id ON closed_positions(closed_at DESC, id DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_closed_positions_segment_closed ON closed_positions(segment, closed_at DESC, id DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_weekly_runs_created_id ON weekly_runs(created_at DESC, id DESC)"
                )
                try:
                    conn.execute("ALTER TABLE closed_positions ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE weekly_runs ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
                _normalize_open_position_sides(conn)
                _backfill_segments(conn, "open_positions")
                _backfill_segments(conn, "closed_positions")
                _dedupe_closed_positions(conn)
                conn.commit()
                _initialized_dbs.add(key)
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                conn.rollback()
                if "locked" not in str(exc).lower() or attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))
            finally:
                conn.close()

        if last_error is not None:
            raise last_error


def save_week_result(summary: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    champion = summary["champion"]
    champion_metrics = champion_trade_metrics_from_summary(summary)
    conn = _connect_db(db_path)
    try:
        conn.execute(
            """
            INSERT INTO weekly_runs (
                segment, week, generation, market_source, symbol, interval,
                champion_model_id, champion_model, champion_score, champion_sortino,
                champion_calmar, champion_max_dd, champion_cvar95, reward_usd
                , champion_closed_trades, champion_win_rate, champion_profit_factor, champion_pnl_czk
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(summary.get("segment", "")) or None,
                int(summary["week"]),
                int(summary["generation"]),
                str(summary.get("market_source", "simulation")),
                str(summary.get("symbol", "BTCUSDT")),
                str(summary.get("interval", "")) or None,
                str(champion_metrics.get("champion_model_id") or "") or None,
                str(champion.get("name", "unknown")),
                float(champion.get("score", 0.0)),
                float(champion.get("sortino", 0.0)),
                float(champion.get("calmar", 0.0)),
                float(champion.get("max_dd", 0.0)),
                float(champion.get("cvar95", 0.0)),
                float(champion.get("reward_usd", 0.0)),
                int(champion_metrics.get("champion_closed_trades", 0) or 0),
                float(champion_metrics.get("champion_win_rate", 0.0) or 0.0),
                float(champion_metrics.get("champion_profit_factor", 0.0) or 0.0),
                float(champion_metrics.get("champion_pnl_czk", 0.0) or 0.0),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_recent_runs(
    limit: int | None = 50,
    segment: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    conn = _connect_db(db_path)
    try:
        query = "SELECT * FROM weekly_runs WHERE hidden = 0"
        params: list[object] = []
        if segment is not None:
            query += " AND segment = ?"
            params.append(str(segment))
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        frame = pd.read_sql_query(query, conn, params=tuple(params))
        return frame
    finally:
        conn.close()


def save_open_positions(summary: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    final_positions = summary.get("final_positions", {})
    final_open_slots = summary.get("final_open_slots", {})
    model_open_positions = summary.get("model_open_positions", {})
    results_df = summary.get("results")
    segment = str(summary.get("segment", "")).strip() or None
    symbol = str(summary.get("symbol", "BTCUSDT"))
    market_source = str(summary.get("market_source", "simulation"))

    if results_df is None or not isinstance(results_df, pd.DataFrame):
        return

    summary_model_ids: set[str] = set()
    if "model_id" in results_df.columns:
        summary_model_ids.update(str(value) for value in results_df["model_id"].dropna().tolist())
    if isinstance(model_open_positions, dict):
        summary_model_ids.update(str(model_id) for model_id in model_open_positions.keys())
    if isinstance(final_positions, dict):
        summary_model_ids.update(str(model_id) for model_id in final_positions.keys())

    model_names = {
        str(row["model_id"]): str(row["name"])
        for _, row in results_df[["model_id", "name"]].iterrows()
    }

    rows_to_insert = []
    if isinstance(model_open_positions, dict) and model_open_positions:
        aggregated_positions: dict[tuple[str, str, str], tuple[str, float]] = {}
        for model_id, positions in model_open_positions.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                side = _normalize_side(position.get("side", ""))
                if side not in {"LONG", "SHORT"}:
                    continue
                symbol_value = str(position.get("symbol", symbol)).upper()
                slot_count = abs(float(position.get("slots", position.get("open_slots", 1.0)) or 0.0))
                if slot_count <= 0:
                    continue
                key = (str(model_id), symbol_value, side)
                model_name = model_names.get(str(model_id), str(position.get("model_name", model_id)))
                previous = aggregated_positions.get(key)
                aggregated_positions[key] = (
                    model_name,
                    slot_count if previous is None else float(previous[1]) + slot_count,
                )
        rows_to_insert.extend(
            (
                segment,
                model_id,
                model_name,
                symbol_value,
                side,
                slot_count,
                market_source,
            )
            for (model_id, symbol_value, side), (model_name, slot_count) in aggregated_positions.items()
        )
    else:
        for model_id, size in final_positions.items():
            size_val = float(size)
            if abs(size_val) < 1e-9:
                continue
            side = _normalize_side("LONG" if size_val > 0 else "SHORT")
            slot_count = abs(float(final_open_slots.get(model_id, 1.0) or 0.0))
            if slot_count <= 0:
                slot_count = 1.0
            rows_to_insert.append(
                (
                    segment,
                    str(model_id),
                    model_names.get(str(model_id), str(model_id)),
                    symbol,
                    side,
                    slot_count,
                    market_source,
                )
            )

    conn = _connect_db(db_path)
    try:
        namespace_prefixes = {
            model_id.split("_", 1)[0]
            for model_id in summary_model_ids
            if "_" in model_id and model_id.split("_", 1)[0]
        }
        namespace_to_delete = None
        if len(namespace_prefixes) == 1 and summary_model_ids and all(
            "_" in model_id and model_id.startswith(f"{next(iter(namespace_prefixes))}_")
            for model_id in summary_model_ids
        ):
            namespace_to_delete = next(iter(namespace_prefixes))

        if namespace_to_delete is not None:
            if segment is not None:
                conn.execute(
                    "DELETE FROM open_positions WHERE segment = ? OR (segment IS NULL AND model_id LIKE ?)",
                    (segment, f"{namespace_to_delete}_%"),
                )
            else:
                conn.execute(
                    "DELETE FROM open_positions WHERE model_id LIKE ?",
                    (f"{namespace_to_delete}_%",),
                )
        elif summary_model_ids:
            placeholders = ", ".join("?" for _ in summary_model_ids)
            delete_sql = f"DELETE FROM open_positions WHERE model_id IN ({placeholders})"
            delete_params: tuple[object, ...] = tuple(sorted(summary_model_ids))
            if segment is not None:
                delete_sql = f"DELETE FROM open_positions WHERE segment = ? OR model_id IN ({placeholders})"
                delete_params = (segment, *delete_params)
            conn.execute(delete_sql, delete_params)
        else:
            if segment is not None:
                conn.execute("DELETE FROM open_positions WHERE segment = ?", (segment,))
            else:
                conn.execute("DELETE FROM open_positions")
        if rows_to_insert:
            conn.executemany(
                """
                INSERT INTO open_positions (
                    segment, model_id, model_name, symbol, side, position_size, market_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
        conn.commit()
    finally:
        conn.close()


def load_open_positions(segment: str | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = _connect_db(db_path)
    try:
        if segment is None:
            frame = pd.read_sql_query(
                "SELECT * FROM open_positions ORDER BY updated_at DESC, model_id ASC",
                conn,
            )
        else:
            frame = pd.read_sql_query(
                "SELECT * FROM open_positions WHERE segment = ? ORDER BY updated_at DESC, model_id ASC",
                conn,
                params=(str(segment),),
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

    conn = _connect_db(db_path)
    try:
        unique_rows: list[tuple] = []
        seen_identities: set[tuple[object, ...]] = set()
        for row in rows_to_insert:
            identity = _closed_trade_identity_from_row(row)
            if identity in seen_identities:
                continue
            if _closed_trade_already_saved(conn, row):
                continue
            seen_identities.add(identity)
            unique_rows.append(row)

        if not unique_rows:
            return

        conn.executemany(
            """
            INSERT INTO closed_positions (
                closed_at, opened_at, model_id, model_name, symbol, side,
                entry_price, exit_price, quantity_slots, pnl_pct, pnl_status, exit_reason,
                market_source, segment, week, generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            unique_rows,
        )
        _dedupe_closed_positions(conn)
        conn.commit()
    finally:
        conn.close()


def load_closed_positions(limit: int | None = 2000, segment: str | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = _connect_db(db_path)
    try:
        if segment is None:
            query = "SELECT * FROM closed_positions WHERE hidden = 0 ORDER BY closed_at DESC, id DESC"
            params: tuple[object, ...] = ()
            if limit is not None:
                query += " LIMIT ?"
                params = (int(limit),)
            return pd.read_sql_query(query, conn, params=params)

        query = "SELECT * FROM closed_positions WHERE hidden = 0 AND segment = ? ORDER BY closed_at DESC, id DESC"
        params = [str(segment)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        return pd.read_sql_query(query, conn, params=tuple(params))
    finally:
        conn.close()


def clear_trade_history(db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = _connect_db(db_path)
    try:
        conn.execute("DELETE FROM weekly_runs")
        conn.execute("DELETE FROM open_positions")
        conn.execute("DELETE FROM closed_positions")
        conn.commit()
    finally:
        conn.close()
