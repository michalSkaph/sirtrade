from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_DB_PATH = Path("data/sirtrade.db")


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
    for model_id, size in final_positions.items():
        size_val = float(size)
        if abs(size_val) < 1e-9:
            continue
        side = "LONG" if size_val > 0 else "SHORT"
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
        return pd.read_sql_query(
            "SELECT * FROM open_positions ORDER BY updated_at DESC, model_id ASC",
            conn,
        )
    finally:
        conn.close()
