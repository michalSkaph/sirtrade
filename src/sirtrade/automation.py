from __future__ import annotations

from dataclasses import asdict

from .config import DEFAULT_CONFIG
from .engine import TradingEngine
from .reporting import export_weekly_report
from .storage import init_db, save_open_positions, save_week_result


def run_automation_cycle(
    market_source: str = "binance",
    symbol: str = "BTCUSDT",
    days: int = 365,
) -> dict:
    init_db()
    engine = TradingEngine(DEFAULT_CONFIG)
    summary = engine.run_week(days=days, market_source=market_source, symbol=symbol)
    save_week_result(summary)
    save_open_positions(summary)
    exports = export_weekly_report(summary, DEFAULT_CONFIG)

    champion = summary["champion"]
    return {
        "week": summary["week"],
        "generation": summary["generation"],
        "market_source": summary["market_source"],
        "symbol": summary["symbol"],
        "champion": {
            "model_id": champion.get("model_id"),
            "name": champion.get("name"),
            "score": champion.get("score"),
        },
        "exports": exports,
    }
