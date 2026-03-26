from __future__ import annotations

from .engine import TradingEngine


def run_segment_cycle(
    engine: TradingEngine,
    segment: str,
    market_source: str,
    symbol: str,
    days: int,
    interval: str,
    previous_summary: dict | None = None,
) -> dict:
    summary = engine.run_week(
        days=days,
        market_source=market_source,
        symbol=symbol,
        interval=interval,
        previous_summary=previous_summary,
    )
    summary["segment"] = segment
    return summary



