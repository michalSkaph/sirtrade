from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PAPER_TRADE_SIZE_CZK


@dataclass
class ProposedOrder:
    model_id: str
    symbol: str
    side: str
    instrument: str
    quantity_czk: float
    confidence: float


def build_dry_run_orders(
    leaderboard: pd.DataFrame,
    symbol: str,
    trade_size_czk: float = PAPER_TRADE_SIZE_CZK,
) -> list[ProposedOrder]:
    orders: list[ProposedOrder] = []
    if leaderboard.empty or "model_open_positions" not in leaderboard.columns:
        return orders

    for _, row in leaderboard.iterrows():
        positions = row.get("model_open_positions", [])
        if not isinstance(positions, list) or not positions:
            continue

        grouped_positions: set[tuple[str, str]] = set()
        for position in positions:
            if not isinstance(position, dict):
                continue
            side = str(position.get("side", "")).upper()
            symbol_value = str(position.get("symbol", symbol)).upper()
            if side not in {"LONG", "SHORT"} or not symbol_value:
                continue
            grouped_positions.add((symbol_value, side))

        confidence = float(max(0.0, min(1.0, 0.5 + float(row["score"]) / 5)))
        for symbol_value, side in sorted(grouped_positions):
            direction = "BUY" if side == "LONG" else "SELL"
            instrument = "spot" if direction == "BUY" else "perpetual"
            qty = float(trade_size_czk)
            orders.append(
                ProposedOrder(
                    model_id=str(row["model_id"]),
                    symbol=symbol_value,
                    side=direction,
                    instrument=instrument,
                    quantity_czk=qty,
                    confidence=confidence,
                )
            )
    return orders
