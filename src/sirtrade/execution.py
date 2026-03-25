from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ProposedOrder:
    model_id: str
    symbol: str
    side: str
    instrument: str
    quantity_usd: float
    confidence: float


def build_dry_run_orders(leaderboard: pd.DataFrame, symbol: str, nav_usd: float = 1000.0) -> list[ProposedOrder]:
    orders: list[ProposedOrder] = []
    if leaderboard.empty or "model_open_positions" not in leaderboard.columns:
        return orders

    for _, row in leaderboard.iterrows():
        positions = row.get("model_open_positions", [])
        if not isinstance(positions, list) or not positions:
            continue

        grouped_positions: dict[tuple[str, str], int] = {}
        for position in positions:
            if not isinstance(position, dict):
                continue
            side = str(position.get("side", "")).upper()
            symbol_value = str(position.get("symbol", symbol)).upper()
            if side not in {"LONG", "SHORT"} or not symbol_value:
                continue
            group_key = (symbol_value, side)
            grouped_positions[group_key] = grouped_positions.get(group_key, 0) + 1

        confidence = float(max(0.0, min(1.0, 0.5 + float(row["score"]) / 5)))
        for (symbol_value, side), slot_count in grouped_positions.items():
            direction = "BUY" if side == "LONG" else "SELL"
            instrument = "spot" if direction == "BUY" else "perpetual"
            qty = max(10.0, nav_usd * 0.02 * slot_count)
            orders.append(
                ProposedOrder(
                    model_id=str(row["model_id"]),
                    symbol=symbol_value,
                    side=direction,
                    instrument=instrument,
                    quantity_usd=qty,
                    confidence=confidence,
                )
            )
    return orders
