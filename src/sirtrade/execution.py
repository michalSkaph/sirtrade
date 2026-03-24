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

        side = str(positions[0].get("side", "")).upper()
        if side not in {"LONG", "SHORT"}:
            continue

        direction = "BUY" if side == "LONG" else "SELL"
        instrument = "spot" if direction == "BUY" else "perpetual"
        qty = max(10.0, nav_usd * 0.02 * len(positions))
        confidence = float(max(0.0, min(1.0, 0.5 + float(row["score"]) / 5)))
        orders.append(
            ProposedOrder(
                model_id=str(row["model_id"]),
                symbol=str(positions[0].get("symbol", symbol)).upper(),
                side=direction,
                instrument=instrument,
                quantity_usd=qty,
                confidence=confidence,
            )
        )
    return orders
