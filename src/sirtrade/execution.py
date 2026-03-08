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
    top = leaderboard.head(3)
    orders: list[ProposedOrder] = []
    for _, row in top.iterrows():
        direction = "BUY" if row["sortino"] >= 0 else "SELL"
        instrument = "perpetual" if direction == "SELL" else "spot"
        qty = max(10.0, nav_usd * 0.02)
        confidence = float(max(0.0, min(1.0, 0.5 + row["score"] / 5)))
        orders.append(
            ProposedOrder(
                model_id=str(row["model_id"]),
                symbol=symbol,
                side=direction,
                instrument=instrument,
                quantity_usd=qty,
                confidence=confidence,
            )
        )
    return orders
