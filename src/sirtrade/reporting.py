from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import AppConfig


def export_weekly_report(summary: dict, config: AppConfig, out_dir: Path | str = "reports") -> dict:
    report_dir = Path(out_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    week = int(summary["week"])
    generation = int(summary["generation"])
    segment = str(summary.get("segment", "unknown")).lower()
    symbol = str(summary.get("symbol", "BTCUSDT"))
    source = str(summary.get("market_source", "simulation"))

    stem = f"week_{week:03d}_gen_{generation:02d}_{segment}_{symbol}_{source}"
    csv_path = report_dir / f"{stem}.csv"
    json_path = report_dir / f"{stem}.json"

    results_df = summary["results"].copy()
    if isinstance(results_df, pd.DataFrame):
        results_df.to_csv(csv_path, index=False)

    payload = {
        "segment": summary.get("segment"),
        "week": week,
        "generation": generation,
        "symbol": symbol,
        "market_source": source,
        "interval": summary.get("interval"),
        "champion": summary.get("champion", {}),
        "decision_matrix": {
            "formula": "S = 0.28*Sortino + 0.22*Calmar - 0.18*CVaR95 - 0.14*MaxDD - 0.10*Cost - 0.08*Turnover",
            "weights": {
                "sortino": config.weights.sortino,
                "calmar": config.weights.calmar,
                "cvar95": config.weights.cvar95,
                "max_dd": config.weights.max_dd,
                "cost": config.weights.cost,
                "turnover": config.weights.turnover,
            },
            "thresholds": {
                "min_sortino": config.thresholds.min_sortino,
                "min_calmar": config.thresholds.min_calmar,
                "max_dd": config.thresholds.max_dd,
                "max_cvar95": config.thresholds.max_cvar95,
            },
        },
        "proposed_orders": summary.get("proposed_orders", []),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"csv": str(csv_path), "json": str(json_path)}
