from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DecisionThresholds, DecisionWeights


SORTINO_CAP = 6.0
CALMAR_CAP = 8.0
CVAR95_CAP = 0.08
MAX_DD_CAP = 0.60
COST_CAP = 0.05
TURNOVER_CAP = 2.50


def _clip(value: float, lower: float, upper: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(np.clip(value, lower, upper))


def bound_metrics(metrics: dict) -> dict[str, float]:
    return {
        "sortino": _clip(float(metrics.get("sortino", 0.0)), -SORTINO_CAP, SORTINO_CAP),
        "calmar": _clip(float(metrics.get("calmar", 0.0)), -CALMAR_CAP, CALMAR_CAP),
        "win_rate": _clip((float(metrics.get("win_rate", 50.0)) - 50.0) / 50.0, -1.0, 1.0),
        "cvar95": _clip(float(metrics.get("cvar95", 0.0)), 0.0, CVAR95_CAP),
        "max_dd": _clip(float(metrics.get("max_dd", 0.0)), 0.0, MAX_DD_CAP),
        "cost": _clip(float(metrics.get("cost", 0.0)), 0.0, COST_CAP),
        "turnover": _clip(float(metrics.get("turnover", 0.0)), 0.0, TURNOVER_CAP),
    }


def sortino_ratio(returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_rms = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    dispersion_floor = max(float(excess.std(ddof=0)) * 0.15, 1e-4)
    effective_downside = max(downside_rms, dispersion_floor)
    if effective_downside <= 0 or np.isnan(effective_downside):
        return 0.0
    raw_ratio = float((excess.mean() / effective_downside) * np.sqrt(365))
    return _clip(raw_ratio, -SORTINO_CAP, SORTINO_CAP)


def calmar_ratio(returns: pd.Series, max_dd: float) -> float:
    ann_ret = float((1 + returns.mean()) ** 365 - 1)
    ann_ret = _clip(ann_ret, -0.95, 3.0)
    effective_drawdown = max(float(max_dd), 0.01)
    if effective_drawdown <= 0:
        return 0.0
    raw_ratio = float(ann_ret / effective_drawdown)
    return _clip(raw_ratio, -CALMAR_CAP, CALMAR_CAP)


def decision_score(metrics: dict, weights: DecisionWeights) -> float:
    bounded = bound_metrics(metrics)
    return (
        weights.sortino * bounded["sortino"]
        + weights.calmar * bounded["calmar"]
        + weights.win_rate * bounded["win_rate"]
        + weights.cvar95 * bounded["cvar95"]
        + weights.max_dd * bounded["max_dd"]
        + weights.cost * bounded["cost"]
        + weights.turnover * bounded["turnover"]
    )


def pass_thresholds(metrics: dict, thresholds: DecisionThresholds) -> bool:
    bounded = bound_metrics(metrics)
    return all(
        [
            bounded["sortino"] >= thresholds.min_sortino,
            bounded["calmar"] >= thresholds.min_calmar,
            bounded["max_dd"] <= thresholds.max_dd,
            bounded["cvar95"] <= thresholds.max_cvar95,
        ]
    )
