from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DecisionThresholds, DecisionWeights


def sortino_ratio(returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    downside = excess[excess < 0]
    dd = downside.std()
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float((excess.mean() / dd) * np.sqrt(365))


def calmar_ratio(returns: pd.Series, max_dd: float) -> float:
    ann_ret = (1 + returns.mean()) ** 365 - 1
    if max_dd <= 1e-9:
        return 0.0
    return float(ann_ret / max_dd)


def decision_score(metrics: dict, weights: DecisionWeights) -> float:
    return (
        weights.sortino * metrics["sortino"]
        + weights.calmar * metrics["calmar"]
        + weights.cvar95 * metrics["cvar95"]
        + weights.max_dd * metrics["max_dd"]
        + weights.cost * metrics["cost"]
        + weights.turnover * metrics["turnover"]
    )


def pass_thresholds(metrics: dict, thresholds: DecisionThresholds) -> bool:
    return all(
        [
            metrics["sortino"] >= thresholds.min_sortino,
            metrics["calmar"] >= thresholds.min_calmar,
            metrics["max_dd"] <= thresholds.max_dd,
            metrics["cvar95"] <= thresholds.max_cvar95,
        ]
    )
