from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RiskPolicy


def annualized_vol(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(365))


def drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return (equity_curve / running_max) - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    dd = drawdown(equity_curve)
    return float(abs(dd.min())) if not dd.empty else 0.0


def cvar95(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    var = returns.quantile(0.05)
    tail = returns[returns <= var]
    return float(abs(tail.mean())) if not tail.empty else float(abs(var))


def apply_risk_controls(raw_position: pd.Series, returns: pd.Series, policy: RiskPolicy) -> pd.Series:
    vol = returns.rolling(20).std().fillna(returns.std() if returns.std() > 0 else 0.01)
    target_daily_vol = policy.target_vol_annual / np.sqrt(365)
    scaler = np.clip(target_daily_vol / (vol + 1e-6), 0.1, 1.5)

    controlled = raw_position * scaler
    controlled = controlled.clip(-policy.max_asset_exposure, policy.max_asset_exposure)
    return controlled
