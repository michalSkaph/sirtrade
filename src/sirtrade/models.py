from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


ModelKind = Literal[
    "trend_vol",
    "xsec_momentum",
    "mean_reversion",
    "onchain_sentiment_overlay",
    "meta_ensemble",
    "copy_trader",
]


@dataclass
class ModelSpec:
    model_id: str
    name: str
    kind: ModelKind
    generation: int


def default_model_specs(namespace: str = "", label_prefix: str = "") -> list[ModelSpec]:
    ns = str(namespace).strip().upper()
    prefix = f"{label_prefix.strip()} | " if str(label_prefix).strip() else ""

    def _model_id(base_id: str) -> str:
        return f"{ns}_{base_id}" if ns else base_id

    return [
        ModelSpec(_model_id("M1"), f"{prefix}Trend + cílení volatility", "trend_vol", 1),
        ModelSpec(_model_id("M2"), f"{prefix}Průřezové momentum + carry", "xsec_momentum", 1),
        ModelSpec(_model_id("M3"), f"{prefix}Swing návrat k průměru", "mean_reversion", 1),
        ModelSpec(_model_id("M4"), f"{prefix}On-chain + sentimentní vrstva", "onchain_sentiment_overlay", 1),
        ModelSpec(_model_id("M5"), f"{prefix}Meta ansámbl", "meta_ensemble", 1),
        ModelSpec(_model_id("MC"), f"{prefix}Kopie lead tradera Binance", "copy_trader", 1),
    ]


def generate_signals(model: ModelSpec, market: pd.DataFrame, seed: int = 0) -> pd.Series:
    close = market["close"].astype(float)
    returns = market["ret"].fillna(0.0)

    # Scalp models (SC_ prefix) use faster indicator periods
    is_scalp = model.model_id.startswith("SC_")
    ema_fast_span = 3 if is_scalp else 8
    ema_slow_span = 8 if is_scalp else 21
    roc_fast_period = 3 if is_scalp else 6
    roc_slow_period = 8 if is_scalp else 24
    bb_period = 10 if is_scalp else 20
    rsi_alpha = 1 / 7 if is_scalp else 1 / 14
    signal_smooth_span = 2 if is_scalp else 3

    vol = returns.rolling(bb_period).std().fillna(returns.std() if returns.std() > 0 else 0.01)
    ema_fast = close.ewm(span=ema_fast_span, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_span, adjust=False).mean()
    trend_gap = ((ema_fast - ema_slow) / (close + 1e-6)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    roc_fast = close.pct_change(roc_fast_period).fillna(0.0)
    roc_slow = close.pct_change(roc_slow_period).fillna(0.0)
    rolling_mean = close.rolling(bb_period).mean().bfill()
    rolling_std = close.rolling(bb_period).std(ddof=0).replace(0.0, np.nan)
    price_z = ((close - rolling_mean) / (rolling_std + 1e-6)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    delta = close.diff().fillna(0.0)
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.ewm(alpha=rsi_alpha, adjust=False).mean()
    avg_loss = losses.ewm(alpha=rsi_alpha, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))

    if model.kind == "trend_vol":
        momentum_stack = 0.65 * roc_fast + 0.35 * roc_slow
        signal = np.tanh((momentum_stack / (vol + 1e-6)) + (3.5 * trend_gap))
    elif model.kind == "xsec_momentum":
        breakout_bias = (close / (close.rolling(bb_period).max().shift(1) + 1e-6) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        signal = np.tanh((1.1 * roc_fast / (vol + 1e-6)) + (0.9 * breakout_bias * 10.0) + (2.5 * trend_gap))
    elif model.kind == "mean_reversion":
        rsi_centered = (rsi - 50.0) / 12.5
        signal = -np.tanh(0.9 * price_z + 0.6 * rsi_centered)
    elif model.kind == "onchain_sentiment_overlay":
        overlay = market["sentiment"].fillna(0.0) * 0.4 + market["onchain"].fillna(0.0) * 0.6
        signal = np.tanh(overlay + (2.0 * trend_gap) + (0.25 * np.sign(roc_fast)))
    elif model.kind == "copy_trader":
        signal = pd.Series(0.0, index=market.index, dtype=float)
    else:
        trend_component = np.tanh((roc_slow / (vol + 1e-6)) + (3.0 * trend_gap))
        reversion_component = -np.tanh(price_z)
        signal = (0.55 * trend_component) + (0.25 * reversion_component) + (0.20 * np.tanh((rsi - 50.0) / 10.0))

    smoothed = pd.Series(signal, index=market.index, dtype=float).ewm(span=signal_smooth_span, adjust=False).mean()
    return smoothed.clip(-1.0, 1.0)
