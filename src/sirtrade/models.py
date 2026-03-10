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
    ]


def generate_signals(model: ModelSpec, market: pd.DataFrame, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed + hash(model.model_id) % 10_000)
    returns = market["ret"].fillna(0.0)
    vol = returns.rolling(20).std().fillna(returns.std() if returns.std() > 0 else 0.01)

    if model.kind == "trend_vol":
        signal = np.sign(returns.rolling(10).mean().fillna(0.0)) * (0.02 / (vol + 1e-6))
    elif model.kind == "xsec_momentum":
        signal = np.sign(returns.rolling(5).mean().fillna(0.0)) * 0.8
    elif model.kind == "mean_reversion":
        z = (returns - returns.rolling(20).mean().fillna(0.0)) / (vol + 1e-6)
        signal = -np.tanh(z)
    elif model.kind == "onchain_sentiment_overlay":
        overlay = market["sentiment"].fillna(0.0) * 0.4 + market["onchain"].fillna(0.0) * 0.6
        signal = np.tanh(overlay)
    else:
        trend = np.sign(returns.rolling(15).mean().fillna(0.0))
        rev = -np.tanh((returns - returns.rolling(20).mean().fillna(0.0)) / (vol + 1e-6))
        signal = 0.6 * trend + 0.4 * rev

    noise = rng.normal(0, 0.08, len(market))
    return pd.Series(np.clip(signal + noise, -1.0, 1.0), index=market.index)
