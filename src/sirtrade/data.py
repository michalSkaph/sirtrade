from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


BINANCE_BASE_URL = "https://api.binance.com"

INTERVAL_TO_STEPS_PER_DAY = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "1h": 24,
    "4h": 6,
    "1d": 1,
}

INTERVAL_TO_PANDAS_FREQ = {
    "1m": "min",
    "5m": "5min",
    "15m": "15min",
    "1h": "h",
    "4h": "4h",
    "1d": "D",
}


def _http_get_json(base_url: str, path: str, params: dict | None = None) -> list | dict:
    query = urlencode(params or {})
    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{query}"
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def simulate_market(days: int = 365, seed: int = 42, interval: str = "1d") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps_per_day = INTERVAL_TO_STEPS_PER_DAY.get(interval, 1)
    periods = max(30, min(days * steps_per_day, 50_000))
    freq = INTERVAL_TO_PANDAS_FREQ.get(interval, "D")
    idx = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=periods, freq=freq)

    regime = rng.choice([0, 1, 2], size=periods, p=[0.5, 0.35, 0.15])
    drift = np.select([regime == 0, regime == 1, regime == 2], [0.0005, 0.001, -0.001], default=0.0)
    vol = np.select([regime == 0, regime == 1, regime == 2], [0.02, 0.035, 0.05], default=0.03)
    scale = np.sqrt(max(1, steps_per_day))
    ret = (drift / scale) + rng.normal(0, vol / scale)

    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * (1 - ret[0])
    intraday = np.clip(np.abs(rng.normal(0.01, 0.006, periods)), 0.002, 0.06)
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    sentiment = np.clip(rng.normal(0, 1, days) + 0.5 * np.sign(pd.Series(ret).rolling(5).mean().fillna(0)), -3, 3)
    onchain = np.clip(rng.normal(0, 1, days) + 0.7 * np.sign(pd.Series(ret).rolling(10).mean().fillna(0)), -3, 3)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "ret": ret,
            "sentiment": sentiment,
            "onchain": onchain,
            "regime": regime,
        },
        index=idx,
    )
    return df


def fetch_binance_market(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 365) -> pd.DataFrame:
    rows = _http_get_json(
        BINANCE_BASE_URL,
        "/api/v3/klines",
        {"symbol": symbol.upper(), "interval": interval, "limit": max(30, min(limit, 1000))},
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Binance returned no kline data.")

    frame = pd.DataFrame(rows)
    open_ = frame[1].astype(float)
    high = frame[2].astype(float)
    low = frame[3].astype(float)
    close = frame[4].astype(float)
    ret = close.pct_change().fillna(0.0)
    sentiment = ret.rolling(5).mean().fillna(0.0) * 10
    onchain = ret.rolling(10).mean().fillna(0.0) * 12

    out = pd.DataFrame(
        {
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "ret": ret.values,
            "sentiment": sentiment.clip(-3, 3).values,
            "onchain": onchain.clip(-3, 3).values,
            "regime": np.where(ret.abs().rolling(20).mean().fillna(ret.abs().mean()) > 0.03, 2, 0),
        },
        index=pd.to_datetime(frame[0], unit="ms", utc=True),
    )
    return out


def get_market_data(source: str, days: int, symbol: str, seed: int, interval: str = "1d") -> pd.DataFrame:
    steps_per_day = INTERVAL_TO_STEPS_PER_DAY.get(interval, 1)
    limit = max(30, min(days * steps_per_day, 1000))
    if source == "binance":
        try:
            return fetch_binance_market(symbol=symbol, interval=interval, limit=limit)
        except Exception:
            return simulate_market(days=days, seed=seed, interval=interval)
    return simulate_market(days=days, seed=seed, interval=interval)


def scan_long_tail_opportunities(seed: int = 0, universe_size: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [f"ALT{i:03d}USDT" for i in range(1, universe_size + 1)]
    momentum = rng.normal(0.0, 1.0, universe_size)
    liquidity = np.clip(rng.lognormal(mean=10, sigma=1.0, size=universe_size), 1e3, 1e8)
    spread_bps = np.clip(rng.normal(18, 9, universe_size), 3, 80)
    compliance_risk = np.clip(rng.beta(2, 8, universe_size), 0, 1)

    score = 0.5 * momentum + 0.3 * np.log1p(liquidity) - 0.2 * (spread_bps / 10) - 0.8 * compliance_risk
    df = pd.DataFrame(
        {
            "symbol": symbols,
            "momentum": momentum,
            "liquidity": liquidity,
            "spread_bps": spread_bps,
            "compliance_risk": compliance_risk,
            "opportunity_score": score,
        }
    )
    return df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def scan_binance_long_tail(top_n: int = 20) -> pd.DataFrame:
    try:
        tickers = _http_get_json(BINANCE_BASE_URL, "/api/v3/ticker/24hr")
        if not isinstance(tickers, list):
            raise ValueError("Unexpected ticker response")

        records = []
        for item in tickers:
            symbol = item.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            quote_volume = float(item.get("quoteVolume", 0.0))
            change_pct = float(item.get("priceChangePercent", 0.0))
            count = float(item.get("count", 0.0))
            if quote_volume <= 0:
                continue
            liquidity = quote_volume
            spread_proxy_bps = max(2.0, 2000.0 / np.sqrt(liquidity + 1.0))
            compliance_risk = 0.2 if quote_volume > 5_000_000 else 0.45
            momentum = change_pct / 100.0
            score = 0.5 * momentum + 0.3 * np.log1p(liquidity) - 0.2 * (spread_proxy_bps / 10) - 0.8 * compliance_risk
            records.append(
                {
                    "symbol": symbol,
                    "momentum": momentum,
                    "liquidity": liquidity,
                    "spread_bps": spread_proxy_bps,
                    "compliance_risk": compliance_risk,
                    "opportunity_score": score,
                    "trades_24h": count,
                }
            )

        if not records:
            raise ValueError("No suitable tickers found")
        frame = pd.DataFrame(records).sort_values("opportunity_score", ascending=False)
        return frame.head(top_n).reset_index(drop=True)
    except Exception:
        return scan_long_tail_opportunities(seed=7, universe_size=300).head(top_n)
