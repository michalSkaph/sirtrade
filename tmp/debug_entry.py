from unittest.mock import patch
import pandas as pd, numpy as np
from src.sirtrade.engine import TradingEngine
from src.sirtrade.models import ModelSpec, generate_signals
from src.sirtrade.risk import apply_risk_controls
from src.sirtrade.config import DEFAULT_CONFIG

def _build_market_frame():
    index = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    close = np.linspace(100.0, 106.0, 60)
    close[13] = 104.0; close[14] = 105.0
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.998
    ret = pd.Series(close).pct_change().fillna(0.0).to_numpy()
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "ret": ret, "sentiment": np.full(60, 0.6), "onchain": np.full(60, 0.8), "regime": np.zeros(60)}, index=index)

engine = TradingEngine()
engine.models = [ModelSpec("M1", "Trend", "trend_vol", 1)]
market_a = _build_market_frame()
extra_idx = market_a.index[-1] + pd.Timedelta(hours=1)
extra_row = market_a.iloc[[-1]].copy(); extra_row.index = pd.DatetimeIndex([extra_idx])
market_b = pd.concat([market_a, extra_row])
universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
sig_a = pd.Series(1.0, index=market_a.index)
sig_b = pd.Series(1.0, index=market_b.index)

model = engine.models[0]
raw = generate_signals(model, market_b, seed=0)
controlled = apply_risk_controls(raw, market_b["ret"], DEFAULT_CONFIG.risk)
confl, req_votes, reset_votes, atr_pct = engine._build_entry_confluence(model, market_b, controlled)
ts = market_b.index[-1]
sig = float(controlled.loc[ts])
lv = int(confl.loc[ts, "long_votes"])
sv = int(confl.loc[ts, "short_votes"])
print(f"Signal={sig:.4f} long_votes={lv} short_votes={sv} required={req_votes} reset={reset_votes}")
print(f"Would enter? armed=True lv>={req_votes}={lv>=req_votes} sig>0={sig>0} lv>sv={lv>sv}")
