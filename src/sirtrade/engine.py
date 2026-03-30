from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig, DEFAULT_CONFIG, INITIAL_PAPER_WALLET_CZK, PAPER_TRADE_SIZE_CZK
from .copy_trading import load_top_copy_trader_snapshot
from .data import get_market_data, scan_binance_long_tail, scan_long_tail_opportunities
from .execution import build_dry_run_orders
from .models import ModelSpec, default_model_specs, generate_signals
from .research import StudyInsight, daily_deep_research
from .risk import annualized_vol, apply_risk_controls, cvar95, max_drawdown
from .scoring import calmar_ratio, decision_score, pass_thresholds, sortino_ratio


BLOCKED_TRADING_SYMBOLS = {"USDCUSDT"}
MAX_POSITIONS_PER_MODEL = 5
STANDARD_MODEL_BLUEPRINTS: list[tuple[str, str, str]] = [
    ("M1", "Trend + cílení volatility", "trend_vol"),
    ("M2", "Průřezové momentum + carry", "xsec_momentum"),
    ("M3", "Swing návrat k průměru", "mean_reversion"),
    ("M4", "On-chain + sentimentní vrstva", "onchain_sentiment_overlay"),
    ("M5", "Meta ansámbl", "meta_ensemble"),
]


@dataclass
class ModelResult:
    model_id: str
    name: str
    generation: int
    symbol: str
    sortino: float
    calmar: float
    cvar95: float
    max_dd: float
    cost: float
    turnover: float
    score: float
    passed: bool


def _current_execution_timestamp() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class TradingEngine:
    def __init__(self, config: AppConfig = DEFAULT_CONFIG, model_namespace: str = "", model_label_prefix: str = ""):
        self.config = config
        self.model_namespace = str(model_namespace)
        self.model_label_prefix = str(model_label_prefix)
        self.models: list[ModelSpec] = default_model_specs(namespace=model_namespace, label_prefix=model_label_prefix)
        self.generation = 1
        self.week = 0
        self._live_snapshot_cache: dict[tuple[str, int, str], dict] = {}
        self._model_symbol_memory: dict[str, str] = {}

    def _market_seed(self, symbol: str, offset: int = 0) -> int:
        digest = hashlib.md5(f"{symbol}:{self.week}:{offset}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _build_candidate_universe(
        self,
        market_source: str,
        fallback_symbol: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        if market_source in {"binance", "binance_copy"}:
            long_tail = scan_binance_long_tail(top_n=50)
        else:
            long_tail = scan_long_tail_opportunities(seed=self.week, universe_size=300).head(50)

        candidate_symbols: list[str] = []
        if isinstance(long_tail, pd.DataFrame) and "symbol" in long_tail.columns:
            candidate_symbols = [str(value).upper() for value in long_tail["symbol"].dropna().tolist()]

        candidate_symbols = [symbol for symbol in candidate_symbols if symbol not in BLOCKED_TRADING_SYMBOLS]
        if isinstance(long_tail, pd.DataFrame) and "symbol" in long_tail.columns:
            long_tail = long_tail[~long_tail["symbol"].astype(str).str.upper().isin(BLOCKED_TRADING_SYMBOLS)].reset_index(drop=True)

        candidate_symbols = list(dict.fromkeys([symbol for symbol in candidate_symbols if symbol]))
        if not candidate_symbols:
            candidate_symbols = [fallback_symbol.upper()]
        return long_tail, candidate_symbols

    def _get_live_market_snapshot(
        self,
        market_source: str,
        days: int,
        interval: str,
        fallback_symbol: str,
    ) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame], dict[str, float]]:
        cache_key = (market_source, int(days), str(interval))
        cache_ttl_seconds = 15.0 if market_source in {"binance", "binance_copy"} else 5.0
        cached = self._live_snapshot_cache.get(cache_key)
        now = time.time()
        if cached and (now - float(cached.get("timestamp", 0.0))) < cache_ttl_seconds:
            return (
                cached["long_tail"],
                cached["candidate_symbols"],
                cached["markets_by_symbol"],
                cached["latest_prices"],
            )

        long_tail, candidate_symbols = self._build_candidate_universe(
            market_source=market_source,
            fallback_symbol=fallback_symbol,
        )
        markets_by_symbol: dict[str, pd.DataFrame] = {}
        latest_prices: dict[str, float] = {}
        for index, candidate_symbol in enumerate(candidate_symbols):
            market = get_market_data(
                source=market_source,
                days=days,
                symbol=candidate_symbol,
                seed=self._market_seed(candidate_symbol, offset=index),
                interval=interval,
            )
            markets_by_symbol[candidate_symbol] = market
            if not market.empty and "close" in market.columns:
                latest_prices[candidate_symbol] = float(pd.to_numeric(market["close"], errors="coerce").dropna().iloc[-1])

        snapshot = {
            "timestamp": now,
            "long_tail": long_tail,
            "candidate_symbols": candidate_symbols,
            "markets_by_symbol": markets_by_symbol,
            "latest_prices": latest_prices,
        }
        self._live_snapshot_cache[cache_key] = snapshot
        return long_tail, candidate_symbols, markets_by_symbol, latest_prices

    def _inactive_copy_trader_run(self, model: ModelSpec, symbol: str) -> dict:
        result = ModelResult(
            model_id=model.model_id,
            name=model.name,
            generation=model.generation,
            symbol=symbol,
            sortino=0.0,
            calmar=0.0,
            cvar95=1.0,
            max_dd=1.0,
            cost=0.0,
            turnover=0.0,
            score=-1.0,
            passed=False,
        )
        return {
            "result": result,
            "events": [],
            "final_position": 0.0,
            "final_open_slots": 0,
            "market": pd.DataFrame(),
            "opportunity_score": 0.0,
            "open_positions_override": [],
        }

    def _copy_trader_slot_allocations(self, positions: list[dict[str, object]], slot_budget: int = 5) -> list[int]:
        if not positions or slot_budget <= 0:
            return []

        capped_positions = positions[:slot_budget]
        return [1] * len(capped_positions)

    def _build_copy_trader_run(
        self,
        model: ModelSpec,
        snapshot: dict[str, object] | None,
        markets_by_symbol: dict[str, pd.DataFrame],
        fallback_symbol: str,
    ) -> dict:
        if not isinstance(snapshot, dict):
            return self._inactive_copy_trader_run(model, fallback_symbol)

        leader = snapshot.get("leader", {})
        raw_positions = snapshot.get("positions", [])
        if not isinstance(leader, dict) or not isinstance(raw_positions, list):
            return self._inactive_copy_trader_run(model, fallback_symbol)

        ranked_positions = []
        for position in raw_positions:
            if not isinstance(position, dict):
                continue
            symbol = str(position.get("symbol", "")).upper()
            if symbol in BLOCKED_TRADING_SYMBOLS:
                continue
            market = markets_by_symbol.get(symbol)
            if not symbol or market is None or market.empty:
                continue
            ranked_positions.append(position)

        ranked_positions = sorted(
            ranked_positions,
            key=lambda item: float(item.get("notional_usd", 0.0)),
            reverse=True,
        )
        if not ranked_positions:
            return self._inactive_copy_trader_run(model, fallback_symbol)

        allocations = self._copy_trader_slot_allocations(ranked_positions, slot_budget=5)
        slot_size = PAPER_TRADE_SIZE_CZK / INITIAL_PAPER_WALLET_CZK
        events: list[dict] = []
        open_positions_override: list[dict[str, object]] = []
        open_positions_by_key: dict[tuple[str, str], dict[str, object]] = {}
        portfolio_pnl: pd.Series | None = None
        total_open_slots = 0
        net_final_position = 0.0
        primary_symbol = str(ranked_positions[0].get("symbol", fallback_symbol)).upper()
        primary_market = markets_by_symbol.get(primary_symbol, pd.DataFrame())
        leader_name = str(leader.get("nickname") or leader.get("trader_id") or model.name).strip()

        for position, slots in zip(ranked_positions, allocations):
            if slots <= 0:
                continue

            symbol = str(position.get("symbol", "")).upper()
            side = str(position.get("side", "")).upper()
            market = markets_by_symbol[symbol]
            direction = 1.0 if side == "LONG" else -1.0
            position_size = direction * slots * slot_size
            total_open_slots += int(slots)
            net_final_position += position_size

            pnl_series = market["ret"].fillna(0.0).astype(float) * position_size
            portfolio_pnl = pnl_series if portfolio_pnl is None else portfolio_pnl.add(pnl_series, fill_value=0.0)
            price = float(pd.to_numeric(market["close"], errors="coerce").dropna().iloc[-1])
            timestamp = market.index[-1]
            position_key = (symbol, side)
            entry_price = float(position.get("entry_price", price) or price)
            existing_position = open_positions_by_key.get(position_key)
            if existing_position is None:
                open_positions_by_key[position_key] = {
                    "symbol": symbol,
                    "side": side,
                    "slots": int(slots),
                    "model_id": model.model_id,
                    "model_name": f"{model.name} | {leader_name}",
                    "leader_id": leader.get("trader_id"),
                    "leader_name": leader_name,
                    "entry_price": entry_price,
                }
            else:
                previous_slots = int(existing_position.get("slots", 0))
                total_slots = previous_slots + int(slots)
                previous_entry_price = float(existing_position.get("entry_price", entry_price) or entry_price)
                existing_position["slots"] = total_slots
                existing_position["entry_price"] = (
                    (previous_entry_price * previous_slots) + (entry_price * int(slots))
                ) / max(total_slots, 1)

            events.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "model_id": model.model_id,
                    "model_name": f"{model.name} | {leader_name}",
                    "akce": f"Vstup {side} (+{slots})",
                    "strana": side,
                    "cena": price,
                    "pozice": position_size,
                    "sloty": int(slots),
                    "leader_id": leader.get("trader_id"),
                    "leader_name": leader_name,
                }
            )

        open_positions_override = list(open_positions_by_key.values())

        if portfolio_pnl is None or portfolio_pnl.empty:
            return self._inactive_copy_trader_run(model, primary_symbol)

        equity = (1 + portfolio_pnl).cumprod()
        mdd = max_drawdown(equity)
        srt = sortino_ratio(portfolio_pnl)
        calmar = calmar_ratio(portfolio_pnl, mdd)
        cv = cvar95(portfolio_pnl)
        metrics = {
            "sortino": srt,
            "calmar": calmar,
            "cvar95": cv,
            "max_dd": mdd,
            "cost": 0.0,
            "turnover": 0.0,
        }
        score = decision_score(metrics, self.config.weights) + float(leader.get("score", 0.0))
        passed = pass_thresholds(metrics, self.config.thresholds)
        result = ModelResult(
            model_id=model.model_id,
            name=f"{model.name} | {leader_name}",
            generation=model.generation,
            symbol=primary_symbol,
            sortino=srt,
            calmar=calmar,
            cvar95=cv,
            max_dd=mdd,
            cost=0.0,
            turnover=0.0,
            score=score,
            passed=passed,
        )
        return {
            "result": result,
            "events": events,
            "final_position": float(net_final_position),
            "final_open_slots": int(total_open_slots),
            "market": primary_market,
            "opportunity_score": 0.0,
            "open_positions_override": open_positions_override,
        }

    def _build_entry_confluence(
        self,
        model: ModelSpec,
        market: pd.DataFrame,
        controlled_signal: pd.Series,
    ) -> tuple[pd.DataFrame, int, int, pd.Series]:
        close = market["close"].astype(float)
        high = market["high"].astype(float)
        low = market["low"].astype(float)
        returns = market["ret"].fillna(0.0).astype(float)
        sentiment = market.get("sentiment", pd.Series(0.0, index=market.index)).fillna(0.0).astype(float)
        onchain = market.get("onchain", pd.Series(0.0, index=market.index)).fillna(0.0).astype(float)

        # Scalp segment uses faster indicator periods for quick entries/exits
        is_scalp = self.model_namespace == "SC"
        ema_fast_span = 3 if is_scalp else 8
        ema_slow_span = 8 if is_scalp else 21
        ema_anchor_span = 21 if is_scalp else 55
        mom_fast_win = 2 if is_scalp else 3
        mom_slow_win = 4 if is_scalp else 8
        breakout_period = 3 if is_scalp else 5
        bb_period = 10 if is_scalp else 20
        rsi_alpha = 1 / 7 if is_scalp else 1 / 14
        macd_fast_span = 5 if is_scalp else 12
        macd_slow_span = 13 if is_scalp else 26
        macd_signal_span = 4 if is_scalp else 9
        atr_period = 7 if is_scalp else 14
        donchian_period = 10 if is_scalp else 20
        bb_width = 1.2 if is_scalp else 1.4

        ema_fast = close.ewm(span=ema_fast_span, adjust=False).mean()
        ema_slow = close.ewm(span=ema_slow_span, adjust=False).mean()
        ema_anchor = close.ewm(span=ema_anchor_span, adjust=False).mean()
        momentum_fast = returns.rolling(mom_fast_win).mean().fillna(0.0)
        momentum_slow = returns.rolling(mom_slow_win).mean().fillna(0.0)
        breakout = close.pct_change(breakout_period).fillna(0.0)
        rolling_mean = close.rolling(bb_period).mean().bfill()
        rolling_std = close.rolling(bb_period).std(ddof=0).replace(0.0, np.nan)
        price_z = ((close - rolling_mean) / (rolling_std + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        overlay_bias = ((0.6 * onchain) + (0.4 * sentiment)).clip(-3.0, 3.0)
        upper_band = rolling_mean + (bb_width * rolling_std.fillna(0.0))
        lower_band = rolling_mean - (bb_width * rolling_std.fillna(0.0))

        delta = close.diff().fillna(0.0)
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = gains.ewm(alpha=rsi_alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=rsi_alpha, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = (100 - (100 / (1 + rs))).fillna(50.0)

        macd_fast = close.ewm(span=macd_fast_span, adjust=False).mean()
        macd_slow = close.ewm(span=macd_slow_span, adjust=False).mean()
        macd_line = macd_fast - macd_slow
        macd_signal = macd_line.ewm(span=macd_signal_span, adjust=False).mean()
        macd_hist = (macd_line - macd_signal).fillna(0.0)

        prev_close = close.shift(1).fillna(close)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(atr_period).mean().fillna(true_range.expanding().mean()).fillna(0.0)
        atr_pct = (atr / (close.abs() + 1e-6)).clip(lower=0.0015, upper=0.08)

        donchian_high = high.rolling(donchian_period).max().shift(1).fillna(high.expanding().max())
        donchian_low = low.rolling(donchian_period).min().shift(1).fillna(low.expanding().min())
        breakout_up = close >= donchian_high
        breakout_down = close <= donchian_low

        trend_up = (ema_fast > ema_slow) & (close >= ema_fast)
        trend_down = (ema_fast < ema_slow) & (close <= ema_fast)
        broad_trend_up = close >= ema_anchor
        broad_trend_down = close <= ema_anchor
        momentum_up = momentum_fast > 0
        momentum_down = momentum_fast < 0
        acceleration_up = momentum_fast >= momentum_slow
        acceleration_down = momentum_fast <= momentum_slow

        if model.kind == "mean_reversion":
            sig_thresh = 0.08 if is_scalp else 0.16
            z_thresh = 0.90 if is_scalp else 1.35
            rsi_lo = 40 if is_scalp else 36
            rsi_hi = 60 if is_scalp else 64
            long_checks = [
                controlled_signal >= sig_thresh,
                price_z <= -z_thresh,
                close <= lower_band,
                rsi <= rsi_lo,
                acceleration_up,
            ]
            short_checks = [
                controlled_signal <= -sig_thresh,
                price_z >= z_thresh,
                close >= upper_band,
                rsi >= rsi_hi,
                acceleration_down,
            ]
            required_votes = 3 if is_scalp else 4
            reset_votes = 1 if is_scalp else 2
        elif model.kind == "onchain_sentiment_overlay":
            sig_thresh = 0.10 if is_scalp else 0.18
            bias_thresh = 0.15 if is_scalp else 0.25
            long_checks = [
                controlled_signal >= sig_thresh,
                overlay_bias >= bias_thresh,
                trend_up,
                momentum_up,
                macd_hist >= 0,
            ]
            short_checks = [
                controlled_signal <= -sig_thresh,
                overlay_bias <= -bias_thresh,
                trend_down,
                momentum_down,
                macd_hist <= 0,
            ]
            required_votes = 3 if is_scalp else 4
            reset_votes = 1 if is_scalp else 2
        elif model.kind == "xsec_momentum":
            sig_thresh = 0.12 if is_scalp else 0.24
            long_checks = [
                controlled_signal >= sig_thresh,
                momentum_up,
                acceleration_up,
                breakout_up | (breakout > 0),
                trend_up,
                broad_trend_up,
                macd_hist > 0,
            ]
            short_checks = [
                controlled_signal <= -sig_thresh,
                momentum_down,
                acceleration_down,
                breakout_down | (breakout < 0),
                trend_down,
                broad_trend_down,
                macd_hist < 0,
            ]
            required_votes = 4 if is_scalp else 5
            reset_votes = 1 if is_scalp else 2
        elif model.kind == "meta_ensemble":
            sig_thresh = 0.10 if is_scalp else 0.20
            long_checks = [
                controlled_signal >= sig_thresh,
                trend_up,
                momentum_up,
                acceleration_up,
                overlay_bias >= -0.10,
                macd_hist >= 0,
                rsi.between(50, 72) if not is_scalp else rsi.between(45, 75),
            ]
            short_checks = [
                controlled_signal <= -sig_thresh,
                trend_down,
                momentum_down,
                acceleration_down,
                overlay_bias <= 0.10,
                macd_hist <= 0,
                rsi.between(28, 50) if not is_scalp else rsi.between(25, 55),
            ]
            required_votes = 4 if is_scalp else 5
            reset_votes = 1 if is_scalp else 2
        else:
            sig_thresh = 0.12 if is_scalp else 0.22
            long_checks = [
                controlled_signal >= sig_thresh,
                trend_up,
                broad_trend_up,
                momentum_up,
                acceleration_up,
                breakout_up | (breakout > 0),
                macd_hist > 0,
            ]
            short_checks = [
                controlled_signal <= -sig_thresh,
                trend_down,
                broad_trend_down,
                momentum_down,
                acceleration_down,
                breakout_down | (breakout < 0),
                macd_hist < 0,
            ]
            required_votes = 5
            reset_votes = 2

        long_votes = sum(check.astype(int) for check in long_checks)
        short_votes = sum(check.astype(int) for check in short_checks)
        confluence = pd.DataFrame(
            {
                "long_votes": long_votes,
                "short_votes": short_votes,
                "long_confidence": long_votes / float(len(long_checks)),
                "short_confidence": short_votes / float(len(short_checks)),
            },
            index=market.index,
        )
        return confluence, required_votes, reset_votes, atr_pct

    def _trade_profile(self, model: ModelSpec) -> tuple[float, float, float, float]:
        # Scalp segment: very tight TP/SL so trades resolve in seconds-minutes
        if self.model_namespace == "SC":
            if model.kind == "mean_reversion":
                return 0.35, 0.55, 0.0008, 0.03
            if model.kind == "xsec_momentum":
                return 0.45, 0.70, 0.0010, 0.04
            if model.kind == "onchain_sentiment_overlay":
                return 0.40, 0.60, 0.0010, 0.03
            if model.kind == "meta_ensemble":
                return 0.45, 0.70, 0.0010, 0.04
            return 0.50, 0.75, 0.0010, 0.04
        if model.kind == "mean_reversion":
            return 1.35, 2.75, 0.0025, 0.08
        if model.kind == "xsec_momentum":
            return 1.85, 3.90, 0.0030, 0.10
        if model.kind == "onchain_sentiment_overlay":
            return 1.70, 3.40, 0.0030, 0.09
        if model.kind == "meta_ensemble":
            return 1.95, 4.10, 0.0032, 0.09
        return 2.10, 4.40, 0.0035, 0.10

    def _latest_setup_snapshot(self, model: ModelSpec, market: pd.DataFrame) -> dict[str, float | int]:
        raw = generate_signals(model, market, seed=0)
        controlled_signal = apply_risk_controls(raw, market["ret"], self.config.risk)
        confluence, required_votes, reset_votes, atr_pct = self._build_entry_confluence(model, market, controlled_signal)
        latest_ts = market.index[-1]
        _, _, min_stop_floor, signal_reset_floor = self._trade_profile(model)
        vol_step = float(atr_pct.loc[latest_ts])
        if np.isnan(vol_step) or vol_step <= 0:
            vol_step = min_stop_floor

        return {
            "signal_value": float(controlled_signal.loc[latest_ts]),
            "long_votes": int(confluence.loc[latest_ts, "long_votes"]),
            "short_votes": int(confluence.loc[latest_ts, "short_votes"]),
            "long_confidence": float(confluence.loc[latest_ts, "long_confidence"]),
            "short_confidence": float(confluence.loc[latest_ts, "short_confidence"]),
            "required_votes": int(required_votes),
            "reset_votes": int(reset_votes),
            "vol_step": float(vol_step),
            "min_stop_floor": float(min_stop_floor),
            "signal_reset_floor": float(signal_reset_floor),
        }

    def _scalp_invalidation_reason(
        self,
        side: str,
        signal_value: float,
        long_votes: int,
        short_votes: int,
        required_votes: int,
        reset_votes: int,
        signal_reset_floor: float,
    ) -> str | None:
        if self.model_namespace != "SC":
            return None

        normalized_side = str(side).upper()
        if normalized_side not in {"LONG", "SHORT"}:
            return None

        same_votes = long_votes if normalized_side == "LONG" else short_votes
        opposite_votes = short_votes if normalized_side == "LONG" else long_votes
        reversal_signal = signal_value <= -signal_reset_floor if normalized_side == "LONG" else signal_value >= signal_reset_floor
        edge_faded = abs(signal_value) <= signal_reset_floor and same_votes <= reset_votes
        opposite_pressure = opposite_votes >= max(reset_votes + 1, required_votes - 1)
        setup_lost = (
            same_votes < max(2, required_votes - 1)
            and opposite_votes >= same_votes
            and abs(signal_value) <= (signal_reset_floor * 1.5)
        )

        if reversal_signal or opposite_pressure:
            return "SCALP_REVERSAL"
        if edge_faded or setup_lost:
            return "SCALP_INVALIDATION"
        return None

    def _build_trade_events(self, model: ModelSpec, prices: pd.Series, position: pd.Series) -> tuple[list[dict], int]:
        events: list[dict] = []
        slot_size = PAPER_TRADE_SIZE_CZK / INITIAL_PAPER_WALLET_CZK
        position_filled = position.fillna(0.0)
        slot_series = np.floor((position_filled.abs() / (slot_size + 1e-9))).clip(0, 5).astype(int)
        side_series = np.sign(position_filled).astype(int)

        prev_side = 0
        prev_slots = 0
        for ts in position_filled.index:
            current_side = int(side_series.loc[ts])
            current_slots = int(slot_series.loc[ts])
            price = float(prices.loc[ts])
            if prev_side == 0 and current_side != 0 and current_slots > 0:
                events.append(
                    {
                        "timestamp": ts,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": f"Vstup { 'LONG' if current_side > 0 else 'SHORT' } (+{current_slots})",
                        "strana": "LONG" if current_side > 0 else "SHORT",
                        "cena": price,
                        "pozice": float(position_filled.loc[ts]),
                        "sloty": current_slots,
                    }
                )
            elif prev_side != 0 and current_side == 0 and prev_slots > 0:
                events.append(
                    {
                        "timestamp": ts,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": f"Výstup { 'LONG' if prev_side > 0 else 'SHORT' } (-{prev_slots})",
                        "strana": "LONG" if prev_side > 0 else "SHORT",
                        "cena": price,
                        "pozice": 0.0,
                        "sloty": 0,
                    }
                )
            elif prev_side != 0 and current_side != 0 and prev_side != current_side:
                events.append(
                    {
                        "timestamp": ts,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": f"Výstup { 'LONG' if prev_side > 0 else 'SHORT' } (-{prev_slots})",
                        "strana": "LONG" if prev_side > 0 else "SHORT",
                        "cena": price,
                        "pozice": 0.0,
                        "sloty": 0,
                    }
                )
                if current_slots > 0:
                    events.append(
                        {
                            "timestamp": ts,
                            "model_id": model.model_id,
                            "model_name": model.name,
                            "akce": f"Vstup { 'LONG' if current_side > 0 else 'SHORT' } (+{current_slots})",
                            "strana": "LONG" if current_side > 0 else "SHORT",
                            "cena": price,
                            "pozice": float(position_filled.loc[ts]),
                            "sloty": current_slots,
                        }
                    )
            elif current_side == prev_side and current_side != 0 and current_slots != prev_slots:
                if current_slots > prev_slots:
                    delta = current_slots - prev_slots
                    events.append(
                        {
                            "timestamp": ts,
                            "model_id": model.model_id,
                            "model_name": model.name,
                            "akce": f"Vstup { 'LONG' if current_side > 0 else 'SHORT' } (+{delta})",
                            "strana": "LONG" if current_side > 0 else "SHORT",
                            "cena": price,
                            "pozice": float(position_filled.loc[ts]),
                            "sloty": current_slots,
                        }
                    )
                else:
                    delta = prev_slots - current_slots
                    events.append(
                        {
                            "timestamp": ts,
                            "model_id": model.model_id,
                            "model_name": model.name,
                            "akce": f"Výstup { 'LONG' if current_side > 0 else 'SHORT' } (-{delta})",
                            "strana": "LONG" if current_side > 0 else "SHORT",
                            "cena": price,
                            "pozice": float(position_filled.loc[ts]),
                            "sloty": current_slots,
                        }
                    )

            prev_side = current_side
            prev_slots = current_slots
        return events, prev_slots

    def _simulate_model(
        self,
        model: ModelSpec,
        market: pd.DataFrame,
        symbol: str,
    ) -> tuple[ModelResult, list[dict], float, int]:
        raw = generate_signals(model, market, seed=self.week)
        controlled_signal = apply_risk_controls(raw, market["ret"], self.config.risk)
        confluence, required_votes, reset_votes, atr_pct = self._build_entry_confluence(model, market, controlled_signal)

        close = market["close"].astype(float)
        high = market["high"].astype(float)
        low = market["low"].astype(float)

        slot_size = PAPER_TRADE_SIZE_CZK / INITIAL_PAPER_WALLET_CZK
        warmup_bars = min(48, max(12, int(len(market) * 0.1)))
        if self.model_namespace == "SC":
            warmup_bars = min(20, max(8, int(len(market) * 0.05)))
        stop_atr_multiplier, target_atr_multiplier, min_stop_floor, signal_reset_floor = self._trade_profile(model)

        pos = pd.Series(0.0, index=market.index, dtype=float)
        side = 0
        position_size = 0.0
        current_slots = 0
        stop_price = None
        target_price = None
        events: list[dict] = []
        entry_armed = True

        for step, ts in enumerate(market.index):
            signal_value = float(controlled_signal.loc[ts])
            long_votes = int(confluence.loc[ts, "long_votes"])
            short_votes = int(confluence.loc[ts, "short_votes"])
            long_confidence = float(confluence.loc[ts, "long_confidence"])
            short_confidence = float(confluence.loc[ts, "short_confidence"])
            close_price = float(close.loc[ts])
            high_price = float(high.loc[ts])
            low_price = float(low.loc[ts])
            vol_step = float(atr_pct.loc[ts])
            if np.isnan(vol_step) or vol_step <= 0:
                vol_step = min_stop_floor

            if step < warmup_bars:
                pos.loc[ts] = 0.0
                continue

            if side == 0:
                if not entry_armed:
                    setup_reset = max(long_votes, short_votes) <= reset_votes or abs(signal_value) <= signal_reset_floor
                    if setup_reset:
                        entry_armed = True
                    pos.loc[ts] = 0.0
                    continue

                direction = 0
                confidence = 0.0
                if long_votes >= required_votes and signal_value > 0 and long_votes > short_votes:
                    direction = 1
                    confidence = long_confidence
                if short_votes >= required_votes and signal_value < 0 and short_votes > long_votes:
                    direction = -1
                    confidence = short_confidence

                if direction != 0:
                    slots = 1
                    position_size = float(direction * slots * slot_size)
                    side = direction
                    current_slots = slots

                    stop_dist = max(min_stop_floor, stop_atr_multiplier * max(vol_step, min_stop_floor))
                    target_dist = max(stop_dist * 1.8, target_atr_multiplier * max(vol_step, min_stop_floor))
                    if side > 0:
                        stop_price = close_price * (1.0 - stop_dist)
                        target_price = close_price * (1.0 + target_dist)
                    else:
                        stop_price = close_price * (1.0 + stop_dist)
                        target_price = close_price * (1.0 - target_dist)

                    events.append(
                        {
                            "timestamp": ts,
                            "symbol": symbol,
                            "model_id": model.model_id,
                            "model_name": model.name,
                            "akce": f"Vstup { 'LONG' if side > 0 else 'SHORT' } (+{current_slots})",
                            "strana": "LONG" if side > 0 else "SHORT",
                            "cena": close_price,
                            "pozice": position_size,
                            "sloty": current_slots,
                        }
                    )

                    entry_armed = False
                    pos.loc[ts] = position_size
                else:
                    pos.loc[ts] = 0.0
                continue

            hit_exit = False
            exit_reason = None
            if side > 0:
                if stop_price is not None and low_price <= stop_price:
                    hit_exit = True
                    exit_reason = "STOP"
                if target_price is not None and high_price >= target_price:
                    hit_exit = True
                    exit_reason = "TARGET" if exit_reason is None else exit_reason
            else:
                if stop_price is not None and high_price >= stop_price:
                    hit_exit = True
                    exit_reason = "STOP"
                if target_price is not None and low_price <= target_price:
                    hit_exit = True
                    exit_reason = "TARGET" if exit_reason is None else exit_reason

            if not hit_exit:
                scalp_exit_reason = self._scalp_invalidation_reason(
                    side="LONG" if side > 0 else "SHORT",
                    signal_value=signal_value,
                    long_votes=long_votes,
                    short_votes=short_votes,
                    required_votes=required_votes,
                    reset_votes=reset_votes,
                    signal_reset_floor=signal_reset_floor,
                )
                if scalp_exit_reason is not None:
                    hit_exit = True
                    exit_reason = scalp_exit_reason

            if hit_exit:
                exit_side = "LONG" if side > 0 else "SHORT"
                events.append(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": f"Výstup {exit_side} (-{current_slots})",
                        "strana": exit_side,
                        "cena": close_price,
                        "pozice": 0.0,
                        "sloty": 0,
                        "duvod_vystupu": exit_reason or "NEURČENO",
                    }
                )
                pos.loc[ts] = 0.0
                side = 0
                position_size = 0.0
                current_slots = 0
                stop_price = None
                target_price = None
                entry_armed = False
            else:
                pos.loc[ts] = position_size

        turnover = float(pos.diff().abs().fillna(0).mean())
        fee_cost = turnover * (self.config.fee_bps_assumption / 10_000)
        pnl = pos.shift(1).fillna(0) * market["ret"] - fee_cost
        equity = (1 + pnl).cumprod()

        mdd = max_drawdown(equity)
        srt = sortino_ratio(pnl)
        calmar = calmar_ratio(pnl, mdd)
        cv = cvar95(pnl)

        metrics = {
            "sortino": srt,
            "calmar": calmar,
            "cvar95": cv,
            "max_dd": mdd,
            "cost": fee_cost,
            "turnover": turnover,
        }
        score = decision_score(metrics, self.config.weights)
        passed = pass_thresholds(metrics, self.config.thresholds)

        result = ModelResult(
            model_id=model.model_id,
            name=model.name,
            generation=model.generation,
            symbol=symbol,
            sortino=srt,
            calmar=calmar,
            cvar95=cv,
            max_dd=mdd,
            cost=fee_cost,
            turnover=turnover,
            score=score,
            passed=passed,
        )
        final_position = float(pos.iloc[-1]) if not pos.empty else 0.0
        final_open_slots = int(current_slots if side != 0 else 0)
        return result, events, final_position, int(final_open_slots)

    def _build_model_open_positions(
        self,
        results_df: pd.DataFrame,
        final_positions: dict[str, float],
        final_open_slots: dict[str, int],
    ) -> dict[str, list[dict]]:
        model_open_positions: dict[str, list[dict]] = {}
        for _, row in results_df.reset_index(drop=True).iterrows():
            model_id = str(row["model_id"])
            model_name = str(row["name"])
            model_symbol = str(row.get("symbol", "")).upper()
            slots = max(0, int(final_open_slots.get(model_id, 0)))
            position_value = float(final_positions.get(model_id, 0.0))
            side = "LONG" if position_value > 0 else ("SHORT" if position_value < 0 else "-")
            if slots <= 0 or side == "-" or not model_symbol:
                model_open_positions[model_id] = []
                continue

            model_open_positions[model_id] = [
                {
                    "symbol": model_symbol,
                    "side": side,
                    "slots": slots,
                    "model_id": model_id,
                    "model_name": model_name,
                }
            ]

        return model_open_positions

    def _previous_live_state(self, previous_summary: dict[str, Any] | None, model_id: str) -> dict[str, Any]:
        if not isinstance(previous_summary, dict):
            return {}
        live_state = previous_summary.get("live_model_state", {})
        if isinstance(live_state, dict):
            state = live_state.get(model_id)
            if isinstance(state, dict):
                result = dict(state)
                # Migrate old single-position format to positions list
                if "positions" not in result:
                    side = str(result.get("side", "")).upper()
                    symbol = str(result.get("symbol", "")).upper()
                    open_slots = int(result.get("open_slots", 0) or 0)
                    if side in {"LONG", "SHORT"} and symbol and open_slots > 0:
                        result["positions"] = [
                            {
                                "symbol": symbol,
                                "side": side,
                                "open_slots": open_slots,
                                "entry_price": float(result.get("entry_price", 0.0) or 0.0),
                                "stop_price": float(result.get("stop_price", 0.0) or 0.0),
                                "target_price": float(result.get("target_price", 0.0) or 0.0),
                                "opened_at": result.get("opened_at"),
                            }
                        ]
                    else:
                        result["positions"] = []
                return result

        model_positions = previous_summary.get("model_open_positions", {})
        if not isinstance(model_positions, dict):
            return {}
        positions = model_positions.get(model_id, [])
        if not isinstance(positions, list) or not positions:
            return {}

        valid_positions = [item for item in positions if isinstance(item, dict)]
        if not valid_positions:
            return {}

        # Group positions by (symbol, side) to build unique position entries
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for pos in valid_positions:
            sym = str(pos.get("symbol", "")).upper()
            sd = str(pos.get("side", "")).upper()
            if sd not in {"LONG", "SHORT"} or not sym:
                continue
            key = (sym, sd)
            if key not in grouped:
                grouped[key] = {
                    "symbol": sym,
                    "side": sd,
                    "open_slots": 0,
                    "entry_price": float(pos.get("entry_price", 0.0) or 0.0),
                    "stop_price": float(pos.get("stop_price", 0.0) or 0.0),
                    "target_price": float(pos.get("target_price", 0.0) or 0.0),
                    "opened_at": pos.get("opened_at"),
                }
            grouped[key]["open_slots"] += max(1, int(pos.get("slots", 1) or 1))

        pos_list = list(grouped.values())
        total_slots = sum(int(p.get("open_slots", 0)) for p in pos_list)
        first = pos_list[0] if pos_list else {}
        return {
            "entry_armed": False,
            "symbol": str(first.get("symbol", "")).upper(),
            "side": str(first.get("side", "")).upper(),
            "open_slots": total_slots,
            "entry_price": float(first.get("entry_price", 0.0) or 0.0),
            "opened_at": first.get("opened_at"),
            "stop_price": float(first.get("stop_price", 0.0) or 0.0),
            "target_price": float(first.get("target_price", 0.0) or 0.0),
            "positions": pos_list,
        }

    def _build_incremental_live_state(
        self,
        selected_runs: list[dict[str, Any]],
        previous_summary: dict[str, Any] | None,
        markets_by_symbol: dict[str, pd.DataFrame],
        effective_symbol: str,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]],
        dict[str, float],
        dict[str, int],
        dict[str, list[dict[str, Any]]],
        dict[str, dict[str, Any]],
    ]:
        previous_trades = previous_summary.get("model_trades", {}) if isinstance(previous_summary, dict) else {}
        model_map = {model.model_id: model for model in self.models}
        slot_size = PAPER_TRADE_SIZE_CZK / INITIAL_PAPER_WALLET_CZK

        cumulative_trades: dict[str, list[dict[str, Any]]] = {}
        trade_events_delta: dict[str, list[dict[str, Any]]] = {}
        final_positions: dict[str, float] = {}
        final_open_slots: dict[str, int] = {}
        model_open_positions: dict[str, list[dict[str, Any]]] = {}
        live_model_state: dict[str, dict[str, Any]] = {}
        active_exposures: set[tuple[str, str]] = set()
        claimed_symbols: set[str] = set()

        # Pre-populate active_exposures from ALL models' previous positions
        if isinstance(previous_summary, dict):
            previous_live_state = previous_summary.get("live_model_state", {})
            if isinstance(previous_live_state, dict):
                for state in previous_live_state.values():
                    if not isinstance(state, dict):
                        continue
                    # New multi-position format
                    for pos in state.get("positions", []):
                        if not isinstance(pos, dict):
                            continue
                        p_side = str(pos.get("side", "")).upper()
                        p_sym = str(pos.get("symbol", "")).upper()
                        p_slots = int(pos.get("open_slots", 0) or 0)
                        if p_side in {"LONG", "SHORT"} and p_sym and p_slots > 0:
                            active_exposures.add((p_sym, p_side))
                    # Fallback: old single-position format
                    if not state.get("positions"):
                        side = str(state.get("side", "")).upper()
                        symbol = str(state.get("symbol", "")).upper()
                        open_slots = int(state.get("open_slots", 0) or 0)
                        if side in {"LONG", "SHORT"} and symbol and open_slots > 0:
                            active_exposures.add((symbol, side))

        for run in selected_runs:
            result = run["result"]
            model_id = str(result.model_id)
            model = model_map.get(model_id)
            previous_state = self._previous_live_state(previous_summary, model_id)
            model_trade_history = list(previous_trades.get(model_id, [])) if isinstance(previous_trades, dict) else []
            delta_events: list[dict[str, Any]] = []

            open_positions_override = run.get("open_positions_override", [])
            if open_positions_override:
                cumulative_trades[model_id] = model_trade_history + list(run.get("events", []))
                trade_events_delta[model_id] = list(run.get("events", []))
                final_positions[model_id] = float(run.get("final_position", 0.0))
                final_open_slots[model_id] = int(run.get("final_open_slots", 0))
                model_open_positions[model_id] = list(open_positions_override)
                live_model_state[model_id] = {
                    "entry_armed": False,
                    "symbol": str(result.symbol).upper(),
                    "side": "",
                    "open_slots": int(run.get("final_open_slots", 0)),
                    "entry_price": 0.0,
                    "opened_at": None,
                    "stop_price": 0.0,
                    "target_price": 0.0,
                    "positions": [],
                }
                for position in open_positions_override:
                    if not isinstance(position, dict):
                        continue
                    side = str(position.get("side", "")).upper()
                    symbol = str(position.get("symbol", result.symbol)).upper()
                    slots = int(position.get("slots", 0) or 0)
                    if side in {"LONG", "SHORT"} and symbol and slots > 0:
                        active_exposures.add((symbol, side))
                continue

            if model is None:
                cumulative_trades[model_id] = model_trade_history
                trade_events_delta[model_id] = []
                final_positions[model_id] = 0.0
                final_open_slots[model_id] = 0
                model_open_positions[model_id] = []
                live_model_state[model_id] = {"entry_armed": True, "positions": []}
                continue

            # --- Build list of existing positions from previous state ---
            prev_positions: list[dict[str, Any]] = list(previous_state.get("positions", []))
            entry_armed = bool(previous_state.get("entry_armed", True))
            prev_setup_active = bool(previous_state.get("setup_active", False))
            prev_setup_direction = int(previous_state.get("setup_direction", 0) or 0)
            current_setup_active = prev_setup_active
            current_setup_direction = prev_setup_direction

            # Determine a reference timestamp from the run's market
            run_market = run.get("market")
            ref_symbol = str(result.symbol).upper()
            ref_market = markets_by_symbol.get(ref_symbol, run_market)
            if ref_market is None or ref_market.empty:
                cumulative_trades[model_id] = model_trade_history
                trade_events_delta[model_id] = []
                final_positions[model_id] = 0.0
                final_open_slots[model_id] = 0
                model_open_positions[model_id] = []
                live_model_state[model_id] = {"entry_armed": True, "positions": []}
                continue

            ts = ref_market.index[-1]
            ts_key = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            execution_ts = _current_execution_timestamp()
            execution_ts_key = execution_ts.isoformat() if hasattr(execution_ts, "isoformat") else str(execution_ts)
            setup_snapshot_cache: dict[str, dict[str, float | int]] = {}

            # --- EXIT logic: evaluate each existing position independently ---
            surviving_positions: list[dict[str, Any]] = []
            for pos in prev_positions:
                if not isinstance(pos, dict):
                    continue
                pos_symbol = str(pos.get("symbol", "")).upper()
                pos_side = str(pos.get("side", "")).upper()
                pos_slots = int(pos.get("open_slots", 0) or 0)
                if pos_side not in {"LONG", "SHORT"} or not pos_symbol or pos_slots <= 0:
                    continue

                pos_entry_price = float(pos.get("entry_price", 0.0) or 0.0)
                pos_stop_price = float(pos.get("stop_price", 0.0) or 0.0)
                pos_target_price = float(pos.get("target_price", 0.0) or 0.0)
                pos_opened_at = pos.get("opened_at")

                # Get market data for this position's symbol
                pos_market = markets_by_symbol.get(pos_symbol)
                if pos_market is None or pos_market.empty:
                    surviving_positions.append(pos)
                    continue

                setup_snapshot = setup_snapshot_cache.get(pos_symbol)
                if setup_snapshot is None:
                    setup_snapshot = self._latest_setup_snapshot(model, pos_market)
                    setup_snapshot_cache[pos_symbol] = setup_snapshot

                pos_close = float(pd.to_numeric(pos_market["close"], errors="coerce").iloc[-1])
                pos_high = float(pd.to_numeric(pos_market["high"], errors="coerce").iloc[-1])
                pos_low = float(pd.to_numeric(pos_market["low"], errors="coerce").iloc[-1])

                forced_exit_reason = None
                if pos_symbol in BLOCKED_TRADING_SYMBOLS:
                    forced_exit_reason = "BLOCKED_SYMBOL"
                elif pos_symbol in claimed_symbols:
                    forced_exit_reason = "DUPLICATE_SYMBOL"

                hit_exit = False
                exit_reason = forced_exit_reason
                if forced_exit_reason is not None:
                    hit_exit = True
                if pos_side == "LONG":
                    if pos_stop_price > 0 and pos_low <= pos_stop_price:
                        hit_exit = True
                        exit_reason = "STOP"
                    if pos_target_price > 0 and pos_high >= pos_target_price and exit_reason is None:
                        hit_exit = True
                        exit_reason = "TARGET"
                else:
                    if pos_stop_price > 0 and pos_high >= pos_stop_price:
                        hit_exit = True
                        exit_reason = "STOP"
                    if pos_target_price > 0 and pos_low <= pos_target_price and exit_reason is None:
                        hit_exit = True
                        exit_reason = "TARGET"

                if not hit_exit:
                    scalp_exit_reason = self._scalp_invalidation_reason(
                        side=pos_side,
                        signal_value=float(setup_snapshot["signal_value"]),
                        long_votes=int(setup_snapshot["long_votes"]),
                        short_votes=int(setup_snapshot["short_votes"]),
                        required_votes=int(setup_snapshot["required_votes"]),
                        reset_votes=int(setup_snapshot["reset_votes"]),
                        signal_reset_floor=float(setup_snapshot["signal_reset_floor"]),
                    )
                    if scalp_exit_reason is not None:
                        hit_exit = True
                        exit_reason = scalp_exit_reason

                if hit_exit:
                    active_exposures.discard((pos_symbol, pos_side))
                    delta_events.append(
                        {
                            "timestamp": ts,
                            "market_timestamp": ts_key,
                            "executed_at": execution_ts_key,
                            "symbol": pos_symbol,
                            "model_id": model_id,
                            "model_name": result.name,
                            "akce": f"Výstup {pos_side} (-{pos_slots})",
                            "strana": pos_side,
                            "cena": pos_close,
                            "pozice": 0.0,
                            "sloty": 0,
                            "duvod_vystupu": exit_reason or "NEURČENO",
                            "opened_at": pos_opened_at,
                            "entry_price": pos_entry_price,
                            "quantity_slots": pos_slots,
                        }
                    )
                else:
                    surviving_positions.append(pos)
                    claimed_symbols.add(pos_symbol)

            # --- ENTRY logic: evaluate if model has capacity for more positions ---
            total_used_slots = sum(int(p.get("open_slots", 0)) for p in surviving_positions)
            remaining_slot_budget = max(0, 5 - total_used_slots)
            if len(surviving_positions) < MAX_POSITIONS_PER_MODEL and remaining_slot_budget > 0:
                entry_symbol = str(result.symbol).upper()
                # Only evaluate entry on a symbol not already held by this model
                model_held_symbols = {str(p.get("symbol", "")).upper() for p in surviving_positions}
                if entry_symbol not in model_held_symbols:
                    market = markets_by_symbol.get(entry_symbol, run_market)
                    if market is not None and not market.empty:
                        close_price = float(pd.to_numeric(market["close"], errors="coerce").iloc[-1])
                        setup_snapshot = setup_snapshot_cache.get(entry_symbol)
                        if setup_snapshot is None:
                            setup_snapshot = self._latest_setup_snapshot(model, market)
                            setup_snapshot_cache[entry_symbol] = setup_snapshot

                        signal_value = float(setup_snapshot["signal_value"])
                        long_votes = int(setup_snapshot["long_votes"])
                        short_votes = int(setup_snapshot["short_votes"])
                        long_confidence = float(setup_snapshot["long_confidence"])
                        short_confidence = float(setup_snapshot["short_confidence"])
                        required_votes = int(setup_snapshot["required_votes"])
                        reset_votes = int(setup_snapshot["reset_votes"])
                        vol_step = float(setup_snapshot["vol_step"])
                        min_stop_floor = float(setup_snapshot["min_stop_floor"])
                        signal_reset_floor = float(setup_snapshot["signal_reset_floor"])

                        direction = 0
                        confidence = 0.0
                        if long_votes >= required_votes and signal_value > 0 and long_votes > short_votes:
                            direction = 1
                            confidence = long_confidence
                        if short_votes >= required_votes and signal_value < 0 and short_votes > long_votes:
                            direction = -1
                            confidence = short_confidence

                        setup_reset = max(long_votes, short_votes) <= reset_votes or abs(signal_value) <= signal_reset_floor
                        if setup_reset:
                            entry_armed = True
                            current_setup_active = False
                            current_setup_direction = 0
                        else:
                            current_setup_active = direction != 0
                            current_setup_direction = direction

                        setup_just_activated = current_setup_active and (
                            not prev_setup_active or prev_setup_direction != current_setup_direction
                        )

                        if direction != 0:
                            proposed_side = "LONG" if direction > 0 else "SHORT"
                            proposed_symbol = entry_symbol
                            exposure_key = (proposed_symbol, proposed_side)
                            if exposure_key in active_exposures:
                                direction = 0

                        if direction != 0 and entry_armed and setup_just_activated:
                            conviction = max(abs(signal_value), confidence)
                            new_open_slots = int(np.clip(np.ceil(conviction * 5), 1, 5))
                            new_side = "LONG" if direction > 0 else "SHORT"
                            stop_dist = max(min_stop_floor, self._trade_profile(model)[0] * max(vol_step, min_stop_floor))
                            target_dist = max(stop_dist * 1.8, self._trade_profile(model)[1] * max(vol_step, min_stop_floor))
                            if new_side == "LONG":
                                new_stop = close_price * (1.0 - stop_dist)
                                new_target = close_price * (1.0 + target_dist)
                            else:
                                new_stop = close_price * (1.0 + stop_dist)
                                new_target = close_price * (1.0 - target_dist)

                            delta_events.append(
                                {
                                    "timestamp": ts,
                                    "market_timestamp": ts_key,
                                    "executed_at": execution_ts_key,
                                    "symbol": entry_symbol,
                                    "model_id": model_id,
                                    "model_name": result.name,
                                    "akce": f"Vstup {new_side} (+{new_open_slots})",
                                    "strana": new_side,
                                    "cena": close_price,
                                    "pozice": float(direction * new_open_slots * slot_size),
                                    "sloty": new_open_slots,
                                }
                            )
                            active_exposures.add((entry_symbol, new_side))
                            surviving_positions.append(
                                {
                                    "symbol": entry_symbol,
                                    "side": new_side,
                                    "open_slots": new_open_slots,
                                    "entry_price": close_price,
                                    "stop_price": new_stop,
                                    "target_price": new_target,
                                    "opened_at": execution_ts_key,
                                }
                            )
                            entry_armed = False
                            claimed_symbols.add(entry_symbol)

            # --- Build final state ---
            total_slots = sum(int(p.get("open_slots", 0)) for p in surviving_positions)
            net_signed = sum(
                (1 if p.get("side") == "LONG" else -1) * int(p.get("open_slots", 0))
                for p in surviving_positions
            )
            primary = surviving_positions[0] if surviving_positions else {}

            cumulative_trades[model_id] = model_trade_history + delta_events
            trade_events_delta[model_id] = delta_events
            final_open_slots[model_id] = total_slots
            final_positions[model_id] = float(net_signed * slot_size)
            live_model_state[model_id] = {
                "entry_armed": entry_armed,
                "last_bar_ts": ts_key,
                "setup_active": current_setup_active,
                "setup_direction": int(current_setup_direction),
                "symbol": str(primary.get("symbol", result.symbol)).upper(),
                "side": str(primary.get("side", "")).upper(),
                "open_slots": total_slots,
                "entry_price": float(primary.get("entry_price", 0.0) or 0.0),
                "opened_at": primary.get("opened_at"),
                "stop_price": float(primary.get("stop_price", 0.0) or 0.0),
                "target_price": float(primary.get("target_price", 0.0) or 0.0),
                "positions": surviving_positions,
            }

            if surviving_positions:
                pos_entries: list[dict[str, Any]] = []
                for pos in surviving_positions:
                    p_sym = str(pos.get("symbol", "")).upper()
                    p_side = str(pos.get("side", "")).upper()
                    p_slots = int(pos.get("open_slots", 0) or 0)
                    for slot_idx in range(p_slots):
                        pos_entries.append(
                            {
                                "slot": len(pos_entries) + 1,
                                "slots": 1,
                                "symbol": p_sym,
                                "side": p_side,
                                "model_id": model_id,
                                "model_name": result.name,
                                "entry_price": float(pos.get("entry_price", 0.0) or 0.0),
                                "opened_at": pos.get("opened_at"),
                                "stop_price": float(pos.get("stop_price", 0.0) or 0.0),
                                "target_price": float(pos.get("target_price", 0.0) or 0.0),
                            }
                        )
                model_open_positions[model_id] = pos_entries
            else:
                model_open_positions[model_id] = []

        return cumulative_trades, trade_events_delta, final_positions, final_open_slots, model_open_positions, live_model_state

    def _select_candidate_run(self, candidate_runs: list[dict]) -> dict:
        actionable = [item for item in candidate_runs if int(item.get("final_open_slots", 0)) > 0]
        pool = actionable or candidate_runs
        return max(
            pool,
            key=lambda item: (
                bool(item["result"].passed),
                float(item["result"].score),
                float(item.get("opportunity_score", 0.0)),
                abs(float(item.get("final_position", 0.0))),
            ),
        )

    def _shortlist_candidate_symbols(
        self,
        model: ModelSpec,
        candidate_symbols: list[str],
        opportunity_scores: dict[str, float],
        excluded_symbols: set[str] | None = None,
    ) -> list[str]:
        excluded = {str(symbol).upper() for symbol in (excluded_symbols or set())}
        ranked = sorted(
            [symbol for symbol in candidate_symbols if symbol not in BLOCKED_TRADING_SYMBOLS and symbol not in excluded],
            key=lambda symbol: float(opportunity_scores.get(symbol, 0.0)),
            reverse=True,
        )
        shortlist_size = 5
        shortlist = ranked[:shortlist_size]

        previous_symbol = self._model_symbol_memory.get(model.model_id)
        if previous_symbol and previous_symbol in candidate_symbols and previous_symbol not in BLOCKED_TRADING_SYMBOLS and previous_symbol not in excluded and previous_symbol not in shortlist:
            shortlist = [previous_symbol] + shortlist[:-1]

        if model.kind == "mean_reversion" and len(ranked) > shortlist_size:
            tail_candidate = ranked[min(len(ranked) - 1, shortlist_size + 1)]
            if tail_candidate not in shortlist:
                shortlist.append(tail_candidate)

        fallback_candidates = [symbol for symbol in candidate_symbols if symbol not in BLOCKED_TRADING_SYMBOLS]
        return list(dict.fromkeys(shortlist or fallback_candidates[:1]))

    def run_week(
        self,
        days: int = 365,
        market_source: str | None = None,
        symbol: str | None = None,
        interval: str = "1d",
        previous_summary: dict[str, Any] | None = None,
    ) -> dict:
        self.week += 1
        effective_source = market_source or self.config.market_data_source
        effective_symbol = (symbol or self.config.default_symbol).upper()
        market_data_source = "binance" if effective_source == "binance_copy" else effective_source

        long_tail, candidate_symbols, markets_by_symbol, latest_prices = self._get_live_market_snapshot(
            market_source=market_data_source,
            days=days,
            interval=interval,
            fallback_symbol=effective_symbol,
        )
        opportunity_scores = {}
        if isinstance(long_tail, pd.DataFrame) and {"symbol", "opportunity_score"}.issubset(long_tail.columns):
            opportunity_scores = {
                str(row["symbol"]).upper(): float(row["opportunity_score"])
                for _, row in long_tail[["symbol", "opportunity_score"]].iterrows()
            }

        copy_snapshot = None
        if isinstance(previous_summary, dict):
            previous_positions = previous_summary.get("model_open_positions", {})
            if isinstance(previous_positions, dict):
                previous_symbols = [
                    str(position.get("symbol", "")).upper()
                    for positions in previous_positions.values()
                    if isinstance(positions, list)
                    for position in positions
                    if isinstance(position, dict)
                ]
                for previous_symbol in previous_symbols:
                    if not previous_symbol:
                        continue
                    if previous_symbol not in candidate_symbols:
                        candidate_symbols.append(previous_symbol)
                    if previous_symbol not in markets_by_symbol:
                        market = get_market_data(
                            source=market_data_source,
                            days=days,
                            symbol=previous_symbol,
                            seed=self._market_seed(previous_symbol, offset=len(markets_by_symbol)),
                            interval=interval,
                        )
                        markets_by_symbol[previous_symbol] = market
                        if not market.empty and "close" in market.columns:
                            latest_prices[previous_symbol] = float(pd.to_numeric(market["close"], errors="coerce").dropna().iloc[-1])

        if effective_source in {"binance", "binance_copy"}:
            copy_snapshot = load_top_copy_trader_snapshot(
                allow_shorts=self.config.allow_shorts and self.config.allow_leverage,
                allow_leverage=self.config.allow_leverage,
            )
            if isinstance(copy_snapshot, dict):
                position_symbols = [
                    str(position.get("symbol", "")).upper()
                    for position in copy_snapshot.get("positions", [])
                    if isinstance(position, dict)
                ]
                for position_symbol in position_symbols:
                    if not position_symbol:
                        continue
                    if position_symbol not in candidate_symbols:
                        candidate_symbols.append(position_symbol)
                    if position_symbol in markets_by_symbol:
                        continue
                    market = get_market_data(
                        source="binance",
                        days=days,
                        symbol=position_symbol,
                        seed=self._market_seed(position_symbol, offset=len(markets_by_symbol)),
                        interval=interval,
                    )
                    markets_by_symbol[position_symbol] = market
                    if not market.empty and "close" in market.columns:
                        latest_prices[position_symbol] = float(pd.to_numeric(market["close"], errors="coerce").dropna().iloc[-1])

        selected_runs: list[dict] = []
        models_to_run = list(self.models)
        reserved_symbols: set[str] = set()
        previous_live_states = {
            model.model_id: self._previous_live_state(previous_summary, model.model_id)
            for model in models_to_run
        }

        # Reserve ALL symbols held in any model's positions (multi-position aware)
        for state in previous_live_states.values():
            if not isinstance(state, dict):
                continue
            for pos in state.get("positions", []):
                if not isinstance(pos, dict):
                    continue
                p_sym = str(pos.get("symbol", "")).upper()
                p_slots = int(pos.get("open_slots", 0) or 0)
                if p_sym and p_slots > 0:
                    reserved_symbols.add(p_sym)
            # Fallback: old single-position format
            if not state.get("positions"):
                if int(state.get("open_slots", 0) or 0) > 0:
                    previous_symbol = str(state.get("symbol", "")).upper()
                    if previous_symbol:
                        reserved_symbols.add(previous_symbol)

        for model in models_to_run:
            previous_state = previous_live_states.get(model.model_id, {})
            if model.kind == "copy_trader":
                chosen_run = self._build_copy_trader_run(
                    model=model,
                    snapshot=copy_snapshot,
                    markets_by_symbol=markets_by_symbol,
                    fallback_symbol=effective_symbol,
                )
                self._model_symbol_memory[model.model_id] = str(chosen_run["result"].symbol).upper()
                selected_runs.append(chosen_run)
                for position in chosen_run.get("open_positions_override", []):
                    if isinstance(position, dict):
                        symbol_value = str(position.get("symbol", "")).upper()
                        if symbol_value:
                            reserved_symbols.add(symbol_value)
                result_symbol = str(chosen_run["result"].symbol).upper()
                if result_symbol:
                    reserved_symbols.add(result_symbol)
                continue

            candidate_runs: list[dict] = []

            # Collect symbols this model already holds (multi-position)
            model_held_symbols: set[str] = set()
            for pos in previous_state.get("positions", []):
                if isinstance(pos, dict):
                    p_sym = str(pos.get("symbol", "")).upper()
                    p_slots = int(pos.get("open_slots", 0) or 0)
                    if p_sym and p_slots > 0:
                        model_held_symbols.add(p_sym)
            # Fallback: old single-position format
            if not model_held_symbols and isinstance(previous_state, dict) and int(previous_state.get("open_slots", 0) or 0) > 0:
                p_sym = str(previous_state.get("symbol", "")).upper()
                if p_sym:
                    model_held_symbols.add(p_sym)

            # If model is at max capacity, pin to primary symbol (no new entry possible)
            if len(model_held_symbols) >= MAX_POSITIONS_PER_MODEL:
                primary_symbol = str(previous_state.get("symbol", "")).upper()
                if primary_symbol and primary_symbol in markets_by_symbol:
                    shortlist_symbols = [primary_symbol]
                else:
                    shortlist_symbols = list(model_held_symbols)[:1]
            else:
                # Choose a NEW symbol not already held by this model
                model_exclusions = reserved_symbols | model_held_symbols
                shortlist_symbols = self._shortlist_candidate_symbols(
                    model,
                    candidate_symbols,
                    opportunity_scores,
                    excluded_symbols=model_exclusions,
                )
                if not shortlist_symbols:
                    shortlist_symbols = self._shortlist_candidate_symbols(model, candidate_symbols, opportunity_scores)
            for candidate_symbol in shortlist_symbols:
                if candidate_symbol in BLOCKED_TRADING_SYMBOLS:
                    continue
                market = markets_by_symbol[candidate_symbol]
                result, events, final_position, final_open_slots = self._simulate_model(model, market, symbol=candidate_symbol)
                candidate_runs.append(
                    {
                        "result": result,
                        "events": events,
                        "final_position": final_position,
                        "final_open_slots": final_open_slots,
                        "market": market,
                        "opportunity_score": float(opportunity_scores.get(candidate_symbol, 0.0)),
                    }
                )
            if not candidate_runs:
                continue
            chosen_run = self._select_candidate_run(candidate_runs)
            self._model_symbol_memory[model.model_id] = str(chosen_run["result"].symbol).upper()
            selected_runs.append(chosen_run)
            chosen_symbol = str(chosen_run["result"].symbol).upper()
            if chosen_symbol:
                reserved_symbols.add(chosen_symbol)

        results = [run["result"] for run in selected_runs]
        model_trades = {run["result"].model_id: run["events"] for run in selected_runs}
        trade_events_delta = {run["result"].model_id: list(run["events"]) for run in selected_runs}
        final_positions = {run["result"].model_id: run["final_position"] for run in selected_runs}
        final_open_slots = {run["result"].model_id: run["final_open_slots"] for run in selected_runs}
        model_markets = {run["result"].model_id: run["market"] for run in selected_runs}
        results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("score", ascending=False)
        model_selected_symbols = {
            str(row["model_id"]): str(row.get("symbol", "")).upper()
            for _, row in results_df.iterrows()
        }

        champion = results_df.iloc[0].to_dict()
        champion_score = float(champion.get("score", 1.0))
        champion["reward_usd"] = max(1.0, champion_score * 10.0)
        champion_model_id = str(champion["model_id"])
        champion_symbol = str(champion.get("symbol", effective_symbol)).upper()
        champion_market = model_markets.get(champion_model_id, markets_by_symbol.get(champion_symbol))

        live_model_state: dict[str, dict[str, Any]] = {}
        if effective_source in {"binance", "binance_copy"}:
            (
                model_trades,
                trade_events_delta,
                final_positions,
                final_open_slots,
                model_open_positions,
                live_model_state,
            ) = self._build_incremental_live_state(
                selected_runs=selected_runs,
                previous_summary=previous_summary,
                markets_by_symbol=markets_by_symbol,
                effective_symbol=effective_symbol,
            )
            for model_id, state in live_model_state.items():
                if not isinstance(state, dict):
                    continue
                active_symbol = str(state.get("symbol", "")).upper()
                if not active_symbol:
                    continue
                model_selected_symbols[str(model_id)] = active_symbol
                if active_symbol in markets_by_symbol:
                    model_markets[str(model_id)] = markets_by_symbol[active_symbol]
            if not results_df.empty and "model_id" in results_df.columns:
                results_df["symbol"] = results_df["model_id"].astype(str).map(model_selected_symbols).fillna(results_df["symbol"])
                refreshed_champion_row = results_df[results_df["model_id"].astype(str) == champion_model_id]
                if not refreshed_champion_row.empty:
                    champion = refreshed_champion_row.iloc[0].to_dict()
                    champion["reward_usd"] = max(1.0, float(champion.get("score", champion_score or 1.0)) * 10.0)
                    champion_symbol = str(champion.get("symbol", effective_symbol)).upper()
                    champion_market = model_markets.get(champion_model_id, markets_by_symbol.get(champion_symbol))
        else:
            model_open_positions = self._build_model_open_positions(
                results_df=results_df,
                final_positions=final_positions,
                final_open_slots=final_open_slots,
            )

        research: list[StudyInsight] = daily_deep_research(seed=10_000 + self.week)
        for run in selected_runs:
            open_positions_override = run.get("open_positions_override", [])
            if open_positions_override:
                model_open_positions[str(run["result"].model_id)] = list(open_positions_override)
        order_source = results_df.copy()
        order_source["model_open_positions"] = order_source["model_id"].map(model_open_positions)
        proposed_orders = build_dry_run_orders(
            order_source,
            symbol=effective_symbol,
            trade_size_czk=PAPER_TRADE_SIZE_CZK,
        )

        if self.week % self.config.generation_horizon_weeks == 0:
            self._evolve_generation(results_df)

        summary = {
            "week": self.week,
            "generation": self.generation,
            "portfolio_vol_annual": float(annualized_vol(champion_market["ret"])) if champion_market is not None else 0.0,
            "market_source": effective_source,
            "symbol": champion_symbol,
            "interval": interval,
            "champion": champion,
            "results": results_df,
            "market": champion_market,
            "model_markets": model_markets,
            "model_trades": model_trades,
            "champion_trades": model_trades.get(champion_model_id, []),
            "trade_events_delta": trade_events_delta,
            "final_positions": final_positions,
            "final_open_slots": final_open_slots,
            "model_open_positions": model_open_positions,
            "live_model_state": live_model_state,
            "model_selected_symbols": model_selected_symbols,
            "candidate_symbols": candidate_symbols,
            "latest_prices": latest_prices,
            "research": research,
            "long_tail": long_tail,
            "proposed_orders": [o.__dict__ for o in proposed_orders],
        }
        return summary

    def _evolve_generation(self, leaderboard: pd.DataFrame) -> None:
        self.generation += 1

        def _model_id(base_id: str) -> str:
            namespace = self.model_namespace.strip().upper()
            return f"{namespace}_{base_id}" if namespace else base_id

        prefix = f"{self.model_label_prefix.strip()} | " if self.model_label_prefix.strip() else ""

        model_by_id = {model.model_id: model for model in self.models if model.kind != "copy_trader"}
        scored_rows: list[tuple[str, float, str]] = []
        if isinstance(leaderboard, pd.DataFrame) and {"model_id", "score"}.issubset(leaderboard.columns):
            for _, row in leaderboard.iterrows():
                model_id = str(row["model_id"])
                model = model_by_id.get(model_id)
                if model is None:
                    continue
                scored_rows.append((model.kind, float(row.get("score", 0.0)), model.name))

        best_name_by_kind: dict[str, str] = {}
        for kind, _score, model_name in sorted(scored_rows, key=lambda item: item[1], reverse=True):
            if kind in best_name_by_kind:
                continue
            best_name_by_kind[kind] = str(model_name)

        standard_models = [
            ModelSpec(
                _model_id(base_id),
                best_name_by_kind.get(kind, f"{prefix}{default_name}"),
                kind,
                self.generation,
            )
            for base_id, default_name, kind in STANDARD_MODEL_BLUEPRINTS
        ]

        copy_anchor = [
            ModelSpec(_model_id("MC"), f"{prefix}Kopie lead tradera Binance", "copy_trader", self.generation),
        ]

        self.models = standard_models + copy_anchor
