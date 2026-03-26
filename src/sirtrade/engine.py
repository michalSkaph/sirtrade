from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig, DEFAULT_CONFIG, PAPER_TRADE_SIZE_CZK
from .copy_trading import load_top_copy_trader_snapshot
from .data import get_market_data, scan_binance_long_tail, scan_long_tail_opportunities
from .execution import build_dry_run_orders
from .models import ModelSpec, default_model_specs, generate_signals
from .research import StudyInsight, daily_deep_research
from .risk import annualized_vol, apply_risk_controls, cvar95, max_drawdown
from .scoring import calmar_ratio, decision_score, pass_thresholds, sortino_ratio


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
            long_tail = scan_binance_long_tail(top_n=20)
        else:
            long_tail = scan_long_tail_opportunities(seed=self.week, universe_size=300).head(20)

        candidate_symbols: list[str] = []
        if isinstance(long_tail, pd.DataFrame) and "symbol" in long_tail.columns:
            candidate_symbols = [str(value).upper() for value in long_tail["symbol"].dropna().tolist()]

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
        if len(capped_positions) == slot_budget:
            return [1] * slot_budget

        notionals = [max(0.0, float(position.get("notional_usd", 0.0))) for position in capped_positions]
        total_notional = sum(notionals)
        if total_notional <= 0:
            return [1] + [0] * (len(capped_positions) - 1)

        raw_allocations = [(notional / total_notional) * slot_budget for notional in notionals]
        base_allocations = [int(value) for value in raw_allocations]
        remainders = [value - int(value) for value in raw_allocations]

        allocated = sum(base_allocations)
        while allocated < slot_budget:
            next_index = max(range(len(remainders)), key=lambda idx: remainders[idx])
            base_allocations[next_index] += 1
            remainders[next_index] = 0.0
            allocated += 1

        while allocated > slot_budget:
            next_index = max(range(len(base_allocations)), key=lambda idx: base_allocations[idx])
            if base_allocations[next_index] <= 0:
                break
            base_allocations[next_index] -= 1
            allocated -= 1

        if all(value == 0 for value in base_allocations):
            base_allocations[0] = 1
        return base_allocations

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
        slot_size = self.config.risk.max_asset_exposure / 5
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

        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()
        ema_anchor = close.ewm(span=55, adjust=False).mean()
        momentum_fast = returns.rolling(3).mean().fillna(0.0)
        momentum_slow = returns.rolling(8).mean().fillna(0.0)
        breakout = close.pct_change(5).fillna(0.0)
        rolling_mean = close.rolling(20).mean().bfill()
        rolling_std = close.rolling(20).std(ddof=0).replace(0.0, np.nan)
        price_z = ((close - rolling_mean) / (rolling_std + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        overlay_bias = ((0.6 * onchain) + (0.4 * sentiment)).clip(-3.0, 3.0)
        upper_band = rolling_mean + (1.4 * rolling_std.fillna(0.0))
        lower_band = rolling_mean - (1.4 * rolling_std.fillna(0.0))

        delta = close.diff().fillna(0.0)
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = gains.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = (100 - (100 / (1 + rs))).fillna(50.0)

        macd_fast = close.ewm(span=12, adjust=False).mean()
        macd_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = macd_fast - macd_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
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
        atr = true_range.rolling(14).mean().fillna(true_range.expanding().mean()).fillna(0.0)
        atr_pct = (atr / (close.abs() + 1e-6)).clip(lower=0.0015, upper=0.08)

        donchian_high = high.rolling(20).max().shift(1).fillna(high.expanding().max())
        donchian_low = low.rolling(20).min().shift(1).fillna(low.expanding().min())
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
            long_checks = [
                controlled_signal >= 0.16,
                price_z <= -1.35,
                close <= lower_band,
                rsi <= 36,
                acceleration_up,
            ]
            short_checks = [
                controlled_signal <= -0.16,
                price_z >= 1.35,
                close >= upper_band,
                rsi >= 64,
                acceleration_down,
            ]
            required_votes = 4
            reset_votes = 2
        elif model.kind == "onchain_sentiment_overlay":
            long_checks = [
                controlled_signal >= 0.18,
                overlay_bias >= 0.25,
                trend_up,
                momentum_up,
                macd_hist >= 0,
            ]
            short_checks = [
                controlled_signal <= -0.18,
                overlay_bias <= -0.25,
                trend_down,
                momentum_down,
                macd_hist <= 0,
            ]
            required_votes = 4
            reset_votes = 2
        elif model.kind == "xsec_momentum":
            long_checks = [
                controlled_signal >= 0.24,
                momentum_up,
                acceleration_up,
                breakout_up | (breakout > 0),
                trend_up,
                broad_trend_up,
                macd_hist > 0,
            ]
            short_checks = [
                controlled_signal <= -0.24,
                momentum_down,
                acceleration_down,
                breakout_down | (breakout < 0),
                trend_down,
                broad_trend_down,
                macd_hist < 0,
            ]
            required_votes = 5
            reset_votes = 2
        elif model.kind == "meta_ensemble":
            long_checks = [
                controlled_signal >= 0.20,
                trend_up,
                momentum_up,
                acceleration_up,
                overlay_bias >= -0.10,
                macd_hist >= 0,
                rsi.between(50, 72),
            ]
            short_checks = [
                controlled_signal <= -0.20,
                trend_down,
                momentum_down,
                acceleration_down,
                overlay_bias <= 0.10,
                macd_hist <= 0,
                rsi.between(28, 50),
            ]
            required_votes = 5
            reset_votes = 2
        else:
            long_checks = [
                controlled_signal >= 0.22,
                trend_up,
                broad_trend_up,
                momentum_up,
                acceleration_up,
                breakout_up | (breakout > 0),
                macd_hist > 0,
            ]
            short_checks = [
                controlled_signal <= -0.22,
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
        if model.kind == "mean_reversion":
            return 1.35, 2.75, 0.0025, 0.08
        if model.kind == "xsec_momentum":
            return 1.85, 3.90, 0.0030, 0.10
        if model.kind == "onchain_sentiment_overlay":
            return 1.70, 3.40, 0.0030, 0.09
        if model.kind == "meta_ensemble":
            return 1.95, 4.10, 0.0032, 0.09
        return 2.10, 4.40, 0.0035, 0.10

    def _build_trade_events(self, model: ModelSpec, prices: pd.Series, position: pd.Series) -> tuple[list[dict], int]:
        events: list[dict] = []
        slot_size = self.config.risk.max_asset_exposure / 5
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

        slot_size = self.config.risk.max_asset_exposure / 5
        warmup_bars = min(48, max(12, int(len(market) * 0.1)))
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
                    conviction = max(abs(signal_value), confidence)
                    slots = int(np.clip(np.ceil(conviction * 5), 1, 5))
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
                return dict(state)

        model_positions = previous_summary.get("model_open_positions", {})
        if not isinstance(model_positions, dict):
            return {}
        positions = model_positions.get(model_id, [])
        if not isinstance(positions, list) or not positions:
            return {}

        first_position = next((item for item in positions if isinstance(item, dict)), None)
        if not isinstance(first_position, dict):
            return {}

        return {
            "entry_armed": False,
            "symbol": str(first_position.get("symbol", "")).upper(),
            "side": str(first_position.get("side", "")).upper(),
            "open_slots": int(len([item for item in positions if isinstance(item, dict)])),
            "entry_price": float(first_position.get("entry_price", 0.0) or 0.0),
            "opened_at": first_position.get("opened_at"),
            "stop_price": float(first_position.get("stop_price", 0.0) or 0.0),
            "target_price": float(first_position.get("target_price", 0.0) or 0.0),
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
        slot_size = self.config.risk.max_asset_exposure / 5

        cumulative_trades: dict[str, list[dict[str, Any]]] = {}
        trade_events_delta: dict[str, list[dict[str, Any]]] = {}
        final_positions: dict[str, float] = {}
        final_open_slots: dict[str, int] = {}
        model_open_positions: dict[str, list[dict[str, Any]]] = {}
        live_model_state: dict[str, dict[str, Any]] = {}

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
                }
                continue

            if model is None:
                cumulative_trades[model_id] = model_trade_history
                trade_events_delta[model_id] = []
                final_positions[model_id] = 0.0
                final_open_slots[model_id] = 0
                model_open_positions[model_id] = []
                live_model_state[model_id] = {"entry_armed": True}
                continue

            active_symbol = str(previous_state.get("symbol") or result.symbol or effective_symbol).upper()
            market = markets_by_symbol.get(active_symbol, run.get("market"))
            if market is None or market.empty:
                cumulative_trades[model_id] = model_trade_history
                trade_events_delta[model_id] = []
                final_positions[model_id] = 0.0
                final_open_slots[model_id] = 0
                model_open_positions[model_id] = []
                live_model_state[model_id] = {"entry_armed": True}
                continue

            ts = market.index[-1]
            ts_key = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            prev_last_bar_ts = str(previous_state.get("last_bar_ts", ""))
            bar_is_new = ts_key != prev_last_bar_ts

            close_price = float(pd.to_numeric(market["close"], errors="coerce").iloc[-1])
            high_price = float(pd.to_numeric(market["high"], errors="coerce").iloc[-1])
            low_price = float(pd.to_numeric(market["low"], errors="coerce").iloc[-1])

            entry_armed = bool(previous_state.get("entry_armed", True))
            prev_open_slots = int(previous_state.get("open_slots", 0) or 0)
            prev_side = str(previous_state.get("side", "")).upper()
            prev_entry_price = float(previous_state.get("entry_price", 0.0) or 0.0)
            prev_stop_price = float(previous_state.get("stop_price", 0.0) or 0.0)
            prev_target_price = float(previous_state.get("target_price", 0.0) or 0.0)
            prev_opened_at = previous_state.get("opened_at")

            current_side = prev_side if prev_open_slots > 0 else ""
            current_open_slots = prev_open_slots
            current_entry_price = prev_entry_price
            current_stop_price = prev_stop_price
            current_target_price = prev_target_price
            current_opened_at = prev_opened_at
            current_symbol = active_symbol if prev_open_slots > 0 else str(result.symbol).upper()

            # --- EXIT logic: always evaluate (stop/target can be hit intra-bar) ---
            if current_open_slots > 0 and current_side in {"LONG", "SHORT"}:
                hit_exit = False
                exit_reason = None
                if current_side == "LONG":
                    if current_stop_price > 0 and low_price <= current_stop_price:
                        hit_exit = True
                        exit_reason = "STOP"
                    if current_target_price > 0 and high_price >= current_target_price and exit_reason is None:
                        hit_exit = True
                        exit_reason = "TARGET"
                else:
                    if current_stop_price > 0 and high_price >= current_stop_price:
                        hit_exit = True
                        exit_reason = "STOP"
                    if current_target_price > 0 and low_price <= current_target_price and exit_reason is None:
                        hit_exit = True
                        exit_reason = "TARGET"

                if hit_exit:
                    delta_events.append(
                        {
                            "timestamp": ts,
                            "symbol": current_symbol,
                            "model_id": model_id,
                            "model_name": result.name,
                            "akce": f"Výstup {current_side} (-{current_open_slots})",
                            "strana": current_side,
                            "cena": close_price,
                            "pozice": 0.0,
                            "sloty": 0,
                            "duvod_vystupu": exit_reason or "NEURČENO",
                            "opened_at": current_opened_at,
                            "entry_price": current_entry_price,
                            "quantity_slots": current_open_slots,
                        }
                    )
                    current_side = ""
                    current_open_slots = 0
                    current_entry_price = 0.0
                    current_stop_price = 0.0
                    current_target_price = 0.0
                    current_opened_at = None
                    current_symbol = str(result.symbol).upper()
                    entry_armed = False

            # --- ENTRY logic: only on NEW bars (closed candle boundary) ---
            elif bar_is_new:
                raw = generate_signals(model, market, seed=0)
                controlled_signal = apply_risk_controls(raw, market["ret"], self.config.risk)
                confluence, required_votes, reset_votes, atr_pct = self._build_entry_confluence(model, market, controlled_signal)
                _, _, min_stop_floor, signal_reset_floor = self._trade_profile(model)

                signal_value = float(controlled_signal.loc[ts])
                long_votes = int(confluence.loc[ts, "long_votes"])
                short_votes = int(confluence.loc[ts, "short_votes"])
                long_confidence = float(confluence.loc[ts, "long_confidence"])
                short_confidence = float(confluence.loc[ts, "short_confidence"])
                vol_step = float(atr_pct.loc[ts])
                if np.isnan(vol_step) or vol_step <= 0:
                    vol_step = min_stop_floor

                if not entry_armed:
                    setup_reset = max(long_votes, short_votes) <= reset_votes or abs(signal_value) <= signal_reset_floor
                    if setup_reset:
                        entry_armed = True

                direction = 0
                confidence = 0.0
                if entry_armed and long_votes >= required_votes and signal_value > 0 and long_votes > short_votes:
                    direction = 1
                    confidence = long_confidence
                if entry_armed and short_votes >= required_votes and signal_value < 0 and short_votes > long_votes:
                    direction = -1
                    confidence = short_confidence

                if direction != 0:
                    conviction = max(abs(signal_value), confidence)
                    current_open_slots = int(np.clip(np.ceil(conviction * 5), 1, 5))
                    current_side = "LONG" if direction > 0 else "SHORT"
                    current_entry_price = close_price
                    current_opened_at = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    current_symbol = str(result.symbol).upper()
                    stop_dist = max(min_stop_floor, self._trade_profile(model)[0] * max(vol_step, min_stop_floor))
                    target_dist = max(stop_dist * 1.8, self._trade_profile(model)[1] * max(vol_step, min_stop_floor))
                    if current_side == "LONG":
                        current_stop_price = close_price * (1.0 - stop_dist)
                        current_target_price = close_price * (1.0 + target_dist)
                    else:
                        current_stop_price = close_price * (1.0 + stop_dist)
                        current_target_price = close_price * (1.0 - target_dist)

                    delta_events.append(
                        {
                            "timestamp": ts,
                            "symbol": current_symbol,
                            "model_id": model_id,
                            "model_name": result.name,
                            "akce": f"Vstup {current_side} (+{current_open_slots})",
                            "strana": current_side,
                            "cena": close_price,
                            "pozice": float(direction * current_open_slots * slot_size),
                            "sloty": current_open_slots,
                        }
                    )
                    entry_armed = False

            cumulative_trades[model_id] = model_trade_history + delta_events
            trade_events_delta[model_id] = delta_events
            final_open_slots[model_id] = int(current_open_slots)
            side_sign = 1.0 if current_side == "LONG" else (-1.0 if current_side == "SHORT" else 0.0)
            final_positions[model_id] = float(side_sign * current_open_slots * slot_size)
            live_model_state[model_id] = {
                "entry_armed": entry_armed,
                "last_bar_ts": ts_key,
                "symbol": current_symbol,
                "side": current_side,
                "open_slots": int(current_open_slots),
                "entry_price": float(current_entry_price),
                "opened_at": current_opened_at,
                "stop_price": float(current_stop_price),
                "target_price": float(current_target_price),
            }

            if current_open_slots > 0 and current_side in {"LONG", "SHORT"} and current_symbol:
                model_open_positions[model_id] = [
                    {
                        "slot": slot_idx + 1,
                        "slots": 1,
                        "symbol": current_symbol,
                        "side": current_side,
                        "model_id": model_id,
                        "model_name": result.name,
                        "entry_price": float(current_entry_price),
                        "opened_at": current_opened_at,
                        "stop_price": float(current_stop_price),
                        "target_price": float(current_target_price),
                    }
                    for slot_idx in range(current_open_slots)
                ]
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
    ) -> list[str]:
        ranked = sorted(
            candidate_symbols,
            key=lambda symbol: float(opportunity_scores.get(symbol, 0.0)),
            reverse=True,
        )
        shortlist_size = 5
        shortlist = ranked[:shortlist_size]

        previous_symbol = self._model_symbol_memory.get(model.model_id)
        if previous_symbol and previous_symbol in candidate_symbols and previous_symbol not in shortlist:
            shortlist = [previous_symbol] + shortlist[:-1]

        if model.kind == "mean_reversion" and len(ranked) > shortlist_size:
            tail_candidate = ranked[min(len(ranked) - 1, shortlist_size + 1)]
            if tail_candidate not in shortlist:
                shortlist.append(tail_candidate)

        return list(dict.fromkeys(shortlist or candidate_symbols[:1]))

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

        for model in models_to_run:
            if model.kind == "copy_trader":
                chosen_run = self._build_copy_trader_run(
                    model=model,
                    snapshot=copy_snapshot,
                    markets_by_symbol=markets_by_symbol,
                    fallback_symbol=effective_symbol,
                )
                self._model_symbol_memory[model.model_id] = str(chosen_run["result"].symbol).upper()
                selected_runs.append(chosen_run)
                continue

            candidate_runs: list[dict] = []
            shortlist_symbols = self._shortlist_candidate_symbols(model, candidate_symbols, opportunity_scores)
            for candidate_symbol in shortlist_symbols:
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
            chosen_run = self._select_candidate_run(candidate_runs)
            self._model_symbol_memory[model.model_id] = str(chosen_run["result"].symbol).upper()
            selected_runs.append(chosen_run)

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
        top = leaderboard.head(2)
        carry_ids = set(top["model_id"].tolist())

        keep = [m for m in self.models if m.model_id in carry_ids and m.kind != "copy_trader"]

        def _model_id(base_id: str) -> str:
            namespace = self.model_namespace.strip().upper()
            return f"{namespace}_{base_id}" if namespace else base_id

        prefix = f"{self.model_label_prefix.strip()} | " if self.model_label_prefix.strip() else ""

        offspring_pool = [
            ModelSpec(_model_id("M6"), f"{prefix}Potomek A (mutovaný trend)", "trend_vol", self.generation),
            ModelSpec(_model_id("M7"), f"{prefix}Potomek B (mutované momentum)", "xsec_momentum", self.generation),
            ModelSpec(_model_id("M8"), f"{prefix}Potomek C (mutovaný meta model)", "meta_ensemble", self.generation),
            ModelSpec(_model_id("M9"), f"{prefix}Stabilizační kotva MR", "mean_reversion", self.generation),
            ModelSpec(_model_id("M10"), f"{prefix}Stabilizační kotva overlay", "onchain_sentiment_overlay", self.generation),
        ]
        children_needed = max(0, 5 - len(keep))
        children = offspring_pool[:children_needed]

        copy_anchor = [
            ModelSpec(_model_id("MC"), f"{prefix}Kopie lead tradera Binance", "copy_trader", self.generation),
        ]

        self.models = keep + children + copy_anchor
