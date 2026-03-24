from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
import pandas as pd

from .config import AppConfig, DEFAULT_CONFIG
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
        if market_source == "binance":
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
        cache_ttl_seconds = 15.0 if market_source == "binance" else 5.0
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

    def _build_entry_confluence(
        self,
        model: ModelSpec,
        market: pd.DataFrame,
        controlled_signal: pd.Series,
    ) -> tuple[pd.DataFrame, int]:
        close = market["close"].astype(float)
        returns = market["ret"].fillna(0.0).astype(float)
        sentiment = market.get("sentiment", pd.Series(0.0, index=market.index)).fillna(0.0).astype(float)
        onchain = market.get("onchain", pd.Series(0.0, index=market.index)).fillna(0.0).astype(float)

        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()
        momentum_fast = returns.rolling(3).mean().fillna(0.0)
        momentum_slow = returns.rolling(8).mean().fillna(0.0)
        breakout = close.pct_change(5).fillna(0.0)
        rolling_mean = close.rolling(20).mean().bfill()
        rolling_std = close.rolling(20).std(ddof=0).replace(0.0, np.nan)
        price_z = ((close - rolling_mean) / (rolling_std + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        overlay_bias = ((0.6 * onchain) + (0.4 * sentiment)).clip(-3.0, 3.0)

        trend_up = (ema_fast > ema_slow) & (close >= ema_fast)
        trend_down = (ema_fast < ema_slow) & (close <= ema_fast)
        momentum_up = momentum_fast > 0
        momentum_down = momentum_fast < 0
        acceleration_up = momentum_fast >= momentum_slow
        acceleration_down = momentum_fast <= momentum_slow

        if model.kind == "mean_reversion":
            long_checks = [
                controlled_signal >= 0.14,
                price_z <= -1.0,
                close <= rolling_mean,
                acceleration_up,
            ]
            short_checks = [
                controlled_signal <= -0.14,
                price_z >= 1.0,
                close >= rolling_mean,
                acceleration_down,
            ]
            required_votes = 3
        elif model.kind == "onchain_sentiment_overlay":
            long_checks = [
                controlled_signal >= 0.18,
                overlay_bias >= 0.15,
                trend_up,
                momentum_up,
            ]
            short_checks = [
                controlled_signal <= -0.18,
                overlay_bias <= -0.15,
                trend_down,
                momentum_down,
            ]
            required_votes = 3
        elif model.kind == "xsec_momentum":
            long_checks = [
                controlled_signal >= 0.20,
                momentum_up,
                acceleration_up,
                breakout > 0,
                trend_up,
            ]
            short_checks = [
                controlled_signal <= -0.20,
                momentum_down,
                acceleration_down,
                breakout < 0,
                trend_down,
            ]
            required_votes = 4
        elif model.kind == "meta_ensemble":
            long_checks = [
                controlled_signal >= 0.20,
                trend_up,
                momentum_up,
                acceleration_up,
                overlay_bias >= -0.10,
            ]
            short_checks = [
                controlled_signal <= -0.20,
                trend_down,
                momentum_down,
                acceleration_down,
                overlay_bias <= 0.10,
            ]
            required_votes = 4
        else:
            long_checks = [
                controlled_signal >= 0.18,
                trend_up,
                momentum_up,
                acceleration_up,
                breakout > 0,
            ]
            short_checks = [
                controlled_signal <= -0.18,
                trend_down,
                momentum_down,
                acceleration_down,
                breakout < 0,
            ]
            required_votes = 4

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
        return confluence, required_votes

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
        confluence, required_votes = self._build_entry_confluence(model, market, controlled_signal)

        close = market["close"].astype(float)
        high = market["high"].astype(float)
        low = market["low"].astype(float)
        ret_std = market["ret"].fillna(0.0).rolling(20).std().fillna(0.0)

        slot_size = self.config.risk.max_asset_exposure / 5
        warmup_bars = min(48, max(12, int(len(market) * 0.1)))
        stop_multiplier = 1.15 if model.kind == "mean_reversion" else 1.05
        target_multiplier = 2.20 if model.kind == "mean_reversion" else 1.90

        pos = pd.Series(0.0, index=market.index, dtype=float)
        side = 0
        position_size = 0.0
        current_slots = 0
        stop_price = None
        target_price = None
        events: list[dict] = []

        for step, ts in enumerate(market.index):
            signal_value = float(controlled_signal.loc[ts])
            long_votes = int(confluence.loc[ts, "long_votes"])
            short_votes = int(confluence.loc[ts, "short_votes"])
            long_confidence = float(confluence.loc[ts, "long_confidence"])
            short_confidence = float(confluence.loc[ts, "short_confidence"])
            close_price = float(close.loc[ts])
            high_price = float(high.loc[ts])
            low_price = float(low.loc[ts])
            vol_step = float(ret_std.loc[ts])
            if np.isnan(vol_step) or vol_step <= 0:
                vol_step = 0.01

            if step < warmup_bars:
                pos.loc[ts] = 0.0
                continue

            if side == 0:
                direction = 0
                confidence = 0.0
                if long_votes >= required_votes and signal_value > 0:
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

                    stop_dist = max(0.004, stop_multiplier * max(vol_step, 0.004))
                    target_dist = max(stop_dist * 1.5, target_multiplier * max(vol_step, 0.004))
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
                    "slot": slot_idx + 1,
                    "symbol": model_symbol,
                    "side": side,
                    "model_id": model_id,
                    "model_name": model_name,
                }
                for slot_idx in range(slots)
            ]

        return model_open_positions

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
    ) -> dict:
        self.week += 1
        effective_source = market_source or self.config.market_data_source
        effective_symbol = (symbol or self.config.default_symbol).upper()

        long_tail, candidate_symbols, markets_by_symbol, latest_prices = self._get_live_market_snapshot(
            market_source=effective_source,
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

        selected_runs: list[dict] = []
        for model in self.models:
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

        research: list[StudyInsight] = daily_deep_research(seed=10_000 + self.week)
        model_open_positions = self._build_model_open_positions(
            results_df=results_df,
            final_positions=final_positions,
            final_open_slots=final_open_slots,
        )
        order_source = results_df.copy()
        order_source["model_open_positions"] = order_source["model_id"].map(model_open_positions)
        proposed_orders = build_dry_run_orders(order_source, symbol=effective_symbol, nav_usd=1000.0)

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
            "final_positions": final_positions,
            "final_open_slots": final_open_slots,
            "model_open_positions": model_open_positions,
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

        keep = [m for m in self.models if m.model_id in carry_ids]

        children = [
            ModelSpec("M6", "Potomek A (mutovaný trend)", "trend_vol", self.generation),
            ModelSpec("M7", "Potomek B (mutované momentum)", "xsec_momentum", self.generation),
            ModelSpec("M8", "Potomek C (mutovaný meta model)", "meta_ensemble", self.generation),
        ]

        anchor = [
            ModelSpec("M9", "Stabilizační kotva MR", "mean_reversion", self.generation),
            ModelSpec("M10", "Stabilizační kotva overlay", "onchain_sentiment_overlay", self.generation),
        ]

        self.models = keep + children + anchor
