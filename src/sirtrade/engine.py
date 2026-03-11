from __future__ import annotations

from dataclasses import dataclass
import hashlib

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

    def _simulate_model(self, model: ModelSpec, market: pd.DataFrame) -> tuple[ModelResult, list[dict], float, int]:
        model_seed = int(hashlib.md5(model.model_id.encode("utf-8")).hexdigest()[:8], 16)
        week_seed = (self.week * 100_003) + model_seed
        rng = np.random.default_rng(week_seed)

        raw = generate_signals(model, market, seed=week_seed)
        controlled_signal = apply_risk_controls(raw, market["ret"], self.config.risk)

        close = market["close"].astype(float)
        high = market["high"].astype(float)
        low = market["low"].astype(float)
        ret_std = market["ret"].fillna(0.0).rolling(20).std().fillna(0.0)

        slot_size = self.config.risk.max_asset_exposure / 5
        entry_threshold = float(rng.uniform(0.14, 0.34))
        warmup_bars = min(48, max(12, int(len(market) * 0.1)))
        decision_interval = int(rng.integers(1, 4))
        decision_offset = int(rng.integers(0, decision_interval))
        min_hold_bars = int(rng.integers(1, 6))
        cooldown_bars = int(rng.integers(2, 8))
        stop_multiplier = float(rng.uniform(0.9, 1.6))
        target_multiplier = float(rng.uniform(1.8, 3.2))

        pos = pd.Series(0.0, index=market.index, dtype=float)
        side = 0
        position_size = 0.0
        current_slots = 0
        stop_price = None
        target_price = None
        cooldown_remaining = 0
        hold_bars = 0
        events: list[dict] = []

        for step, ts in enumerate(market.index):
            signal_value = float(controlled_signal.loc[ts])
            close_price = float(close.loc[ts])
            high_price = float(high.loc[ts])
            low_price = float(low.loc[ts])
            vol_step = float(ret_std.loc[ts])
            if np.isnan(vol_step) or vol_step <= 0:
                vol_step = 0.01

            if step < warmup_bars:
                pos.loc[ts] = 0.0
                continue

            is_decision_step = ((step - decision_offset) % decision_interval) == 0

            if side == 0:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    pos.loc[ts] = 0.0
                    continue

                if is_decision_step and abs(signal_value) >= entry_threshold:
                    direction = 1 if signal_value > 0 else -1
                    slots = int(np.clip(np.ceil(abs(signal_value) * 5), 1, 5))
                    position_size = float(direction * slots * slot_size)
                    side = direction
                    current_slots = slots
                    hold_bars = 0

                    stop_dist = max(0.004, stop_multiplier * vol_step)
                    target_dist = max(0.008, target_multiplier * vol_step)
                    if side > 0:
                        stop_price = close_price * (1.0 - stop_dist)
                        target_price = close_price * (1.0 + target_dist)
                    else:
                        stop_price = close_price * (1.0 + stop_dist)
                        target_price = close_price * (1.0 - target_dist)

                    events.append(
                        {
                            "timestamp": ts,
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
            hold_bars += 1
            if hold_bars >= min_hold_bars:
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
                hold_bars = 0
                cooldown_remaining = cooldown_bars
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
        long_tail: pd.DataFrame,
        default_symbol: str,
    ) -> dict[str, list[dict]]:
        candidates = [default_symbol]
        if isinstance(long_tail, pd.DataFrame) and "symbol" in long_tail.columns:
            candidates.extend([str(value) for value in long_tail["symbol"].dropna().tolist()])
        unique_candidates = list(dict.fromkeys([symbol.upper() for symbol in candidates if str(symbol).strip()]))
        if not unique_candidates:
            unique_candidates = [default_symbol]

        model_open_positions: dict[str, list[dict]] = {}
        for row_index, row in results_df.reset_index(drop=True).iterrows():
            model_id = str(row["model_id"])
            model_name = str(row["name"])
            slots = max(0, int(final_open_slots.get(model_id, 0)))
            position_value = float(final_positions.get(model_id, 0.0))
            side = "LONG" if position_value > 0 else ("SHORT" if position_value < 0 else "-")
            if slots <= 0 or side == "-":
                model_open_positions[model_id] = []
                continue

            start = (row_index * 3 + self.week) % len(unique_candidates)
            rotated = unique_candidates[start:] + unique_candidates[:start]
            selected_symbols = rotated[:slots]
            model_open_positions[model_id] = [
                {
                    "slot": slot_idx + 1,
                    "symbol": symbol,
                    "side": side,
                    "model_id": model_id,
                    "model_name": model_name,
                }
                for slot_idx, symbol in enumerate(selected_symbols)
            ]

        return model_open_positions

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
        market = get_market_data(
            source=effective_source,
            days=days,
            symbol=effective_symbol,
            seed=100 + self.week,
            interval=interval,
        )

        model_runs = [self._simulate_model(model, market) for model in self.models]
        results = [run[0] for run in model_runs]
        model_trades = {run[0].model_id: run[1] for run in model_runs}
        final_positions = {run[0].model_id: run[2] for run in model_runs}
        final_open_slots = {run[0].model_id: run[3] for run in model_runs}
        results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("score", ascending=False)

        champion = results_df.iloc[0].to_dict()
        champion["reward_usd"] = 1.0
        champion_model_id = str(champion["model_id"])

        research: list[StudyInsight] = daily_deep_research(seed=10_000 + self.week)
        if effective_source == "binance":
            long_tail = scan_binance_long_tail(top_n=20)
        else:
            long_tail = scan_long_tail_opportunities(seed=self.week, universe_size=300).head(20)
        model_open_positions = self._build_model_open_positions(
            results_df=results_df,
            final_positions=final_positions,
            final_open_slots=final_open_slots,
            long_tail=long_tail,
            default_symbol=effective_symbol,
        )
        proposed_orders = build_dry_run_orders(results_df, symbol=effective_symbol, nav_usd=1000.0)

        if self.week % self.config.generation_horizon_weeks == 0:
            self._evolve_generation(results_df)

        summary = {
            "week": self.week,
            "generation": self.generation,
            "portfolio_vol_annual": float(annualized_vol(market["ret"])),
            "market_source": effective_source,
            "symbol": effective_symbol,
            "interval": interval,
            "champion": champion,
            "results": results_df,
            "market": market,
            "model_trades": model_trades,
            "champion_trades": model_trades.get(champion_model_id, []),
            "final_positions": final_positions,
            "final_open_slots": final_open_slots,
            "model_open_positions": model_open_positions,
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
