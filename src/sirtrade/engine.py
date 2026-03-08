from __future__ import annotations

from dataclasses import dataclass

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
    def __init__(self, config: AppConfig = DEFAULT_CONFIG):
        self.config = config
        self.models: list[ModelSpec] = default_model_specs()
        self.generation = 1
        self.week = 0

    def _build_trade_events(self, model: ModelSpec, prices: pd.Series, position: pd.Series) -> list[dict]:
        events: list[dict] = []
        prev = 0.0
        for ts, current in position.fillna(0.0).items():
            price = float(prices.loc[ts])
            if prev == 0 and current != 0:
                events.append(
                    {
                        "timestamp": ts,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": "Vstup LONG" if current > 0 else "Vstup SHORT",
                        "strana": "LONG" if current > 0 else "SHORT",
                        "cena": price,
                        "pozice": float(current),
                    }
                )
            elif prev != 0 and current == 0:
                events.append(
                    {
                        "timestamp": ts,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": "Výstup LONG" if prev > 0 else "Výstup SHORT",
                        "strana": "LONG" if prev > 0 else "SHORT",
                        "cena": price,
                        "pozice": float(current),
                    }
                )
            elif prev * current < 0:
                events.append(
                    {
                        "timestamp": ts,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": "Výstup LONG" if prev > 0 else "Výstup SHORT",
                        "strana": "LONG" if prev > 0 else "SHORT",
                        "cena": price,
                        "pozice": 0.0,
                    }
                )
                events.append(
                    {
                        "timestamp": ts,
                        "model_id": model.model_id,
                        "model_name": model.name,
                        "akce": "Vstup LONG" if current > 0 else "Vstup SHORT",
                        "strana": "LONG" if current > 0 else "SHORT",
                        "cena": price,
                        "pozice": float(current),
                    }
                )
            prev = float(current)
        return events

    def _simulate_model(self, model: ModelSpec, market: pd.DataFrame) -> tuple[ModelResult, list[dict], float]:
        raw = generate_signals(model, market, seed=self.week)
        pos = apply_risk_controls(raw, market["ret"], self.config.risk)

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
        events = self._build_trade_events(model, market["close"], pos)
        final_position = float(pos.iloc[-1]) if not pos.empty else 0.0
        return result, events, final_position

    def run_week(self, days: int = 365, market_source: str | None = None, symbol: str | None = None) -> dict:
        self.week += 1
        effective_source = market_source or self.config.market_data_source
        effective_symbol = (symbol or self.config.default_symbol).upper()
        market = get_market_data(
            source=effective_source,
            days=days,
            symbol=effective_symbol,
            seed=100 + self.week,
        )

        model_runs = [self._simulate_model(model, market) for model in self.models]
        results = [run[0] for run in model_runs]
        model_trades = {run[0].model_id: run[1] for run in model_runs}
        final_positions = {run[0].model_id: run[2] for run in model_runs}
        results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("score", ascending=False)

        champion = results_df.iloc[0].to_dict()
        champion["reward_usd"] = 1.0
        champion_model_id = str(champion["model_id"])

        research: list[StudyInsight] = daily_deep_research(seed=10_000 + self.week)
        if effective_source == "binance":
            long_tail = scan_binance_long_tail(top_n=20)
        else:
            long_tail = scan_long_tail_opportunities(seed=self.week, universe_size=300).head(20)
        proposed_orders = build_dry_run_orders(results_df, symbol=effective_symbol, nav_usd=1000.0)

        if self.week % self.config.generation_horizon_weeks == 0:
            self._evolve_generation(results_df)

        summary = {
            "week": self.week,
            "generation": self.generation,
            "portfolio_vol_annual": float(annualized_vol(market["ret"])),
            "market_source": effective_source,
            "symbol": effective_symbol,
            "champion": champion,
            "results": results_df,
            "market": market,
            "model_trades": model_trades,
            "champion_trades": model_trades.get(champion_model_id, []),
            "final_positions": final_positions,
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
