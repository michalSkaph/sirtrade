from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.sirtrade.engine import TradingEngine
from src.sirtrade.execution import build_dry_run_orders
from src.sirtrade.engine import ModelResult
from src.sirtrade.live_worker import _apply_trade_cutoff
from src.sirtrade.copy_trading import LeadTraderProfile, select_best_lead_trader
from src.sirtrade.models import ModelSpec
from src.sirtrade.storage import clear_trade_history, init_db


def _build_market_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    close = np.linspace(100.0, 106.0, 60)
    close[13] = 104.0
    close[14] = 105.0
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.998
    ret = pd.Series(close).pct_change().fillna(0.0).to_numpy()

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "ret": ret,
            "sentiment": np.full(len(index), 0.6),
            "onchain": np.full(len(index), 0.8),
            "regime": np.zeros(len(index)),
        },
        index=index,
    )


class TradingLogicTests(unittest.TestCase):
    def test_trade_cutoff_removes_pre_reset_open_positions(self) -> None:
        summary = {
            "symbol": "BTCUSDT",
            "champion": {"model_id": "M1"},
            "results": pd.DataFrame([
                {"model_id": "M1", "name": "Trend", "symbol": "BTCUSDT", "score": 1.0}
            ]),
            "model_trades": {
                "M1": [
                    {
                        "timestamp": "2026-03-24T08:00:00Z",
                        "symbol": "BTCUSDT",
                        "akce": "Vstup LONG (+2)",
                        "strana": "LONG",
                        "cena": 100.0,
                    }
                ]
            },
            "final_positions": {"M1": 0.1},
            "final_open_slots": {"M1": 2},
            "model_open_positions": {"M1": [{"symbol": "BTCUSDT", "side": "LONG"}]},
            "proposed_orders": [{"model_id": "M1"}],
        }

        filtered = _apply_trade_cutoff(summary, "2026-03-24T09:00:00Z")

        self.assertEqual(filtered["model_trades"]["M1"], [])
        self.assertEqual(filtered["final_open_slots"]["M1"], 0)
        self.assertEqual(filtered["final_positions"]["M1"], 0.0)
        self.assertEqual(filtered["model_open_positions"]["M1"], [])
        self.assertEqual(filtered["proposed_orders"], [])

    def test_clear_trade_history_resets_all_paper_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sirtrade.db"
            init_db(db_path=db_path)

            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO weekly_runs (
                        segment, week, generation, market_source, symbol, interval,
                        champion_model, champion_score, champion_sortino,
                        champion_calmar, champion_max_dd, champion_cvar95, reward_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "Swing",
                        1,
                        3,
                        "binance",
                        "BTCUSDT",
                        "4h",
                        "Swing | Test",
                        1.0,
                        1.0,
                        1.0,
                        0.1,
                        0.1,
                        1.0,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO open_positions (
                        model_id, model_name, symbol, side, position_size, market_source
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("SW_1", "Swing | Test", "BTCUSDT", "LONG", 1.0, "binance"),
                )
                conn.execute(
                    """
                    INSERT INTO closed_positions (
                        closed_at, opened_at, model_id, model_name, symbol, side,
                        entry_price, exit_price, quantity_slots, pnl_pct, pnl_status,
                        exit_reason, market_source, week, generation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "2026-01-01T00:00:00Z",
                        "2025-12-31T23:00:00Z",
                        "SW_1",
                        "Swing | Test",
                        "BTCUSDT",
                        "LONG",
                        100.0,
                        101.0,
                        1.0,
                        1.0,
                        "ZISK",
                        "TARGET",
                        "binance",
                        1,
                        3,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            clear_trade_history(db_path=db_path)

            conn = sqlite3.connect(db_path)
            try:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM weekly_runs").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM closed_positions").fetchone()[0], 0)
            finally:
                conn.close()

    def test_open_positions_use_evaluated_symbol_only(self) -> None:
        engine = TradingEngine()
        results = pd.DataFrame(
            [
                {"model_id": "M1", "name": "Trend", "score": 1.5, "symbol": "BTCUSDT"},
            ]
        )

        open_positions = engine._build_model_open_positions(
            results_df=results,
            final_positions={"M1": 0.15},
            final_open_slots={"M1": 3},
        )

        self.assertEqual(len(open_positions["M1"]), 3)
        self.assertTrue(all(item["symbol"] == "BTCUSDT" for item in open_positions["M1"]))

    def test_dry_run_orders_follow_actual_open_positions(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "M1",
                    "score": 2.0,
                    "model_open_positions": [{"symbol": "BTCUSDT", "side": "LONG"}],
                },
                {
                    "model_id": "M2",
                    "score": 1.0,
                    "model_open_positions": [],
                },
            ]
        )

        orders = build_dry_run_orders(leaderboard, symbol="BTCUSDT", nav_usd=1000.0)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].model_id, "M1")
        self.assertEqual(orders[0].symbol, "BTCUSDT")
        self.assertEqual(orders[0].side, "BUY")

    def test_dry_run_orders_support_multiple_symbols_per_model(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "MC",
                    "score": 2.5,
                    "model_open_positions": [
                        {"symbol": "BTCUSDT", "side": "LONG"},
                        {"symbol": "BTCUSDT", "side": "LONG"},
                        {"symbol": "ETHUSDT", "side": "LONG"},
                    ],
                }
            ]
        )

        orders = build_dry_run_orders(leaderboard, symbol="BTCUSDT", nav_usd=1000.0)

        self.assertEqual(len(orders), 2)
        self.assertEqual({order.symbol for order in orders}, {"BTCUSDT", "ETHUSDT"})
        self.assertTrue(all(order.side == "BUY" for order in orders))

    def test_target_exit_is_not_delayed_by_hold_interval(self) -> None:
        engine = TradingEngine()
        market = _build_market_frame()
        model = ModelSpec("M1", "Trend", "trend_vol", 1)
        strong_signal = pd.Series(1.0, index=market.index)

        with patch("src.sirtrade.engine.generate_signals", return_value=strong_signal):
            _, events, _, _ = engine._simulate_model(model, market, symbol="ETHUSDT")

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]["akce"].split()[0], "Vstup")
        self.assertEqual(events[0].get("symbol"), "ETHUSDT")
        self.assertEqual(events[1].get("duvod_vystupu"), "TARGET")
        entry_time = pd.Timestamp(events[0]["timestamp"])
        exit_time = pd.Timestamp(events[1]["timestamp"])
        self.assertLessEqual(exit_time, entry_time + pd.Timedelta(hours=1))

    def test_run_week_selects_coin_from_dynamic_top20(self) -> None:
        engine = TradingEngine()
        universe = pd.DataFrame(
            [
                {"symbol": "ETHUSDT", "opportunity_score": 0.8},
                {"symbol": "SOLUSDT", "opportunity_score": 1.2},
            ]
        )
        market = _build_market_frame()

        def fake_simulate(_engine: TradingEngine, model: ModelSpec, market_frame: pd.DataFrame, symbol: str):
            score = 3.0 if symbol == "SOLUSDT" else 1.0
            result = ModelResult(
                model_id=model.model_id,
                name=model.name,
                generation=model.generation,
                symbol=symbol,
                sortino=score,
                calmar=score,
                cvar95=0.01,
                max_dd=0.02,
                cost=0.0,
                turnover=0.0,
                score=score,
                passed=True,
            )
            return result, [{"timestamp": market_frame.index[-1], "symbol": symbol, "akce": "Vstup LONG (+1)", "strana": "LONG", "cena": float(market_frame["close"].iloc[-1])}], 0.05, 1

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch.object(TradingEngine, "_simulate_model", side_effect=fake_simulate, autospec=True):
            summary = engine.run_week(days=30, market_source="binance", symbol="BTCUSDT", interval="15m")

        self.assertIn("SOLUSDT", summary.get("candidate_symbols", []))
        self.assertEqual(summary["champion"]["symbol"], "SOLUSDT")
        self.assertTrue(all(row_symbol == "SOLUSDT" for row_symbol in summary["results"]["symbol"].tolist()))

    def test_select_best_lead_trader_prefers_higher_weighted_score(self) -> None:
        conservative = LeadTraderProfile(
            trader_id="A",
            nickname="Conservative",
            roi=0.25,
            pnl_usd=800.0,
            win_rate=0.62,
            max_drawdown=0.04,
            followers=120.0,
            score=0.65,
        )
        aggressive = LeadTraderProfile(
            trader_id="B",
            nickname="Aggressive",
            roi=0.90,
            pnl_usd=1500.0,
            win_rate=0.60,
            max_drawdown=0.45,
            followers=300.0,
            score=0.92,
        )

        best = select_best_lead_trader([conservative, aggressive])

        self.assertIsNotNone(best)
        self.assertEqual(best.trader_id, "B")

    def test_run_week_binance_copy_uses_copy_trader_positions_without_leverage(self) -> None:
        engine = TradingEngine(model_namespace="", model_label_prefix="")
        universe = pd.DataFrame([
            {"symbol": "BTCUSDT", "opportunity_score": 1.0},
        ])
        market = _build_market_frame()
        snapshot = {
            "leader": {
                "trader_id": "leader-1",
                "nickname": "Top Trader",
                "score": 1.2,
            },
            "positions": [
                {"symbol": "BTCUSDT", "side": "LONG", "leverage": 1.0, "notional_usd": 4000.0, "entry_price": 100.0},
                {"symbol": "ETHUSDT", "side": "LONG", "leverage": 1.0, "notional_usd": 2000.0, "entry_price": 101.0},
            ],
        }

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=snapshot):
            summary = engine.run_week(days=30, market_source="binance_copy", symbol="BTCUSDT", interval="15m")

        self.assertEqual(summary["market_source"], "binance_copy")
        self.assertEqual(len(summary["results"]), 1)
        self.assertEqual(summary["results"].iloc[0]["model_id"], "MC")
        self.assertEqual({item["symbol"] for item in summary["model_open_positions"]["MC"]}, {"BTCUSDT", "ETHUSDT"})
        self.assertEqual({order["symbol"] for order in summary["proposed_orders"]}, {"BTCUSDT", "ETHUSDT"})


if __name__ == "__main__":
    unittest.main()