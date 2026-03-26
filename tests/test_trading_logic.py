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
from src.sirtrade.health_server import _build_worker_health_payload
from src.sirtrade.live_worker import _apply_trade_cutoff, should_start_embedded_worker
from src.sirtrade.models import ModelSpec
from src.sirtrade.storage import clear_trade_history, init_db, load_open_positions, save_open_positions
from src.sirtrade.ui_state import load_worker_status, save_worker_status


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


def _build_flat_market_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC")
    close = np.linspace(100.0, 100.6, len(index))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.0005
    low = np.minimum(open_, close) * 0.9995
    ret = pd.Series(close).pct_change().fillna(0.0).to_numpy()

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "ret": ret,
            "sentiment": np.full(len(index), 0.5),
            "onchain": np.full(len(index), 0.4),
            "regime": np.zeros(len(index)),
        },
        index=index,
    )


class TradingLogicTests(unittest.TestCase):
    def test_embedded_worker_can_be_disabled_by_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(should_start_embedded_worker())

        with patch.dict("os.environ", {"SIRTRADE_ENABLE_EMBEDDED_WORKER": "0"}, clear=True):
            self.assertFalse(should_start_embedded_worker())

    def test_worker_health_payload_reports_fresh_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "worker_status.json"
            save_worker_status(
                {
                    "status": "ok",
                    "heartbeat_at": pd.Timestamp.now(tz="UTC"),
                    "message": "healthy",
                },
                file_path=status_path,
            )

            saved = load_worker_status(file_path=status_path)
            self.assertEqual(saved.get("status"), "ok")

            with patch("src.sirtrade.health_server.load_worker_status", return_value=saved):
                is_fresh, payload = _build_worker_health_payload()

            self.assertTrue(is_fresh)
            self.assertTrue(bool(payload["worker"]["fresh"]))
            self.assertEqual(payload["worker"]["status"], "ok")
    def test_live_cycle_blocks_duplicate_symbol_side_exposure_across_models(self) -> None:
        engine = TradingEngine()
        engine.models = [
            ModelSpec("M1", "Trend", "trend_vol", 1),
            ModelSpec("M2", "Momentum", "xsec_momentum", 1),
        ]
        market_a = _build_market_frame()
        extra_idx = market_a.index[-1] + pd.Timedelta(hours=1)
        extra_row = market_a.iloc[[-1]].copy()
        extra_row.index = pd.DatetimeIndex([extra_idx])
        market_b = pd.concat([market_a, extra_row])
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])

        def _mock_confluence(model, market, controlled_signal):
            index = market.index
            confluence = pd.DataFrame(
                {
                    "long_votes": [6] * len(index),
                    "short_votes": [0] * len(index),
                    "long_confidence": [0.8] * len(index),
                    "short_confidence": [0.0] * len(index),
                },
                index=index,
            )
            atr_pct = pd.Series(0.01, index=index)
            return confluence, 5, 2, atr_pct

        strong_signal_a = pd.Series(1.0, index=market_a.index)
        strong_signal_b = pd.Series(1.0, index=market_b.index)

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market_a
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal_a
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_mock_confluence):
            first = engine.run_week(days=7, market_source="binance", symbol="BTCUSDT", interval="15m")

        self.assertEqual(first["final_open_slots"]["M1"], 0)
        self.assertEqual(first["final_open_slots"]["M2"], 0)

        engine._live_snapshot_cache.clear()

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market_b
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal_b
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_mock_confluence):
            second = engine.run_week(
                days=7,
                market_source="binance",
                symbol="BTCUSDT",
                interval="15m",
                previous_summary=first,
            )

        self.assertTrue(second["trade_events_delta"]["M1"])
        self.assertEqual(second["final_open_slots"]["M1"], 4)
        self.assertEqual(second["trade_events_delta"]["M2"], [])
        self.assertEqual(second["final_open_slots"]["M2"], 0)

    def test_init_db_is_idempotent_and_normalizes_legacy_sides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sirtrade.db"

            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE open_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        model_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        position_size REAL NOT NULL,
                        market_source TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO open_positions (
                        model_id, model_name, symbol, side, position_size, market_source
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("M1", "Trend", "BTCUSDT", "BUY", 1.0, "binance"),
                )
                conn.commit()
            finally:
                conn.close()

            init_db(db_path=db_path)
            init_db(db_path=db_path)

            conn = sqlite3.connect(db_path)
            try:
                side = conn.execute("SELECT side FROM open_positions WHERE model_id = ?", ("M1",)).fetchone()[0]
            finally:
                conn.close()

            self.assertEqual(side, "LONG")

    def test_save_open_positions_keeps_other_segment_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sirtrade.db"
            init_db(db_path=db_path)

            scalp_summary = {
                "symbol": "BTCUSDT",
                "market_source": "binance",
                "results": pd.DataFrame(
                    [
                        {"model_id": "SC_M1", "name": "Scalp Trend"},
                        {"model_id": "SC_M2", "name": "Scalp MR"},
                    ]
                ),
                "model_open_positions": {
                    "SC_M1": [
                        {"symbol": "BTCUSDT", "side": "LONG", "slots": 2, "model_name": "Scalp Trend"}
                    ]
                },
                "final_positions": {"SC_M1": 2.0, "SC_M2": 0.0},
            }
            swing_summary = {
                "symbol": "ETHUSDT",
                "market_source": "binance",
                "results": pd.DataFrame(
                    [
                        {"model_id": "SW_M1", "name": "Swing Trend"},
                        {"model_id": "SW_M2", "name": "Swing MR"},
                    ]
                ),
                "model_open_positions": {
                    "SW_M1": [
                        {"symbol": "ETHUSDT", "side": "SHORT", "slots": 1, "model_name": "Swing Trend"}
                    ]
                },
                "final_positions": {"SW_M1": -1.0, "SW_M2": 0.0},
            }

            save_open_positions(scalp_summary, db_path=db_path)
            save_open_positions(swing_summary, db_path=db_path)

            combined = load_open_positions(db_path=db_path)
            self.assertEqual(set(combined["model_id"].tolist()), {"SC_M1", "SW_M1"})

            scalp_summary["model_open_positions"] = {"SC_M1": [], "SC_M2": []}
            scalp_summary["final_positions"] = {"SC_M1": 0.0, "SC_M2": 0.0}
            save_open_positions(scalp_summary, db_path=db_path)

            remaining = load_open_positions(db_path=db_path)
            self.assertEqual(set(remaining["model_id"].tolist()), {"SW_M1"})

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
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM weekly_runs WHERE hidden = 0").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM weekly_runs WHERE hidden = 1").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM closed_positions WHERE hidden = 0").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM closed_positions WHERE hidden = 1").fetchone()[0], 1)
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

        self.assertEqual(len(open_positions["M1"]), 1)
        self.assertEqual(open_positions["M1"][0]["symbol"], "BTCUSDT")
        self.assertEqual(open_positions["M1"][0]["slots"], 3)

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

        orders = build_dry_run_orders(leaderboard, symbol="BTCUSDT")

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].model_id, "M1")
        self.assertEqual(orders[0].symbol, "BTCUSDT")
        self.assertEqual(orders[0].side, "BUY")
        self.assertEqual(orders[0].quantity_czk, 1000.0)

    def test_dry_run_orders_support_multiple_symbols_per_model(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "MC",
                    "score": 2.5,
                    "model_open_positions": [
                        {"symbol": "BTCUSDT", "side": "LONG", "slots": 2},
                        {"symbol": "ETHUSDT", "side": "LONG", "slots": 1},
                    ],
                }
            ]
        )

        orders = build_dry_run_orders(leaderboard, symbol="BTCUSDT")

        self.assertEqual(len(orders), 2)
        self.assertEqual({order.symbol for order in orders}, {"BTCUSDT", "ETHUSDT"})
        self.assertTrue(all(order.side == "BUY" for order in orders))
        self.assertEqual(next(order.quantity_czk for order in orders if order.symbol == "BTCUSDT"), 2000.0)
        self.assertEqual(next(order.quantity_czk for order in orders if order.symbol == "ETHUSDT"), 1000.0)

    def test_target_exit_is_not_delayed_by_hold_interval(self) -> None:
        engine = TradingEngine()
        market = _build_market_frame()
        model = ModelSpec("M1", "Trend", "trend_vol", 1)
        strong_signal = pd.Series(1.0, index=market.index)
        forced_confluence = pd.DataFrame(
            {
                "long_votes": np.full(len(market.index), 6),
                "short_votes": np.zeros(len(market.index)),
                "long_confidence": np.full(len(market.index), 1.0),
                "short_confidence": np.zeros(len(market.index)),
            },
            index=market.index,
        )
        atr_pct = pd.Series(0.0018, index=market.index)

        with patch("src.sirtrade.engine.generate_signals", return_value=strong_signal), patch.object(
            TradingEngine,
            "_build_entry_confluence",
            return_value=(forced_confluence, 4, 2, atr_pct),
            autospec=True,
        ), patch.object(
            TradingEngine,
            "_trade_profile",
            return_value=(0.8, 1.2, 0.0015, 0.05),
            autospec=True,
        ):
            _, events, _, _ = engine._simulate_model(model, market, symbol="ETHUSDT")

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]["akce"].split()[0], "Vstup")
        self.assertEqual(events[0].get("symbol"), "ETHUSDT")
        self.assertIn(events[1].get("duvod_vystupu"), {"TARGET", "STOP"})
        entry_time = pd.Timestamp(events[0]["timestamp"])
        exit_time = pd.Timestamp(events[1]["timestamp"])
        self.assertLessEqual(exit_time, entry_time + pd.Timedelta(hours=1))

    def test_position_is_not_closed_only_because_next_bar_arrives(self) -> None:
        engine = TradingEngine()
        market = _build_flat_market_frame()
        model = ModelSpec("M1", "Trend", "trend_vol", 1)
        strong_signal = pd.Series(1.0, index=market.index)

        with patch("src.sirtrade.engine.generate_signals", return_value=strong_signal):
            _, events, final_position, final_open_slots = engine._simulate_model(model, market, symbol="BTCUSDT")

        entry_events = [event for event in events if str(event.get("akce", "")).startswith("Vstup")]
        exit_events = [event for event in events if str(event.get("akce", "")).startswith("Výstup")]
        self.assertGreaterEqual(len(entry_events), 1)
        self.assertEqual(len(exit_events), 0)
        self.assertGreater(final_open_slots, 0)
        self.assertGreater(final_position, 0.0)

    def test_model_does_not_reenter_without_signal_reset_after_target_exit(self) -> None:
        engine = TradingEngine()
        market = _build_market_frame().copy()
        market["high"] = market["close"] * 1.03
        market["low"] = market["close"] * 0.999
        model = ModelSpec("M1", "Trend", "trend_vol", 1)
        strong_signal = pd.Series(1.0, index=market.index)

        forced_confluence = pd.DataFrame(
            {
                "long_votes": np.full(len(market.index), 6),
                "short_votes": np.zeros(len(market.index)),
                "long_confidence": np.full(len(market.index), 1.0),
                "short_confidence": np.zeros(len(market.index)),
            },
            index=market.index,
        )
        atr_pct = pd.Series(0.003, index=market.index)

        with patch("src.sirtrade.engine.generate_signals", return_value=strong_signal), patch.object(
            TradingEngine,
            "_build_entry_confluence",
            return_value=(forced_confluence, 4, 2, atr_pct),
            autospec=True,
        ):
            _, events, _, _ = engine._simulate_model(model, market, symbol="BTCUSDT")

        entry_events = [event for event in events if str(event.get("akce", "")).startswith("Vstup")]
        exit_events = [event for event in events if str(event.get("akce", "")).startswith("Výstup")]
        self.assertEqual(len(entry_events), 1)
        self.assertEqual(len(exit_events), 1)
        self.assertEqual(exit_events[0].get("duvod_vystupu"), "TARGET")

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
        standard_rows = summary["results"][summary["results"]["model_id"] != "MC"]
        self.assertEqual(len(summary["results"]), 6)
        self.assertTrue(all(row_symbol == "SOLUSDT" for row_symbol in standard_rows["symbol"].tolist()))

    def test_evolve_generation_keeps_fixed_model_budget_per_segment(self) -> None:
        engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        leaderboard = pd.DataFrame(
            [
                {"model_id": "SC_M1", "score": 2.0},
                {"model_id": "SC_M2", "score": 1.8},
                {"model_id": "SC_M3", "score": 1.5},
                {"model_id": "SC_M4", "score": 1.2},
                {"model_id": "SC_M5", "score": 1.0},
                {"model_id": "SC_MC", "score": 0.9},
            ]
        )

        engine._evolve_generation(leaderboard)

        self.assertEqual(len(engine.models), 6)
        self.assertEqual(sum(model.kind == "copy_trader" for model in engine.models), 1)
        self.assertEqual(sum(model.kind != "copy_trader" for model in engine.models), 5)

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
        self.assertEqual(len(summary["results"]), 6)
        self.assertIn("MC", summary["results"]["model_id"].tolist())
        self.assertEqual({item["symbol"] for item in summary["model_open_positions"]["MC"]}, {"BTCUSDT", "ETHUSDT"})
        self.assertEqual(sum(int(item.get("slots", 0)) for item in summary["model_open_positions"]["MC"]), 5)
        self.assertEqual({order["symbol"] for order in summary["proposed_orders"]}, {"BTCUSDT", "ETHUSDT"})

    def test_live_cycle_does_not_reopen_position_each_worker_tick(self) -> None:
        """Entry fires only when bar changes; same bar = no new entry."""
        engine = TradingEngine()
        engine.models = [ModelSpec("M1", "Trend", "trend_vol", 1)]
        market_a = _build_market_frame()
        # market_b has one extra bar at the end → different last-bar timestamp
        extra_idx = market_a.index[-1] + pd.Timedelta(hours=1)
        extra_row = market_a.iloc[[-1]].copy()
        extra_row.index = pd.DatetimeIndex([extra_idx])
        market_b = pd.concat([market_a, extra_row])

        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])

        # Cycle 1: first ever run → seeds last_bar_ts, no entry (no prior bar to compare)
        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market_a
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            first = engine.run_week(days=7, market_source="binance", symbol="BTCUSDT", interval="15m")

        self.assertEqual(len(first["trade_events_delta"]["M1"]), 0, "First cycle must not enter (no prior bar)")
        self.assertTrue(first["live_model_state"]["M1"].get("last_bar_ts"), "last_bar_ts must be seeded")

        # Clear market-data cache so cycle 2 sees market_b
        engine._live_snapshot_cache.clear()

        # Cycle 2: new bar (market_b) → entry should fire
        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market_b
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            second = engine.run_week(
                days=7, market_source="binance", symbol="BTCUSDT", interval="15m",
                previous_summary=first,
            )

        self.assertTrue(second["trade_events_delta"]["M1"], "New bar must trigger entry")
        self.assertGreater(second["final_open_slots"]["M1"], 0)

        # Cycle 3: same bar (market_b again) → no new entry
        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market_b
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            third = engine.run_week(
                days=7, market_source="binance", symbol="BTCUSDT", interval="15m",
                previous_summary=second,
            )

        self.assertEqual(len(third["trade_events_delta"]["M1"]), 0, "Same bar must not trigger entry")
        self.assertGreater(third["final_open_slots"]["M1"], 0, "Position must remain open")

    def test_same_bar_does_not_trigger_new_entry(self) -> None:
        """Even with entry_armed=True and strong signal, same bar = no entry."""
        engine = TradingEngine()
        engine.models = [ModelSpec("M1", "Trend", "trend_vol", 1)]
        market = _build_market_frame()
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
        strong_signal = pd.Series(1.0, index=market.index)

        # Build a fake previous_summary with last_bar_ts matching current market's last bar
        last_ts = market.index[-1]
        ts_key = last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts)
        prev = {
            "model_trades": {"M1": []},
            "live_model_state": {
                "M1": {
                    "entry_armed": True,
                    "last_bar_ts": ts_key,
                    "symbol": "BTCUSDT",
                    "side": "",
                    "open_slots": 0,
                    "entry_price": 0.0,
                    "opened_at": None,
                    "stop_price": 0.0,
                    "target_price": 0.0,
                }
            },
            "model_open_positions": {"M1": []},
        }

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal
        ):
            result = engine.run_week(
                days=7, market_source="binance", symbol="BTCUSDT", interval="5m",
                previous_summary=prev,
            )

        self.assertEqual(len(result["trade_events_delta"]["M1"]), 0)
        self.assertEqual(result["final_open_slots"]["M1"], 0)


if __name__ == "__main__":
    unittest.main()