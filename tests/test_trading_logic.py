from __future__ import annotations

import ast
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.sirtrade.data import simulate_market
from src.sirtrade.engine import TradingEngine
from src.sirtrade.engine import ModelResult
from src.sirtrade.execution import build_dry_run_orders
from src.sirtrade.copy_trading import LeadTraderProfile, select_best_lead_trader
from src.sirtrade.config import DEFAULT_CONFIG
from src.sirtrade.health_server import _build_worker_health_payload
from src.sirtrade.live_worker import (
    _apply_trade_cutoff,
    _choose_segments,
    _get_binance_cadence_seconds,
    _WorkerProcessLock,
    should_start_embedded_worker,
)
from src.sirtrade.market_stream import apply_stream_kline_to_market
from src.sirtrade.models import ModelSpec
from src.sirtrade.scoring import decision_score
from src.sirtrade.storage import _build_closed_positions_rows, _dedupe_closed_positions, clear_trade_history, init_db, load_open_positions, save_closed_positions, save_open_positions
from src.sirtrade.ui_state import (
    load_segment_runs,
    load_worker_status,
    sanitize_runtime_state_for_ui_boot,
    save_segment_runs,
    save_worker_status,
)


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
    def test_decision_score_bounds_extreme_ratio_metrics(self) -> None:
        score = decision_score(
            {
                "sortino": 1_000_000.0,
                "calmar": 1_000_000.0,
                "cvar95": 0.001,
                "max_dd": 0.001,
                "cost": 0.0,
                "turnover": 0.0,
            },
            DEFAULT_CONFIG.weights,
        )

        self.assertLessEqual(score, (DEFAULT_CONFIG.weights.sortino * 6.0) + (DEFAULT_CONFIG.weights.calmar * 8.0))
        self.assertGreater(score, 0.0)

    def test_decision_score_rewards_higher_win_rate_when_other_metrics_match(self) -> None:
        base_metrics = {
            "sortino": 1.6,
            "calmar": 1.1,
            "cvar95": 0.01,
            "max_dd": 0.04,
            "cost": 0.002,
            "turnover": 0.10,
        }

        low_win_rate_score = decision_score({**base_metrics, "win_rate": 42.0}, DEFAULT_CONFIG.weights)
        high_win_rate_score = decision_score({**base_metrics, "win_rate": 68.0}, DEFAULT_CONFIG.weights)

        self.assertGreater(high_win_rate_score, low_win_rate_score)

    def test_trade_analytics_summary_reports_expectancy_and_exit_reasons(self) -> None:
        engine = TradingEngine()
        analytics = engine._summarize_trade_analytics(
            {
                "M1": [
                    {
                        "timestamp": "2026-03-01T10:00:00Z",
                        "symbol": "BTCUSDT",
                        "model_id": "M1",
                        "model_name": "Trend",
                        "akce": "Vstup LONG (+2)",
                        "strana": "LONG",
                        "cena": 100.0,
                        "sloty": 2,
                    },
                    {
                        "timestamp": "2026-03-01T11:00:00Z",
                        "symbol": "BTCUSDT",
                        "model_id": "M1",
                        "model_name": "Trend",
                        "akce": "Výstup LONG (-2)",
                        "strana": "LONG",
                        "cena": 103.0,
                        "opened_at": "2026-03-01T10:00:00Z",
                        "entry_price": 100.0,
                        "quantity_slots": 2,
                        "duvod_vystupu": "TARGET",
                    },
                    {
                        "timestamp": "2026-03-01T12:00:00Z",
                        "symbol": "ETHUSDT",
                        "model_id": "M1",
                        "model_name": "Trend",
                        "akce": "Vstup LONG (+1)",
                        "strana": "LONG",
                        "cena": 100.0,
                        "sloty": 1,
                    },
                    {
                        "timestamp": "2026-03-01T13:00:00Z",
                        "symbol": "ETHUSDT",
                        "model_id": "M1",
                        "model_name": "Trend",
                        "akce": "Výstup LONG (-1)",
                        "strana": "LONG",
                        "cena": 99.0,
                        "opened_at": "2026-03-01T12:00:00Z",
                        "entry_price": 100.0,
                        "quantity_slots": 1,
                        "duvod_vystupu": "STOP",
                    },
                ]
            }
        )

        self.assertEqual(int(analytics["closed_trades"]), 2)
        self.assertEqual(int(analytics["wins"]), 1)
        self.assertEqual(int(analytics["losses"]), 1)
        self.assertAlmostEqual(float(analytics["win_rate"]), 50.0, places=6)
        self.assertAlmostEqual(float(analytics["expectancy_pct"]), 1.0, places=6)
        self.assertGreater(float(analytics["profit_factor"]), 1.0)
        exit_reasons = {item["reason"]: int(item["count"]) for item in analytics["exit_reasons"]}
        self.assertEqual(exit_reasons, {"TARGET": 1, "STOP": 1})

    def test_runtime_state_sanitization_disables_stale_running_segments_by_default(self) -> None:
        runtime_state = {
            "simulation_running": True,
            "simulation_running_by_segment": {"Scalp": True, "Intraday": False, "Swing": True},
            "last_simulation_tick": 123.45,
            "active_segment": "Swing",
        }

        sanitized = sanitize_runtime_state_for_ui_boot(runtime_state)

        self.assertFalse(bool(sanitized["simulation_running"]))
        self.assertEqual(
            sanitized["simulation_running_by_segment"],
            {"Scalp": False, "Intraday": False, "Swing": False},
        )
        self.assertEqual(float(sanitized["last_simulation_tick"]), 0.0)
        self.assertEqual(str(sanitized["active_segment"]), "Swing")

    def test_runtime_state_sanitization_preserves_running_segments_when_enabled(self) -> None:
        runtime_state = {
            "simulation_running": True,
            "simulation_running_by_segment": {"Scalp": True},
            "last_simulation_tick": 123.45,
        }

        sanitized = sanitize_runtime_state_for_ui_boot(runtime_state, resume_running_segments=True)

        self.assertEqual(sanitized, runtime_state)

    def test_app_boot_uses_runtime_state_sanitizer(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(app_path.read_text(encoding="utf-8"))

        boot_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_reset_embedded_worker_boot_state"
        )
        called_names = {
            node.func.id
            for node in ast.walk(boot_func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        self.assertIn("sanitize_runtime_state_for_ui_boot", called_names)

    def test_app_resets_embedded_worker_state_only_before_first_worker_start(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(app_path.read_text(encoding="utf-8"))

        should_start_if = next(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "should_start_embedded_worker"
        )

        guarded_reset = False
        for node in should_start_if.body:
            if not isinstance(node, ast.If):
                continue
            if not isinstance(node.test, ast.UnaryOp) or not isinstance(node.test.op, ast.Not):
                continue
            operand = node.test.operand
            if not (
                isinstance(operand, ast.Call)
                and isinstance(operand.func, ast.Name)
                and operand.func.id == "is_live_worker_started"
            ):
                continue
            guarded_reset = any(
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "_reset_embedded_worker_boot_state"
                for stmt in node.body
            )
            if guarded_reset:
                break

        self.assertTrue(guarded_reset)

    def test_app_position_loaders_are_not_streamlit_cached(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(app_path.read_text(encoding="utf-8"))
        target_names = {"_load_open_positions_cached", "_load_closed_positions_cached"}

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in target_names:
                continue

            decorator_names = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorator_names.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    decorator_names.append(decorator.attr)
                elif isinstance(decorator, ast.Call):
                    func = decorator.func
                    if isinstance(func, ast.Name):
                        decorator_names.append(func.id)
                    elif isinstance(func, ast.Attribute):
                        decorator_names.append(func.attr)

            self.assertNotIn("cache_data", decorator_names, msg=f"{node.name} must not use st.cache_data")

    def test_segment_run_persistence_trims_market_payload(self) -> None:
        market_index = pd.date_range("2026-01-01", periods=500, freq="min", tz="UTC")
        market = pd.DataFrame(
            {
                "open": np.linspace(100.0, 101.0, len(market_index)),
                "high": np.linspace(100.5, 101.5, len(market_index)),
                "low": np.linspace(99.5, 100.5, len(market_index)),
                "close": np.linspace(100.1, 101.1, len(market_index)),
            },
            index=market_index,
        )
        summary = {
            "segment": "Scalp",
            "week": 1,
            "generation": 1,
            "portfolio_vol_annual": 0.1,
            "market_source": "simulation",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "champion": {},
            "research": [],
            "proposed_orders": [],
            "model_trades": {},
            "trade_events_delta": {},
            "final_positions": {},
            "final_open_slots": {},
            "model_open_positions": {},
            "live_model_state": {},
            "model_selected_symbols": {},
            "candidate_symbols": [],
            "latest_prices": {},
            "results": pd.DataFrame(),
            "long_tail": pd.DataFrame(),
            "market": market,
            "model_markets": {"SC_M1": market},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            segment_file = Path(temp_dir) / "segment_runs.json"
            save_segment_runs({"Scalp": summary}, file_path=segment_file)
            restored = load_segment_runs(file_path=segment_file)

        restored_market = restored["Scalp"]["market"]
        restored_model_market = restored["Scalp"]["model_markets"]["SC_M1"]
        self.assertLessEqual(len(restored_market), 360)
        self.assertLessEqual(len(restored_model_market), 360)

    def test_closed_positions_do_not_mix_symbols_for_same_model(self) -> None:
        summary = {
            "symbol": "BTCUSDT",
            "market_source": "binance",
            "week": 1,
            "generation": 1,
            "results": pd.DataFrame([
                {"model_id": "M1", "name": "Trend"},
            ]),
            "trade_events_delta": {
                "M1": [
                    {
                        "timestamp": "2026-03-26T20:30:00Z",
                        "symbol": "USDCUSDT",
                        "model_id": "M1",
                        "akce": "Vstup LONG (+5)",
                        "strana": "LONG",
                        "cena": 1.0005,
                        "pozice": 0.25,
                        "sloty": 5,
                    },
                    {
                        "timestamp": "2026-03-26T21:30:00Z",
                        "symbol": "BTCUSDT",
                        "model_id": "M1",
                        "akce": "Výstup LONG (-5)",
                        "strana": "LONG",
                        "cena": 69185.05,
                        "pozice": 0.0,
                        "sloty": 0,
                    },
                ]
            },
        }

        rows = _build_closed_positions_rows(summary)
        self.assertEqual(rows, [])

    def test_closed_positions_use_direct_entry_metadata_for_live_exits(self) -> None:
        summary = {
            "symbol": "BTCUSDT",
            "market_source": "binance",
            "week": 1,
            "generation": 1,
            "results": pd.DataFrame([
                {"model_id": "M1", "name": "Trend"},
            ]),
            "trade_events_delta": {
                "M1": [
                    {
                        "timestamp": "2026-03-26T21:30:00Z",
                        "symbol": "SOLUSDT",
                        "model_id": "M1",
                        "akce": "Výstup LONG (-5)",
                        "strana": "LONG",
                        "cena": 87.10,
                        "pozice": 0.0,
                        "sloty": 0,
                        "opened_at": "2026-03-26T20:20:00Z",
                        "entry_price": 86.66,
                        "quantity_slots": 5,
                        "duvod_vystupu": "TARGET",
                    },
                ]
            },
        }

        rows = _build_closed_positions_rows(summary)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row[2], "M1")
        self.assertEqual(row[4], "SOLUSDT")
        self.assertAlmostEqual(row[6], 86.66)
        self.assertAlmostEqual(row[7], 87.10)

    def test_apply_stream_kline_to_market_appends_new_bar(self) -> None:
        market = simulate_market(days=2, seed=7, interval="1h").tail(5)
        last_ts = market.index[-1]
        next_ts = last_ts + pd.Timedelta(hours=1)
        merged = apply_stream_kline_to_market(
            market,
            {
                "open_time": int(next_ts.timestamp() * 1000),
                "open": 123.0,
                "high": 125.0,
                "low": 122.0,
                "close": 124.0,
            },
        )

        self.assertEqual(len(merged), len(market) + 1)
        self.assertAlmostEqual(float(merged.loc[next_ts, "close"]), 124.0)
        self.assertIn("ret", merged.columns)

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
    def test_live_cycle_allows_same_symbol_side_across_models(self) -> None:
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

        def _weak_confluence(model, market, controlled_signal):
            index = market.index
            confluence = pd.DataFrame(
                {
                    "long_votes": [0] * len(index),
                    "short_votes": [0] * len(index),
                    "long_confidence": [0.0] * len(index),
                    "short_confidence": [0.0] * len(index),
                },
                index=index,
            )
            atr_pct = pd.Series(0.01, index=index)
            return confluence, 5, 2, atr_pct

        strong_signal_a = pd.Series(0.0, index=market_a.index)
        strong_signal_b = pd.Series(1.0, index=market_b.index)

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market_a
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal_a
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_weak_confluence):
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
        self.assertTrue(second["trade_events_delta"]["M2"])
        self.assertEqual(second["final_open_slots"]["M2"], 4)

    def test_live_cycle_keeps_duplicate_symbol_positions_across_models(self) -> None:
        engine = TradingEngine()
        engine.models = [
            ModelSpec("M1", "Trend", "trend_vol", 1),
            ModelSpec("M2", "Momentum", "xsec_momentum", 1),
        ]
        market = _build_market_frame()
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
        ts_key = market.index[-1].isoformat()
        prev = {
            "model_trades": {"M1": [], "M2": []},
            "live_model_state": {
                "M1": {
                    "entry_armed": False,
                    "last_bar_ts": ts_key,
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "open_slots": 2,
                    "entry_price": 100.0,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_price": 95.0,
                    "target_price": 110.0,
                },
                "M2": {
                    "entry_armed": False,
                    "last_bar_ts": ts_key,
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "open_slots": 1,
                    "entry_price": 100.0,
                    "opened_at": "2026-01-01T00:05:00Z",
                    "stop_price": 95.0,
                    "target_price": 110.0,
                },
            },
            "model_open_positions": {
                "M1": [{"symbol": "BTCUSDT", "side": "LONG", "slots": 2}],
                "M2": [{"symbol": "BTCUSDT", "side": "LONG", "slots": 1}],
            },
        }

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            result = engine.run_week(days=7, market_source="binance", symbol="BTCUSDT", interval="15m", previous_summary=prev)

        self.assertGreater(result["final_open_slots"]["M1"], 0)
        self.assertGreater(result["final_open_slots"]["M2"], 0)
        self.assertEqual(result["trade_events_delta"]["M1"], [])
        self.assertEqual(result["trade_events_delta"]["M2"], [])

    def test_run_week_pins_open_live_symbol_and_filters_usdcusdt(self) -> None:
        engine = TradingEngine()
        engine.models = [ModelSpec("M1", "Trend", "trend_vol", 1)]
        btc_market = _build_market_frame()
        sol_market = _build_flat_market_frame()
        universe = pd.DataFrame(
            [
                {"symbol": "USDCUSDT", "opportunity_score": 5.0},
                {"symbol": "BTCUSDT", "opportunity_score": 2.0},
            ]
        )
        prev = {
            "live_model_state": {
                "M1": {
                    "entry_armed": False,
                    "symbol": "SOLUSDT",
                    "side": "LONG",
                    "open_slots": 1,
                    "entry_price": 100.0,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_price": 98.0,
                    "target_price": 102.0,
                }
            },
            "model_open_positions": {"M1": [{"symbol": "SOLUSDT", "side": "LONG", "slots": 1}]},
        }

        def market_for_symbol(source, days, symbol, seed, interval="1d"):
            return sol_market if symbol == "SOLUSDT" else btc_market

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", side_effect=market_for_symbol
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            summary = engine.run_week(days=7, market_source="binance", symbol="BTCUSDT", interval="15m", previous_summary=prev)

        self.assertNotIn("USDCUSDT", summary["candidate_symbols"])
        self.assertEqual(summary["model_selected_symbols"]["M1"], "SOLUSDT")
        self.assertIs(summary["model_markets"]["M1"], sol_market)

    def test_choose_segments_runs_all_runnable_segments_in_binance_mode(self) -> None:
        worker_state = {"segment_cursor": 1}

        selected = _choose_segments(
            ["Scalp", "Intraday", "Swing"],
            active_segment="Swing",
            worker_state=worker_state,
            data_source="binance",
        )

        self.assertEqual(selected, ["Scalp", "Intraday", "Swing"])
        self.assertEqual(worker_state["segment_cursor"], 1)

    def test_choose_segments_round_robins_in_simulation_mode(self) -> None:
        worker_state = {"segment_cursor": 1}

        selected = _choose_segments(
            ["Scalp", "Intraday", "Swing"],
            active_segment="Swing",
            worker_state=worker_state,
            data_source="simulation",
        )

        self.assertEqual(selected, ["Intraday"])
        self.assertEqual(worker_state["segment_cursor"], 2)

    def test_worker_process_lock_allows_only_one_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "worker.lock"
            first = _WorkerProcessLock(lock_path)
            second = _WorkerProcessLock(lock_path)

            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())

            first.release()

            third = _WorkerProcessLock(lock_path)
            self.assertTrue(third.acquire())
            third.release()

    def test_binance_cadence_uses_segment_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_get_binance_cadence_seconds("Scalp"), 5)
            self.assertEqual(_get_binance_cadence_seconds("Intraday"), 10)
            self.assertEqual(_get_binance_cadence_seconds("Swing"), 30)

    def test_binance_cadence_accepts_env_override(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SIRTRADE_BINANCE_DECISION_SECONDS": "12",
                "SIRTRADE_BINANCE_DECISION_SECONDS_SCALP": "3",
            },
            clear=False,
        ):
            self.assertEqual(_get_binance_cadence_seconds("Scalp"), 3)
            self.assertEqual(_get_binance_cadence_seconds("Intraday"), 12)

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
            slot_sizes = {
                str(row["model_id"]): float(row["position_size"])
                for _, row in combined.iterrows()
            }
            self.assertEqual(slot_sizes["SC_M1"], 2.0)
            self.assertEqual(slot_sizes["SW_M1"], 1.0)

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
            "trade_events_delta": {
                "M1": [
                    {
                        "timestamp": "2026-03-24T08:00:00Z",
                        "symbol": "BTCUSDT",
                        "akce": "Výstup LONG (-2)",
                        "strana": "LONG",
                        "cena": 101.0,
                        "opened_at": "2026-03-24T07:00:00Z",
                        "entry_price": 100.0,
                        "quantity_slots": 2,
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
        self.assertEqual(filtered["trade_events_delta"]["M1"], [])
        self.assertEqual(filtered["champion_trades"], [])
        self.assertEqual(filtered["final_open_slots"]["M1"], 0)
        self.assertEqual(filtered["final_positions"]["M1"], 0.0)
        self.assertEqual(filtered["model_open_positions"]["M1"], [])

    def test_save_closed_positions_is_idempotent_for_same_logical_trade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sirtrade.db"
            init_db(db_path=db_path)

            first_summary = {
                "segment": "Scalp",
                "market_source": "binance",
                "week": 1,
                "generation": 1,
                "results": pd.DataFrame([{"model_id": "SC_M1", "name": "Scalp Trend"}]),
                "trade_events_delta": {
                    "SC_M1": [
                        {
                            "timestamp": "2026-04-04T18:26:51.046587+00:00",
                            "opened_at": "2026-04-04T18:21:32.865332+00:00",
                            "symbol": "STOUSDT",
                            "model_id": "SC_M1",
                            "model_name": "Scalp Trend",
                            "akce": "Výstup LONG (-5)",
                            "strana": "LONG",
                            "cena": 0.2628,
                            "entry_price": 0.2489,
                            "quantity_slots": 5,
                            "duvod_vystupu": "TARGET",
                        }
                    ]
                },
            }
            duplicate_summary = {
                **first_summary,
                "trade_events_delta": {
                    "SC_M1": [
                        {
                            "timestamp": "2026-04-04T18:26:52.741495+00:00",
                            "opened_at": "2026-04-04T18:21:32.865332+00:00",
                            "symbol": "STOUSDT",
                            "model_id": "SC_M1",
                            "model_name": "Scalp Trend",
                            "akce": "Výstup LONG (-5)",
                            "strana": "LONG",
                            "cena": 0.2631,
                            "entry_price": 0.2489,
                            "quantity_slots": 5,
                            "duvod_vystupu": "TARGET",
                        }
                    ]
                },
            }

            save_closed_positions(first_summary, db_path=db_path)
            save_closed_positions(duplicate_summary, db_path=db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    "SELECT closed_at, exit_price, hidden FROM closed_positions ORDER BY id ASC"
                ).fetchall()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "2026-04-04T18:26:51.046587+00:00")
            self.assertAlmostEqual(float(rows[0][1]), 0.2628)
            self.assertEqual(int(rows[0][2]), 0)

    def test_dedupe_closed_positions_hides_existing_duplicate_trade_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sirtrade.db"
            init_db(db_path=db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                conn.executemany(
                    """
                    INSERT INTO closed_positions (
                        closed_at, opened_at, segment, model_id, model_name, symbol, side,
                        entry_price, exit_price, quantity_slots, pnl_pct, pnl_status, exit_reason,
                        market_source, week, generation, hidden
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-04-04T18:26:51.046587+00:00",
                            "2026-04-04T18:21:32.865332+00:00",
                            "Scalp",
                            "SC_M1",
                            "Scalp Trend",
                            "STOUSDT",
                            "LONG",
                            0.2489,
                            0.2628,
                            5.0,
                            5.584572,
                            "ZISK",
                            "TARGET",
                            "binance",
                            1,
                            1,
                            0,
                        ),
                        (
                            "2026-04-04T18:26:52.741495+00:00",
                            "2026-04-04T18:21:32.865332+00:00",
                            "Scalp",
                            "SC_M1",
                            "Scalp Trend",
                            "STOUSDT",
                            "LONG",
                            0.2489,
                            0.2631,
                            5.0,
                            5.705102,
                            "ZISK",
                            "TARGET",
                            "binance",
                            1,
                            1,
                            0,
                        ),
                    ],
                )
                _dedupe_closed_positions(conn)
                conn.commit()
                rows = conn.execute(
                    "SELECT closed_at, exit_price, hidden FROM closed_positions ORDER BY id ASC"
                ).fetchall()

            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[0][2]), 0)
            self.assertEqual(int(rows[1][2]), 1)
            self.assertEqual(rows[0][0], "2026-04-04T18:26:51.046587+00:00")

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
        ), patch.object(
            TradingEngine,
            "_estimated_round_trip_cost_pct",
            return_value=0.0002,
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

    def test_scalp_target_distance_enforces_three_to_one_ratio(self) -> None:
        engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        market = _build_flat_market_frame()
        model = ModelSpec("SC_M1", "Scalp Trend", "trend_vol", 1)

        stop_dist, target_dist = engine._resolve_trade_distances(
            model=model,
            market=market,
            close_price=float(market["close"].iloc[-1]),
            vol_step=0.004,
            min_stop_floor=0.001,
        )

        self.assertAlmostEqual(stop_dist, 0.002, places=8)
        self.assertAlmostEqual(target_dist, stop_dist * 3.0, places=8)

    def test_non_scalp_target_distance_enforces_three_to_one_ratio(self) -> None:
        engine = TradingEngine()
        market = _build_market_frame()
        model = ModelSpec("M1", "Trend", "trend_vol", 1)

        stop_dist, target_dist = engine._resolve_trade_distances(
            model=model,
            market=market,
            close_price=float(market["close"].iloc[-1]),
            vol_step=0.01,
            min_stop_floor=0.001,
        )

        self.assertGreater(stop_dist, 0.0)
        self.assertAlmostEqual(target_dist, stop_dist * 3.0, places=8)

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

    def test_scalp_simulation_exits_when_momentum_thesis_invalidates(self) -> None:
        engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        model = ModelSpec("SC_M1", "Scalp Trend", "trend_vol", 1)
        index = pd.date_range("2026-01-01", periods=80, freq="min", tz="UTC")
        close = np.full(len(index), 100.0)
        open_ = close.copy()
        high = close * 1.0002
        low = close * 0.9998
        ret = pd.Series(close).pct_change().fillna(0.0).to_numpy()
        market = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "ret": ret,
                "sentiment": np.full(len(index), 0.4),
                "onchain": np.full(len(index), 0.4),
                "regime": np.zeros(len(index)),
            },
            index=index,
        )
        signal = pd.Series(np.concatenate([np.full(20, 0.8), np.zeros(len(index) - 20)]), index=index)
        confluence = pd.DataFrame(
            {
                "long_votes": np.concatenate([np.full(20, 5), np.zeros(len(index) - 20)]),
                "short_votes": np.zeros(len(index)),
                "long_confidence": np.concatenate([np.full(20, 0.8), np.zeros(len(index) - 20)]),
                "short_confidence": np.zeros(len(index)),
            },
            index=index,
        )
        atr_pct = pd.Series(0.0015, index=index)

        with patch("src.sirtrade.engine.generate_signals", return_value=signal), patch.object(
            TradingEngine,
            "_build_entry_confluence",
            return_value=(confluence, 4, 1, atr_pct),
            autospec=True,
        ), patch.object(
            TradingEngine,
            "_scalp_target_context_cap",
            return_value=0.01,
            autospec=True,
        ), patch.object(
            TradingEngine,
            "_estimated_round_trip_cost_pct",
            return_value=0.0002,
            autospec=True,
        ):
            _, events, final_position, final_open_slots = engine._simulate_model(model, market, symbol="BTCUSDT")

        exit_events = [event for event in events if str(event.get("akce", "")).startswith("Výstup")]
        self.assertTrue(exit_events)
        self.assertEqual(exit_events[0].get("duvod_vystupu"), "SCALP_INVALIDATION")
        self.assertEqual(final_open_slots, 0)
        self.assertEqual(final_position, 0.0)

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
        self.assertIn("SOLUSDT", standard_rows["symbol"].tolist())
        self.assertGreaterEqual(len(set(standard_rows["symbol"].tolist())), 2)

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
        standard_kinds = [model.kind for model in engine.models if model.kind != "copy_trader"]
        self.assertEqual(len(set(standard_kinds)), 5)

    def test_select_candidate_run_prefers_better_sampled_candidate(self) -> None:
        engine = TradingEngine()
        under_sampled = {
            "result": ModelResult(
                model_id="M1",
                name="Under-sampled",
                generation=1,
                symbol="BTCUSDT",
                sortino=1.0,
                calmar=1.0,
                cvar95=0.01,
                max_dd=0.03,
                cost=0.0,
                turnover=0.0,
                score=1.25,
                passed=True,
                closed_trades=1,
                win_rate=100.0,
            ),
            "final_position": 0.05,
            "final_open_slots": 1,
            "opportunity_score": 0.4,
        }
        better_sampled = {
            "result": ModelResult(
                model_id="M1",
                name="Better-sampled",
                generation=1,
                symbol="ETHUSDT",
                sortino=1.0,
                calmar=1.0,
                cvar95=0.01,
                max_dd=0.03,
                cost=0.0,
                turnover=0.0,
                score=1.25,
                passed=True,
                closed_trades=8,
                win_rate=62.0,
            ),
            "final_position": 0.05,
            "final_open_slots": 1,
            "opportunity_score": 0.4,
        }

        selected = engine._select_candidate_run([under_sampled, better_sampled])

        self.assertEqual(selected["result"].name, "Better-sampled")

    def test_segment_minimum_closed_trades_are_configured_per_segment(self) -> None:
        scalp_engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        intraday_engine = TradingEngine(model_namespace="ID", model_label_prefix="Intraday")
        swing_engine = TradingEngine(model_namespace="SW", model_label_prefix="Swing")

        self.assertEqual(scalp_engine._min_closed_trades_for_evolution(), 50)
        self.assertEqual(intraday_engine._min_closed_trades_for_evolution(), 20)
        self.assertEqual(swing_engine._min_closed_trades_for_evolution(), 15)

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

    def test_live_cycle_enters_when_setup_activates_without_waiting_for_new_bar(self) -> None:
        """A newly active setup can enter on the current live candle."""
        engine = TradingEngine()
        engine.models = [ModelSpec("M1", "Trend", "trend_vol", 1)]
        market = _build_market_frame()

        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
        weak_signal = pd.Series(0.0, index=market.index)
        strong_signal = pd.Series(1.0, index=market.index)

        def _weak_confluence(model, market, controlled_signal):
            index = market.index
            confluence = pd.DataFrame(
                {
                    "long_votes": [0] * len(index),
                    "short_votes": [0] * len(index),
                    "long_confidence": [0.0] * len(index),
                    "short_confidence": [0.0] * len(index),
                },
                index=index,
            )
            atr_pct = pd.Series(0.01, index=index)
            return confluence, 5, 2, atr_pct

        def _strong_confluence(model, market, controlled_signal):
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

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            with patch("src.sirtrade.engine.generate_signals", return_value=weak_signal), patch.object(
                TradingEngine,
                "_build_entry_confluence",
                side_effect=_weak_confluence,
            ):
                first = engine.run_week(days=7, market_source="binance", symbol="BTCUSDT", interval="15m")

        self.assertEqual(len(first["trade_events_delta"]["M1"]), 0)
        self.assertTrue(first["live_model_state"]["M1"].get("entry_armed"))
        engine._live_snapshot_cache.clear()

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            with patch("src.sirtrade.engine.generate_signals", return_value=strong_signal), patch.object(
                TradingEngine,
                "_build_entry_confluence",
                side_effect=_strong_confluence,
            ):
                second = engine.run_week(
                    days=7,
                    market_source="binance",
                    symbol="BTCUSDT",
                    interval="15m",
                    previous_summary=first,
                )

        self.assertTrue(second["trade_events_delta"]["M1"])
        self.assertGreater(second["final_open_slots"]["M1"], 0)
        self.assertEqual(second["live_model_state"]["M1"].get("last_bar_ts"), first["live_model_state"]["M1"].get("last_bar_ts"))

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None):
            with patch("src.sirtrade.engine.generate_signals", return_value=strong_signal), patch.object(
                TradingEngine,
                "_build_entry_confluence",
                side_effect=_strong_confluence,
            ):
                third = engine.run_week(
                    days=7,
                    market_source="binance",
                    symbol="BTCUSDT",
                    interval="15m",
                    previous_summary=second,
                )

        self.assertEqual(len(third["trade_events_delta"]["M1"]), 0)
        self.assertGreater(third["final_open_slots"]["M1"], 0)

    def test_same_active_setup_does_not_retrigger_entry_on_same_bar(self) -> None:
        """A setup that stays active must not emit duplicate entries on worker ticks."""
        engine = TradingEngine()
        engine.models = [ModelSpec("M1", "Trend", "trend_vol", 1)]
        market = _build_market_frame()
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
        strong_signal = pd.Series(1.0, index=market.index)

        def _strong_confluence(model, market, controlled_signal):
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

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_strong_confluence):
            first = engine.run_week(days=7, market_source="binance", symbol="BTCUSDT", interval="5m")

        self.assertEqual(len(first["trade_events_delta"]["M1"]), 1)
        self.assertGreater(first["final_open_slots"]["M1"], 0)

        engine._live_snapshot_cache.clear()

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_strong_confluence):
            second = engine.run_week(
                days=7,
                market_source="binance",
                symbol="BTCUSDT",
                interval="5m",
                previous_summary=first,
            )

        self.assertEqual(len(second["trade_events_delta"]["M1"]), 0)
        self.assertEqual(second["final_open_slots"]["M1"], first["final_open_slots"]["M1"])

    def test_live_entry_records_execution_time_separately_from_bar_time(self) -> None:
        engine = TradingEngine()
        engine.models = [ModelSpec("M1", "Trend", "trend_vol", 1)]
        market = _build_market_frame()
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
        strong_signal = pd.Series(1.0, index=market.index)
        execution_ts = pd.Timestamp("2026-01-03T12:34:56Z")

        def _strong_confluence(model, market, controlled_signal):
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

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_strong_confluence), patch(
            "src.sirtrade.engine._current_execution_timestamp",
            return_value=execution_ts,
        ):
            result = engine.run_week(days=7, market_source="binance", symbol="BTCUSDT", interval="5m")

        event = result["trade_events_delta"]["M1"][0]
        self.assertEqual(str(event["executed_at"]), execution_ts.isoformat())
        self.assertEqual(str(result["live_model_state"]["M1"]["opened_at"]), execution_ts.isoformat())
        self.assertEqual(
            pd.to_datetime(event["market_timestamp"], utc=True),
            pd.to_datetime(event["timestamp"], utc=True),
        )
        self.assertNotEqual(str(event["executed_at"]), str(event["timestamp"]))

    def test_live_scalp_cycle_exits_when_setup_invalidates(self) -> None:
        engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        engine.models = [ModelSpec("SC_M1", "Scalp Trend", "trend_vol", 1)]
        market = _build_flat_market_frame()
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
        weak_signal = pd.Series(0.0, index=market.index)

        def _weak_confluence(model, market, controlled_signal):
            index = market.index
            confluence = pd.DataFrame(
                {
                    "long_votes": [0] * len(index),
                    "short_votes": [0] * len(index),
                    "long_confidence": [0.0] * len(index),
                    "short_confidence": [0.0] * len(index),
                },
                index=index,
            )
            atr_pct = pd.Series(0.0015, index=index)
            return confluence, 4, 1, atr_pct

        previous_summary = {
            "model_trades": {"SC_M1": []},
            "live_model_state": {
                "SC_M1": {
                    "entry_armed": False,
                    "setup_active": True,
                    "setup_direction": 1,
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "open_slots": 2,
                    "entry_price": 100.0,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_price": 98.50,
                    "target_price": 101.50,
                    "positions": [
                        {
                            "symbol": "BTCUSDT",
                            "side": "LONG",
                            "open_slots": 2,
                            "entry_price": 100.0,
                            "stop_price": 98.50,
                            "target_price": 101.50,
                            "opened_at": "2026-01-01T00:00:00Z",
                        }
                    ],
                }
            },
            "model_open_positions": {
                "SC_M1": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "slots": 2,
                        "entry_price": 100.0,
                        "opened_at": "2026-01-01T00:00:00Z",
                        "stop_price": 98.50,
                        "target_price": 101.50,
                    }
                ]
            },
        }

        def _fake_simulate(_engine: TradingEngine, model: ModelSpec, market_frame: pd.DataFrame, symbol: str):
            result = ModelResult(
                model_id=model.model_id,
                name=model.name,
                generation=model.generation,
                symbol=symbol,
                sortino=1.0,
                calmar=1.0,
                cvar95=0.01,
                max_dd=0.01,
                cost=0.0,
                turnover=0.0,
                score=1.0,
                passed=True,
            )
            return result, [], 0.0, 0

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=weak_signal
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_weak_confluence), patch.object(
            TradingEngine,
            "_simulate_model",
            side_effect=_fake_simulate,
            autospec=True,
        ):
            result = engine.run_week(days=1, market_source="binance", symbol="BTCUSDT", interval="1m", previous_summary=previous_summary)

        self.assertEqual(result["final_open_slots"]["SC_M1"], 0)
        self.assertEqual(result["trade_events_delta"]["SC_M1"][0]["duvod_vystupu"], "SCALP_INVALIDATION")

    def test_live_scalp_cycle_rearms_on_new_symbol_with_same_setup_direction(self) -> None:
        engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        engine.models = [ModelSpec("SC_M1", "Scalp Trend", "trend_vol", 1)]
        market = _build_market_frame()
        universe = pd.DataFrame([{"symbol": "ETHUSDT", "opportunity_score": 1.0}])
        strong_signal = pd.Series(1.0, index=market.index)

        def _strong_confluence(model, market, controlled_signal):
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

        previous_summary = {
            "model_trades": {"SC_M1": []},
            "live_model_state": {
                "SC_M1": {
                    "entry_armed": False,
                    "setup_active": True,
                    "setup_direction": 1,
                    "symbol": "BTCUSDT",
                    "side": "",
                    "open_slots": 0,
                    "entry_price": 0.0,
                    "opened_at": None,
                    "stop_price": 0.0,
                    "target_price": 0.0,
                    "positions": [],
                }
            },
            "model_open_positions": {"SC_M1": []},
        }

        def _fake_simulate(_engine: TradingEngine, model: ModelSpec, market_frame: pd.DataFrame, symbol: str):
            result = ModelResult(
                model_id=model.model_id,
                name=model.name,
                generation=model.generation,
                symbol=symbol,
                sortino=1.0,
                calmar=1.0,
                cvar95=0.01,
                max_dd=0.01,
                cost=0.0,
                turnover=0.0,
                score=1.0,
                passed=True,
            )
            return result, [], 0.0, 0

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_strong_confluence), patch.object(
            TradingEngine,
            "_simulate_model",
            side_effect=_fake_simulate,
            autospec=True,
        ):
            result = engine.run_week(days=1, market_source="binance", symbol="BTCUSDT", interval="1m", previous_summary=previous_summary)

        self.assertEqual(len(result["trade_events_delta"]["SC_M1"]), 1)
        self.assertIn("Vstup LONG", result["trade_events_delta"]["SC_M1"][0]["akce"])
        self.assertEqual(result["trade_events_delta"]["SC_M1"][0]["symbol"], "ETHUSDT")
        self.assertEqual(result["live_model_state"]["SC_M1"].get("setup_symbol"), "ETHUSDT")
        self.assertGreater(result["final_open_slots"]["SC_M1"], 0)

    def test_live_scalp_entry_skips_trade_when_micro_range_cannot_support_three_to_one(self) -> None:
        engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        engine.models = [ModelSpec("SC_M1", "Scalp Trend", "trend_vol", 1)]
        market = _build_flat_market_frame()
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])
        strong_signal = pd.Series(1.0, index=market.index)

        def _strong_confluence(model, market, controlled_signal):
            index = market.index
            confluence = pd.DataFrame(
                {
                    "long_votes": [5] * len(index),
                    "short_votes": [0] * len(index),
                    "long_confidence": [0.8] * len(index),
                    "short_confidence": [0.0] * len(index),
                },
                index=index,
            )
            atr_pct = pd.Series(0.004, index=index)
            return confluence, 4, 1, atr_pct

        def _fake_simulate(_engine: TradingEngine, model: ModelSpec, market_frame: pd.DataFrame, symbol: str):
            result = ModelResult(
                model_id=model.model_id,
                name=model.name,
                generation=model.generation,
                symbol=symbol,
                sortino=1.0,
                calmar=1.0,
                cvar95=0.01,
                max_dd=0.01,
                cost=0.0,
                turnover=0.0,
                score=1.0,
                passed=True,
            )
            return result, [], 0.0, 0

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch(
            "src.sirtrade.engine.generate_signals", return_value=strong_signal
        ), patch.object(TradingEngine, "_build_entry_confluence", side_effect=_strong_confluence), patch.object(
            TradingEngine,
            "_simulate_model",
            side_effect=_fake_simulate,
            autospec=True,
        ):
            result = engine.run_week(days=1, market_source="binance", symbol="BTCUSDT", interval="1m")

        live_state = result["live_model_state"]["SC_M1"]

        self.assertEqual(result["trade_events_delta"]["SC_M1"], [])
        self.assertEqual(result["final_open_slots"]["SC_M1"], 0)
        self.assertEqual(float(live_state["entry_price"]), 0.0)
        self.assertEqual(float(live_state["stop_price"]), 0.0)
        self.assertEqual(float(live_state["target_price"]), 0.0)

    def test_run_week_handles_inactive_copy_trader_champion_without_ret_crash(self) -> None:
        engine = TradingEngine(model_namespace="SC", model_label_prefix="Scalp")
        market = _build_market_frame()
        universe = pd.DataFrame([{"symbol": "BTCUSDT", "opportunity_score": 1.0}])

        def _very_weak_simulate(_engine: TradingEngine, model: ModelSpec, market_frame: pd.DataFrame, symbol: str):
            score = -2.0 if model.model_id != "SC_MC" else -1.0
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
                score=score,
                passed=False,
            )
            return result, [], 0.0, 0

        with patch("src.sirtrade.engine.scan_binance_long_tail", return_value=universe), patch(
            "src.sirtrade.engine.get_market_data", return_value=market
        ), patch("src.sirtrade.engine.load_top_copy_trader_snapshot", return_value=None), patch.object(
            TradingEngine,
            "_simulate_model",
            side_effect=_very_weak_simulate,
            autospec=True,
        ):
            result = engine.run_week(days=1, market_source="binance", symbol="BTCUSDT", interval="1m")

        self.assertIn("ret", result["market"].columns)
        self.assertEqual(result["champion"]["model_id"], "SC_MC")
        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertGreaterEqual(result["portfolio_vol_annual"], 0.0)


if __name__ == "__main__":
    unittest.main()