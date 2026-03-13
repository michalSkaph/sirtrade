from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.sirtrade.config import DEFAULT_CONFIG
from src.sirtrade.data import fetch_binance_market
from src.sirtrade.engine import TradingEngine
from src.sirtrade.reporting import export_weekly_report
from src.sirtrade.storage import (
    init_db,
    load_closed_positions,
    load_open_positions,
    load_recent_runs,
    save_closed_positions,
    save_open_positions,
    save_week_result,
)
from src.sirtrade.ui_state import (
    clear_segment_runs,
    clear_last_ui_run,
    clear_runtime_state,
    load_last_ui_run,
    load_segment_runs,
    load_runtime_state,
    save_last_ui_run,
    save_segment_runs,
    save_runtime_state,
)

st.set_page_config(page_title="SirTrade", page_icon="📈", layout="wide")
init_db()


@st.cache_data(ttl=3, show_spinner=False)
def _load_closed_positions_cached(limit: int) -> pd.DataFrame:
    return load_closed_positions(limit=limit)


@st.cache_data(ttl=3, show_spinner=False)
def _load_open_positions_cached() -> pd.DataFrame:
    return load_open_positions()


@st.cache_data(ttl=10, show_spinner=False)
def _load_recent_runs_cached(limit: int) -> pd.DataFrame:
    return load_recent_runs(limit=limit)


@st.cache_data(ttl=1, show_spinner=False)
def _fetch_binance_market_cached(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    return fetch_binance_market(symbol=symbol, interval=interval, limit=limit)


CLOSED_POSITIONS_LIMIT = 10000
SIMULATION_WEEKS_PER_CYCLE = 1
FIXED_LIVE_REFRESH_SECONDS = 2
FIXED_SIMULATION_CYCLE_SECONDS = 10


def _split_datetime_column(frame: pd.DataFrame, source_column: str, label_prefix: str) -> pd.DataFrame:
    if source_column not in frame.columns:
        return frame

    out = frame.copy()
    ts = pd.to_datetime(out[source_column], errors="coerce")
    insert_at = out.columns.get_loc(source_column)
    out.insert(insert_at, f"{label_prefix} - Datum", ts.dt.strftime("%d.%m.%y").where(ts.notna(), None))
    out.insert(insert_at + 1, f"{label_prefix} - Čas", ts.dt.strftime("%H:%M").where(ts.notna(), None))
    out = out.drop(columns=[source_column])
    return out


def _load_segment_closed_positions(segment: str, limit: int = CLOSED_POSITIONS_LIMIT) -> pd.DataFrame:
    closed_positions = _load_closed_positions_cached(limit=limit)
    if closed_positions.empty or "model_id" not in closed_positions.columns:
        return pd.DataFrame()

    namespace = f"{SEGMENT_DEFAULTS[segment]['namespace']}_"
    segment_closed = closed_positions[closed_positions["model_id"].astype(str).str.startswith(namespace)].copy()
    if segment_closed.empty:
        return segment_closed

    if "closed_at" in segment_closed.columns:
        segment_closed["closed_at"] = pd.to_datetime(segment_closed["closed_at"], errors="coerce")
        segment_closed = segment_closed[segment_closed["closed_at"].notna()].copy()
    if "opened_at" in segment_closed.columns:
        segment_closed["opened_at"] = pd.to_datetime(segment_closed["opened_at"], errors="coerce")
    if "pnl_pct" in segment_closed.columns:
        segment_closed["pnl_pct"] = pd.to_numeric(segment_closed["pnl_pct"], errors="coerce")
    return segment_closed


def _compute_closed_position_metrics(frame: pd.DataFrame) -> tuple[str, str, int, int, int]:
    if frame.empty or "pnl_pct" not in frame.columns:
        return "N/A", "N/A", 0, 0, 0

    pnl = pd.to_numeric(frame["pnl_pct"], errors="coerce").dropna()
    if pnl.empty:
        return "N/A", "N/A", 0, 0, 0

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    decided = max(1, wins + losses)
    win_rate_label = f"{(wins / decided) * 100:.1f}%"
    avg_pnl_label = f"{pnl.mean():.3f}%"
    return win_rate_label, avg_pnl_label, wins, losses, len(pnl)


def _infer_segment_name(summary: dict[str, object] | None) -> str:
    if isinstance(summary, dict):
        segment = str(summary.get("segment", "")).strip()
        if segment in SEGMENT_DEFAULTS:
            return segment
        interval = str(summary.get("interval", "")).strip().lower()
        for segment_name, cfg in SEGMENT_DEFAULTS.items():
            if str(cfg.get("interval", "")).strip().lower() == interval:
                return segment_name
    return "Swing"


def _infer_segment_from_model_name(model_name: object) -> str | None:
    normalized = str(model_name).strip()
    for segment_name in SEGMENT_DEFAULTS:
        if normalized.startswith(f"{segment_name} |") or normalized.startswith(segment_name):
            return segment_name
    return None


def _segment_namespace(segment: str) -> str:
    return f"{SEGMENT_DEFAULTS[segment]['namespace']}_"


def _placeholder_market_frame(last_price: float, interval: str, periods: int = 120) -> pd.DataFrame:
    price = float(last_price) if pd.notna(last_price) and float(last_price) > 0 else 1.0
    freq_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }
    freq = freq_map.get(str(interval).lower(), "1h")
    index = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=periods, freq=freq)
    return pd.DataFrame(
        {
            "open": [price] * len(index),
            "high": [price] * len(index),
            "low": [price] * len(index),
            "close": [price] * len(index),
            "volume": [0.0] * len(index),
        },
        index=index,
    )


def _restore_missing_segments_from_storage(existing_segments: set[str]) -> dict[str, dict[str, object]]:
    restored: dict[str, dict[str, object]] = {}
    recent_runs = _load_recent_runs_cached(limit=300)
    if recent_runs.empty:
        return restored

    open_positions = _load_open_positions_cached()
    closed_positions = _load_closed_positions_cached(limit=5000)

    recent_runs = recent_runs.copy()
    if "segment" not in recent_runs.columns:
        recent_runs["segment"] = None
    if "interval" not in recent_runs.columns:
        recent_runs["interval"] = None

    inferred_segments = []
    for _, row in recent_runs.iterrows():
        segment_value = str(row.get("segment", "")).strip()
        if segment_value not in SEGMENT_DEFAULTS:
            segment_value = _infer_segment_from_model_name(row.get("champion_model")) or ""
        inferred_segments.append(segment_value)
    recent_runs["segment_inferred"] = inferred_segments

    usable_runs = recent_runs[recent_runs["segment_inferred"].isin(SEGMENT_DEFAULTS.keys())].copy()
    if usable_runs.empty:
        return restored

    usable_runs = usable_runs.sort_values(by=["id"], ascending=False)
    latest_by_segment = usable_runs.drop_duplicates(subset=["segment_inferred"], keep="first")

    def _normalize_side(value: object) -> str:
        side = str(value).strip().upper()
        if side == "BUY":
            return "LONG"
        if side == "SELL":
            return "SHORT"
        return side

    for _, row in latest_by_segment.iterrows():
        segment = str(row["segment_inferred"])
        if segment in existing_segments or segment in restored:
            continue

        namespace = _segment_namespace(segment)
        segment_interval = str(row.get("interval") or SEGMENT_DEFAULTS[segment]["interval"])
        symbol = str(row.get("symbol") or "BTCUSDT")
        market_source = str(row.get("market_source") or "simulation")
        champion_name = str(row.get("champion_model") or f"{segment} | Restored champion")

        segment_open = pd.DataFrame()
        if not open_positions.empty and "model_id" in open_positions.columns:
            segment_open = open_positions[open_positions["model_id"].astype(str).str.startswith(namespace)].copy()

        segment_closed = pd.DataFrame()
        if not closed_positions.empty and "model_id" in closed_positions.columns:
            segment_closed = closed_positions[closed_positions["model_id"].astype(str).str.startswith(namespace)].copy()

        model_rows: list[dict[str, object]] = []
        model_open_positions: dict[str, list[dict[str, object]]] = {}
        final_positions: dict[str, float] = {}
        final_open_slots: dict[str, int] = {}
        known_models: dict[str, str] = {}

        if not segment_open.empty:
            for _, open_row in segment_open.iterrows():
                model_id = str(open_row.get("model_id", ""))
                model_name = str(open_row.get("model_name", model_id))
                known_models[model_id] = model_name
                side = _normalize_side(open_row.get("side", ""))
                qty = float(open_row.get("position_size", 0.0) or 0.0)
                signed_qty = qty if side == "LONG" else (-qty if side == "SHORT" else 0.0)
                final_positions[model_id] = final_positions.get(model_id, 0.0) + signed_qty
                final_open_slots[model_id] = final_open_slots.get(model_id, 0) + int(round(abs(qty)))
                model_open_positions.setdefault(model_id, []).append(
                    {
                        "symbol": str(open_row.get("symbol", symbol)).upper(),
                        "side": side,
                        "model_name": model_name,
                    }
                )

        if not segment_closed.empty:
            for _, closed_row in segment_closed.iterrows():
                model_id = str(closed_row.get("model_id", ""))
                model_name = str(closed_row.get("model_name", model_id))
                known_models.setdefault(model_id, model_name)

        champion_model_id = next(
            (model_id for model_id, model_name in known_models.items() if model_name == champion_name),
            f"{namespace}RESTORED",
        )
        known_models.setdefault(champion_model_id, champion_name)

        for model_id, model_name in known_models.items():
            is_champion = model_id == champion_model_id
            model_rows.append(
                {
                    "model_id": model_id,
                    "name": model_name,
                    "generation": int(row.get("generation") or 0),
                    "sortino": float(row.get("champion_sortino") or 0.0) if is_champion else 0.0,
                    "calmar": float(row.get("champion_calmar") or 0.0) if is_champion else 0.0,
                    "cvar95": float(row.get("champion_cvar95") or 0.0) if is_champion else 0.0,
                    "max_dd": float(row.get("champion_max_dd") or 0.0) if is_champion else 0.0,
                    "cost": 0.0,
                    "turnover": 0.0,
                    "score": float(row.get("champion_score") or 0.0) if is_champion else 0.0,
                    "passed": bool(float(row.get("champion_score") or 0.0) > 0.0) if is_champion else False,
                }
            )

        if not model_rows:
            model_rows.append(
                {
                    "model_id": champion_model_id,
                    "name": champion_name,
                    "generation": int(row.get("generation") or 0),
                    "sortino": float(row.get("champion_sortino") or 0.0),
                    "calmar": float(row.get("champion_calmar") or 0.0),
                    "cvar95": float(row.get("champion_cvar95") or 0.0),
                    "max_dd": float(row.get("champion_max_dd") or 0.0),
                    "cost": 0.0,
                    "turnover": 0.0,
                    "score": float(row.get("champion_score") or 0.0),
                    "passed": bool(float(row.get("champion_score") or 0.0) > 0.0),
                }
            )

        last_price = None
        if not segment_closed.empty and "exit_price" in segment_closed.columns:
            last_price = pd.to_numeric(segment_closed["exit_price"], errors="coerce").dropna().iloc[0] if not pd.to_numeric(segment_closed["exit_price"], errors="coerce").dropna().empty else None

        market_frame = pd.DataFrame()
        if market_source == "binance":
            try:
                market_frame = _fetch_binance_market_cached(symbol=symbol, interval=segment_interval, limit=500)
            except Exception:
                market_frame = pd.DataFrame()
        if market_frame.empty:
            market_frame = _placeholder_market_frame(last_price if last_price is not None else 1.0, segment_interval)

        restored[segment] = {
            "segment": segment,
            "week": int(row.get("week") or 0),
            "generation": int(row.get("generation") or 0),
            "portfolio_vol_annual": 0.0,
            "market_source": market_source,
            "symbol": symbol,
            "interval": segment_interval,
            "champion": {
                "model_id": champion_model_id,
                "name": champion_name,
                "generation": int(row.get("generation") or 0),
                "sortino": float(row.get("champion_sortino") or 0.0),
                "calmar": float(row.get("champion_calmar") or 0.0),
                "cvar95": float(row.get("champion_cvar95") or 0.0),
                "max_dd": float(row.get("champion_max_dd") or 0.0),
                "cost": 0.0,
                "turnover": 0.0,
                "score": float(row.get("champion_score") or 0.0),
                "passed": bool(float(row.get("champion_score") or 0.0) > 0.0),
                "reward_usd": float(row.get("reward_usd") or 0.0),
            },
            "research": [],
            "proposed_orders": [],
            "model_trades": {str(item["model_id"]): [] for item in model_rows},
            "final_positions": final_positions,
            "final_open_slots": final_open_slots,
            "model_open_positions": model_open_positions,
            "results": pd.DataFrame(model_rows),
            "long_tail": pd.DataFrame(
                columns=["symbol", "momentum", "liquidity", "spread_bps", "compliance_risk", "opportunity_score", "trades_24h"]
            ),
            "market": market_frame,
        }

    return restored

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:10px;margin:0 0 0.4rem 0;">
        <div style="width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 55%,#22c55e 100%);display:flex;align-items:center;justify-content:center;color:#ffffff;font-size:16px;font-weight:700;box-shadow:0 8px 20px rgba(29,78,216,0.25);">S</div>
        <div style="font-size:1.35rem;font-weight:700;line-height:1;">SirTrade</div>
    </div>
    """,
    unsafe_allow_html=True,
)

runtime_state = load_runtime_state()
SEGMENT_DEFAULTS = {
    "Scalp": {"interval": "5m", "sim_days": 3, "namespace": "SC"},
    "Intraday": {"interval": "15m", "sim_days": 7, "namespace": "ID"},
    "Swing": {"interval": "4h", "sim_days": 30, "namespace": "SW"},
}

if "engines" not in st.session_state:
    st.session_state.engines = {
        segment: TradingEngine(
            DEFAULT_CONFIG,
            model_namespace=cfg["namespace"],
            model_label_prefix=segment,
        )
        for segment, cfg in SEGMENT_DEFAULTS.items()
    }

if "history_by_segment" not in st.session_state:
    st.session_state.history_by_segment = {segment: [] for segment in SEGMENT_DEFAULTS.keys()}
    normalized_restored: dict[str, dict[str, object]] = {}
    restored_by_segment = load_segment_runs()
    if restored_by_segment:
        for segment, restored in restored_by_segment.items():
            restored_segment = _infer_segment_name(restored)
            if restored_segment in st.session_state.history_by_segment:
                normalized_restored[restored_segment] = restored
    else:
        restored = load_last_ui_run()
        if restored:
            restored_segment = _infer_segment_name(restored)
            if restored_segment in st.session_state.history_by_segment:
                restored["segment"] = restored_segment
                normalized_restored[restored_segment] = restored

    recovered_from_storage = _restore_missing_segments_from_storage(set(normalized_restored.keys()))
    normalized_restored.update(recovered_from_storage)

    for segment, restored in normalized_restored.items():
        st.session_state.history_by_segment[segment] = [restored]

    if normalized_restored:
        save_segment_runs(normalized_restored)

if "active_segment" not in st.session_state:
    st.session_state.active_segment = str(runtime_state.get("active_segment", "Swing"))
if "interval" not in st.session_state:
    st.session_state.interval = SEGMENT_DEFAULTS["Swing"]["interval"]
if "simulation_running_by_segment" not in st.session_state:
    persisted_segment_state = runtime_state.get("simulation_running_by_segment", {})
    fallback_running = bool(runtime_state.get("simulation_running", False))
    st.session_state.simulation_running_by_segment = {
        segment: bool(persisted_segment_state.get(segment, fallback_running))
        for segment in SEGMENT_DEFAULTS.keys()
    }
if "auto_center_last_candle" not in st.session_state:
    st.session_state.auto_center_last_candle = bool(runtime_state.get("auto_center_last_candle", True))
if "data_source" not in st.session_state:
    st.session_state.data_source = str(runtime_state.get("data_source", "simulation"))
if "symbol" not in st.session_state:
    st.session_state.symbol = str(runtime_state.get("symbol", "BTCUSDT"))
if "live_refresh_enabled" not in st.session_state:
    st.session_state.live_refresh_enabled = bool(runtime_state.get("live_refresh_enabled", True))
if "live_refresh_seconds" not in st.session_state:
    st.session_state.live_refresh_seconds = FIXED_LIVE_REFRESH_SECONDS
if "live_refresh_when_stopped" not in st.session_state:
    st.session_state.live_refresh_when_stopped = bool(runtime_state.get("live_refresh_when_stopped", True))
if "simulation_cycle_seconds" not in st.session_state:
    st.session_state.simulation_cycle_seconds = FIXED_SIMULATION_CYCLE_SECONDS
if "active_view" not in st.session_state:
    st.session_state.active_view = str(runtime_state.get("active_view", "Dashboard"))
if "last_simulation_tick" not in st.session_state:
    st.session_state.last_simulation_tick = float(runtime_state.get("last_simulation_tick", 0.0))
if "ui_active_segment" not in st.session_state:
    st.session_state.ui_active_segment = st.session_state.active_segment
if "ui_active_view" not in st.session_state:
    st.session_state.ui_active_view = st.session_state.active_view

st.session_state.live_refresh_seconds = FIXED_LIVE_REFRESH_SECONDS
st.session_state.simulation_cycle_seconds = FIXED_SIMULATION_CYCLE_SECONDS

prev_active_segment = str(st.session_state.active_segment)
prev_active_view = str(st.session_state.active_view)
prev_data_source = str(st.session_state.data_source)

active_segment_running = bool(st.session_state.simulation_running_by_segment.get(st.session_state.active_segment, False))
has_running_segments = any(st.session_state.simulation_running_by_segment.values())
force_simulation_cycle = False
status_run = "BĚŽÍ" if active_segment_running else "STOP"
status_source = "Simulace" if st.session_state.data_source == "simulation" else "Binance"
status_profile = st.session_state.active_segment
status_symbol = st.session_state.symbol
status_interval = SEGMENT_DEFAULTS.get(st.session_state.active_segment, SEGMENT_DEFAULTS["Swing"])["interval"]

with st.sidebar:
    st.header("Nastavení")
    st.selectbox(
        "Segment",
        ["Scalp", "Intraday", "Swing"],
        index=["Scalp", "Intraday", "Swing"].index(st.session_state.ui_active_segment)
        if st.session_state.ui_active_segment in ["Scalp", "Intraday", "Swing"]
        else 2,
        help="Vybere detail segmentu v hlavní části aplikace.",
        key="ui_active_segment",
    )
    st.session_state.active_segment = st.session_state.ui_active_segment

    data_source = st.selectbox(
        "Data",
        ["simulation", "binance"],
        index=["simulation", "binance"].index(st.session_state.data_source)
        if st.session_state.data_source in ["simulation", "binance"]
        else 0,
        format_func=lambda value: {"simulation": "Simulace", "binance": "Binance"}.get(value, value),
    )
    st.session_state.data_source = data_source

    symbol = st.session_state.symbol
    st.caption(f"Trh: {symbol}")
    st.caption(
        f"Simulace běží do ručního vypnutí. Každý cyklus přidá {SIMULATION_WEEKS_PER_CYCLE} týden. Graf se obnovuje po {FIXED_LIVE_REFRESH_SECONDS} s, přepočet po {FIXED_SIMULATION_CYCLE_SECONDS} s."
    )

    st.session_state.auto_center_last_candle = st.checkbox(
        "Držet graf na konci",
        value=st.session_state.auto_center_last_candle,
        help="Po každém kroku posune graf na nejnovější data.",
    )
    st.session_state.live_refresh_enabled = st.checkbox(
        "Živý graf",
        value=st.session_state.live_refresh_enabled,
        help="Průběžně obnovuje sekci Grafy s aktuální cenou.",
    )
    st.session_state.live_refresh_when_stopped = st.checkbox(
        "Živý graf i při stopu",
        value=st.session_state.live_refresh_when_stopped,
        help="Nechá graf běžet i při zastavené simulaci.",
    )
    st.markdown("### Běh segmentů")
    for segment in ["Scalp", "Intraday", "Swing"]:
        is_running = bool(st.session_state.simulation_running_by_segment.get(segment, False))
        run_label = f"{segment}: {'Zastavit' if is_running else 'Spustit'} simulaci"
        if st.button(run_label, key=f"run_toggle_{segment}", use_container_width=True):
            st.session_state.simulation_running_by_segment[segment] = not is_running
            force_simulation_cycle = force_simulation_cycle or (not is_running)
    reset_btn = st.button("Resetovat")

    st.markdown("---")
    st.subheader("Riziková politika")
    cfg = DEFAULT_CONFIG
    st.write(f"Cílová volatilita: {cfg.risk.target_vol_annual:.0%}")
    st.write(f"Měkký DD limit: {cfg.risk.soft_dd_alert:.0%}")
    st.write(f"Tvrdý DD limit: {cfg.risk.hard_dd_limit:.0%}")
    st.write(f"Maximální expozice na aktivum: {cfg.risk.max_asset_exposure:.0%}")

if reset_btn:
    st.session_state.engines = {
        segment: TradingEngine(
            DEFAULT_CONFIG,
            model_namespace=cfg["namespace"],
            model_label_prefix=segment,
        )
        for segment, cfg in SEGMENT_DEFAULTS.items()
    }
    st.session_state.history_by_segment = {segment: [] for segment in SEGMENT_DEFAULTS.keys()}
    st.session_state.simulation_running_by_segment = {segment: False for segment in SEGMENT_DEFAULTS.keys()}
    clear_last_ui_run()
    clear_segment_runs()
    clear_runtime_state()
    st.rerun()

view_options = ["Dashboard", "Grafy", "Pozice", "Uzavřené pozice", "Analýza", "Historie & Export"]
st.radio(
    "Sekce",
    view_options,
    index=view_options.index(st.session_state.ui_active_view)
    if st.session_state.ui_active_view in view_options
    else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="ui_active_view",
)
st.session_state.active_view = st.session_state.ui_active_view

ui_interaction_detected = any(
    [
        st.session_state.active_segment != prev_active_segment,
        st.session_state.active_view != prev_active_view,
        st.session_state.data_source != prev_data_source,
    ]
)

active_segment_running = bool(st.session_state.simulation_running_by_segment.get(st.session_state.active_segment, False))
has_running_segments = any(st.session_state.simulation_running_by_segment.values())
now_ts = time.time()
min_cycle_seconds = max(1, int(st.session_state.simulation_cycle_seconds))
should_run_simulation = bool(
    has_running_segments
    and (force_simulation_cycle or st.session_state.active_view == "Dashboard")
    and (force_simulation_cycle or not ui_interaction_detected)
    and (
        force_simulation_cycle
        or (now_ts - float(st.session_state.last_simulation_tick)) >= float(min_cycle_seconds)
    )
)
if should_run_simulation:
    st.session_state.last_simulation_tick = now_ts

save_runtime_state(
    {
        "simulation_running": has_running_segments,
        "simulation_running_by_segment": st.session_state.simulation_running_by_segment,
        "auto_center_last_candle": st.session_state.auto_center_last_candle,
        "active_segment": st.session_state.active_segment,
        "data_source": st.session_state.data_source,
        "symbol": st.session_state.symbol,
        "live_refresh_enabled": st.session_state.live_refresh_enabled,
        "live_refresh_seconds": int(st.session_state.live_refresh_seconds),
        "live_refresh_when_stopped": st.session_state.live_refresh_when_stopped,
        "simulation_cycle_seconds": int(st.session_state.simulation_cycle_seconds),
        "active_view": st.session_state.active_view,
        "last_simulation_tick": float(st.session_state.last_simulation_tick),
    }
)

if should_run_simulation:
    for _ in range(SIMULATION_WEEKS_PER_CYCLE):
        for segment, cfg in SEGMENT_DEFAULTS.items():
            if not st.session_state.simulation_running_by_segment.get(segment, False):
                continue
            result = st.session_state.engines[segment].run_week(
                days=int(cfg["sim_days"]),
                market_source=data_source,
                symbol=symbol,
                interval=str(cfg["interval"]),
            )
            result["segment"] = segment
            st.session_state.history_by_segment[segment].append(result)
            save_week_result(result)
            save_open_positions(result)
            save_closed_positions(result)
            if segment == st.session_state.active_segment:
                save_last_ui_run(result)
                st.session_state.last_exports = export_weekly_report(result, DEFAULT_CONFIG)
    _load_open_positions_cached.clear()
    _load_closed_positions_cached.clear()
    _load_recent_runs_cached.clear()

latest_runs_by_segment = {
    segment: history[-1]
    for segment, history in st.session_state.history_by_segment.items()
    if history
}
if latest_runs_by_segment:
    save_segment_runs(latest_runs_by_segment)

has_any_history = any(len(history) > 0 for history in st.session_state.history_by_segment.values())
if not has_any_history:
    st.caption("Spusť simulaci pro první týdenní vyhodnocení.")
else:
    active_history = st.session_state.history_by_segment.get(st.session_state.active_segment, [])
    if not active_history:
        st.caption(f"Segment {st.session_state.active_segment} zatím nemá data.")
        st.stop()

    latest = active_history[-1]
    latest_by_segment = {
        segment: history[-1]
        for segment, history in st.session_state.history_by_segment.items()
        if history
    }
    live_market_price = None
    live_market_change_pct = None

    source_label = {"simulation": "Simulace", "binance": "Binance"}.get(latest["market_source"], latest["market_source"])

    refreshable_views = {"Grafy"}
    if (
        st.session_state.live_refresh_enabled
        and st.session_state.active_view in refreshable_views
        and (active_segment_running or st.session_state.live_refresh_when_stopped)
    ):
        refresh_ms = max(1, int(st.session_state.live_refresh_seconds)) * 1000
        st_autorefresh(interval=refresh_ms, key="sirtrade_live_refresh")

    if (
        st.session_state.live_refresh_enabled
        and st.session_state.active_view == "Grafy"
        and latest.get("market_source") == "binance"
    ):
        try:
            live_market = _fetch_binance_market_cached(
                symbol=latest.get("symbol", st.session_state.symbol),
                interval=latest.get("interval", st.session_state.interval),
                limit=2,
            )
            if not live_market.empty:
                live_market_price = float(live_market["close"].iloc[-1])
                if len(live_market) > 1 and float(live_market["close"].iloc[-2]) != 0.0:
                    prev_price = float(live_market["close"].iloc[-2])
                    live_market_change_pct = ((live_market_price - prev_price) / prev_price) * 100
        except Exception:
            live_market_price = None
            live_market_change_pct = None

    if st.session_state.active_view == "Dashboard":
        st.subheader(f"Detail segmentu: {st.session_state.active_segment}")
        dashboard_closed_positions = _load_segment_closed_positions(st.session_state.active_segment)
        win_rate_label, avg_pnl_label, _, _, _ = _compute_closed_position_metrics(dashboard_closed_positions)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Týden", latest["week"])
        c2.metric("Generace", latest["generation"])
        c3.metric("Win-rate segmentu", win_rate_label)
        c4.metric("Avg PnL segmentu", avg_pnl_label)
        c5, c6 = st.columns(2)
        c5.metric("Volatilita portfolia (roč.)", f"{latest['portfolio_vol_annual']:.2%}")
        c6.metric("Odměna vítěze", "$1")
        if live_market_price is not None:
            st.metric(
                f"Aktuální cena {latest['symbol']}",
                f"{live_market_price:.6f}",
                None if live_market_change_pct is None else f"{live_market_change_pct:.3f}%",
            )
        st.caption(
            f"Zdroj dat: {source_label} | Symbol: {latest['symbol']} | Timeframe: {latest.get('interval', '1d')} | Exekuce: pouze dry-run"
        )
        if (
            active_segment_running
            and st.session_state.live_refresh_enabled
            and st.session_state.active_view in refreshable_views
        ):
            st.caption(f"Live refresh aktivní: každých {st.session_state.live_refresh_seconds}s")

        st.subheader("Vítěz týdne")
        champ = latest["champion"]
        st.write(
            {
                "Model": champ["name"],
                "Skóre": round(champ["score"], 4),
                "Sortino": round(champ["sortino"], 3),
                "Calmar": round(champ["calmar"], 3),
                "Max DD": round(champ["max_dd"], 4),
                "CVaR95": round(champ["cvar95"], 4),
                "Odměna (USD)": champ["reward_usd"],
            }
        )

        st.subheader("Leaderboard modelů")
        leaderboard = latest["results"].rename(
            columns={
                "model_id": "ID modelu",
                "name": "Název modelu",
                "generation": "Generace",
                "sortino": "Sortino",
                "calmar": "Calmar",
                "cvar95": "CVaR95",
                "max_dd": "Max DD",
                "cost": "Náklad",
                "turnover": "Obrat",
                "score": "Skóre",
                "passed": "Splnil limity",
            }
        )
        leaderboard_config = {
            "ID modelu": st.column_config.TextColumn("ID modelu", help="Interní identifikátor modelu."),
            "Název modelu": st.column_config.TextColumn("Název modelu", help="Název obchodního modelu."),
            "Generace": st.column_config.NumberColumn("Generace", help="Generace evolučního cyklu modelů."),
            "Sortino": st.column_config.NumberColumn("Sortino", help="Výnos očištěný o downside volatilitu. Vyšší je lepší."),
            "Calmar": st.column_config.NumberColumn("Calmar", help="Poměr výnosu k max drawdownu. Vyšší je lepší."),
            "CVaR95": st.column_config.NumberColumn("CVaR95", help="Průměrná ztráta v nejhorších 5 % scénářů. Nižší je lepší."),
            "Max DD": st.column_config.NumberColumn("Max DD", help="Největší pokles equity křivky od maxima. Nižší je lepší."),
            "Náklad": st.column_config.NumberColumn("Náklad", help="Odhad transakčních nákladů po započtení poplatků."),
            "Obrat": st.column_config.NumberColumn("Obrat", help="Intenzita změn pozic (turnover). Vyšší obrat zvyšuje náklady."),
            "Skóre": st.column_config.NumberColumn("Skóre", help="Výsledné skóre decision matrix pro pořadí modelu."),
            "Splnil limity": st.column_config.CheckboxColumn("Splnil limity", help="Zda model splnil minimální riskové prahy."),
        }
        st.dataframe(leaderboard, use_container_width=True, column_config=leaderboard_config)

    if st.session_state.active_view == "Pozice":
        st.subheader("Detail pozic modelů")
        st.caption("Modely alokují pozice autonomně napříč více coiny (max 5 slotů na model).")
        model_position_rows = []
        latest_market_price = float(latest["market"]["close"].iloc[-1])
        if live_market_price is not None:
            latest_market_price = live_market_price
        vol_latest = float(latest["market"]["close"].pct_change().rolling(20).std().iloc[-1])
        if pd.isna(vol_latest) or vol_latest <= 0:
            vol_latest = 0.015

        for _, row in latest["results"].iterrows():
            model_id = str(row["model_id"])
            model_name = str(row["name"])
            position_value = float(latest.get("final_positions", {}).get(model_id, 0.0))
            open_slots = int(latest.get("final_open_slots", {}).get(model_id, 0))
            side = "LONG" if position_value > 0 else ("SHORT" if position_value < 0 else "-")
            is_open = abs(position_value) > 1e-9

            entry_price = None
            opened_at = None
            target_price = None
            stop_price = None
            pnl_pct = None

            trades_model = pd.DataFrame(latest.get("model_trades", {}).get(model_id, []))
            if is_open and not trades_model.empty:
                entry_events = trades_model[trades_model["akce"].str.contains("Vstup")]
                if not entry_events.empty:
                    last_entry = entry_events.sort_values("timestamp").iloc[-1]
                    entry_price = float(last_entry["cena"])
                    opened_at = str(last_entry["timestamp"])
                    stop_dist = max(0.005, 1.0 * vol_latest)
                    target_dist = max(0.01, 2.0 * vol_latest)
                    if position_value > 0:
                        stop_price = entry_price * (1 - stop_dist)
                        target_price = entry_price * (1 + target_dist)
                        pnl_pct = ((latest_market_price - entry_price) / entry_price) * 100
                    else:
                        stop_price = entry_price * (1 + stop_dist)
                        target_price = entry_price * (1 - target_dist)
                        pnl_pct = ((entry_price - latest_market_price) / entry_price) * 100

            model_position_rows.append(
                {
                    "ID modelu": model_id,
                    "Model": model_name,
                    "Symbol": ", ".join(
                        [
                            str(item.get("symbol", latest["symbol"]))
                            for item in latest.get("model_open_positions", {}).get(model_id, [])
                        ]
                    )
                    or latest["symbol"],
                    "Pozice otevřená": "ANO" if is_open else "NE",
                    "Směr": side,
                    "Sloty": f"{open_slots}/5",
                    "Vstupní cena": round(entry_price, 6) if entry_price is not None else None,
                    "Aktuální cena": round(latest_market_price, 6),
                    "Target": round(target_price, 6) if target_price is not None else None,
                    "Stop": round(stop_price, 6) if stop_price is not None else None,
                    "Nerealizované PnL %": round(pnl_pct, 3) if pnl_pct is not None else None,
                    "Otevřeno od": opened_at,
                }
            )

        model_positions_df = pd.DataFrame(model_position_rows)
        model_positions_df = _split_datetime_column(model_positions_df, "Otevřeno od", "Otevřeno")

        def _style_side(value):
            if value == "LONG":
                return "background-color: #14532d; color: #dcfce7; font-weight: 600;"
            if value == "SHORT":
                return "background-color: #7f1d1d; color: #fee2e2; font-weight: 600;"
            return "background-color: #374151; color: #e5e7eb;"

        def _style_pnl(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            if value > 0:
                return "background-color: #14532d; color: #dcfce7;"
            if value < 0:
                return "background-color: #7f1d1d; color: #fee2e2;"
            return ""

        styled_positions = (
            model_positions_df.style
            .map(_style_side, subset=["Směr"])
            .map(_style_pnl, subset=["Nerealizované PnL %"])
        )
        st.dataframe(styled_positions, use_container_width=True)

        st.subheader("Otevřené pozice (držené i po vypnutí aplikace)")
        open_positions = _load_open_positions_cached()
        namespace = f"{SEGMENT_DEFAULTS[st.session_state.active_segment]['namespace']}_"
        if not open_positions.empty and "model_id" in open_positions.columns:
            open_positions = open_positions[open_positions["model_id"].astype(str).str.startswith(namespace)].copy()
        if open_positions.empty:
            st.info(f"Pro segment {st.session_state.active_segment} momentálně nejsou evidované žádné otevřené paper pozice.")
        else:
            open_view = open_positions.rename(
                columns={
                    "id": "ID",
                    "updated_at": "Naposledy aktualizováno",
                    "model_id": "ID modelu",
                    "model_name": "Název modelu",
                    "symbol": "Symbol",
                    "side": "Směr",
                    "position_size": "Velikost pozice",
                    "market_source": "Zdroj dat",
                }
            )
            open_view["Zdroj dat"] = open_view["Zdroj dat"].replace({"simulation": "Simulace", "binance": "Binance"})
            open_view = _split_datetime_column(open_view, "Naposledy aktualizováno", "Aktualizace")
            if "Směr" in open_view.columns:
                open_view["Směr"] = (
                    open_view["Směr"]
                    .astype(str)
                    .str.upper()
                    .replace({
                        "BUY": "LONG",
                        "SELL": "SHORT",
                    })
                )

            def _style_side_open(value):
                if value == "LONG":
                    return "background-color: #14532d; color: #dcfce7; font-weight: 600;"
                if value == "SHORT":
                    return "background-color: #7f1d1d; color: #fee2e2; font-weight: 600;"
                return "background-color: #374151; color: #e5e7eb;"

            styled_open_view = open_view.style.map(_style_side_open, subset=["Směr"])
            st.dataframe(styled_open_view, use_container_width=True)

    if st.session_state.active_view == "Uzavřené pozice":
        st.subheader("Přehled uzavřených pozic")
        closed_positions = _load_segment_closed_positions(st.session_state.active_segment)
        if closed_positions.empty:
            st.info(f"Pro segment {st.session_state.active_segment} zatím nejsou uložené žádné uzavřené obchody.")
        else:
            valid_closed = closed_positions.copy()

            if valid_closed.empty:
                st.info("Uzavřené obchody nemají validní datum uzavření.")
            else:
                min_date = valid_closed["closed_at"].dt.date.min()
                max_date = valid_closed["closed_at"].dt.date.max()
                d1, d2 = st.columns(2)
                date_from = d1.date_input("Od data", value=min_date, min_value=min_date, max_value=max_date)
                date_to = d2.date_input("Do data", value=max_date, min_value=min_date, max_value=max_date)

                filtered = valid_closed[
                    (valid_closed["closed_at"].dt.date >= date_from)
                    & (valid_closed["closed_at"].dt.date <= date_to)
                ].copy()

                if filtered.empty:
                    st.warning("Pro zvolené období nejsou žádné uzavřené obchody.")
                else:
                    filtered["pnl_status"] = filtered["pnl_status"].astype(str).str.upper()
                    filtered["side"] = filtered["side"].astype(str).str.upper().replace({"BUY": "LONG", "SELL": "SHORT"})

                    overview = filtered.rename(
                        columns={
                            "closed_at": "Uzavřeno",
                            "opened_at": "Otevřeno",
                            "model_id": "ID modelu",
                            "model_name": "Model",
                            "symbol": "Symbol",
                            "side": "Směr",
                            "entry_price": "Vstupní cena",
                            "exit_price": "Výstupní cena",
                            "quantity_slots": "Sloty",
                            "pnl_pct": "PnL %",
                            "pnl_status": "Výsledek",
                            "market_source": "Zdroj dat",
                            "week": "Týden",
                            "generation": "Generace",
                        }
                    )
                    if "exit_reason" in overview.columns:
                        overview = overview.drop(columns=["exit_reason"])
                    overview["Zdroj dat"] = overview["Zdroj dat"].replace({"simulation": "Simulace", "binance": "Binance"})
                    overview = _split_datetime_column(overview, "Uzavřeno", "Uzavřeno")
                    overview = _split_datetime_column(overview, "Otevřeno", "Otevřeno")

                    win_rate_label, avg_pnl_label, _, _, _ = _compute_closed_position_metrics(filtered)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Uzavřené obchody", len(overview))
                    m2.metric("Win rate", win_rate_label)
                    m3.metric("Průměrné PnL", avg_pnl_label)

                    def _style_result(value):
                        if value == "ZISK":
                            return "background-color: #14532d; color: #dcfce7; font-weight: 600;"
                        if value == "ZTRÁTA":
                            return "background-color: #7f1d1d; color: #fee2e2; font-weight: 600;"
                        return "background-color: #374151; color: #e5e7eb;"

                    def _style_pnl(value):
                        if pd.isna(value):
                            return ""
                        if float(value) > 0:
                            return "background-color: #14532d; color: #dcfce7;"
                        if float(value) < 0:
                            return "background-color: #7f1d1d; color: #fee2e2;"
                        return ""

                    styled_overview = overview.style.map(_style_result, subset=["Výsledek"]).map(_style_pnl, subset=["PnL %"])
                    st.dataframe(styled_overview, use_container_width=True)

    if st.session_state.active_view == "Grafy":
        st.subheader("Graf ceny a obchody modelu")
        if st.session_state.live_refresh_enabled:
            st.caption(f"Realtime aktivní: aktualizace ceny a grafu každých {st.session_state.live_refresh_seconds}s.")
        if "chart_interval" not in st.session_state:
            st.session_state.chart_interval = latest.get("interval", st.session_state.interval)

        chart_intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
        interval_col, _ = st.columns([1, 3])
        st.session_state.chart_interval = interval_col.selectbox(
            "Časový rámec grafu",
            chart_intervals,
            index=chart_intervals.index(st.session_state.chart_interval)
            if st.session_state.chart_interval in chart_intervals
            else chart_intervals.index("1h"),
            help="Mění timeframe interního realtime grafu.",
            key="chart_interval_selector",
        )

        market_df = latest["market"].copy()
        model_options = [(row["model_id"], row["name"]) for _, row in latest["results"].iterrows()]
        default_model = next((i for i, opt in enumerate(model_options) if opt[0] == latest["champion"]["model_id"]), 0)
        selected_model = st.selectbox(
            "Model pro vykreslení obchodů",
            options=model_options,
            index=default_model,
            format_func=lambda item: f"{item[0]} — {item[1]}",
        )

        selected_model_id = selected_model[0]
        trades = latest["model_trades"].get(selected_model_id, [])
        trades_df = pd.DataFrame(trades)
        model_coin_positions = latest.get("model_open_positions", {}).get(selected_model_id, [])
        model_coin_symbols = [str(item.get("symbol", latest["symbol"])).upper() for item in model_coin_positions]

        def _extract_slots_from_action(action: str, is_entry: bool) -> int:
            pattern = r"\(\+(\d+)\)" if is_entry else r"\(-?(\d+)\)"
            match = re.search(pattern, str(action))
            if not match:
                return 1
            try:
                return max(1, int(match.group(1)))
            except Exception:
                return 1

        open_legs: list[dict] = []
        if not trades_df.empty and {"timestamp", "akce", "strana", "cena"}.issubset(trades_df.columns):
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce")
            trades_df = trades_df[trades_df["timestamp"].notna()].sort_values("timestamp")
            for _, event in trades_df.iterrows():
                action = str(event.get("akce", ""))
                side = str(event.get("strana", "")).upper()
                if side not in {"LONG", "SHORT"}:
                    side = "LONG" if "LONG" in action.upper() else ("SHORT" if "SHORT" in action.upper() else "LONG")
                price = float(event.get("cena", 0.0))
                ts = event["timestamp"]

                if "Vstup" in action:
                    qty = _extract_slots_from_action(action, is_entry=True)
                    for _ in range(qty):
                        open_legs.append(
                            {
                                "side": side,
                                "entry_price": price,
                                "entry_time": ts,
                            }
                        )
                elif "Výstup" in action:
                    qty = _extract_slots_from_action(action, is_entry=False)
                    same_side_idx = [i for i, leg in enumerate(open_legs) if leg["side"] == side]
                    for index in same_side_idx[:qty]:
                        open_legs[index]["_close"] = True
                    open_legs = [leg for leg in open_legs if not leg.get("_close")]

        if model_coin_symbols:
            for idx, leg in enumerate(open_legs):
                leg["symbol"] = model_coin_symbols[idx % len(model_coin_symbols)]

        position_options = []
        for idx, leg in enumerate(open_legs, start=1):
            entry_time = pd.to_datetime(leg["entry_time"]).strftime("%Y-%m-%d %H:%M")
            leg_symbol = str(leg.get("symbol", latest["symbol"])).upper()
            label = f"{leg_symbol} | {leg['side']} | pozice #{idx} | vstup {entry_time}"
            position_options.append((idx - 1, label))

        selected_leg = None
        if position_options:
            selected_position_option = st.selectbox(
                "Vybraná otevřená pozice pro overlay",
                options=position_options,
                index=0,
                format_func=lambda item: item[1],
                help="Vyber konkrétní otevřenou pozici. Overlay target/stop se přepočítá podle ní.",
            )
            selected_leg = open_legs[selected_position_option[0]]

        available_symbols = [str(latest["symbol"]).upper()]
        available_symbols.extend([str(symbol).upper() for symbol in model_coin_symbols if symbol])
        if selected_leg is not None:
            available_symbols.append(str(selected_leg.get("symbol", latest["symbol"])).upper())
        available_symbols = sorted(list(dict.fromkeys(available_symbols)))

        default_symbol = str(selected_leg.get("symbol", latest["symbol"])).upper() if selected_leg else str(latest["symbol"]).upper()
        selected_symbol_for_overlay = st.selectbox(
            "Coin pro realtime graf",
            options=available_symbols,
            index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        )

        overlay_market_df = market_df.copy()
        if latest.get("market_source") == "binance":
            try:
                overlay_market_df = _fetch_binance_market_cached(
                    symbol=selected_symbol_for_overlay,
                    interval=st.session_state.chart_interval,
                    limit=1000,
                )
            except Exception:
                overlay_market_df = market_df.copy()

        for col_name in ["open", "high", "low", "close"]:
            if col_name in overlay_market_df.columns:
                overlay_market_df[col_name] = pd.to_numeric(overlay_market_df[col_name], errors="coerce")
        overlay_market_df = overlay_market_df.dropna(subset=["open", "high", "low", "close"]).sort_index()
        if overlay_market_df.empty:
            st.warning("Pro zvolený časový rámec nejsou dostupná data grafu.")
            st.stop()

        current_coin_price = float(overlay_market_df["close"].iloc[-1])
        current_coin_change = None
        if len(overlay_market_df) > 1 and float(overlay_market_df["close"].iloc[-2]) != 0.0:
            prev_coin_price = float(overlay_market_df["close"].iloc[-2])
            current_coin_change = ((current_coin_price - prev_coin_price) / prev_coin_price) * 100
        st.metric(
            f"Aktuální cena {selected_symbol_for_overlay}",
            f"{current_coin_price:.6f}",
            None if current_coin_change is None else f"{current_coin_change:.3f}%",
        )

        if selected_leg is not None:
            selected_entry = float(selected_leg["entry_price"])
            selected_side = str(selected_leg["side"]).upper()
            selected_current = float(overlay_market_df["close"].iloc[-1])
            if selected_side == "LONG":
                selected_pnl_pct = ((selected_current - selected_entry) / selected_entry) * 100 if selected_entry != 0 else 0.0
            else:
                selected_pnl_pct = ((selected_entry - selected_current) / selected_entry) * 100 if selected_entry != 0 else 0.0

            p_sel1, p_sel2, p_sel3 = st.columns(3)
            p_sel1.metric("Vybraná pozice", f"{selected_symbol_for_overlay} | {selected_side}")
            p_sel2.metric("Vstup → Aktuální", f"{selected_entry:.6f}", f"{selected_current:.6f}")
            p_sel3.metric("Průběžné PnL", f"{selected_pnl_pct:.3f}%")

        final_positions = latest.get("final_positions", {})
        final_open_slots = latest.get("final_open_slots", {})
        selected_position = float(final_positions.get(selected_model_id, 0.0))
        selected_slots = int(final_open_slots.get(selected_model_id, 0))
        aktivni_pozice = "ANO" if abs(selected_position) > 1e-9 else "NE"
        smer_pozice = "LONG" if selected_position > 0 else ("SHORT" if selected_position < 0 else "-")
        pocet_vstupu = int(trades_df["akce"].str.contains("Vstup").sum()) if not trades_df.empty else 0
        pocet_vystupu = int(trades_df["akce"].str.contains("Výstup").sum()) if not trades_df.empty else 0

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Aktivní pozice modelu", aktivni_pozice)
        p2.metric("Směr", smer_pozice)
        p3.metric("Počet vstupů", pocet_vstupu)
        p4.metric("Počet výstupů", pocet_vystupu)
        p5.metric("Otevřené pozice", f"{selected_slots}/5")
        st.caption("Každý model může mít současně otevřeno maximálně 5 pozic (slotů) na zvoleném symbolu.")

        st.markdown("**Realtime interní graf: vstupy/výstupy + target/stop**")

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=overlay_market_df.index,
                open=overlay_market_df["open"],
                high=overlay_market_df["high"],
                low=overlay_market_df["low"],
                close=overlay_market_df["close"],
                name="Cena",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            )
        )

        if not trades_df.empty:
            entries = trades_df[trades_df["akce"].str.contains("Vstup")]
            exits = trades_df[trades_df["akce"].str.contains("Výstup")]

            if not entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=entries["timestamp"],
                        y=entries["cena"],
                        mode="markers",
                        marker=dict(size=10, symbol="triangle-up", color="#16a34a"),
                        name="Vstup",
                        text=entries["akce"],
                        hovertemplate="%{text}<br>Cena: %{y:.4f}<extra></extra>",
                    )
                )

            if abs(selected_position) > 1e-9 and not entries.empty:
                if selected_leg is not None:
                    entry_price = float(selected_leg["entry_price"])
                    selected_leg_side = str(selected_leg["side"]).upper()
                else:
                    last_entry = entries.sort_values("timestamp").iloc[-1]
                    entry_price = float(last_entry["cena"])
                    selected_leg_side = "LONG" if selected_position > 0 else "SHORT"

                vol = float(overlay_market_df["close"].pct_change().rolling(20).std().iloc[-1])
                if pd.isna(vol) or vol <= 0:
                    vol = 0.015
                stop_dist = max(0.005, 1.0 * vol)
                target_dist = max(0.01, 2.0 * vol)

                if selected_leg_side == "LONG":
                    stop_price = entry_price * (1 - stop_dist)
                    target_price = entry_price * (1 + target_dist)
                else:
                    stop_price = entry_price * (1 + stop_dist)
                    target_price = entry_price * (1 - target_dist)

                fig.add_hline(
                    y=entry_price,
                    line_width=1,
                    line_dash="dot",
                    line_color="#60a5fa",
                    annotation_text="Vstupní cena",
                    annotation_position="top left",
                )
                fig.add_hline(
                    y=target_price,
                    line_width=1,
                    line_dash="dash",
                    line_color="#16a34a",
                    annotation_text="Cílová cena (target)",
                    annotation_position="top right",
                )
                fig.add_hline(
                    y=stop_price,
                    line_width=1,
                    line_dash="dash",
                    line_color="#dc2626",
                    annotation_text="Stop úroveň",
                    annotation_position="bottom right",
                )
            if not exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits["timestamp"],
                        y=exits["cena"],
                        mode="markers",
                        marker=dict(size=10, symbol="triangle-down", color="#dc2626"),
                        name="Výstup",
                        text=exits["akce"],
                        hovertemplate="%{text}<br>Cena: %{y:.4f}<extra></extra>",
                    )
                )

        fig.update_layout(
            height=760,
            xaxis_title="Datum",
            yaxis_title="Cena",
            dragmode="zoom",
            xaxis_rangeslider_visible=True,
            xaxis_rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=30, label="30d", step="day", stepmode="backward"),
                        dict(count=90, label="90d", step="day", stepmode="backward"),
                        dict(step="all", label="Vše"),
                    ]
                )
            ),
            legend_title_text="Události",
        )
        fig.update_yaxes(fixedrange=False, autorange=True)
        fig.update_xaxes(fixedrange=False)

        if st.session_state.auto_center_last_candle and len(overlay_market_df.index) > 10:
            visible_bars = min(120, len(overlay_market_df.index))
            range_start = overlay_market_df.index[-visible_bars]
            range_end = overlay_market_df.index[-1]
            fig.update_xaxes(range=[range_start, range_end])

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
            },
        )

    if st.session_state.active_view == "Analýza":
        st.subheader("Long-tail příležitosti (Top 20)")
        long_tail = latest["long_tail"].rename(
            columns={
                "symbol": "Symbol",
                "momentum": "Momentum",
                "liquidity": "Likvidita",
                "spread_bps": "Spread (bps)",
                "compliance_risk": "Compliance riziko",
                "opportunity_score": "Skóre příležitosti",
                "trades_24h": "Obchody 24h",
            }
        )
        long_tail_config = {
            "Symbol": st.column_config.TextColumn("Symbol", help="Obchodní pár na burze."),
            "Momentum": st.column_config.NumberColumn("Momentum", help="Krátkodobá směrová síla pohybu ceny."),
            "Likvidita": st.column_config.NumberColumn("Likvidita", help="Odhad obchodovatelnosti aktiva (vyšší je lepší)."),
            "Spread (bps)": st.column_config.NumberColumn("Spread (bps)", help="Odhad bid-ask spreadu v bazických bodech."),
            "Compliance riziko": st.column_config.NumberColumn("Compliance riziko", help="Odhad regulatorního/etického rizika (nižší je lepší)."),
            "Skóre příležitosti": st.column_config.NumberColumn("Skóre příležitosti", help="Kombinované skóre atraktivity aktiva."),
            "Obchody 24h": st.column_config.NumberColumn("Obchody 24h", help="Počet obchodů za posledních 24 hodin."),
        }
        st.dataframe(long_tail, use_container_width=True, column_config=long_tail_config)

        st.subheader("Dry-run navržené ordery (bez exekuce)")
        orders = latest["proposed_orders"].copy()
        for order in orders:
            order["side"] = {"BUY": "NÁKUP", "SELL": "PRODEJ"}.get(order.get("side"), order.get("side"))
            order["instrument"] = {
                "spot": "Spot",
                "perpetual": "Perpetual",
            }.get(order.get("instrument"), order.get("instrument"))
        orders_df = pd.DataFrame(orders).rename(
            columns={
                "model_id": "ID modelu",
                "symbol": "Symbol",
                "side": "Směr",
                "instrument": "Instrument",
                "quantity_usd": "Objem (USD)",
                "confidence": "Důvěra",
            }
        )
        orders_config = {
            "ID modelu": st.column_config.TextColumn("ID modelu", help="Model, který order navrhl."),
            "Symbol": st.column_config.TextColumn("Symbol", help="Obchodovaný pár."),
            "Směr": st.column_config.TextColumn("Směr", help="NÁKUP/PRODEJ podle signálu modelu."),
            "Instrument": st.column_config.TextColumn("Instrument", help="Spot nebo perpetual větev pro exekuci."),
            "Objem (USD)": st.column_config.NumberColumn("Objem (USD)", help="Navržený objem orderu v USD ekvivalentu."),
            "Důvěra": st.column_config.NumberColumn("Důvěra", help="Modelová důvěra v signál (0 až 1)."),
        }
        st.dataframe(orders_df, use_container_width=True, column_config=orders_config)

        st.subheader("Denní návrhy z hluboké analýzy (evidence-based)")
        for insight in latest["research"]:
            title = insight["title"] if isinstance(insight, dict) else insight.title
            year = insight["year"] if isinstance(insight, dict) else insight.year
            evidence_strength = insight["evidence_strength"] if isinstance(insight, dict) else insight.evidence_strength
            overfit_risk = insight["overfit_risk"] if isinstance(insight, dict) else insight.overfit_risk
            limitations = insight["limitations"] if isinstance(insight, dict) else insight.limitations
            proposal = insight["proposal"] if isinstance(insight, dict) else insight.proposal
            st.markdown(
                f"- **{title} ({year})** | síla důkazu: {evidence_strength} | "
                f"riziko overfittingu: {overfit_risk}  \n"
                f"  Limity: {limitations}  \n"
                f"  Návrh: {proposal}"
            )

        st.subheader("Rozhodovací matice")
        st.latex(r"S = 0.28Sortino + 0.22Calmar - 0.18CVaR_{95} - 0.14MaxDD - 0.10Cost - 0.08Turnover")
        st.write(
            {
                "Prahové hodnoty": {
                    "Minimum Sortino": DEFAULT_CONFIG.thresholds.min_sortino,
                    "Minimum Calmar": DEFAULT_CONFIG.thresholds.min_calmar,
                    "Maximum Max DD": DEFAULT_CONFIG.thresholds.max_dd,
                    "Maximum CVaR95": DEFAULT_CONFIG.thresholds.max_cvar95,
                }
            }
        )

    if st.session_state.active_view == "Historie & Export":
        st.subheader("Persisted historie (SQLite)")
        persisted = _load_recent_runs_cached(limit=25)
        segment_prefix = f"{st.session_state.active_segment} | "
        if not persisted.empty and "champion_model" in persisted.columns:
            persisted = persisted[persisted["champion_model"].astype(str).str.startswith(segment_prefix)].copy()
        if persisted.empty:
            st.info(f"Pro segment {st.session_state.active_segment} zatím není v historii žádný uložený běh.")
            st.stop()
        persisted_view = persisted.rename(
        columns={
            "id": "ID",
            "created_at": "Vytvořeno",
            "week": "Týden",
            "generation": "Generace",
            "market_source": "Zdroj dat",
            "symbol": "Symbol",
            "champion_model": "Vítězný model",
            "champion_score": "Skóre vítěze",
            "champion_sortino": "Sortino vítěze",
            "champion_calmar": "Calmar vítěze",
            "champion_max_dd": "Max DD vítěze",
            "champion_cvar95": "CVaR95 vítěze",
            "reward_usd": "Odměna (USD)",
        }
    )
        persisted_view["Zdroj dat"] = persisted_view["Zdroj dat"].replace({"simulation": "Simulace", "binance": "Binance"})
        persisted_view = _split_datetime_column(persisted_view, "Vytvořeno", "Vytvořeno")
        persisted_config = {
        "ID": st.column_config.NumberColumn("ID", help="Interní ID uloženého běhu."),
        "Vytvořeno - Datum": st.column_config.TextColumn("Vytvořeno - Datum", help="Datum uložení záznamu (dd.mm.yy)."),
        "Vytvořeno - Čas": st.column_config.TextColumn("Vytvořeno - Čas", help="Čas uložení záznamu (hh:mm)."),
        "Týden": st.column_config.NumberColumn("Týden", help="Pořadí týdenního vyhodnocení."),
        "Generace": st.column_config.NumberColumn("Generace", help="Generace modelové populace."),
        "Zdroj dat": st.column_config.TextColumn("Zdroj dat", help="Použitý zdroj tržních dat (Simulace/Binance)."),
        "Symbol": st.column_config.TextColumn("Symbol", help="Hlavní obchodovaný symbol pro běh."),
        "Vítězný model": st.column_config.TextColumn("Vítězný model", help="Model s nejvyšším skóre v daném týdnu."),
        "Skóre vítěze": st.column_config.NumberColumn("Skóre vítěze", help="Výstup decision matrix vítězného modelu."),
        "Sortino vítěze": st.column_config.NumberColumn("Sortino vítěze", help="Sortino ratio vítězného modelu."),
        "Calmar vítěze": st.column_config.NumberColumn("Calmar vítěze", help="Calmar ratio vítězného modelu."),
        "Max DD vítěze": st.column_config.NumberColumn("Max DD vítěze", help="Největší pokles vítězného modelu."),
        "CVaR95 vítěze": st.column_config.NumberColumn("CVaR95 vítěze", help="Tail-risk metrika vítěze (95 %)."),
        "Odměna (USD)": st.column_config.NumberColumn("Odměna (USD)", help="Gamifikovaná odměna vítězi kola."),
    }
        st.dataframe(persisted_view, use_container_width=True, column_config=persisted_config)

        st.warning(
            "Pokud je notebook vypnutý, pozice zůstanou uložené v databázi. "
            "Nové vstupy/výstupy se ale vyhodnotí až při dalším spuštění aplikace."
        )

        st.subheader("Export reportů")
        if st.session_state.last_exports:
            st.write(st.session_state.last_exports)
            csv_path = Path(st.session_state.last_exports.get("csv", ""))
            json_path = Path(st.session_state.last_exports.get("json", ""))

            c_csv, c_json = st.columns(2)
            if csv_path.exists():
                c_csv.download_button(
                    label="Stáhnout poslední CSV",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                )
            if json_path.exists():
                c_json.download_button(
                    label="Stáhnout poslední JSON",
                    data=json_path.read_bytes(),
                    file_name=json_path.name,
                    mime="application/json",
                )

            if csv_path.exists() and json_path.exists():
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(csv_path.name, csv_path.read_bytes())
                    zf.writestr(json_path.name, json_path.read_bytes())
                st.download_button(
                    label="Stáhnout vše jako ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"{csv_path.stem}.zip",
                    mime="application/zip",
                )
        else:
            st.info("Po dalším běhu se automaticky uloží CSV + JSON do složky reports/.")
