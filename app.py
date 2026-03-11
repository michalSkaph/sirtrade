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
from src.sirtrade.status import read_automation_status
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
    clear_last_ui_run,
    clear_runtime_state,
    load_last_ui_run,
    load_runtime_state,
    save_last_ui_run,
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


def _format_datetime_label(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "N/A"
    return ts.strftime("%d.%m.%y %H:%M")


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

st.title("SirTrade — Autonomní obchodování krypta (Paper režim)")
st.caption("Bezpečný simulační režim: spot + perpetuals logika, bez páky, týdenní governance.")
st.info(
    "Aktuálně jde o paper režim: aplikace neposílá reálné ordery na Binance a nepracuje s reálnými penězi. "
    "V režimu Binance se načítají pouze veřejná tržní data."
)

automation_status = read_automation_status()
st.subheader("Stav automatizačního workeru")
if automation_status is None:
    st.warning("Stav workeru zatím není k dispozici. Spusť alespoň jednou automatizační běh.")
elif automation_status.get("ok"):
    result = automation_status.get("result", {})
    champ = result.get("champion", {})
    st.success(
        f"Worker běží správně | Poslední aktualizace: {_format_datetime_label(automation_status.get('updated_at'))} | "
        f"Symbol: {result.get('symbol', 'N/A')} | Vítěz: {champ.get('name', 'N/A')}"
    )
else:
    st.error(
        f"Worker hlásí chybu | Poslední aktualizace: {_format_datetime_label(automation_status.get('updated_at'))} | "
        f"Chyba: {automation_status.get('error', 'Neznámá chyba')}"
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
    restored = load_last_ui_run()
    if restored:
        restored_segment = str(restored.get("segment", "Swing"))
        if restored_segment in st.session_state.history_by_segment:
            st.session_state.history_by_segment[restored_segment] = [restored]

if "active_segment" not in st.session_state:
    st.session_state.active_segment = str(runtime_state.get("active_segment", "Swing"))
if "interval" not in st.session_state:
    st.session_state.interval = SEGMENT_DEFAULTS["Swing"]["interval"]
if "last_exports" not in st.session_state:
    st.session_state.last_exports = {}
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
if "weeks_to_run" not in st.session_state:
    st.session_state.weeks_to_run = int(runtime_state.get("weeks_to_run", 1))
if "live_refresh_enabled" not in st.session_state:
    st.session_state.live_refresh_enabled = bool(runtime_state.get("live_refresh_enabled", True))
if "live_refresh_seconds" not in st.session_state:
    st.session_state.live_refresh_seconds = int(runtime_state.get("live_refresh_seconds", 1))
if "live_refresh_when_stopped" not in st.session_state:
    st.session_state.live_refresh_when_stopped = bool(runtime_state.get("live_refresh_when_stopped", True))
if "simulation_cycle_seconds" not in st.session_state:
    st.session_state.simulation_cycle_seconds = int(runtime_state.get("simulation_cycle_seconds", 10))
if "active_view" not in st.session_state:
    st.session_state.active_view = str(runtime_state.get("active_view", "Dashboard"))
if "last_simulation_tick" not in st.session_state:
    st.session_state.last_simulation_tick = float(runtime_state.get("last_simulation_tick", 0.0))
if "ui_active_segment" not in st.session_state:
    st.session_state.ui_active_segment = st.session_state.active_segment
if "ui_active_view" not in st.session_state:
    st.session_state.ui_active_view = st.session_state.active_view

prev_active_segment = str(st.session_state.active_segment)
prev_active_view = str(st.session_state.active_view)
prev_weeks_to_run = int(st.session_state.weeks_to_run)
prev_data_source = str(st.session_state.data_source)
prev_live_refresh_seconds = int(st.session_state.live_refresh_seconds)
prev_simulation_cycle_seconds = int(st.session_state.simulation_cycle_seconds)

active_segment_running = bool(st.session_state.simulation_running_by_segment.get(st.session_state.active_segment, False))
has_running_segments = any(st.session_state.simulation_running_by_segment.values())
force_simulation_cycle = False
status_run = "BĚŽÍ" if active_segment_running else "STOP"
status_source = "Simulace" if st.session_state.data_source == "simulation" else "Binance"
status_profile = st.session_state.active_segment
status_symbol = st.session_state.symbol
status_interval = SEGMENT_DEFAULTS.get(st.session_state.active_segment, SEGMENT_DEFAULTS["Swing"])["interval"]

run_bg = "#16a34a" if active_segment_running else "#dc2626"
run_text = "#ffffff"
status_badges = st.empty()

with st.sidebar:
    st.header("Ovládání")
    st.selectbox(
        "Zobrazený segment",
        ["Scalp", "Intraday", "Swing"],
        index=["Scalp", "Intraday", "Swing"].index(st.session_state.ui_active_segment)
        if st.session_state.ui_active_segment in ["Scalp", "Intraday", "Swing"]
        else 2,
        help="Simulace běží současně pro všechny 3 segmenty. Tady vybíráš, který segment se má zobrazit v detailech.",
        key="ui_active_segment",
    )
    st.session_state.active_segment = st.session_state.ui_active_segment

    data_source = st.selectbox(
        "Zdroj dat",
        ["simulation", "binance"],
        index=["simulation", "binance"].index(st.session_state.data_source)
        if st.session_state.data_source in ["simulation", "binance"]
        else 0,
        format_func=lambda value: {"simulation": "Simulace", "binance": "Binance"}.get(value, value),
    )
    st.session_state.data_source = data_source

    symbol = st.session_state.symbol
    st.caption(f"Referenční trh (automaticky): {symbol}")
    st.caption("Časové rámce běží paralelně: Scalp 5m/3d, Intraday 15m/7d, Swing 4h/30d.")

    weeks_to_run = st.slider("Kolik týdnů spustit", 1, 12, int(st.session_state.weeks_to_run))
    st.session_state.weeks_to_run = weeks_to_run
    st.session_state.auto_center_last_candle = st.checkbox(
        "Auto-center na poslední svíčku",
        value=st.session_state.auto_center_last_candle,
        help="Po každém kroku simulace automaticky posune overlay graf na nejnovější část dat.",
    )
    st.session_state.live_refresh_enabled = st.checkbox(
        "Realtime ceny + graf",
        value=st.session_state.live_refresh_enabled,
        help="Průběžně obnovuje pouze sekci Grafy pro aktuální cenu a vývoj vybraného coinu.",
    )
    st.session_state.live_refresh_seconds = st.slider(
        "Interval realtime cen (s)",
        1,
        10,
        int(st.session_state.live_refresh_seconds),
        1,
    )
    st.session_state.simulation_cycle_seconds = st.slider(
        "Interval přepočtu simulace (s)",
        3,
        60,
        int(st.session_state.simulation_cycle_seconds),
        1,
        help="Řídí, jak často běží nové simulační kroky segmentů. Je oddělený od realtime cen.",
    )
    st.session_state.live_refresh_when_stopped = st.checkbox(
        "Realtime i při STOP simulace",
        value=st.session_state.live_refresh_when_stopped,
        help="Ponechá aktualizaci ceny a grafu i při zastavené simulaci.",
    )
    st.markdown("### Řízení běhu segmentů")
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
        int(st.session_state.weeks_to_run) != prev_weeks_to_run,
        st.session_state.data_source != prev_data_source,
        int(st.session_state.live_refresh_seconds) != prev_live_refresh_seconds,
        int(st.session_state.simulation_cycle_seconds) != prev_simulation_cycle_seconds,
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

status_run = "BĚŽÍ" if active_segment_running else "STOP"
status_source = "Simulace" if st.session_state.data_source == "simulation" else "Binance"
status_profile = st.session_state.active_segment
status_symbol = st.session_state.symbol
status_interval = SEGMENT_DEFAULTS.get(st.session_state.active_segment, SEGMENT_DEFAULTS["Swing"])["interval"]
run_bg = "#16a34a" if active_segment_running else "#dc2626"
run_text = "#ffffff"

status_badges.markdown(
    f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px 0;">
        <span style="background:{run_bg};color:{run_text};padding:6px 10px;border-radius:999px;font-weight:700;">Běh: {status_run}</span>
        <span style="background:#1f2937;color:#e5e7eb;padding:6px 10px;border-radius:999px;">Profil: {status_profile}</span>
        <span style="background:#1f2937;color:#e5e7eb;padding:6px 10px;border-radius:999px;">Zdroj: {status_source}</span>
        <span style="background:#1f2937;color:#e5e7eb;padding:6px 10px;border-radius:999px;">Referenční trh: {status_symbol}</span>
        <span style="background:#1f2937;color:#e5e7eb;padding:6px 10px;border-radius:999px;">Timeframe: {status_interval}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

save_runtime_state(
    {
        "simulation_running": has_running_segments,
        "simulation_running_by_segment": st.session_state.simulation_running_by_segment,
        "auto_center_last_candle": st.session_state.auto_center_last_candle,
        "active_segment": st.session_state.active_segment,
        "data_source": st.session_state.data_source,
        "symbol": st.session_state.symbol,
        "weeks_to_run": int(st.session_state.weeks_to_run),
        "live_refresh_enabled": st.session_state.live_refresh_enabled,
        "live_refresh_seconds": int(st.session_state.live_refresh_seconds),
        "live_refresh_when_stopped": st.session_state.live_refresh_when_stopped,
        "simulation_cycle_seconds": int(st.session_state.simulation_cycle_seconds),
        "active_view": st.session_state.active_view,
        "last_simulation_tick": float(st.session_state.last_simulation_tick),
    }
)

if should_run_simulation:
    for _ in range(weeks_to_run):
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

has_any_history = any(len(history) > 0 for history in st.session_state.history_by_segment.values())
if not has_any_history:
    st.info("Spusť simulaci u vybraného segmentu pro první týdenní vyhodnocení.")
else:
    if has_running_segments:
        running_segments = [segment for segment, running in st.session_state.simulation_running_by_segment.items() if running]
        st.success(f"Simulace běží pro segmenty: {', '.join(running_segments)}")
    else:
        st.info("Všechny segmenty jsou zastavené. Poslední výsledky zůstávají zobrazené.")

    active_history = st.session_state.history_by_segment.get(st.session_state.active_segment, [])
    if not active_history:
        st.warning(f"Segment {st.session_state.active_segment} zatím nemá žádná data. Spusť simulaci pro získání výsledků.")
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
        namespace = f"{SEGMENT_DEFAULTS[st.session_state.active_segment]['namespace']}_"
        closed_positions_all = _load_closed_positions_cached(limit=10000)
        win_rate_label = "N/A"
        avg_pnl_label = "N/A"
        if not closed_positions_all.empty and "model_id" in closed_positions_all.columns:
            segment_closed = closed_positions_all[
                closed_positions_all["model_id"].astype(str).str.startswith(namespace)
            ].copy()
            if not segment_closed.empty and "pnl_pct" in segment_closed.columns:
                pnl = pd.to_numeric(segment_closed["pnl_pct"], errors="coerce").dropna()
                if not pnl.empty:
                    wins = int((pnl > 0).sum())
                    losses = int((pnl < 0).sum())
                    decided = max(1, wins + losses)
                    win_rate_label = f"{(wins / decided) * 100:.1f}%"
                    avg_pnl_label = f"{pnl.mean():.3f}%"

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
        closed_positions = _load_closed_positions_cached(limit=5000)
        namespace = f"{SEGMENT_DEFAULTS[st.session_state.active_segment]['namespace']}_"
        if not closed_positions.empty and "model_id" in closed_positions.columns:
            closed_positions = closed_positions[closed_positions["model_id"].astype(str).str.startswith(namespace)].copy()
        if closed_positions.empty:
            st.info(f"Pro segment {st.session_state.active_segment} zatím nejsou uložené žádné uzavřené obchody.")
        else:
            closed_positions["closed_at"] = pd.to_datetime(closed_positions["closed_at"], errors="coerce")
            closed_positions["opened_at"] = pd.to_datetime(closed_positions["opened_at"], errors="coerce")
            valid_closed = closed_positions[closed_positions["closed_at"].notna()].copy()

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

                    wins = int((overview["PnL %"] > 0).sum())
                    losses = int((overview["PnL %"] < 0).sum())
                    win_rate = (wins / max(1, wins + losses)) * 100
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Uzavřené obchody", len(overview))
                    m2.metric("Win rate", f"{win_rate:.1f}%")
                    m3.metric("Průměrné PnL", f"{overview['PnL %'].mean():.3f}%")

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
