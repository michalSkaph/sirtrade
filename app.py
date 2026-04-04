from __future__ import annotations

import io
import json
import html
import urllib.error
import urllib.request
import re
import subprocess
import time
import zipfile
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.sirtrade.config import DEFAULT_CONFIG, INITIAL_PAPER_WALLET_CZK, PAPER_TRADE_SIZE_CZK
from src.sirtrade.data import fetch_binance_market
from src.sirtrade.engine import TradingEngine
from src.sirtrade.health_server import DEFAULT_HEALTH_PORT, ensure_health_server_started
from src.sirtrade.live_worker import ensure_live_worker_started, is_live_worker_started, should_start_embedded_worker
from src.sirtrade.reporting import export_weekly_report
from src.sirtrade.storage import (
    clear_trade_history,
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
    clear_segment_runs,
    load_last_ui_run,
    load_runtime_state,
    load_segment_runs,
    save_last_ui_run,
    save_runtime_state,
    save_segment_runs,
    sanitize_runtime_state_for_ui_boot,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _reset_embedded_worker_boot_state() -> None:
    runtime_state = load_runtime_state()
    sanitized_state = sanitize_runtime_state_for_ui_boot(
        runtime_state,
        resume_running_segments=_env_flag("SIRTRADE_RESUME_SEGMENTS_ON_UI_BOOT", False),
    )
    if sanitized_state == runtime_state:
        return
    save_runtime_state(sanitized_state)


def _clear_optional_streamlit_cache(func: object) -> None:
    clear = getattr(func, "clear", None)
    if callable(clear):
        clear()


st.set_page_config(page_title="SirTrade", page_icon="S", layout="wide")
init_db()
if should_start_embedded_worker():
    if not is_live_worker_started():
        _reset_embedded_worker_boot_state()
    ensure_live_worker_started()
ensure_health_server_started()


@st.cache_data(ttl=30, show_spinner=False)
def _get_last_commit_info() -> str:
    env_val = os.getenv("SIRTRADE_LAST_COMMIT")
    if env_val:
        return env_val

    for fname in ("LAST_COMMIT", ".last_commit"):
        path = Path(fname)
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if content:
            return content

    try:
        if Path(".git").exists():
            output = subprocess.check_output(
                ["git", "log", "-1", "--format=%cI"],
                stderr=subprocess.DEVNULL,
            )
            value = output.decode().strip()
            if value:
                return value
    except Exception:
        pass

    return "unknown"


try:
    _last_commit_label = _get_last_commit_info()
    st.markdown(
        (
            "<div style='position:fixed;right:12px;bottom:8px;z-index:9999;"
            "background:rgba(255,255,255,0.85);padding:6px 8px;border-radius:6px;"
            "box-shadow:0 1px 4px rgba(0,0,0,0.12);font-size:12px;color:#333;'>"
            f"Last commit: {_last_commit_label}</div>"
        ),
        unsafe_allow_html=True,
    )
except Exception:
    pass


@st.cache_data(ttl=2, show_spinner=False)
def _load_runtime_state_cached() -> dict[str, object]:
    state = load_runtime_state()
    return state if isinstance(state, dict) else {}


@st.cache_data(ttl=2, show_spinner=False)
def _load_last_ui_run_cached() -> dict[str, object] | None:
    summary = load_last_ui_run()
    return summary if isinstance(summary, dict) else None


@st.cache_data(ttl=2, show_spinner=False)
def _load_segment_runs_cached() -> dict[str, dict[str, object]]:
    runs = load_segment_runs()
    return runs if isinstance(runs, dict) else {}


def _load_closed_positions_cached(limit: int, segment: str | None = None) -> pd.DataFrame:
    return load_closed_positions(limit=limit, segment=segment)


def _load_open_positions_cached(segment: str | None = None) -> pd.DataFrame:
    return load_open_positions(segment=segment)


@st.cache_data(ttl=10, show_spinner=False)
def _load_recent_runs_cached(limit: int) -> pd.DataFrame:
    return load_recent_runs(limit=limit)


@st.cache_data(ttl=1, show_spinner=False)
def _fetch_binance_market_cached(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    return fetch_binance_market(symbol=symbol, interval=interval, limit=limit)


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_platform_status_cached(health_port: int) -> dict[str, object]:
    base_url = f"http://127.0.0.1:{health_port}"
    payload: dict[str, object] = {"health": None, "status": None, "error": None}

    for key, path in (("health", "/health"), ("status", "/status")):
        try:
            with urllib.request.urlopen(f"{base_url}{path}", timeout=2) as response:
                payload[key] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            try:
                payload[key] = json.loads(body) if body else {"status": "http_error", "code": exc.code}
            except json.JSONDecodeError:
                payload[key] = {"status": "http_error", "code": exc.code, "detail": body}
        except Exception as exc:
            payload["error"] = str(exc)
            break

    return payload


def _save_runtime_state_if_changed(state: dict[str, object]) -> None:
    serialized = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    previous = st.session_state.get("_runtime_state_signature")
    if previous == serialized:
        return
    save_runtime_state(state)
    st.session_state["_runtime_state_signature"] = serialized
    _load_runtime_state_cached.clear()


CLOSED_POSITIONS_LIMIT = 10000
SIMULATION_WEEKS_PER_CYCLE = 1
FIXED_LIVE_REFRESH_SECONDS = 2
FIXED_SIMULATION_CYCLE_SECONDS = 10
FIXED_BINANCE_DECISION_SECONDS = 30
PRAGUE_TIMEZONE = "Europe/Prague"


def _report_paths_from_summary(summary: dict[str, object] | None) -> dict[str, str]:
    if not isinstance(summary, dict):
        return {}
    try:
        week = int(summary.get("week", 0))
        generation = int(summary.get("generation", 0))
    except (TypeError, ValueError):
        return {}
    segment = str(summary.get("segment", "unknown")).lower()
    symbol = str(summary.get("symbol", "BTCUSDT"))
    source = str(summary.get("market_source", "simulation"))
    stem = f"week_{week:03d}_gen_{generation:02d}_{segment}_{symbol}_{source}"
    return {
        "csv": str(Path("reports") / f"{stem}.csv"),
        "json": str(Path("reports") / f"{stem}.json"),
    }


def _reset_summary_trade_state(summary: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(summary, dict):
        return None

    reset_summary = dict(summary)
    results = summary.get("results")
    model_ids: list[str] = []
    if isinstance(results, pd.DataFrame) and "model_id" in results.columns:
        model_ids = [str(value) for value in results["model_id"].tolist()]

    reset_summary["model_trades"] = {model_id: [] for model_id in model_ids}
    reset_summary["champion_trades"] = []
    reset_summary["final_positions"] = {model_id: 0.0 for model_id in model_ids}
    reset_summary["final_open_slots"] = {model_id: 0 for model_id in model_ids}
    reset_summary["model_open_positions"] = {model_id: [] for model_id in model_ids}
    reset_summary["trade_events_delta"] = {model_id: [] for model_id in model_ids}
    reset_summary["live_model_state"] = {model_id: {"entry_armed": True} for model_id in model_ids}
    reset_summary["proposed_orders"] = []
    return reset_summary


def _to_prague_timestamps(values: object) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    return timestamps.dt.tz_convert(PRAGUE_TIMEZONE)


def _format_prague_timestamp(value: object, fmt: str = "%Y-%m-%d %H:%M") -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return "–"
    return timestamp.tz_convert(PRAGUE_TIMEZONE).strftime(fmt)


def _split_datetime_column(frame: pd.DataFrame, source_column: str, label_prefix: str) -> pd.DataFrame:
    if source_column not in frame.columns:
        return frame

    out = frame.copy()
    timestamps = _to_prague_timestamps(out[source_column])
    insert_at = out.columns.get_loc(source_column)
    out.insert(insert_at, f"{label_prefix} - Datum", timestamps.dt.strftime("%d.%m.%y").where(timestamps.notna(), None))
    out.insert(insert_at + 1, f"{label_prefix} - Čas", timestamps.dt.strftime("%H:%M").where(timestamps.notna(), None))
    out = out.drop(columns=[source_column])
    return out


def _format_datetime_column(frame: pd.DataFrame, source_column: str, label: str) -> pd.DataFrame:
    """Replace a datetime column with a single 'dd.mm.yy · HH:MM' string in Europe/Prague tz."""
    if source_column not in frame.columns:
        return frame
    out = frame.copy()
    local_ts = _to_prague_timestamps(out[source_column])
    formatted = local_ts.dt.strftime("%d.%m.%y  ·  %H:%M").where(local_ts.notna(), "–")
    out[source_column] = formatted
    if source_column != label:
        out = out.rename(columns={source_column: label})
    return out


def _compute_trade_levels(entry_price: float, side: str, vol: float) -> tuple[float, float]:
    normalized_side = str(side).upper()
    normalized_vol = float(vol)
    if pd.isna(normalized_vol) or normalized_vol <= 0:
        normalized_vol = 0.015

    stop_dist = max(0.005, 1.0 * normalized_vol)
    target_dist = max(0.01, 2.0 * normalized_vol)

    if normalized_side == "LONG":
        return entry_price * (1 - stop_dist), entry_price * (1 + target_dist)
    return entry_price * (1 + stop_dist), entry_price * (1 - target_dist)


def _build_current_open_positions(
    open_positions: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    summary_positions = summary.get("model_open_positions", {}) if isinstance(summary, dict) else {}
    summary_lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    if isinstance(summary_positions, dict):
        for model_id, positions in summary_positions.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                if not isinstance(position, dict):
                    continue
                key = (
                    str(model_id),
                    str(position.get("symbol", "")).upper(),
                    str(position.get("side", "")).upper(),
                )
                summary_lookup[key] = position

    if open_positions.empty:
        return {
            str(model_id): [position for position in positions if isinstance(position, dict)]
            for model_id, positions in summary_positions.items()
            if isinstance(positions, list) and positions
        } if isinstance(summary_positions, dict) else {}

    current_positions: dict[str, list[dict[str, object]]] = {}
    normalized = open_positions.copy()
    for _, row in normalized.iterrows():
        model_id = str(row.get("model_id", ""))
        symbol = str(row.get("symbol", "")).upper()
        side = str(row.get("side", "")).upper()
        if side not in {"LONG", "SHORT"} or not model_id or not symbol:
            continue

        try:
            slots = max(1, int(abs(float(row.get("position_size", 0.0) or 0.0))))
        except Exception:
            slots = 1

        merged_position: dict[str, object] = {
            "symbol": symbol,
            "side": side,
            "slots": slots,
            "model_id": model_id,
            "model_name": str(row.get("model_name", model_id)),
        }
        merged_position.update(summary_lookup.get((model_id, symbol, side), {}))
        current_positions.setdefault(model_id, []).append(merged_position)

    return current_positions


def _chart_unix_time(value: object) -> int | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return int(timestamp.timestamp())


def _build_live_chart_payload(
    overlay_market_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    chart_open_legs: list[dict[str, object]],
    selected_symbol_for_overlay: str,
    chart_interval: str,
    refresh_seconds: int,
    auto_center_last_candle: bool,
    selected_leg: dict[str, object] | None,
) -> dict[str, object]:
    candles: list[dict[str, float | int]] = []
    for timestamp, row in overlay_market_df.iterrows():
        unix_time = _chart_unix_time(timestamp)
        if unix_time is None:
            continue
        candles.append(
            {
                "time": unix_time,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )

    markers: list[dict[str, object]] = []
    if not trades_df.empty and {"timestamp", "akce", "cena"}.issubset(trades_df.columns):
        marker_rows = trades_df.copy()
        marker_time_column = "market_timestamp" if "market_timestamp" in marker_rows.columns else "timestamp"
        marker_rows[marker_time_column] = pd.to_datetime(marker_rows[marker_time_column], utc=True, errors="coerce")
        marker_rows = marker_rows[marker_rows[marker_time_column].notna()].sort_values(marker_time_column)
        for _, row in marker_rows.iterrows():
            unix_time = _chart_unix_time(row[marker_time_column])
            if unix_time is None:
                continue
            action = str(row.get("akce", ""))
            is_entry = "Vstup" in action
            markers.append(
                {
                    "time": unix_time,
                    "position": "belowBar" if is_entry else "aboveBar",
                    "color": "#16a34a" if is_entry else "#dc2626",
                    "shape": "arrowUp" if is_entry else "arrowDown",
                    "text": action,
                }
            )

    overlay_series: list[dict[str, object]] = []
    price_lines: list[dict[str, object]] = []
    line_styles = {"dot": 1, "dash": 2}
    first_candle_time = candles[0]["time"] if candles else None
    last_candle_time = candles[-1]["time"] if candles else None

    close_vol = pd.to_numeric(overlay_market_df.get("close"), errors="coerce")
    rolling_vol = close_vol.pct_change().rolling(20).std().dropna()
    volatility = float(rolling_vol.iloc[-1]) if not rolling_vol.empty else 0.015

    if chart_open_legs:
        level_specs = [
            ("entry", "Vstup", "#2563eb", "dot"),
            ("target", "Target", "#16a34a", "dash"),
            ("stop", "Stop-loss", "#dc2626", "dash"),
        ]
        for leg_index, leg in enumerate(chart_open_legs, start=1):
            try:
                entry_price = float(leg["entry_price"])
                leg_side = str(leg["side"]).upper()
            except Exception:
                continue
            stop_price, target_price = _compute_trade_levels(entry_price, leg_side, volatility)
            level_values = {"entry": entry_price, "target": target_price, "stop": stop_price}
            for level_key, level_label, level_color, level_dash in level_specs:
                suffix = f" #{leg_index}" if len(chart_open_legs) > 1 else ""
                level_price = float(level_values[level_key])
                price_lines.append(
                    {
                        "price": level_price,
                        "color": level_color,
                        "lineWidth": 1,
                        "lineStyle": line_styles[level_dash],
                        "axisLabelVisible": True,
                        "title": f"{level_label}{suffix}",
                    }
                )
                if first_candle_time is not None and last_candle_time is not None:
                    overlay_series.append(
                        {
                            "id": f"{level_key}-{leg_index}",
                            "label": f"{level_label}{suffix}",
                            "price": level_price,
                            "color": level_color,
                            "lineStyle": line_styles[level_dash],
                            "data": [
                                {"time": first_candle_time, "value": level_price},
                                {"time": last_candle_time, "value": level_price},
                            ],
                        }
                    )

    selected_position = None
    if selected_leg is not None:
        try:
            selected_position = {
                "entryPrice": float(selected_leg["entry_price"]),
                "side": str(selected_leg["side"]).upper(),
                "symbol": selected_symbol_for_overlay,
            }
        except Exception:
            selected_position = None

    health_port = int(os.getenv("SIRTRADE_HEALTH_PORT", str(DEFAULT_HEALTH_PORT)))
    return {
        "symbol": selected_symbol_for_overlay,
        "interval": chart_interval,
        "candles": candles,
        "markers": markers,
        "overlaySeries": overlay_series,
        "priceLines": price_lines,
        "selectedPosition": selected_position,
        "refreshMs": max(1, int(refresh_seconds)) * 1000,
        "autoCenter": bool(auto_center_last_candle),
        "visibleBars": min(120, max(30, len(candles) if candles else 30)),
        "liveEnabled": bool(st.session_state.get("live_refresh_enabled", True)),
        "apiPort": health_port,
        "theme": {
            "bg": "#ffffff",
            "text": "#111827",
            "grid": "rgba(148, 163, 184, 0.22)",
            "up": "#16a34a",
            "down": "#dc2626",
            "wickUp": "#15803d",
            "wickDown": "#b91c1c",
            "accent": "#0f172a",
        },
    }


def _build_live_chart_html(payload: dict[str, object], height: int = 980) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False)
        chart_height = max(520, height - 210)
        return f"""
<div id=\"sirtrade-live-root\" style=\"font-family:Segoe UI, Arial, sans-serif;background:#ffffff;border:1px solid rgba(15,23,42,0.08);border-radius:18px;padding:16px 16px 10px 16px;\">
    <div style=\"display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:14px;\">
        <div>
            <div id=\"sirtrade-symbol\" style=\"font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;\"></div>
            <div id=\"sirtrade-price\" style=\"font-size:34px;line-height:1.1;font-weight:700;color:#0f172a;margin-top:6px;\">--</div>
        </div>
        <div style=\"display:flex;gap:10px;flex-wrap:wrap;\">
            <div style=\"min-width:140px;padding:10px 12px;border-radius:14px;background:#f8fafc;border:1px solid rgba(148,163,184,0.18);\">
                <div style=\"font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;font-weight:700;\">Změna</div>
                <div id=\"sirtrade-delta\" style=\"font-size:20px;font-weight:700;color:#0f172a;margin-top:6px;\">--</div>
            </div>
            <div id=\"sirtrade-position-card\" style=\"display:none;min-width:220px;padding:10px 12px;border-radius:14px;background:#f8fafc;border:1px solid rgba(148,163,184,0.18);\">
                <div id=\"sirtrade-position-label\" style=\"font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;font-weight:700;\">Pozice</div>
                <div id=\"sirtrade-position-value\" style=\"font-size:18px;font-weight:700;color:#0f172a;margin-top:6px;\">--</div>
            </div>
        </div>
    </div>
    <div id=\"sirtrade-levels\" style=\"display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px 0;\"></div>
    <div id=\"sirtrade-live-chart\" style=\"width:100%;height:{chart_height}px;\"></div>
    <div id=\"sirtrade-status\" style=\"margin-top:8px;font-size:12px;color:#64748b;\">Live chart inicializace…</div>
</div>
<script src=\"https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js\"></script>
<script>
(() => {{
    const payload = {payload_json};
    const chartNode = document.getElementById('sirtrade-live-chart');
    const priceNode = document.getElementById('sirtrade-price');
    const deltaNode = document.getElementById('sirtrade-delta');
    const symbolNode = document.getElementById('sirtrade-symbol');
    const statusNode = document.getElementById('sirtrade-status');
    const levelsNode = document.getElementById('sirtrade-levels');
    const positionCardNode = document.getElementById('sirtrade-position-card');
    const positionLabelNode = document.getElementById('sirtrade-position-label');
    const positionValueNode = document.getElementById('sirtrade-position-value');

    if (!window.LightweightCharts) {{
        statusNode.textContent = 'Lightweight Charts se nepodařilo načíst.';
        return;
    }}

    const formatPrice = (value) => Number(value).toLocaleString('cs-CZ', {{ minimumFractionDigits: 4, maximumFractionDigits: 6 }});
    const formatPct = (value) => `${{value >= 0 ? '+' : ''}}${{Number(value).toLocaleString('cs-CZ', {{ minimumFractionDigits: 3, maximumFractionDigits: 3 }})}}%`;
    const updateStatus = (text) => {{ statusNode.textContent = text; }};

    symbolNode.textContent = `${{payload.symbol}} • ${{payload.interval}}`;

    const chart = window.LightweightCharts.createChart(chartNode, {{
        width: chartNode.clientWidth,
        height: chartNode.clientHeight,
        layout: {{
            background: {{ color: payload.theme.bg }},
            textColor: payload.theme.text,
            fontFamily: 'Segoe UI, Arial, sans-serif',
        }},
        grid: {{
            vertLines: {{ color: payload.theme.grid }},
            horzLines: {{ color: payload.theme.grid }},
        }},
        rightPriceScale: {{ borderColor: 'rgba(148,163,184,0.22)' }},
        timeScale: {{ borderColor: 'rgba(148,163,184,0.22)', rightOffset: 6, timeVisible: true, secondsVisible: false }},
        crosshair: {{ mode: 0 }},
        handleScroll: true,
        handleScale: true,
    }});

    const candleSeries = chart.addCandlestickSeries({{
        upColor: payload.theme.up,
        downColor: payload.theme.down,
        borderUpColor: payload.theme.up,
        borderDownColor: payload.theme.down,
        wickUpColor: payload.theme.wickUp,
        wickDownColor: payload.theme.wickDown,
        priceLineVisible: true,
        lastValueVisible: true,
    }});

    let candles = Array.isArray(payload.candles) ? [...payload.candles] : [];
    candleSeries.setData(candles);

    const overlaySeriesRecords = [];
    if (Array.isArray(payload.overlaySeries)) {{
        payload.overlaySeries.forEach((overlay) => {{
            const series = chart.addLineSeries({{
                color: overlay.color,
                lineWidth: 2,
                lineStyle: overlay.lineStyle,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            series.setData(Array.isArray(overlay.data) ? overlay.data : []);
            overlaySeriesRecords.push({{ ...overlay, series }});
        }});
    }}

    if (Array.isArray(payload.markers) && payload.markers.length > 0 && typeof candleSeries.setMarkers === 'function') {{
        candleSeries.setMarkers(payload.markers);
    }}

    if (Array.isArray(payload.priceLines)) {{
        payload.priceLines.forEach((line) => candleSeries.createPriceLine(line));
    }}

    if (levelsNode && Array.isArray(payload.overlaySeries) && payload.overlaySeries.length > 0) {{
        levelsNode.innerHTML = payload.overlaySeries.map((overlay) => `
            <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:#f8fafc;border:1px solid rgba(148,163,184,0.18);font-size:12px;color:#334155;font-weight:600;">
                <span style="width:10px;height:10px;border-radius:999px;background:${{overlay.color}};"></span>
                <span>${{overlay.label}}</span>
                <span>${{formatPrice(overlay.price)}}</span>
            </span>
        `).join('');
    }}

    const renderHeader = () => {{
        if (!candles.length) {{
            priceNode.textContent = '--';
            deltaNode.textContent = '--';
            return;
        }}

        const lastBar = candles[candles.length - 1];
        const prevBar = candles.length > 1 ? candles[candles.length - 2] : null;
        priceNode.textContent = formatPrice(lastBar.close);

        if (prevBar && Number(prevBar.close) !== 0) {{
            const deltaPct = ((Number(lastBar.close) - Number(prevBar.close)) / Number(prevBar.close)) * 100;
            deltaNode.textContent = formatPct(deltaPct);
            deltaNode.style.color = deltaPct >= 0 ? payload.theme.up : payload.theme.down;
        }} else {{
            deltaNode.textContent = '--';
            deltaNode.style.color = payload.theme.text;
        }}

        if (payload.selectedPosition && Number.isFinite(Number(payload.selectedPosition.entryPrice))) {{
            const entryPrice = Number(payload.selectedPosition.entryPrice);
            const currentPrice = Number(lastBar.close);
            const isLong = String(payload.selectedPosition.side || '').toUpperCase() === 'LONG';
            const pnlPct = entryPrice === 0 ? 0 : (isLong ? ((currentPrice - entryPrice) / entryPrice) : ((entryPrice - currentPrice) / entryPrice)) * 100;
            positionCardNode.style.display = 'block';
            positionLabelNode.textContent = `${{payload.selectedPosition.symbol}} • ${{payload.selectedPosition.side}}`;
            positionValueNode.textContent = `${{formatPrice(entryPrice)}} → ${{formatPrice(currentPrice)}} • ${{formatPct(pnlPct)}}`;
            positionValueNode.style.color = pnlPct >= 0 ? payload.theme.up : payload.theme.down;
        }}
    }};

    const applyRealtimeViewport = () => {{
        if (!payload.autoCenter) {{
            return;
        }}
        chart.timeScale().scrollToRealTime();
        if (candles.length > payload.visibleBars) {{
            chart.timeScale().setVisibleLogicalRange({{ from: candles.length - payload.visibleBars, to: candles.length + 4 }});
        }} else {{
            chart.timeScale().fitContent();
        }}
    }};

    const syncOverlaySeries = () => {{
        const firstBar = candles.length ? candles[0] : null;
        const lastBar = candles.length ? candles[candles.length - 1] : null;
        if (!firstBar || !lastBar) {{
            return;
        }}
        overlaySeriesRecords.forEach((overlay) => {{
            overlay.series.setData([
                {{ time: firstBar.time, value: overlay.price }},
                {{ time: lastBar.time, value: overlay.price }},
            ]);
        }});
    }};

    renderHeader();
    syncOverlaySeries();
    applyRealtimeViewport();
    updateStatus(payload.liveEnabled ? 'Live chart běží bez rerenderu stránky.' : 'Live refresh je vypnutý.');

    const resizeObserver = new ResizeObserver(() => {{
        chart.applyOptions({{ width: chartNode.clientWidth, height: chartNode.clientHeight }});
    }});
    resizeObserver.observe(chartNode);

    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const host = window.location.hostname || '127.0.0.1';
    const pollUrl = `${{protocol}}//${{host}}:${{payload.apiPort}}/market-chart?symbol=${{encodeURIComponent(payload.symbol)}}&interval=${{encodeURIComponent(payload.interval)}}&limit=2`;

    let timerId = null;
    let inflight = false;

    const mergeLastBars = (incomingCandles) => {{
        if (!Array.isArray(incomingCandles) || incomingCandles.length === 0) {{
            return;
        }}

        incomingCandles.forEach((bar) => {{
            const lastLocalBar = candles.length ? candles[candles.length - 1] : null;
            if (!lastLocalBar || Number(bar.time) > Number(lastLocalBar.time)) {{
                candles.push(bar);
                candleSeries.update(bar);
                return;
            }}
            if (Number(bar.time) === Number(lastLocalBar.time)) {{
                candles[candles.length - 1] = bar;
                candleSeries.update(bar);
            }}
        }});
        syncOverlaySeries();
    }};

    const pollMarket = async () => {{
        if (!payload.liveEnabled || inflight) {{
            return;
        }}
        inflight = true;
        try {{
            const response = await fetch(pollUrl, {{ cache: 'no-store' }});
            if (!response.ok) {{
                throw new Error(`HTTP ${{response.status}}`);
            }}
            const marketPayload = await response.json();
            mergeLastBars(marketPayload.candles || []);
            renderHeader();
            const timestampLabel = marketPayload.updated_at ? new Date(marketPayload.updated_at).toLocaleTimeString('cs-CZ') : 'n/a';
            updateStatus(`Poslední synchronizace: ${{timestampLabel}}`);
        }} catch (error) {{
            updateStatus(`Live feed nedostupný: ${{error.message}}`);
        }} finally {{
            inflight = false;
        }}
    }};

    if (payload.liveEnabled) {{
        timerId = window.setInterval(pollMarket, payload.refreshMs);
    }}

    window.addEventListener('beforeunload', () => {{
        if (timerId) {{
            window.clearInterval(timerId);
        }}
        resizeObserver.disconnect();
    }});
}})();
</script>
"""


def _load_segment_closed_positions(segment: str, limit: int = CLOSED_POSITIONS_LIMIT) -> pd.DataFrame:
    closed_positions = _load_closed_positions_cached(limit=limit, segment=segment)
    if closed_positions.empty:
        return pd.DataFrame()

    segment_closed = closed_positions.copy()
    if segment_closed.empty:
        return segment_closed

    if "closed_at" in segment_closed.columns:
        segment_closed["closed_at"] = pd.to_datetime(segment_closed["closed_at"], errors="coerce", utc=True)
        segment_closed = segment_closed[segment_closed["closed_at"].notna()].copy()
    if "opened_at" in segment_closed.columns:
        segment_closed["opened_at"] = pd.to_datetime(segment_closed["opened_at"], errors="coerce", utc=True)
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


def _compute_trade_analytics(frame: pd.DataFrame) -> dict[str, object]:
    empty_exit_frame = pd.DataFrame(columns=["Důvod výstupu", "Počet"])
    default_payload = {
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_pnl_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "expectancy_pct": 0.0,
        "expectancy_czk": 0.0,
        "profit_factor": 0.0,
        "avg_holding_minutes": 0.0,
        "exit_reason_frame": empty_exit_frame,
    }
    if frame.empty or "pnl_pct" not in frame.columns:
        return default_payload

    pnl_pct = pd.to_numeric(frame["pnl_pct"], errors="coerce").dropna()
    if pnl_pct.empty:
        return default_payload

    aligned = frame.loc[pnl_pct.index].copy()
    slots = pd.to_numeric(aligned.get("quantity_slots"), errors="coerce").fillna(0.0).abs()
    pnl_czk = (slots * float(PAPER_TRADE_SIZE_CZK) * (pnl_pct / 100.0)).astype(float)
    wins = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct < 0]
    trade_count = int(len(pnl_pct))
    win_rate = (len(wins) / trade_count) if trade_count > 0 else 0.0
    avg_win_pct = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_pct = float(losses.mean()) if not losses.empty else 0.0
    expectancy_pct = (win_rate * avg_win_pct) + ((1.0 - win_rate) * avg_loss_pct)
    gross_profit = float(pnl_czk[pnl_czk > 0].sum())
    gross_loss = float(abs(pnl_czk[pnl_czk < 0].sum()))

    opened_at = pd.to_datetime(aligned.get("opened_at"), utc=True, errors="coerce")
    closed_at = pd.to_datetime(aligned.get("closed_at"), utc=True, errors="coerce")
    holding_minutes = ((closed_at - opened_at).dt.total_seconds() / 60.0).dropna()

    exit_reason_frame = empty_exit_frame
    if "exit_reason" in aligned.columns:
        reason_counts = aligned["exit_reason"].fillna("NEURČENO").astype(str).value_counts().reset_index()
        reason_counts.columns = ["Důvod výstupu", "Počet"]
        exit_reason_frame = reason_counts

    return {
        "closed_trades": trade_count,
        "wins": int((pnl_pct > 0).sum()),
        "losses": int((pnl_pct < 0).sum()),
        "win_rate": float(win_rate * 100.0),
        "avg_pnl_pct": float(pnl_pct.mean()),
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "expectancy_pct": float(expectancy_pct),
        "expectancy_czk": float(pnl_czk.mean()) if len(pnl_czk) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit > 0),
        "avg_holding_minutes": float(holding_minutes.mean()) if not holding_minutes.empty else 0.0,
        "exit_reason_frame": exit_reason_frame,
    }


def _format_holding_time(minutes_value: float) -> str:
    if minutes_value <= 0:
        return "N/A"
    if minutes_value < 60:
        return f"{minutes_value:.0f} min"
    hours = minutes_value / 60.0
    if hours < 24:
        return f"{hours:.1f} h"
    return f"{(hours / 24.0):.1f} d"


def _format_czk(value: float) -> str:
    return f"{value:,.0f} Kč".replace(",", " ")


def _format_czk_delta(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f} Kč".replace(",", " ")


def _render_segment_header_card(label: str, value: str, detail: str | None = None) -> None:
    detail_html = ""
    if detail:
        detail_html = (
            "<div style='margin-top:6px;font-size:0.8rem;line-height:1.2;color:#64748b;'>"
            f"{html.escape(detail)}"
            "</div>"
        )

    st.markdown(
        (
            "<div style='height:100%;min-height:86px;padding:12px 14px;border-radius:14px;"
            "background:linear-gradient(180deg,rgba(248,250,252,0.96) 0%,rgba(241,245,249,0.96) 100%);"
            "border:1px solid rgba(148,163,184,0.18);box-shadow:0 8px 24px rgba(15,23,42,0.06);'>"
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;"
            "color:#64748b;line-height:1.15;'>"
            f"{html.escape(label)}"
            "</div>"
            "<div style='margin-top:8px;font-size:1.18rem;font-weight:700;line-height:1.15;color:#0f172a;"
            "word-break:break-word;'>"
            f"{html.escape(value)}"
            "</div>"
            f"{detail_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _compute_realized_pnl_czk(
    closed_positions: pd.DataFrame,
    trade_size_czk: float = PAPER_TRADE_SIZE_CZK,
) -> float:
    if closed_positions.empty or {"quantity_slots", "pnl_pct"}.issubset(closed_positions.columns) is False:
        return 0.0

    slot_counts = pd.to_numeric(closed_positions["quantity_slots"], errors="coerce").fillna(0.0).abs()
    pnl_pct = pd.to_numeric(closed_positions["pnl_pct"], errors="coerce").fillna(0.0)
    return float(((slot_counts * float(trade_size_czk)) * (pnl_pct / 100.0)).sum())


def _compute_paper_wallet_state(
    open_positions: pd.DataFrame,
    closed_positions: pd.DataFrame,
    initial_wallet_czk: float = INITIAL_PAPER_WALLET_CZK,
    trade_size_czk: float = PAPER_TRADE_SIZE_CZK,
) -> dict[str, float]:
    open_slots = 0.0
    if not open_positions.empty and "position_size" in open_positions.columns:
        open_slots = float(
            pd.to_numeric(open_positions["position_size"], errors="coerce").fillna(0.0).abs().sum()
        )

    realized_pnl_czk = _compute_realized_pnl_czk(closed_positions, trade_size_czk=trade_size_czk)

    locked_capital_czk = open_slots * float(trade_size_czk)
    equity_czk = float(initial_wallet_czk) + realized_pnl_czk
    available_cash_czk = equity_czk - locked_capital_czk

    return {
        "initial_wallet_czk": float(initial_wallet_czk),
        "trade_size_czk": float(trade_size_czk),
        "open_slots": open_slots,
        "locked_capital_czk": locked_capital_czk,
        "realized_pnl_czk": realized_pnl_czk,
        "equity_czk": equity_czk,
        "available_cash_czk": available_cash_czk,
    }


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
            aggregated_open_positions: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
            for _, open_row in segment_open.iterrows():
                model_id = str(open_row.get("model_id", ""))
                model_name = str(open_row.get("model_name", model_id))
                known_models[model_id] = model_name
                side = _normalize_side(open_row.get("side", ""))
                qty = float(open_row.get("position_size", 0.0) or 0.0)
                slot_count = int(round(abs(qty))) if abs(qty) > 1e-9 else 0
                signed_qty = float(slot_count) if side == "LONG" else (-float(slot_count) if side == "SHORT" else 0.0)
                final_positions[model_id] = final_positions.get(model_id, 0.0) + signed_qty
                final_open_slots[model_id] = final_open_slots.get(model_id, 0) + slot_count
                symbol_value = str(open_row.get("symbol", symbol)).upper()
                position_key = (symbol_value, side)
                model_positions = aggregated_open_positions.setdefault(model_id, {})
                existing_position = model_positions.get(position_key)
                if existing_position is None:
                    model_positions[position_key] = {
                        "symbol": symbol_value,
                        "side": side,
                        "slots": slot_count,
                        "model_name": model_name,
                    }
                else:
                    existing_position["slots"] = int(existing_position.get("slots", 0)) + slot_count

            for model_id, positions in aggregated_open_positions.items():
                model_open_positions[model_id] = list(positions.values())

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


def _get_active_latest_summary() -> dict[str, object] | None:
    history_by_segment = st.session_state.get("history_by_segment", {})
    active_history = history_by_segment.get(st.session_state.active_segment, [])
    if not active_history:
        return None
    latest = active_history[-1]
    return latest if isinstance(latest, dict) else None


def _render_graph_view_body() -> None:
    latest = _get_active_latest_summary()
    if latest is None:
        st.warning("Pro aktivní segment zatím nejsou dostupná data grafu.")
        return

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
    model_markets = latest.get("model_markets", {})
    model_options = [(row["model_id"], row["name"]) for _, row in latest["results"].iterrows()]
    default_model = next((i for i, opt in enumerate(model_options) if opt[0] == latest["champion"]["model_id"]), 0)
    selected_model = st.selectbox(
        "Model pro vykreslení obchodů",
        options=model_options,
        index=default_model,
        format_func=lambda item: f"{item[0]} — {item[1]}",
    )

    selected_model_id = selected_model[0]
    market_df = model_markets.get(selected_model_id, latest["market"]).copy()
    trades = latest["model_trades"].get(selected_model_id, [])
    trades_df = pd.DataFrame(trades)
    model_coin_positions = latest.get("model_open_positions", {}).get(selected_model_id, [])
    default_model_symbol = str(latest.get("model_selected_symbols", {}).get(selected_model_id, latest["symbol"])).upper()
    model_coin_symbols = [str(item.get("symbol", default_model_symbol)).upper() for item in model_coin_positions]

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
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce", utc=True)
        trades_df = trades_df[trades_df["timestamp"].notna()].sort_values("timestamp")
        for _, event in trades_df.iterrows():
            action = str(event.get("akce", ""))
            side = str(event.get("strana", "")).upper()
            if side not in {"LONG", "SHORT"}:
                side = "LONG" if "LONG" in action.upper() else ("SHORT" if "SHORT" in action.upper() else "LONG")
            price = float(event.get("cena", 0.0))
            ts = pd.to_datetime(event.get("executed_at", event.get("timestamp")), errors="coerce", utc=True)

            if "Vstup" in action:
                qty = _extract_slots_from_action(action, is_entry=True)
                event_symbol = str(event.get("symbol", default_model_symbol)).upper()
                for _ in range(qty):
                    open_legs.append(
                        {
                            "side": side,
                            "symbol": event_symbol,
                            "entry_price": price,
                            "entry_time": ts,
                        }
                    )
            elif "Výstup" in action:
                qty = _extract_slots_from_action(action, is_entry=False)
                event_symbol = str(event.get("symbol", default_model_symbol)).upper()
                same_side_idx = [
                    i for i, leg in enumerate(open_legs)
                    if leg["side"] == side and str(leg.get("symbol", event_symbol)).upper() == event_symbol
                ]
                for index in same_side_idx[:qty]:
                    open_legs[index]["_close"] = True
                open_legs = [leg for leg in open_legs if not leg.get("_close")]

    if model_coin_symbols:
        for idx, leg in enumerate(open_legs):
            leg.setdefault("symbol", model_coin_symbols[idx % len(model_coin_symbols)])

    position_options = []
    for idx, leg in enumerate(open_legs, start=1):
        entry_time = _format_prague_timestamp(leg["entry_time"])
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

    available_symbols = [default_model_symbol]
    available_symbols.extend([str(symbol).upper() for symbol in latest.get("candidate_symbols", []) if symbol])
    available_symbols.extend([str(symbol).upper() for symbol in model_coin_symbols if symbol])
    if selected_leg is not None:
        available_symbols.append(str(selected_leg.get("symbol", default_model_symbol)).upper())
    available_symbols = sorted(list(dict.fromkeys(available_symbols)))

    default_symbol = str(selected_leg.get("symbol", default_model_symbol)).upper() if selected_leg else default_model_symbol
    selected_symbol_for_overlay = st.selectbox(
        "Coin pro realtime graf",
        options=available_symbols,
        index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
    )

    overlay_market_df = market_df.copy()
    if latest.get("market_source") in {"binance", "binance_copy"}:
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
        return

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
    st.caption("Každý model může mít současně otevřeno maximálně 5 slotů na aktuálně vybraném coinu z top 20 univerza.")

    st.markdown("**Realtime interní graf: vstupy/výstupy + target/stop**")
    chart_open_legs = [
        leg
        for leg in open_legs
        if str(leg.get("symbol", default_model_symbol)).upper() == selected_symbol_for_overlay
    ]
    st.caption("Cena, delta i poslední svíce se v panelu níže aktualizují přímo v browseru bez blikání celé stránky.")

    chart_trades_df = pd.DataFrame()
    if selected_leg is not None and not trades_df.empty and {"timestamp", "akce", "cena"}.issubset(trades_df.columns):
        entry_time = selected_leg.get("entry_time")
        entry_price = float(selected_leg.get("entry_price", 0.0))
        if entry_time is not None:
            mask_entry = (
                trades_df["akce"].str.contains("Vstup", na=False)
                & (trades_df["timestamp"] == entry_time)
                & ((trades_df["cena"].astype(float) - entry_price).abs() < 1e-9)
            )
            chart_trades_df = trades_df[mask_entry].head(1)

    chart_payload = _build_live_chart_payload(
        overlay_market_df=overlay_market_df,
        trades_df=chart_trades_df,
        chart_open_legs=[selected_leg] if selected_leg else [],
        selected_symbol_for_overlay=selected_symbol_for_overlay,
        chart_interval=st.session_state.chart_interval,
        refresh_seconds=st.session_state.live_refresh_seconds,
        auto_center_last_candle=st.session_state.auto_center_last_candle,
        selected_leg=selected_leg,
    )
    components.html(
        _build_live_chart_html(chart_payload, height=980),
        height=980,
        scrolling=False,
    )


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hydrate_engines_from_history(
    engines: dict[str, TradingEngine],
    history_by_segment: dict[str, list[dict[str, object]]],
) -> None:
    for segment, engine in engines.items():
        history = history_by_segment.get(segment, [])
        if not history:
            continue

        latest = history[-1]
        latest_week = max(0, _coerce_int(latest.get("week"), 0))
        latest_generation = max(1, _coerce_int(latest.get("generation"), 1))

        engine.week = max(engine.week, latest_week)
        engine.generation = max(engine.generation, latest_generation)
        for model in engine.models:
            model.generation = engine.generation

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:10px;margin:0 0 0.4rem 0;">
        <div style="width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 55%,#22c55e 100%);display:flex;align-items:center;justify-content:center;color:#ffffff;font-size:16px;font-weight:700;box-shadow:0 8px 20px rgba(29,78,216,0.25);">S</div>
        <div style="font-size:1.35rem;font-weight:700;line-height:1;">SirTrade</div>
    </div>
    """,
    unsafe_allow_html=True,
)

runtime_state = _load_runtime_state_cached()
SEGMENT_DEFAULTS = {
    "Scalp": {"interval": "1m", "sim_days": 1, "namespace": "SC"},
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
    restored_by_segment = _load_segment_runs_cached()
    if restored_by_segment:
        for segment, restored in restored_by_segment.items():
            restored_segment = _infer_segment_name(restored)
            if restored_segment in st.session_state.history_by_segment:
                normalized_restored[restored_segment] = restored
    else:
        restored = _load_last_ui_run_cached()
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
        _load_segment_runs_cached.clear()

_hydrate_engines_from_history(st.session_state.engines, st.session_state.history_by_segment)

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
    st.session_state.data_source = str(runtime_state.get("data_source", "binance"))
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
if "live_segment_cursor" not in st.session_state:
    st.session_state.live_segment_cursor = int(runtime_state.get("live_segment_cursor", 0))
if "last_exports" not in st.session_state:
    st.session_state.last_exports = {}
if "reset_token" not in st.session_state:
    st.session_state.reset_token = int(runtime_state.get("reset_token", 0))
if "paper_trade_cutoff_ts" not in st.session_state:
    st.session_state.paper_trade_cutoff_ts = runtime_state.get("paper_trade_cutoff_ts")

st.session_state.live_refresh_seconds = FIXED_LIVE_REFRESH_SECONDS
st.session_state.simulation_cycle_seconds = FIXED_SIMULATION_CYCLE_SECONDS

active_segment_running = bool(st.session_state.simulation_running_by_segment.get(st.session_state.active_segment, False))
has_running_segments = any(st.session_state.simulation_running_by_segment.values())
force_simulation_cycle = False
health_port = int(os.getenv("SIRTRADE_HEALTH_PORT", str(DEFAULT_HEALTH_PORT)))
platform_status = _fetch_platform_status_cached(health_port)
health_payload = platform_status.get("health") if isinstance(platform_status, dict) else None
status_payload = platform_status.get("status") if isinstance(platform_status, dict) else None

if isinstance(health_payload, dict):
    worker_payload = health_payload.get("worker", {}) if isinstance(health_payload.get("worker"), dict) else {}
    if health_payload.get("status") != "ok":
        st.error(
            "Worker je degradovaný nebo bez heartbeat. "
            f"Stáří heartbeat: {worker_payload.get('heartbeat_age_seconds', 'N/A')} s. "
            f"Detail: {worker_payload.get('message') or worker_payload.get('detail') or 'neznámý'}"
        )
    elif worker_payload.get("fresh") is False:
        st.warning("Worker heartbeat není čerstvý. Live vyhodnocení může být opožděné.")
elif platform_status.get("error"):
    st.error(f"Health endpoint není dostupný: {platform_status['error']}")

with st.sidebar:
    st.header("Nastavení")
    if isinstance(health_payload, dict):
        worker_payload = health_payload.get("worker", {}) if isinstance(health_payload.get("worker"), dict) else {}
        market_stream = health_payload.get("market_stream", {}) if isinstance(health_payload.get("market_stream"), dict) else {}
        st.markdown("---")
        st.subheader("Runtime stav")
        st.write(f"Worker: {worker_payload.get('status', 'N/A')}")
        st.write(f"Heartbeat age: {worker_payload.get('heartbeat_age_seconds', 'N/A')} s")
        st.write(f"Streamy: {market_stream.get('stream_count', 0)}")
        if isinstance(status_payload, dict):
            worker_status = status_payload.get("worker", {}) if isinstance(status_payload.get("worker"), dict) else {}
            if worker_status.get("active_segment"):
                st.write(f"Aktivní segment workeru: {worker_status['active_segment']}")

    segment_options = ["Scalp", "Intraday", "Swing"]
    if st.session_state.active_segment not in segment_options:
        st.session_state.active_segment = "Swing"
    st.selectbox(
        "Segment",
        segment_options,
        key="active_segment",
        help="Vybere detail segmentu v hlavní části aplikace.",
    )

    data_source_options = ["simulation", "binance", "binance_copy"]
    if st.session_state.data_source not in data_source_options:
        st.session_state.data_source = "simulation"
    st.selectbox(
        "Data",
        data_source_options,
        key="data_source",
        format_func=lambda value: {"simulation": "Simulace", "binance": "Binance", "binance_copy": "Binance Copy"}.get(value, value),
    )

    if st.session_state.data_source == "binance":
        st.caption("Univerzum: dynamické Top 20 coiny z Binance podle aktuální atraktivity a likvidity.")
        st.caption(
            f"Paper-trading běží nad živými Binance daty. Graf se obnovuje po {FIXED_LIVE_REFRESH_SECONDS} s, rozhodovací přepočet po {FIXED_BINANCE_DECISION_SECONDS} s."
        )
    elif st.session_state.data_source == "binance_copy":
        st.caption("Zdroj: externí leaderboard lead traderů Binance a jejich otevřené pozice, mapované do paper režimu bez leverage.")
        st.caption(
            f"Režim Binance Copy používá živá Binance data pro ocenění pozic. Graf se obnovuje po {FIXED_LIVE_REFRESH_SECONDS} s, rozhodovací přepočet po {FIXED_BINANCE_DECISION_SECONDS} s."
        )
    else:
        st.caption("Univerzum: dynamické Top 20 simulovaných coinů podle aktuální atraktivity.")
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
    st.session_state.simulation_running_by_segment = {segment: False for segment in SEGMENT_DEFAULTS.keys()}
    st.session_state.reset_token += 1
    st.session_state.paper_trade_cutoff_ts = pd.Timestamp.utcnow().isoformat()
    clear_trade_history()
    clear_last_ui_run()

    reset_segment_runs: dict[str, dict[str, object]] = {}
    for segment in SEGMENT_DEFAULTS.keys():
        history = st.session_state.history_by_segment.get(segment, [])
        latest_summary = history[-1] if history else None
        reset_summary = _reset_summary_trade_state(latest_summary)
        st.session_state.history_by_segment[segment] = [reset_summary] if isinstance(reset_summary, dict) else []
        if isinstance(reset_summary, dict):
            reset_segment_runs[segment] = reset_summary

    if reset_segment_runs:
        save_segment_runs(reset_segment_runs)
        active_reset_summary = reset_segment_runs.get(st.session_state.active_segment)
        if isinstance(active_reset_summary, dict):
            save_last_ui_run(active_reset_summary)
            st.session_state.last_exports = _report_paths_from_summary(active_reset_summary)
    else:
        clear_last_ui_run()
        clear_segment_runs()
        st.session_state.last_exports = {}

    _save_runtime_state_if_changed(
        {
            "simulation_running": False,
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
            "last_simulation_tick": 0.0,
            "live_segment_cursor": int(st.session_state.live_segment_cursor),
            "reset_token": int(st.session_state.reset_token),
            "paper_trade_cutoff_ts": st.session_state.paper_trade_cutoff_ts,
        }
    )
    _clear_optional_streamlit_cache(_load_last_ui_run_cached)
    _clear_optional_streamlit_cache(_load_segment_runs_cached)
    _clear_optional_streamlit_cache(_load_open_positions_cached)
    _clear_optional_streamlit_cache(_load_closed_positions_cached)
    _clear_optional_streamlit_cache(_load_recent_runs_cached)
    st.rerun()

view_options = ["Dashboard", "Grafy", "Pozice", "Uzavřené pozice", "Analýza", "Historie & Export"]
if st.session_state.active_view not in view_options:
    st.session_state.active_view = "Dashboard"

st.radio(
    "Sekce",
    view_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)

active_segment_running = bool(st.session_state.simulation_running_by_segment.get(st.session_state.active_segment, False))
has_running_segments = any(st.session_state.simulation_running_by_segment.values())
status_run = "BĚŽÍ" if active_segment_running else "STOP"
status_source = {"simulation": "Simulace", "binance": "Binance", "binance_copy": "Binance Copy"}.get(
    st.session_state.data_source,
    st.session_state.data_source,
)
status_profile = st.session_state.active_segment
status_symbol = (
    "Top lead trader + jeho otevřené pozice"
    if st.session_state.data_source == "binance_copy"
    else "Dynamické Top 20"
    if st.session_state.data_source == "binance"
    else "Simulační Top 20"
)

_save_runtime_state_if_changed(
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
        "live_segment_cursor": int(st.session_state.live_segment_cursor),
        "reset_token": int(st.session_state.reset_token),
        "paper_trade_cutoff_ts": st.session_state.paper_trade_cutoff_ts,
    }
)

wallet_open_positions = _load_open_positions_cached(segment=st.session_state.active_segment)
wallet_closed_positions = _load_closed_positions_cached(
    limit=CLOSED_POSITIONS_LIMIT,
    segment=st.session_state.active_segment,
)
wallet_state = _compute_paper_wallet_state(wallet_open_positions, wallet_closed_positions)

status1, status2, status3, status4, status5, status6 = st.columns(6)
with status1:
    _render_segment_header_card("Režim", status_run)
with status2:
    _render_segment_header_card("Zdroj dat", status_source)
with status3:
    _render_segment_header_card("Segment", status_profile)
with status4:
    _render_segment_header_card(
        "Peněženka segmentu",
        _format_czk(wallet_state["available_cash_czk"]),
        f"Equity {_format_czk(wallet_state['equity_czk'])}",
    )
with status5:
    _render_segment_header_card(
        "Blokováno v obchodech",
        _format_czk(wallet_state["locked_capital_czk"]),
    )
with status6:
    _render_segment_header_card(
        "Celkový zisk/ztráta",
        _format_czk_delta(wallet_state["realized_pnl_czk"]),
    )
st.caption(
    f"Paper peněženka segmentu {st.session_state.active_segment} je oddělená od ostatních segmentů. "
    f"Start: {_format_czk(wallet_state['initial_wallet_czk'])} | "
    f"Na každý obchod/slot: {_format_czk(wallet_state['trade_size_czk'])} | "
    f"Aktivně blokováno slotů: {int(round(wallet_state['open_slots']))} | "
    f"Univerzum: {status_symbol}"
)

persisted_segment_runs = _load_segment_runs_cached()
if persisted_segment_runs:
    for segment, summary in persisted_segment_runs.items():
        normalized_segment = _infer_segment_name(summary)
        if normalized_segment in st.session_state.history_by_segment:
            current_history = st.session_state.history_by_segment.get(normalized_segment, [])
            current_latest = current_history[-1] if current_history else None
            current_key = (
                int(current_latest.get("week", 0)) if isinstance(current_latest, dict) else -1,
                int(current_latest.get("generation", 0)) if isinstance(current_latest, dict) else -1,
            )
            incoming_key = (
                int(summary.get("week", 0)) if isinstance(summary, dict) else -1,
                int(summary.get("generation", 0)) if isinstance(summary, dict) else -1,
            )
            if incoming_key >= current_key:
                st.session_state.history_by_segment[normalized_segment] = [summary]
                if normalized_segment == st.session_state.active_segment:
                    st.session_state.last_exports = _report_paths_from_summary(summary)

latest_runs_by_segment = {
    segment: history[-1]
    for segment, history in st.session_state.history_by_segment.items()
    if history
}

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

    source_label = {"simulation": "Simulace", "binance": "Binance", "binance_copy": "Binance Copy"}.get(latest["market_source"], latest["market_source"])

    if (
        st.session_state.live_refresh_enabled
        and st.session_state.active_view == "Grafy"
        and latest.get("market_source") in {"binance", "binance_copy"}
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
        dashboard_closed_positions = wallet_closed_positions.copy()
        win_rate_label, avg_pnl_label, _, _, _ = _compute_closed_position_metrics(dashboard_closed_positions)
        dashboard_trade_analytics = _compute_trade_analytics(dashboard_closed_positions)

        st.subheader(f"Detail segmentu: {st.session_state.active_segment}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Týden", latest["week"])
        c2.metric("Generace", latest["generation"])
        c3.metric("Win-rate segmentu", win_rate_label)
        c4.metric("Avg PnL segmentu", avg_pnl_label)
        c6, c7 = st.columns(2)
        c6.metric("Volatilita portfolia (roč.)", f"{latest['portfolio_vol_annual']:.2%}")
        c7.metric("Odměna vítěze", "$1")
        if live_market_price is not None:
            st.metric(
                f"Aktuální cena {latest['symbol']}",
                f"{live_market_price:.6f}",
                None if live_market_change_pct is None else f"{live_market_change_pct:.3f}%",
            )
        st.caption(
            f"Zdroj dat: {source_label} | Champion coin: {latest['symbol']} | Univerzum: {len(latest.get('candidate_symbols', [])) or 1} coinů | Timeframe: {latest.get('interval', '1d')} | Exekuce: pouze dry-run"
        )
        if active_segment_running and st.session_state.live_refresh_enabled:
            st.caption("Live data běží bez automatického refreshování celé stránky.")

        if int(dashboard_trade_analytics["closed_trades"]) > 0:
            st.subheader("Expectancy a kvalita exekuce")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Expectancy / obchod", f"{float(dashboard_trade_analytics['expectancy_pct']):.3f}%")
            a2.metric("Profit factor", f"{float(dashboard_trade_analytics['profit_factor']):.2f}")
            a3.metric(
                "Průměrný zisk / ztráta",
                f"{float(dashboard_trade_analytics['avg_win_pct']):.3f}% / {float(dashboard_trade_analytics['avg_loss_pct']):.3f}%",
            )
            a4.metric("Průměrná doba držení", _format_holding_time(float(dashboard_trade_analytics["avg_holding_minutes"])))

            exit_reason_frame = dashboard_trade_analytics.get("exit_reason_frame", pd.DataFrame())
            if isinstance(exit_reason_frame, pd.DataFrame) and not exit_reason_frame.empty:
                st.caption("Nejčastější důvody výstupu ukazují, zda segment naráží spíš na stop-loss, invalidaci setupu nebo nedokáže dotahovat targety.")
                st.dataframe(exit_reason_frame, use_container_width=True, hide_index=True)

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
                "symbol": "Vybraný coin",
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
            "Vybraný coin": st.column_config.TextColumn("Vybraný coin", help="Coin, který model aktuálně vybral z top 20 univerza."),
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
        st.subheader("Otevřené pozice")
        st.caption("Modely vybírají coin z dynamického top 20 univerza a na něm otevírají paper pozice podle aktuálních Binance dat.")
        model_position_rows = []
        model_markets = latest.get("model_markets", {})
        latest_prices = latest.get("latest_prices", {})
        current_open_positions = _build_current_open_positions(wallet_open_positions, latest)
        for _, row in latest["results"].iterrows():
            model_id = str(row["model_id"])
            model_name = str(row["name"])
            model_symbol = str(row.get("symbol") or latest.get("model_selected_symbols", {}).get(model_id, latest["symbol"])).upper()
            model_live_state = latest.get("live_model_state", {}).get(model_id, {})
            open_pos_list = current_open_positions.get(model_id, [])
            is_open = bool(open_pos_list)

            if not open_pos_list or not is_open:
                # Model without open positions – single inactive row
                model_position_rows.append(
                    {
                        "ID modelu": model_id,
                        "Model": model_name,
                        "Symbol": "-",
                        "Pozice otevřená": "NE",
                        "Směr": "-",
                        "Sloty": "0/5",
                        "Investováno (CZK)": "-",
                        "Vstupní cena": "-",
                        "Aktuální cena": "-",
                        "Target": "-",
                        "Stop": "-",
                        "Nerealizované PnL %": "-",
                        "Otevřeno od": None,
                    }
                )
                continue

            trades_model = pd.DataFrame(latest.get("model_trades", {}).get(model_id, []))

            for pos_item in open_pos_list:
                pos_symbol = str(pos_item.get("symbol", model_symbol)).upper()
                pos_side = str(pos_item.get("side", "")).upper()
                pos_slots = int(pos_item.get("slots", 0) or 0)
                if pos_side not in {"LONG", "SHORT"} or pos_slots <= 0:
                    continue

                invested_czk = round(PAPER_TRADE_SIZE_CZK * pos_slots, 0)

                model_market = model_markets.get(model_id, latest["market"])
                pos_latest_price = latest_prices.get(pos_symbol)
                if pos_latest_price is None:
                    close_series = pd.to_numeric(model_market["close"], errors="coerce").dropna() if not model_market.empty else pd.Series(dtype=float)
                    pos_latest_price = float(close_series.iloc[-1]) if not close_series.empty else 0.0
                vol_latest = float(model_market["close"].pct_change().rolling(20).std().iloc[-1]) if not model_market.empty else 0.0
                if pd.isna(vol_latest) or vol_latest <= 0:
                    vol_latest = 0.015

                entry_price = float(pos_item.get("entry_price", 0.0) or 0.0) or None
                opened_at = pos_item.get("opened_at")
                target_price = float(pos_item.get("target_price", 0.0) or 0.0) or None
                stop_price = float(pos_item.get("stop_price", 0.0) or 0.0) or None
                pnl_pct = None

                # Fallback to trade history for entry price
                if entry_price is None and not trades_model.empty:
                    entry_events = trades_model[trades_model["akce"].str.contains("Vstup")]
                    if not entry_events.empty:
                        last_entry = entry_events.sort_values("timestamp").iloc[-1]
                        entry_price = float(last_entry["cena"])
                        if opened_at is None:
                            opened_at = str(last_entry.get("executed_at", last_entry["timestamp"]))

                # Override from live model state (single-position models only)
                if len(open_pos_list) == 1:
                    live_entry = float(model_live_state.get("entry_price", 0.0) or 0.0)
                    live_opened = model_live_state.get("opened_at")
                    live_stop = float(model_live_state.get("stop_price", 0.0) or 0.0)
                    live_target = float(model_live_state.get("target_price", 0.0) or 0.0)
                    if live_entry > 0:
                        entry_price = live_entry
                    if live_opened:
                        opened_at = str(live_opened)
                    if live_stop > 0:
                        stop_price = live_stop
                    if live_target > 0:
                        target_price = live_target

                if (stop_price is None or target_price is None) and entry_price is not None:
                    stop_price, target_price = _compute_trade_levels(entry_price, pos_side, vol_latest)

                if entry_price is not None and entry_price > 0:
                    if pos_side == "LONG":
                        pnl_pct = ((pos_latest_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - pos_latest_price) / entry_price) * 100

                if opened_at is not None:
                    opened_at = str(opened_at)

                model_position_rows.append(
                    {
                        "ID modelu": model_id,
                        "Model": model_name,
                        "Symbol": pos_symbol,
                        "Pozice otevřená": "ANO",
                        "Směr": pos_side,
                        "Sloty": f"{pos_slots}/5",
                        "Investováno (CZK)": f"{invested_czk:,.0f}",
                        "Vstupní cena": round(entry_price, 6) if entry_price is not None else "-",
                        "Aktuální cena": round(pos_latest_price, 6),
                        "Target": round(target_price, 6) if target_price is not None else "-",
                        "Stop": round(stop_price, 6) if stop_price is not None else "-",
                        "Nerealizované PnL %": round(pnl_pct, 3) if pnl_pct is not None else "-",
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
            if value is None or value == "" or value == "-":
                return ""
            if not isinstance(value, (int, float)) or (isinstance(value, float) and pd.isna(value)):
                return ""
            if value > 0:
                return "background-color: #14532d; color: #dcfce7;"
            if value < 0:
                return "background-color: #7f1d1d; color: #fee2e2;"
            return ""

        def _style_inactive_row(row):
            if row.get("Pozice otevřená") == "ANO":
                return [""] * len(row)
            return ["color: #9ca3af;"] * len(row)

        styled_positions = (
            model_positions_df.style
            .map(_style_side, subset=["Směr"])
            .map(_style_pnl, subset=["Nerealizované PnL %"])
            .apply(_style_inactive_row, axis=1)
        )
        st.dataframe(styled_positions, use_container_width=True)

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
                    filtered_trade_analytics = _compute_trade_analytics(filtered)

                    # Compute financial PnL column before renaming
                    slots = pd.to_numeric(filtered["quantity_slots"], errors="coerce").fillna(0.0).abs()
                    pnl_pct_vals = pd.to_numeric(filtered["pnl_pct"], errors="coerce").fillna(0.0)
                    filtered["pnl_czk"] = (slots * float(PAPER_TRADE_SIZE_CZK) * (pnl_pct_vals / 100.0)).round(1)

                    overview = filtered.rename(
                        columns={
                            "opened_at": "Otevřeno",
                            "closed_at": "Uzavřeno",
                            "model_id": "ID modelu",
                            "model_name": "Model",
                            "symbol": "Symbol",
                            "side": "Směr",
                            "entry_price": "Vstupní cena",
                            "exit_price": "Výstupní cena",
                            "quantity_slots": "Sloty",
                            "pnl_pct": "PnL %",
                            "pnl_czk": "PnL CZK",
                            "pnl_status": "Výsledek",
                            "exit_reason": "Důvod výstupu",
                            "market_source": "Zdroj dat",
                            "week": "Týden",
                            "generation": "Generace",
                        }
                    )
                    overview["Zdroj dat"] = overview["Zdroj dat"].replace({"simulation": "Simulace", "binance": "Binance"})

                    # Format datetimes: combine date+time, opened first
                    overview = _format_datetime_column(overview, "Otevřeno", "Otevřeno")
                    overview = _format_datetime_column(overview, "Uzavřeno", "Uzavřeno")

                    # Ensure column order: Otevřeno → Uzavřeno → rest
                    desired_order = [
                        "Otevřeno", "Uzavřeno", "Model", "Symbol", "Směr",
                        "Vstupní cena", "Výstupní cena", "Sloty",
                        "PnL %", "PnL CZK", "Výsledek",
                        "Zdroj dat", "Týden", "Generace", "ID modelu",
                    ]
                    ordered_cols = [c for c in desired_order if c in overview.columns]
                    remaining_cols = [c for c in overview.columns if c not in ordered_cols]
                    overview = overview[ordered_cols + remaining_cols]

                    win_rate_label, avg_pnl_label, _, _, _ = _compute_closed_position_metrics(filtered)
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Uzavřené obchody", len(overview))
                    m2.metric("Win rate", win_rate_label)
                    m3.metric("Průměrné PnL", avg_pnl_label)
                    m4.metric("Expectancy", f"{float(filtered_trade_analytics['expectancy_pct']):.3f}%")
                    m5.metric("Profit factor", f"{float(filtered_trade_analytics['profit_factor']):.2f}")

                    def _style_result(value):
                        if value == "ZISK":
                            return "background-color: #14532d; color: #dcfce7; font-weight: 600;"
                        if value == "ZTRÁTA":
                            return "background-color: #7f1d1d; color: #fee2e2; font-weight: 600;"
                        return "background-color: #374151; color: #e5e7eb;"

                    def _style_pnl(value):
                        if pd.isna(value):
                            return ""
                        try:
                            v = float(value)
                        except (TypeError, ValueError):
                            return ""
                        if v > 0:
                            return "background-color: #14532d; color: #dcfce7;"
                        if v < 0:
                            return "background-color: #7f1d1d; color: #fee2e2;"
                        return ""

                    def _style_datetime(value):
                        return "color: #9ca3af;"

                    styled_overview = (
                        overview.style
                        .map(_style_result, subset=["Výsledek"])
                        .map(_style_pnl, subset=["PnL %", "PnL CZK"])
                        .map(_style_datetime, subset=[c for c in ["Otevřeno", "Uzavřeno"] if c in overview.columns])
                    )
                    st.dataframe(styled_overview, use_container_width=True)

                    exit_reason_frame = filtered_trade_analytics.get("exit_reason_frame", pd.DataFrame())
                    if isinstance(exit_reason_frame, pd.DataFrame) and not exit_reason_frame.empty:
                        st.caption(
                            f"Průměrná doba držení: {_format_holding_time(float(filtered_trade_analytics['avg_holding_minutes']))} | "
                            f"Expectancy v Kč: {_format_czk_delta(float(filtered_trade_analytics['expectancy_czk']))}"
                        )
                        st.dataframe(exit_reason_frame, use_container_width=True, hide_index=True)

    if st.session_state.active_view == "Grafy":
        _render_graph_view_body()

    if st.session_state.active_view == "Analýza":
        st.subheader("Long-tail příležitosti (Top 20)")
        st.caption("Seznam se průběžně mění podle aktuálních Binance dat a slouží jako obchodní univerzum pro modely.")
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
                "quantity_czk": "Objem (Kč)",
                "confidence": "Důvěra",
            }
        )
        orders_config = {
            "ID modelu": st.column_config.TextColumn("ID modelu", help="Model, který order navrhl."),
            "Symbol": st.column_config.TextColumn("Symbol", help="Obchodovaný pár."),
            "Směr": st.column_config.TextColumn("Směr", help="NÁKUP/PRODEJ podle signálu modelu."),
            "Instrument": st.column_config.TextColumn("Instrument", help="Spot nebo perpetual větev pro exekuci."),
            "Objem (Kč)": st.column_config.NumberColumn("Objem (Kč)", help="Fixní paper alokace 1000 Kč za každý otevřený slot obchodu."),
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
        persisted_view["Zdroj dat"] = persisted_view["Zdroj dat"].replace({"simulation": "Simulace", "binance": "Binance", "binance_copy": "Binance Copy"})
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
