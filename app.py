from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.sirtrade.config import DEFAULT_CONFIG
from src.sirtrade.engine import TradingEngine
from src.sirtrade.reporting import export_weekly_report
from src.sirtrade.status import read_automation_status
from src.sirtrade.storage import init_db, load_open_positions, load_recent_runs, save_open_positions, save_week_result

st.set_page_config(page_title="SirTrade", page_icon="📈", layout="wide")
init_db()

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
        f"Worker běží správně | Poslední aktualizace: {automation_status.get('updated_at')} | "
        f"Symbol: {result.get('symbol', 'N/A')} | Vítěz: {champ.get('name', 'N/A')}"
    )
else:
    st.error(
        f"Worker hlásí chybu | Poslední aktualizace: {automation_status.get('updated_at')} | "
        f"Chyba: {automation_status.get('error', 'Neznámá chyba')}"
    )

if "engine" not in st.session_state:
    st.session_state.engine = TradingEngine(DEFAULT_CONFIG)
if "history" not in st.session_state:
    st.session_state.history = []
if "last_exports" not in st.session_state:
    st.session_state.last_exports = {}

with st.sidebar:
    st.header("Ovládání")
    data_source = st.selectbox(
        "Zdroj dat",
        ["simulation", "binance"],
        index=0,
        format_func=lambda value: {"simulation": "Simulace", "binance": "Binance"}.get(value, value),
    )
    symbol = st.text_input("Obchodovaný symbol", value="BTCUSDT")
    sim_days = st.slider("Délka simulace (dny)", 90, 730, 365, 5)
    weeks_to_run = st.slider("Kolik týdnů spustit", 1, 12, 1)
    run_btn = st.button("Spustit simulaci")
    reset_btn = st.button("Resetovat")

    st.markdown("---")
    st.subheader("Riziková politika")
    cfg = DEFAULT_CONFIG
    st.write(f"Cílová volatilita: {cfg.risk.target_vol_annual:.0%}")
    st.write(f"Měkký DD limit: {cfg.risk.soft_dd_alert:.0%}")
    st.write(f"Tvrdý DD limit: {cfg.risk.hard_dd_limit:.0%}")
    st.write(f"Maximální expozice na aktivum: {cfg.risk.max_asset_exposure:.0%}")

if reset_btn:
    st.session_state.engine = TradingEngine(DEFAULT_CONFIG)
    st.session_state.history = []
    st.rerun()

if run_btn:
    for _ in range(weeks_to_run):
        result = st.session_state.engine.run_week(days=sim_days, market_source=data_source, symbol=symbol)
        st.session_state.history.append(result)
        save_week_result(result)
        save_open_positions(result)
        st.session_state.last_exports = export_weekly_report(result, DEFAULT_CONFIG)

if not st.session_state.history:
    st.info("Klikni na 'Spustit simulaci' pro první týdenní vyhodnocení.")
else:
    latest = st.session_state.history[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Týden", latest["week"])
    c2.metric("Generace", latest["generation"])
    c3.metric("Volatilita portfolia (roč.)", f"{latest['portfolio_vol_annual']:.2%}")
    c4.metric("Odměna vítěze", "$1")
    source_label = {"simulation": "Simulace", "binance": "Binance"}.get(latest["market_source"], latest["market_source"])
    st.caption(f"Zdroj dat: {source_label} | Symbol: {latest['symbol']} | Exekuce: pouze dry-run")

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

    st.subheader("Graf ceny a obchody modelu")
    market_df = latest["market"].copy()
    model_options = [(row["model_id"], row["name"]) for _, row in latest["results"].iterrows()]
    default_model = next((i for i, opt in enumerate(model_options) if opt[0] == latest["champion"]["model_id"]), 0)
    selected_model = st.selectbox(
        "Model pro vykreslení obchodů",
        options=model_options,
        index=default_model,
        format_func=lambda item: f"{item[0]} — {item[1]}",
    )

    st.markdown("**TradingView realtime graf**")
    tv_symbol = f"BINANCE:{latest['symbol'].replace('/', '').upper()}"
    tv_widget = f"""
    <div class=\"tradingview-widget-container\">
        <div id=\"tradingview_chart\"></div>
        <script type=\"text/javascript\" src=\"https://s3.tradingview.com/tv.js\"></script>
        <script type=\"text/javascript\">
            new TradingView.widget({{
                autosize: true,
                symbol: \"{tv_symbol}\",
                interval: \"60\",
                timezone: \"Etc/UTC\",
                theme: \"dark\",
                style: \"1\",
                locale: \"cs\",
                toolbar_bg: \"#1f2937\",
                enable_publishing: false,
                hide_side_toolbar: false,
                withdateranges: true,
                allow_symbol_change: true,
                container_id: \"tradingview_chart\"
            }});
        </script>
    </div>
    """
    components.html(tv_widget, height=560)

    selected_model_id = selected_model[0]
    trades = latest["model_trades"].get(selected_model_id, [])
    trades_df = pd.DataFrame(trades)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=market_df.index,
            open=market_df["open"],
            high=market_df["high"],
            low=market_df["low"],
            close=market_df["close"],
            name="Cena",
        )
    )

    if not trades_df.empty:
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
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
        height=500,
        xaxis_title="Datum",
        yaxis_title="Cena",
        xaxis_rangeslider_visible=False,
        legend_title_text="Události",
    )
    st.plotly_chart(fig, use_container_width=True)

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
        st.markdown(
            f"- **{insight.title} ({insight.year})** | síla důkazu: {insight.evidence_strength} | "
            f"riziko overfittingu: {insight.overfit_risk}  \n"
            f"  Limity: {insight.limitations}  \n"
            f"  Návrh: {insight.proposal}"
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

    st.subheader("Persisted historie (SQLite)")
    persisted = load_recent_runs(limit=25)
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
    persisted_config = {
        "ID": st.column_config.NumberColumn("ID", help="Interní ID uloženého běhu."),
        "Vytvořeno": st.column_config.TextColumn("Vytvořeno", help="Čas uložení záznamu do SQLite."),
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

    st.subheader("Otevřené pozice (držené i po vypnutí aplikace)")
    open_positions = load_open_positions()
    if open_positions.empty:
        st.info("Momentálně nejsou evidované žádné otevřené paper pozice.")
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
        st.dataframe(open_view, use_container_width=True)

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
