# SirTrade (MVP)

Autonomní **paper-trading** aplikace pro krypto (Binance-ready architektura), která:
- provozuje 5 soutěžních modelů,
- vyhodnocuje je pomocí decision matrix,
- evolučně vytváří další generace modelů,
- denně simuluje research návrhy z vědeckých přístupů,
- poskytuje bezpečné UI ve Streamlitu,
- umí běžet na simulovaných datech, na veřejných datech Binance a nově i v režimu Binance Copy přes externí leaderboard/pozice feed.

## Bezpečnostní poznámka
Tato verze je záměrně spuštěná v **simulaci** (paper mode). Neodesílá reálné obchody.
I v režimu Binance jsou ordery pouze **dry-run návrhy** a nejsou posílány na burzu.

## Jak funguje vypínání notebooku
- Aplikace ukládá otevřené paper pozice do SQLite (`data/sirtrade.db`), takže po vypnutí a zapnutí zůstanou zachované.
- Když je notebook vypnutý, neprobíhá nové vyhodnocení trhu.
- Po dalším spuštění aplikace systém naváže na uložený stav.

## Instalace
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Spuštění
```bash
streamlit run app.py
```

## One-click spuštění (Windows)
Spusť soubor `start_sirtrade.bat` v kořeni projektu.
Launcher automaticky:
- najde Python,
- doinstaluje závislosti,
- spustí Streamlit UI.

## Co umí
- Spot + perpetuals logika na úrovni simulace (shorty přes perp větev)
- Páka zakázána (vynuceno konfigurací)
- Přepínat zdroj dat mezi `simulation`, `binance` a `binance_copy`
- TradingView realtime widget přímo v UI
- Týdenní hodnocení modelů, 8týdenní generační cyklus
- Risk policy: vol targeting, DD limity, kill-switch
- Long-tail opportunity scanner
- Persistovat výsledky běhů do SQLite (`data/sirtrade.db`)
- Automaticky exportovat týdenní reporty do `reports/` (CSV leaderboard + JSON decision matrix)

## Binance Copy režim

Režim `binance_copy` zůstává pouze v paper-tradingu. Neodesílá live ordery na burzu a používá externí JSON feed s leaderboardem a otevřenými pozicemi lead traderů.

Nastav tyto proměnné prostředí:

- `SIRTRADE_COPY_TRADER_LIST_URL` - URL vracející seznam traderů
- `SIRTRADE_COPY_TRADER_POSITIONS_URL_TEMPLATE` - URL šablona pro pozice s placeholderem `{trader_id}`
- `SIRTRADE_COPY_TRADER_HEADERS_JSON` - volitelně JSON s HTTP hlavičkami

Nejjednodušší lokální varianta je vytvořit soubor `.env` v kořeni projektu. Aplikace, worker i health server ho načtou automaticky:

```env
SIRTRADE_COPY_TRADER_LIST_URL=https://.../leaderboard
SIRTRADE_COPY_TRADER_POSITIONS_URL_TEMPLATE=https://.../positions/{trader_id}
SIRTRADE_COPY_TRADER_HEADERS_JSON={"Authorization":"Bearer ...","User-Agent":"SirTrade/1.0"}
```

Poznámky:

- Projekt dál vynucuje paper mode a bez leverage.
- Pokud feed vrací leveraged nebo short futures pozice, systém je v copy režimu odfiltruje.
- Vybraný lead trader je určen interním skóre z ROI, PnL, win-rate a drawdownu.
- Když konfigurace chybí nebo feed vrací nepodporovanou odpověď, UI i `/status` nově ukazují diagnostiku místo tichého selhání.

## Deploy na VPS (Docker)

### One-shot produkční deploy
Na VM můžeš spustit vše jedním příkazem:

```bash
cd ~/sirtrade
bash deploy_production.sh
```

### 1) Spusť služby
```bash
docker compose up -d --build
```

Docker Compose nyní předá copy-trader proměnné z host prostředí nebo z lokálního `.env` souboru i do workeru a health serveru.

Tím se spustí:
- `sirtrade-ui` na portu `8501`
- `sirtrade-worker` jako samostatný 24/7 worker pro live vyhodnocení trhu
- `sirtrade-health` na portu `8080` (`/health`, `/market-chart`)

### 2) Jak běží live worker
- UI už v Dockeru worker nespouští samo.
- Live logika běží pouze v kontejneru `sirtrade-worker`.
- Worker pro Binance režimy polluje trh každých 30 sekund a vstupy vyhodnocuje jen na nově uzavřené svíčce.

### 3) Kontrola logů
```bash
docker compose logs -f sirtrade-worker
docker compose logs -f sirtrade-ui
docker compose logs -f sirtrade-health
```

### 4) Health endpointy
- `http://<server>:8080/health` — stav health serveru + čerstvost heartbeat workeru (200 = OK, 503 = degraded)
- `http://<server>:8080/status` — detail runtime stavu workeru, runtime_state a diagnostika Binance streamů
- `http://<server>:8080/market-chart?symbol=BTCUSDT&interval=1m&limit=300` — JSON feed pro live OHLC graf bez refresh blikání UI

### 5) Live data vrstva
- Historie OHLC se dál bere přes Binance REST klines.
- Poslední aktivní kline se průběžně doplňuje přes Binance websocket streamy, pokud jsou dostupné.
- Při výpadku websocketu systém automaticky fallbackne na REST bez zastavení workeru.

### 5) Úpravy z VS Code
- Doporučení: Git workflow (lokální změna -> push -> pull/redeploy na VPS).
- Alternativně VS Code Remote SSH přímo na VPS.

Poznámka: tato verze je stále paper/dry-run, neodesílá live ordery na burzu.

## Struktura
- `app.py` — UI dashboard
- `src/sirtrade/config.py` — konfigurace a risk policy
- `src/sirtrade/engine.py` — simulační engine, model competition
- `src/sirtrade/scoring.py` — decision matrix
- `src/sirtrade/risk.py` — risk guardy
- `src/sirtrade/research.py` — daily deep-research návrhy
- `src/sirtrade/data.py` — tržní simulace + long-tail scan
- `src/sirtrade/models.py` — definice 5 modelů
