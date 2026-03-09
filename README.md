# SirTrade (MVP)

Autonomní **paper-trading** aplikace pro krypto (Binance-ready architektura), která:
- provozuje 5 soutěžních modelů,
- vyhodnocuje je pomocí decision matrix,
- evolučně vytváří další generace modelů,
- denně simuluje research návrhy z vědeckých přístupů,
- poskytuje bezpečné UI ve Streamlitu,
- umí běžet na simulovaných datech i na veřejných datech Binance (bez API klíče).

## Bezpečnostní poznámka
Tato verze je záměrně spuštěná v **simulaci** (paper mode). Neodesílá reálné obchody.
I v režimu Binance jsou ordery pouze **dry-run návrhy** a nejsou posílány na burzu.

## Jak funguje vypínání notebooku
- Aplikace ukládá otevřené paper pozice do SQLite (`data/sirtrade.db`), takže po vypnutí a zapnutí zůstanou zachované.
- Když je notebook vypnutý, neprobíhá nové vyhodnocení trhu.
- Po dalším spuštění aplikace systém naváže na uložený stav.

## Automatický běh bez otevřeného UI
Pro běh bez Streamlit okna použij:

```bash
C:/Users/Lenovo/AppData/Local/Python/pythoncore-3.14-64/python.exe run_automation.py --source binance --symbol BTCUSDT --days 365
```

Tento běh provede vyhodnocení, uloží výsledky, otevřené pozice i reporty.

### Naplánování ve Windows
- V PowerShellu spusť: `./install_scheduler.ps1`
- Výchozí nastavení vytvoří úlohu každých 15 minut.

Poznámka: pokud je počítač úplně vypnutý, úloha neběží. Pro skutečný 24/7 provoz je potřeba server/VPS, který je stále zapnutý.

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
- Přepínat zdroj dat mezi `simulation` a `binance`
- TradingView realtime widget přímo v UI
- Týdenní hodnocení modelů, 8týdenní generační cyklus
- Risk policy: vol targeting, DD limity, kill-switch
- Long-tail opportunity scanner
- Persistovat výsledky běhů do SQLite (`data/sirtrade.db`)
- Automaticky exportovat týdenní reporty do `reports/` (CSV leaderboard + JSON decision matrix)

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

Tím se spustí:
- `sirtrade-ui` na portu `8501`
- `sirtrade-runner` (automatizační worker každých 15 minut)
- `sirtrade-health` na portu `8080` (`/health`, `/status`)

### 2) Změna intervalu workeru
Uprav v `docker-compose.yml` proměnnou `SIRTRADE_INTERVAL_MINUTES`.

### 3) Kontrola logů
```bash
docker compose logs -f sirtrade-runner
docker compose logs -f sirtrade-ui
docker compose logs -f sirtrade-health
```

### 4) Health endpointy
- `http://<server>:8080/health` — rychlý stav služby (200 = OK, 503 = degraded)
- `http://<server>:8080/status` — poslední detailní stav automatizačního běhu

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
