from __future__ import annotations

import argparse
import json

from src.sirtrade.automation import run_automation_cycle
from src.sirtrade.status import write_automation_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spustí automatizační cyklus SirTrade bez UI.")
    parser.add_argument("--source", default="binance", choices=["simulation", "binance"], help="Zdroj tržních dat")
    parser.add_argument("--symbol", default="BTCUSDT", help="Obchodovaný symbol")
    parser.add_argument("--days", type=int, default=365, help="Počet dnů dat pro vyhodnocení")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_automation_cycle(market_source=args.source, symbol=args.symbol.upper(), days=args.days)
    write_automation_status({"ok": True, "result": result})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
