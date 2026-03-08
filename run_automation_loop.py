from __future__ import annotations

import os
import time

from src.sirtrade.automation import run_automation_cycle
from src.sirtrade.status import write_automation_status


def main() -> None:
    source = os.getenv("SIRTRADE_SOURCE", "binance")
    symbol = os.getenv("SIRTRADE_SYMBOL", "BTCUSDT")
    days = int(os.getenv("SIRTRADE_DAYS", "365"))
    interval_minutes = int(os.getenv("SIRTRADE_INTERVAL_MINUTES", "15"))

    while True:
        try:
            result = run_automation_cycle(market_source=source, symbol=symbol, days=days)
            write_automation_status({"ok": True, "result": result})
            print(
                f"[AUTOMATION] week={result['week']} symbol={result['symbol']} "
                f"champion={result['champion']['name']} score={result['champion']['score']:.4f}"
            )
        except Exception as exc:
            write_automation_status({"ok": False, "error": str(exc), "source": source, "symbol": symbol})
            print(f"[AUTOMATION][ERROR] {exc}")
        time.sleep(max(1, interval_minutes) * 60)


if __name__ == "__main__":
    main()
