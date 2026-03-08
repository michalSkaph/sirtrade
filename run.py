from src.sirtrade.engine import TradingEngine


if __name__ == "__main__":
    engine = TradingEngine()
    summary = engine.run_week(days=365)
    print("Week:", summary["week"])
    print("Generation:", summary["generation"])
    print("Champion:", summary["champion"]["name"])
    print("Score:", round(summary["champion"]["score"], 4))
