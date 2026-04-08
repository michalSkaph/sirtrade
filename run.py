from src.sirtrade.engine import TradingEngine
from src.sirtrade.env import load_env_file


if __name__ == "__main__":
    load_env_file()
    engine = TradingEngine()
    summary = engine.run_week(days=365)
    print("Week:", summary["week"])
    print("Generation:", summary["generation"])
    print("Champion:", summary["champion"]["name"])
    print("Score:", round(summary["champion"]["score"], 4))
