import json
import sys
import os

# ensure repo root is on sys.path so `src` package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sirtrade.engine import TradingEngine
from src.sirtrade.config import DEFAULT_CONFIG

if __name__ == '__main__':
    eng = TradingEngine(DEFAULT_CONFIG)
    summary = eng.run_week(days=365, market_source='simulation', symbol='BTCUSDT', interval='1d')
    champ = summary['champion']
    out = {
        'champion_name': champ.get('name'),
        'champion_score': float(champ.get('score', 0.0)),
        'reward_usd': float(champ.get('reward_usd', 0.0)),
        'week': summary.get('week'),
    }
    print(json.dumps(out, ensure_ascii=False))
