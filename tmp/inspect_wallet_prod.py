import sqlite3, pandas as pd
from src.sirtrade.config import INITIAL_PAPER_WALLET_CZK, PAPER_TRADE_SIZE_CZK
conn = sqlite3.connect('/home/Lenovo/sirtrade/data/sirtrade.db')
open_positions = pd.read_sql_query('SELECT * FROM open_positions', conn)
closed_positions = pd.read_sql_query('SELECT * FROM closed_positions', conn)
open_slots = float(pd.to_numeric(open_positions['position_size'], errors='coerce').fillna(0.0).abs().sum()) if not open_positions.empty else 0.0
slot_counts = pd.to_numeric(closed_positions['quantity_slots'], errors='coerce').fillna(0.0).abs() if not closed_positions.empty else pd.Series(dtype=float)
pnl_pct = pd.to_numeric(closed_positions['pnl_pct'], errors='coerce').fillna(0.0) if not closed_positions.empty else pd.Series(dtype=float)
realized = float(((slot_counts * float(PAPER_TRADE_SIZE_CZK)) * (pnl_pct / 100.0)).sum())
equity = float(INITIAL_PAPER_WALLET_CZK) + realized
print({'open_slots': open_slots, 'realized_pnl_czk': realized, 'equity_czk': equity})
conn.close()
