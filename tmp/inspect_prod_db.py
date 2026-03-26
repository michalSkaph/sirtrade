import sqlite3, json
conn = sqlite3.connect('/home/Lenovo/sirtrade/data/sirtrade.db')
out = {}
for table in ['weekly_runs', 'open_positions', 'closed_positions']:
    out[table] = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
out['weekly_latest'] = conn.execute("SELECT segment, week, generation, market_source, symbol FROM weekly_runs ORDER BY id DESC LIMIT 5").fetchall()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
conn.close()
