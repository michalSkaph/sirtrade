import json
from pathlib import Path

data = json.loads(Path("data/ui_segment_runs.json").read_text())
for seg, val in data.items():
    lms = val.get("live_model_state", {})
    for mid, st in lms.items():
        print(f"{seg}/{mid}: last_bar_ts={st.get('last_bar_ts', 'MISSING')} armed={st.get('entry_armed')} slots={st.get('open_slots')} side={st.get('side')}")
