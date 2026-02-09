import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("progress_data")
DATA_DIR.mkdir(exist_ok=True)

def save_session(user_id, payload):
    file = DATA_DIR / f"{user_id}.json"

    if file.exists():
        data = json.loads(file.read_text())
    else:
        data = []

    payload["timestamp"] = datetime.utcnow().isoformat()
    data.append(payload)

    file.write_text(json.dumps(data, indent=2))


def load_sessions(user_id):
    file = DATA_DIR / f"{user_id}.json"
    if not file.exists():
        return []
    return json.loads(file.read_text())

