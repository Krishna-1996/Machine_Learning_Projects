from pathlib import Path
from datetime import datetime
import shutil
from core.paths import sessions_file, sessions_dir

def backup_sessions_file() -> str | None:
    src = sessions_file()
    if not src.exists():
        return None

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = sessions_dir() / f"sessions_backup_{timestamp}.jsonl"
    shutil.copy2(src, dst)
    return str(dst)
