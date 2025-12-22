from pathlib import Path

def project_root() -> Path:
    # This file is in core/, so root is one level above
    return Path(__file__).resolve().parent.parent

def sessions_dir() -> Path:
    p = project_root() / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p

def sessions_file() -> Path:
    return sessions_dir() / "sessions.jsonl"
