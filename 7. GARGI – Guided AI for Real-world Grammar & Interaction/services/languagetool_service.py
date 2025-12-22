import subprocess
import time
import requests
import os

DEFAULT_URL = "http://localhost:8081/v2/check"

def is_languagetool_running(url: str = DEFAULT_URL, timeout: float = 1.5) -> bool:
    try:
        # Lightweight ping using /v2/check with minimal data
        r = requests.post(url, data={"text": "test", "language": "en-US"}, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def start_languagetool_server(
    jar_path: str,
    port: int = 8081,
    java_path: str = "java",
) -> subprocess.Popen:
    """
    Starts LanguageTool server in background.
    Returns the Popen process handle.
    """
    if not os.path.exists(jar_path):
        raise FileNotFoundError(f"LanguageTool JAR not found: {jar_path}")

    cmd = [java_path, "-jar", jar_path, "--port", str(port)]
    # CREATE_NEW_CONSOLE opens separate console window on Windows
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    return proc

def ensure_languagetool(
    jar_path: str,
    port: int = 8081,
    url: str | None = None,
    startup_wait_sec: float = 3.0
) -> bool:
    """
    Ensures LanguageTool is running. Tries to start it if not running.
    Returns True if running, else False.
    """
    url = url or f"http://localhost:{port}/v2/check"

    if is_languagetool_running(url=url):
        return True

    try:
        start_languagetool_server(jar_path=jar_path, port=port)
        time.sleep(startup_wait_sec)
        return is_languagetool_running(url=url)
    except Exception:
        return False
