from threading import Lock

_model = None
_lock = Lock()

def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import whisper
                print("Loading Whisper model (once per container)...")
                _model = whisper.load_model("base")
    return _model


def transcribe_audio(audio_path: str) -> dict:
    model = get_model()
    result = model.transcribe(audio_path)

    return {
        "text": result.get("text", "").strip()
    }
