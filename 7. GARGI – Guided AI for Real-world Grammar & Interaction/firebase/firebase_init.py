import firebase_admin
from firebase_admin import credentials

from pathlib import Path

FIREBASE_KEY_PATH = Path("D:/Google_Cloud_INFo/firebase/serviceAccountKey.json")

def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
import firebase_admin
from firebase_admin import credentials
import os

def init_firebase():
    if not firebase_admin._apps:
        key_path = os.environ.get("FIREBASE_KEY_PATH")
        if not key_path:
            raise RuntimeError("FIREBASE_KEY_PATH not set")

        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
