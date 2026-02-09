import firebase_admin
from firebase_admin import credentials
import os
import json
from google.cloud import secretmanager

def init_firebase():
    if firebase_admin._apps:
        return

    client = secretmanager.SecretManagerServiceClient()
    name = "projects/gargi-cloud/secrets/firebase-service-account/versions/latest"

    response = client.access_secret_version(request={"name": name})
    service_account_info = json.loads(response.payload.data.decode("UTF-8"))

    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
