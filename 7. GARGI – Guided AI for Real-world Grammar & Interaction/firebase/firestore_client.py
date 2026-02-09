from google.cloud import firestore

# 🔹 Explicitly connect to the named database
db = firestore.Client(project="gargi-cloud", database="gargibackend")

def save_session(user_id: str, data: dict):
    db.collection("users").document(user_id).collection("sessions").add(data)

def load_sessions(user_id: str):
    docs = db.collection("users").document(user_id).collection("sessions").stream()
    return [d.to_dict() for d in docs]

def load_all_users():
    users = {}
    user_docs = db.collection("users").stream()
    for u in user_docs:
        sessions = load_sessions(u.id)
        users[u.id] = sessions
    return users
