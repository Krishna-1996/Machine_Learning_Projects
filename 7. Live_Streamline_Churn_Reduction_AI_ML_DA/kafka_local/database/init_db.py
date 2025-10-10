import sqlite3

conn = sqlite3.connect('live_chat.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS live_chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    message_text TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()
print("✅ SQLite DB initialized.")
