import sqlite3

conn = sqlite3.connect('live_chat.db')
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM live_chat_messages")
count = cursor.fetchone()[0]
print(f"📊 Total messages saved: {count}")

cursor.execute("SELECT * FROM live_chat_messages ORDER BY timestamp DESC LIMIT 5")
rows = cursor.fetchall()
print("🕵️ Last 5 messages:")
for row in rows:
    print(row)
