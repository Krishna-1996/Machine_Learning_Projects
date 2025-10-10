from kafka import KafkaConsumer
import sqlite3

conn = sqlite3.connect('live_chat.db')
cursor = conn.cursor()

consumer = KafkaConsumer(
    'live-chat',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='chat-consumer-group'
)

print("🟡 Consuming and storing chat messages...")

for msg in consumer:
    try:
        raw = msg.value.decode('utf-8')  # "user23|Hello"
        user_id, message_text = raw.split('|', 1)
        cursor.execute(
            "INSERT INTO live_chat_messages (user_id, message_text) VALUES (?, ?)",
            (user_id, message_text)
        )
        conn.commit()
        print(f"✅ Stored message from {user_id}: {message_text}")
    except Exception as e:
        print(f"❌ Error processing message: {e}")
