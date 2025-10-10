from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'live-chat',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='chat-group'
)

print("Listening for live chat messages...")

for msg in consumer:
    print(f"[Partition {msg.partition}] New chat message: {msg.value.decode('utf-8')}")
