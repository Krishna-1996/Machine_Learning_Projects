from kafka import KafkaProducer
import random
import time

def simple_hash(user_id, num_partitions):
    return sum(ord(c) for c in user_id) % num_partitions

producer = KafkaProducer(bootstrap_servers='localhost:9092')
topic = 'live-chat'
num_partitions = 10

users = [f"user{i}" for i in range(1, 1001)]
chat_messages = [
    "Wow this tech is amazing!", "This is boring...", "Hello from Germany!",
    "Is this prerecorded?", "AI is going to change everything!", "Lagging so bad right now.",
    "Anyone else confused?", "Love how they explained that!", "Trash presentation.",
    "Can’t wait to try this out!", "Please improve the audio.", "Crazy how far AI has come.",
    "Why is the screen black?", "Nice demo 👏", "This is overhyped tbh.",
    "When is the Q&A?", "Best part of the event so far!", "Terrible camera angle.",
    "Mind blown 🤯", "More marketing, less substance...", "Speaker is 🔥🔥🔥",
    "Nothing new here.", "Finally something useful!", "Too much buzzwords 😩",
    "Sound is perfect now!", "Another hour of fluff?", "Love the live captions feature.",
    "Waiting for the actual product...", "Great job team 👏👏👏", "Why are they repeating content?"
]

try:
    while True:
        user = random.choice(users)
        message_text = random.choice(chat_messages)
        message = f"{user}|{message_text}".encode('utf-8')
        partition = simple_hash(user, num_partitions)
        producer.send(topic, key=user.encode('utf-8'), value=message)
        print(f"🟢 Sent message from {user} to partition {partition}: {message_text}")
        time.sleep(random.uniform(0.5, 2))
except KeyboardInterrupt:
    print("Producer stopped.")
finally:
    producer.flush()
    producer.close()
