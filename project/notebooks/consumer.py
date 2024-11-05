from kafka import KafkaConsumer

consumer = KafkaConsumer("my-topic", bootstrap_servers="localhost:9092")
for msg in consumer:
    value = msg.value.decode()
    print(value)
