import json
import time
import pymongo
from kafka import KafkaProducer
from bson import json_util

# Kafka configuration
KAFKA_TOPIC = 'paysim-transactions'
KAFKA_BROKER = 'localhost:9092'

# MongoDB configuration
MONGO_URI = 'mongodb+srv://andang32822:iXFW6YpPRR8Mpt89@mongodbhostingcluster.wdfv27d.mongodb.net/'
DB_NAME = 'fraudDetection'
COLLECTION_NAME = 'paysimData'

def json_serializer(data):
    return json.dumps(data, default=json_util.default).encode('utf-8')

def main():
    print(f"Connecting to Kafka at {KAFKA_BROKER}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=json_serializer
        )
        print("Successfully connected to Kafka.")
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        return

    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Query a subset of data to stream, sorting by step to simulate time progression
    print("Fetching data from MongoDB...")
    # Fetch 1000 records for simulation
    cursor = collection.find().sort("step", 1).limit(1000)
    
    print(f"Starting to stream transactions to Kafka topic '{KAFKA_TOPIC}'...")
    count = 0
    for doc in cursor:
        # Remove the MongoDB ObjectId as it's not needed for the ML model
        if '_id' in doc:
            del doc['_id']
            
        producer.send(KAFKA_TOPIC, doc)
        count += 1
        print(f"Sent transaction {count}: step={doc.get('step')}, amount={doc.get('amount')}, type={doc.get('type')}")
        
        # Simulate real-time delay (e.g., 0.5 seconds between transactions)
        time.sleep(0.5)

    producer.flush()
    print(f"Finished streaming {count} transactions.")

if __name__ == '__main__':
    main()
