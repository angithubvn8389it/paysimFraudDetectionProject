import pandas as pd
from pymongo import MongoClient

def import_csv_to_mongo(csv_file_path, db_name, collection_name):
    """
    Imports data from a local CSV file into a local MongoDB instance.
    Also creates the necessary indexes for efficient querying.
    """
    # Connect to MongoDB
    client = MongoClient('localhost', 27017)
    db = client[db_name]
    collection = db[collection_name]
    
    # Read CSV using pandas
    print(f"Reading {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Convert DataFrame to a list of dictionaries
    records = df.to_dict(orient='records')
    
    # Insert records into MongoDB
    print(f"Inserting {len(records)} records into MongoDB collection '{collection_name}'...")
    # Clear existing data first
    collection.delete_many({})
    collection.insert_many(records)
    print("Data import completed successfully!")
    
    # Create indexes similar to create_indexes.js
    print("Creating indexes...")
    collection.create_index("step", name="step_index")
    collection.create_index("isFraud", name="fraud_index")
    collection.create_index("type", name="type_index")
    collection.create_index([("amount", -1)], name="amount_desc_index")
    collection.create_index([("isFraud", 1), ("type", 1), ("amount", -1)], name="fraud_type_amount_index")
    print("Indexes created successfully!")

if __name__ == "__main__":
    import_csv_to_mongo(
        csv_file_path="data/raw_data/paysimLog.csv",
        db_name="fraudDetection",
        collection_name="paysimData"
    )
