import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db_names = client.list_database_names()
print("Databases:", db_names)

if "fraudDetection" in db_names:
    db = client["fraudDetection"]
    print("Collections in fraudDetection:", db.list_collection_names())
    if "fraudResults" in db.list_collection_names():
        count = db["fraudResults"].count_documents({})
        print("Count in fraudResults:", count)
        print("Sample:", db["fraudResults"].find_one())
