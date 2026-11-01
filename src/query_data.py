"""Run basic MongoDB queries and verify the collection."""
from common import get_collection

def main():
    client, collection = get_collection()
    try:
        print("MongoDB connection: OK")
        print("Total documents:", collection.count_documents({}))
        print("\nFirst 3 documents:")
        for doc in collection.find({}, {"_id": 0}).limit(3):
            print(doc)
        print("\nSubscription counts:")
        for row in collection.aggregate([{"$group": {"_id": "$y", "count": {"$sum": 1}}}]):
            print(row["_id"], row["count"])
        print("\nExample filter: age >= 60")
        print("Count:", collection.count_documents({"age": {"$gte": 60}}))
    finally:
        client.close()

if __name__ == "__main__":
    main()
