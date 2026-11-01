"""Import bank-full.csv into MongoDB in batches."""
from pathlib import Path
import pandas as pd
from common import get_collection, ROOT

CSV_PATH = ROOT / "data" / "bank-full.csv"
BATCH_SIZE = 1000

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}\nDownload UCI bank.zip and copy bank-full.csv into data/.")
    df = pd.read_csv(CSV_PATH, sep=";", keep_default_na=False)
    if "y" not in df.columns:
        raise ValueError("Expected target column 'y'. Confirm you are using UCI bank-full.csv.")
    records = df.to_dict(orient="records")
    client, collection = get_collection()
    try:
        # Explicit replacement makes reruns reproducible. This deletes only this project's collection.
        collection.delete_many({})
        for start in range(0, len(records), BATCH_SIZE):
            collection.insert_many(records[start:start+BATCH_SIZE], ordered=False)
        count = collection.count_documents({})
        print(f"Imported {count:,} documents into {collection.full_name}.")
        print("First document:", collection.find_one({}, {"_id": 0}))
    finally:
        client.close()

if __name__ == "__main__":
    main()
