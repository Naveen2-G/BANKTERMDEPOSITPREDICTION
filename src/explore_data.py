"""Exploratory analysis on records retrieved from MongoDB."""
import matplotlib.pyplot as plt
import pandas as pd
from common import get_collection, CHART_DIR, OUTPUT_DIR

def main():
    client, collection = get_collection()
    try:
        rows = list(collection.find({}, {"_id": 0}))
    finally:
        client.close()
    if not rows:
        raise RuntimeError("MongoDB collection is empty. Run python src/import_data.py first.")
    df = pd.DataFrame(rows)
    print("Shape:", df.shape)
    print(df.dtypes)
    print("\nTarget counts:\n", df["y"].value_counts())
    print("\nDuplicate records:", int(df.duplicated().sum()))
    df.describe(include="all").transpose().to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    df["y"].value_counts().plot(kind="bar", rot=0)
    plt.title("Bank Term Deposit Subscription")
    plt.xlabel("Subscribed (y)")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "target_distribution.png", dpi=160)
    plt.close()
    print("Saved descriptive_statistics.csv and target_distribution.png")

if __name__ == "__main__":
    main()
