"""Read-only customer access and isolated prediction-history persistence."""
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.common import get_collection


def _with_client(operation):
    client, collection = get_collection()
    try:
        return operation(collection)
    finally:
        client.close()


def customer_statistics() -> dict[str, int | float]:
    def operation(collection):
        total = collection.count_documents({})
        subscribed = collection.count_documents({"y": "yes"})
        return {
            "total": total,
            "subscribed": subscribed,
            "not_subscribed": total - subscribed,
            "subscription_rate": subscribed / total if total else 0.0,
        }

    return _with_client(operation)


def customer_filter_options() -> dict[str, list[str]]:
    def operation(collection):
        return {
            "jobs": sorted(value for value in collection.distinct("job") if value),
            "marital": sorted(value for value in collection.distinct("marital") if value),
            "outcomes": sorted(value for value in collection.distinct("y") if value),
        }

    return _with_client(operation)


def fetch_customers(
    *,
    page: int,
    page_size: int,
    search: str = "",
    outcome: str = "All",
    job: str = "All",
    marital: str = "All",
) -> tuple[list[dict[str, Any]], int]:
    """Fetch one page from MongoDB; no customer collection writes are performed."""
    query: dict[str, Any] = {}
    if outcome != "All":
        query["y"] = outcome
    if job != "All":
        query["job"] = job
    if marital != "All":
        query["marital"] = marital
    if search.strip():
        query["$or"] = [
            {"job": {"$regex": search.strip(), "$options": "i"}},
            {"marital": {"$regex": search.strip(), "$options": "i"}},
            {"education": {"$regex": search.strip(), "$options": "i"}},
            {"contact": {"$regex": search.strip(), "$options": "i"}},
        ]

    def operation(collection):
        total = collection.count_documents(query)
        cursor = (
            collection.find(query, {"_id": 0})
            .sort("_id", 1)
            .skip(max(page - 1, 0) * page_size)
            .limit(page_size)
        )
        return list(cursor), total

    return _with_client(operation)


def fetch_all_customers() -> pd.DataFrame:
    def operation(collection):
        return pd.DataFrame(list(collection.find({}, {"_id": 0})))

    return _with_client(operation)


def save_prediction(record: dict[str, Any]) -> None:
    payload = dict(record)
    payload["timestamp"] = datetime.now(timezone.utc)

    def operation(collection):
        history = collection.database["PredictionHistory"]
        history.insert_one(payload)

    _with_client(operation)


def fetch_prediction_history(
    *, page: int, page_size: int, search: str = "", model_name: str = "All"
) -> tuple[list[dict[str, Any]], int]:
    query: dict[str, Any] = {}
    if model_name != "All":
        query["model_name"] = model_name
    if search.strip():
        query["$or"] = [
            {"predicted_subscription": {"$regex": search.strip(), "$options": "i"}},
            {"model_name": {"$regex": search.strip(), "$options": "i"}},
        ]

    def operation(collection):
        history = collection.database["PredictionHistory"]
        total = history.count_documents(query)
        cursor = (
            history.find(query, {"_id": 0})
            .sort("timestamp", -1)
            .skip(max(page - 1, 0) * page_size)
            .limit(page_size)
        )
        rows = list(cursor)
        for row in rows:
            if isinstance(row.get("timestamp"), datetime):
                row["timestamp"] = row["timestamp"].astimezone(timezone.utc).isoformat()
        return rows, total

    return _with_client(operation)
