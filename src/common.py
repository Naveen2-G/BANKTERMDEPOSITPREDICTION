"""Shared project configuration and helpers."""
from pathlib import Path
import os
from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
MODEL_DIR = ROOT / "models"
for folder in (OUTPUT_DIR, CHART_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "bank_marketing_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "customers")

def get_collection():
    """Return an open MongoDB client and the configured collection; caller closes client."""
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client, client[MONGODB_DATABASE][MONGODB_COLLECTION]
