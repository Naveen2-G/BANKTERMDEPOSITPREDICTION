"""Load existing serialized models and evaluation artifacts without retraining."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.common import MODEL_DIR, OUTPUT_DIR

MODEL_PATHS = {
    "Baseline Random Forest": MODEL_DIR / "random_forest_model.joblib",
    "SMOTENC Random Forest": MODEL_DIR / "random_forest_smote_model.joblib",
}


@lru_cache(maxsize=2)
def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Unknown model: {model_name}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_results() -> dict[str, Any]:
    result = _load_json(OUTPUT_DIR / "evaluation_results.json")
    if "roc_auc" not in result.get("metrics", {}):
        comparison_result = _load_json(
            OUTPUT_DIR / "smote_experiment" / "evaluation_results.json"
        )
        comparison_baseline = comparison_result.get("models", {}).get("baseline", {})
        result["metrics"] = {
            **result.get("metrics", {}),
            "roc_auc": comparison_baseline.get("metrics", {}).get("roc_auc"),
        }
    return result


def smote_results() -> dict[str, Any]:
    return _load_json(OUTPUT_DIR / "smote_experiment" / "evaluation_results.json")


def classification_report_text() -> str:
    path = OUTPUT_DIR / "classification_report.txt"
    if not path.exists():
        return "Classification report is not available."
    return path.read_text(encoding="utf-8")
