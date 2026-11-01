"""Prediction validation and persistence payload construction."""
from typing import Any

import pandas as pd

from .mongodb_service import save_prediction

NUMERIC_FIELDS = {
    "age": (18, 100),
    "balance": (-1_000_000, 10_000_000),
    "day": (1, 31),
    "campaign": (1, 100),
    "pdays": (-1, 10_000),
    "previous": (0, 10_000),
}


def validate_inputs(values: dict[str, Any]) -> list[str]:
    errors = []
    for field, (minimum, maximum) in NUMERIC_FIELDS.items():
        value = values.get(field)
        if value is None or not minimum <= value <= maximum:
            errors.append(f"{field} must be between {minimum} and {maximum}.")
    return errors


def predict(model, values: dict[str, Any]) -> tuple[str, float]:
    errors = validate_inputs(values)
    if errors:
        raise ValueError(" ".join(errors))
    frame = pd.DataFrame([values])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return ("yes" if prediction == 1 else "no"), probability


def record_prediction(
    values: dict[str, Any], result: str, probability: float, model_name: str
) -> None:
    save_prediction(
        {
            "input_features": values,
            "predicted_subscription": result,
            "probability_yes": probability,
            "model_name": model_name,
        }
    )
