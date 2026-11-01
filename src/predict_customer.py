"""Demonstrate loading the trained pipeline and predicting a sample customer."""
import joblib
import pandas as pd
from common import MODEL_DIR

def main():
    path = MODEL_DIR / "random_forest_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Train the model first: python src/train_evaluate.py")
    model = joblib.load(path)
    
    sample = {
        "age": 42, "job": "technician", "marital": "married",
        "education": "secondary", "default": "no", "balance": 1200,
        "housing": "yes", "loan": "no", "contact": "cellular",
        "day": 15, "month": "may", "campaign": 1, "pdays": -1,
        "previous": 0, "poutcome": "unknown"
    }
    frame = pd.DataFrame([sample])
    predicted = int(model.predict(frame)[0])
    labels = {0: "no", 1: "yes"}
    print("Predicted subscription:", labels[predicted])
    if hasattr(model, "predict_proba"):
        print("Estimated probability for yes:", round(float(model.predict_proba(frame)[0][1]), 4))
    print("This is an illustrative prediction, not a guarantee.")

if __name__ == "__main__":
    main()
