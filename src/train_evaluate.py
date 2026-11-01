"""Train Random Forest from MongoDB-retrieved records and evaluate the held-out test set."""
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from common import get_collection, OUTPUT_DIR, CHART_DIR, MODEL_DIR

def main():
    client, collection = get_collection()
    try:
        rows = list(collection.find({}, {"_id": 0}))
    finally:
        client.close()
    if not rows:
        raise RuntimeError("No MongoDB records found. Run python src/import_data.py first.")
    df = pd.DataFrame(rows)
    if "y" not in df:
        raise ValueError("Target column y is missing.")
    y = df["y"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    # Exclude duration: known only after the call; avoids target leakage for pre-call predictions.
    X = df.drop(columns=[c for c in ("y", "duration") if c in df.columns])
    # CSV import uses strings for all fields; recover numeric types using known dataset schema.
    numeric_candidates = ["age", "balance", "day", "campaign", "pdays", "previous"]
    for col in numeric_candidates:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    categorical = [c for c in X.columns if c not in numeric_candidates]
    numeric = [c for c in numeric_candidates if c in X.columns]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])
    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0))
    }
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    report = classification_report(y_test, pred, labels=[0, 1],
                                   target_names=["no", "yes"], zero_division=0)
    results = {
        "project": "Bank Marketing Subscription Prediction",
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model": "RandomForestClassifier",
        "positive_class": "yes",
        "features_used": X.columns.tolist(),
        "metrics": metrics,
        "confusion_matrix_labels": ["no", "yes"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "note": "Metrics were computed from actual held-out test predictions when this script ran."
    }
    (OUTPUT_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(values_format="d")
    plt.title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    joblib.dump(model, MODEL_DIR / "random_forest_model.joblib")
    print("Metrics (positive class = yes):")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", cm)
    print("Saved model and evaluation outputs.")

if __name__ == "__main__":
    main()
