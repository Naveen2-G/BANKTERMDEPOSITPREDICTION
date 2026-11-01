"""Compare the existing Random Forest with a leakage-safe SMOTENC experiment."""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from common import MODEL_DIR, get_collection, ROOT

EXPERIMENT_DIR = ROOT / "outputs" / "smote_experiment"
CHART_DIR = EXPERIMENT_DIR / "charts"
SMOTE_MODEL_PATH = MODEL_DIR / "random_forest_smote_model.joblib"
RANDOM_STATE = 42
NUMERIC_CANDIDATES = ["age", "balance", "day", "campaign", "pdays", "previous"]


def load_dataset() -> pd.DataFrame:
    """Retrieve the original records from MongoDB without modifying them."""
    client, collection = get_collection()
    try:
        rows = list(collection.find({}, {"_id": 0}))
    finally:
        client.close()
    if not rows:
        raise RuntimeError("No MongoDB records found. Run python src/import_data.py first.")
    frame = pd.DataFrame(rows)
    if "y" not in frame.columns:
        raise ValueError("Target column y is missing.")
    return frame


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the same target mapping and duration exclusion as the baseline."""
    target = frame["y"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = target.notna()
    filtered = frame.loc[valid].copy()
    features = filtered.drop(columns=[c for c in ("y", "duration") if c in filtered.columns])
    for column in NUMERIC_CANDIDATES:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features, target.loc[valid].astype(int)


def build_preprocessor(
    numeric: list[str], categorical: list[str], *, ordinal: bool
) -> ColumnTransformer:
    """Build a training-only preprocessing transformer for either model."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    if ordinal:
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        sparse_threshold=0,
    )


def build_baseline_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Build the unchanged baseline configuration in memory for comparison."""
    preprocessor = build_preprocessor(numeric, categorical, ordinal=False)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_smote_model(numeric: list[str], categorical: list[str]) -> ImbPipeline:
    """Build a pipeline that ordinal-encodes and then applies SMOTENC to training data."""
    preprocessor = build_preprocessor(numeric, categorical, ordinal=True)
    categorical_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "sampler",
                SMOTENC(
                    categorical_features=categorical_indices,
                    random_state=RANDOM_STATE,
                    sampling_strategy="auto",
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate positive-class metrics from untouched test records."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["no", "yes"],
            zero_division=0,
        ),
    }


def save_class_distribution(counts: pd.Series, title: str, path) -> None:
    labels = ["No", "Yes"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    axis = plt.subplots(figsize=(6, 4))[1]
    bars = axis.bar(labels, values, color=["#4C78A8", "#F58518"])
    axis.set_title(title)
    axis.set_xlabel("Subscription")
    axis.set_ylabel("Customers")
    axis.bar_label(bars, fmt="%d")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_chart(matrix: list[list[int]], title: str, path) -> None:
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix), display_labels=["No", "Yes"]
    )
    display.plot(values_format="d", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_metric_chart(comparison: pd.DataFrame, path) -> None:
    axis = comparison.plot(
        x="metric",
        y=["baseline", "smote"],
        kind="bar",
        figsize=(8, 5),
        rot=0,
        color=["#4C78A8", "#F58518"],
    )
    axis.set_title("Baseline vs SMOTENC Metrics")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.legend(title="Model")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_dataset()
    features, target = prepare_features(frame)
    numeric = [column for column in NUMERIC_CANDIDATES if column in features.columns]
    categorical = [column for column in features.columns if column not in numeric]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    baseline = build_baseline_model(numeric, categorical)
    baseline.fit(X_train, y_train)
    baseline_metrics = calculate_metrics(baseline, X_test, y_test)

    smote_model = build_smote_model(numeric, categorical)
    smote_model.fit(X_train, y_train)
    smote_metrics = calculate_metrics(smote_model, X_test, y_test)

    preprocessor = smote_model.named_steps["preprocessor"]
    encoded_train = preprocessor.transform(X_train)
    sampler = smote_model.named_steps["sampler"]
    _, resampled_target = sampler.fit_resample(encoded_train, y_train)
    save_class_distribution(
        y_train.value_counts().sort_index(),
        "Original Training Class Distribution",
        CHART_DIR / "original_training_class_distribution.png",
    )
    save_class_distribution(
        pd.Series(resampled_target).value_counts().sort_index(),
        "SMOTENC Resampled Training Class Distribution",
        CHART_DIR / "resampled_training_class_distribution.png",
    )
    save_confusion_chart(
        baseline_metrics["confusion_matrix"],
        "Baseline Random Forest - Confusion Matrix",
        CHART_DIR / "baseline_confusion_matrix.png",
    )
    save_confusion_chart(
        smote_metrics["confusion_matrix"],
        "SMOTENC Random Forest - Confusion Matrix",
        CHART_DIR / "smote_confusion_matrix.png",
    )

    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    comparison = pd.DataFrame(
        {
            "metric": metric_names,
            "baseline": [baseline_metrics[name] for name in metric_names],
            "smote": [smote_metrics[name] for name in metric_names],
        }
    )
    comparison.to_csv(EXPERIMENT_DIR / "model_comparison.csv", index=False)
    save_metric_chart(comparison, CHART_DIR / "baseline_vs_smote_metrics.png")

    confusion_rows = []
    for model_name, metrics in (
        ("baseline", baseline_metrics),
        ("smote", smote_metrics),
    ):
        matrix = metrics["confusion_matrix"]
        confusion_rows.extend(
            [
                {"model": model_name, "actual": "no", "predicted_no": matrix[0][0], "predicted_yes": matrix[0][1]},
                {"model": model_name, "actual": "yes", "predicted_no": matrix[1][0], "predicted_yes": matrix[1][1]},
            ]
        )
    pd.DataFrame(confusion_rows).to_csv(
        EXPERIMENT_DIR / "confusion_matrix.csv", index=False
    )
    report = (
        "Baseline Random Forest\n"
        "======================\n"
        f"{baseline_metrics['classification_report']}\n"
        "SMOTENC Random Forest\n"
        "=====================\n"
        f"{smote_metrics['classification_report']}"
    )
    (EXPERIMENT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    results = {
        "project": "Bank Marketing Subscription Prediction",
        "dataset_rows": int(len(features)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_class": "yes",
        "features_used": features.columns.tolist(),
        "duration_excluded": True,
        "random_state": RANDOM_STATE,
        "test_set_resampled": False,
        "models": {
            "baseline": {
                "configuration": "RandomForestClassifier(class_weight='balanced', n_estimators=250)",
                "metrics": {name: baseline_metrics[name] for name in metric_names},
                "confusion_matrix": baseline_metrics["confusion_matrix"],
            },
            "smote": {
                "configuration": "SMOTENC followed by RandomForestClassifier(n_estimators=250)",
                "metrics": {name: smote_metrics[name] for name in metric_names},
                "confusion_matrix": smote_metrics["confusion_matrix"],
            },
        },
        "note": "Both models were evaluated on the same untouched stratified test split.",
    }
    (EXPERIMENT_DIR / "evaluation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(smote_model, SMOTE_MODEL_PATH)

    print("Baseline metrics:", results["models"]["baseline"]["metrics"])
    print("SMOTENC metrics:", results["models"]["smote"]["metrics"])
    print(f"Saved SMOTENC model to {SMOTE_MODEL_PATH}")


if __name__ == "__main__":
    main()
