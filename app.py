"""BankML interactive Streamlit application."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import mongodb_service as db
from services import model_service as models
from services.prediction_service import (
    NUMERIC_FIELDS,
    predict,
    record_prediction,
    validate_inputs,
)

st.set_page_config(
    page_title="BankML - Machine Learning Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3 {color: #102a43;}
    .metric-card {background: white; border: 1px solid #d9e2ec; border-radius: 12px;
                  padding: 1rem 1.1rem; box-shadow: 0 2px 8px #102a4312;}
    .result-yes {background: #d1fae5; color: #065f46; padding: 1rem; border-radius: 12px;
                 font-size: 1.25rem; font-weight: 700;}
    .result-no {background: #e8eef5; color: #102a43; padding: 1rem; border-radius: 12px;
                font-size: 1.25rem; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def get_stats():
    return db.customer_statistics()


@st.cache_data(ttl=300, show_spinner=False)
def get_filter_options():
    return db.customer_filter_options()


@st.cache_data(ttl=300, show_spinner=False)
def get_analysis_data():
    return db.fetch_all_customers()


def metric_cards(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.markdown(
            f'<div class="metric-card"><small>{label}</small><h2>{value}</h2></div>',
            unsafe_allow_html=True,
        )


def page_dashboard() -> None:
    st.title("Dashboard")
    st.caption("Bank marketing subscription intelligence from the connected MongoDB dataset.")
    try:
        stats = get_stats()
        baseline = models.baseline_results()["metrics"]
        smote = models.smote_results()["models"]["smote"]["metrics"]
    except Exception as exc:
        st.error(f"Could not load dashboard data: {exc}")
        return
    metric_cards(
        [
            ("Total customers", f"{stats['total']:,}"),
            ("Subscribed", f"{stats['subscribed']:,}"),
            ("Not subscribed", f"{stats['not_subscribed']:,}"),
            ("Subscription rate", f"{stats['subscription_rate']:.1%}"),
            ("Baseline accuracy", f"{baseline['accuracy']:.2%}"),
            ("Baseline recall", f"{baseline['recall']:.2%}"),
        ]
    )
    left, right = st.columns(2)
    with left:
        distribution = pd.DataFrame(
            {"Outcome": ["Not subscribed", "Subscribed"], "Customers": [stats["not_subscribed"], stats["subscribed"]]}
        )
        st.plotly_chart(
            px.pie(distribution, names="Outcome", values="Customers", hole=0.55,
                   color_discrete_sequence=["#4c78a8", "#0f9d8a"]),
            use_container_width=True,
        )
    with right:
        metric_frame = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                "Baseline": [baseline[k] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]],
                "SMOTENC": [smote[k] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]],
            }
        ).melt("Metric", var_name="Model", value_name="Score")
        st.plotly_chart(
            px.bar(metric_frame, x="Metric", y="Score", color="Model", barmode="group",
                   text_auto=".1%", color_discrete_sequence=["#12355b", "#f28e2b"]),
            use_container_width=True,
        )
    st.info("The baseline remains preferred for subscriber detection because it has higher recall, F1-score, and ROC-AUC.")


def page_explorer() -> None:
    st.title("Dataset Explorer")
    options = get_filter_options()
    col1, col2, col3 = st.columns(3)
    search = col1.text_input("Search categorical fields")
    outcome = col2.selectbox("Subscription outcome", ["All"] + options["outcomes"])
    job = col3.selectbox("Job", ["All"] + options["jobs"])
    marital = st.selectbox("Marital status", ["All"] + options["marital"])
    page_size = st.selectbox("Rows per page", [10, 25, 50], index=1)
    rows, total = db.fetch_customers(page=1, page_size=page_size, search=search, outcome=outcome, job=job, marital=marital)
    pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    if page != 1:
        rows, total = db.fetch_customers(page=int(page), page_size=page_size, search=search, outcome=outcome, job=job, marital=marital)
    st.caption(f"{total:,} matching records | Page {int(page)} of {pages}")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        selected = st.number_input("View row on this page", min_value=1, max_value=len(rows), value=1, step=1)
        with st.expander("Record details"):
            st.json(rows[int(selected) - 1])
    else:
        st.info("No records match the selected filters.")


def page_analysis() -> None:
    st.title("Data Analysis")
    try:
        frame = get_analysis_data()
    except Exception as exc:
        st.error(f"Could not load analysis data: {exc}")
        return
    analysis_frame = frame.replace("", pd.NA)
    for column in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        if column in analysis_frame:
            analysis_frame[column] = pd.to_numeric(analysis_frame[column], errors="coerce")
    missing = analysis_frame.isna().sum().reset_index()
    missing.columns = ["Column", "Missing values"]
    metric_cards([("Rows", f"{len(analysis_frame):,}"), ("Columns", f"{len(analysis_frame.columns):,}"),
                  ("Duplicates", f"{int(analysis_frame.duplicated().sum()):,}")])
    st.subheader("Schema and missing values")
    st.dataframe(pd.DataFrame({"Column": analysis_frame.columns, "Type": analysis_frame.dtypes.astype(str).values}), use_container_width=True, hide_index=True)
    st.dataframe(missing, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(analysis_frame, x="y", color="y", title="Subscription distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(analysis_frame, x="job", color="y", title="Job distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(analysis_frame, x="age", color="y", nbins=30, title="Age distribution"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(analysis_frame, x="balance", color="y", nbins=40, title="Balance distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(analysis_frame, x="campaign", color="y", title="Campaign analysis"), use_container_width=True)
        st.subheader("Descriptive statistics")
        st.dataframe(analysis_frame.describe(include="all").transpose(), use_container_width=True)


def confusion_figure(matrix, title):
    return px.imshow(matrix, text_auto=True, x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"],
                     color_continuous_scale="Blues", title=title, aspect="auto")


def page_performance() -> None:
    st.title("Model Performance")
    try:
        result = models.baseline_results()
        metrics = result["metrics"]
    except Exception as exc:
        st.error(f"Could not load baseline results: {exc}")
        return
    metric_cards([(label, f"{metrics[key]:.2%}") for label, key in
                  [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1_score"), ("ROC-AUC", "roc_auc")]])
    st.plotly_chart(confusion_figure(result["confusion_matrix"], "Baseline confusion matrix"), use_container_width=True)
    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]
    st.write(f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")
    st.markdown("**Accuracy** is the overall proportion correct. **Precision** measures how many predicted subscribers subscribed. **Recall** measures how many actual subscribers were found. **F1-score** balances precision and recall. **ROC-AUC** measures ranking quality across thresholds.")
    try:
        st.code(models.classification_report_text())
    except Exception as exc:
        st.warning(str(exc))


def page_comparison() -> None:
    st.title("Model Comparison")
    try:
        baseline = models.baseline_results()
        smote = models.smote_results()["models"]["smote"]
    except Exception as exc:
        st.error(f"Could not load comparison results: {exc}")
        return
    b = baseline["metrics"]
    s = smote["metrics"]
    metric_cards([("Baseline recall", f"{b['recall']:.2%}"), ("SMOTENC recall", f"{s['recall']:.2%}"),
                  ("Baseline F1", f"{b['f1_score']:.2%}"), ("SMOTENC F1", f"{s['f1_score']:.2%}")])
    frame = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                          "Baseline": [b[k] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]],
                          "SMOTENC": [s[k] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]]}).melt("Metric", var_name="Model", value_name="Score")
    st.plotly_chart(px.bar(frame, x="Metric", y="Score", color="Model", barmode="group", text_auto=".1%"), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(confusion_figure(baseline["confusion_matrix"], "Baseline"), use_container_width=True)
    right.plotly_chart(confusion_figure(smote["confusion_matrix"], "SMOTENC"), use_container_width=True)
    st.warning("SMOTENC increased accuracy and precision, but reduced recall, F1-score, and ROC-AUC. The baseline is recommended for subscriber detection.")


def page_prediction() -> None:
    st.title("Customer Prediction")
    st.caption("Duration is intentionally excluded because it is known only after a call.")
    model_name = st.selectbox("Model", list(models.MODEL_PATHS))
    with st.form("prediction_form"):
        first, second = st.columns(2)
        values = {
            "age": first.number_input("Age", min_value=18, max_value=100, value=42),
            "job": second.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]),
            "marital": first.selectbox("Marital", ["married", "single", "divorced"]),
            "education": second.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"]),
            "default": first.selectbox("Default", ["no", "yes"]),
            "balance": second.number_input("Balance", value=1200, step=100),
            "housing": first.selectbox("Housing loan", ["no", "yes"]),
            "loan": second.selectbox("Personal loan", ["no", "yes"]),
            "contact": first.selectbox("Contact", ["cellular", "telephone", "unknown"]),
            "day": second.number_input("Day", min_value=1, max_value=31, value=15),
            "month": first.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
            "campaign": second.number_input("Campaign contacts", min_value=1, max_value=100, value=1),
            "pdays": first.number_input("Days since previous contact", min_value=-1, max_value=10000, value=-1),
            "previous": second.number_input("Previous contacts", min_value=0, max_value=10000, value=0),
            "poutcome": first.selectbox("Previous outcome", ["unknown", "failure", "other", "success"]),
        }
        submitted = st.form_submit_button("Predict Subscription", type="primary")
    if submitted:
        errors = validate_inputs(values)
        if errors:
            for error in errors:
                st.error(error)
            return
        try:
            result, probability = predict(models.load_model(model_name), values)
            record_prediction(values, result, probability, model_name)
            st.session_state["last_prediction"] = (result, probability, model_name)
            st.success("Prediction saved to PredictionHistory.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    if "last_prediction" in st.session_state:
        result, probability, selected_model = st.session_state["last_prediction"]
        st.markdown(f'<div class="{"result-yes" if result == "yes" else "result-no"}">Predicted subscription: {result.upper()}<br>Probability of Yes: {probability:.2%}<br>Model: {selected_model}</div>', unsafe_allow_html=True)
        st.caption("This probability is an estimate, not a guarantee.")


def page_history() -> None:
    st.title("Prediction History")
    search, model_name = st.columns(2)
    search_value = search.text_input("Search history")
    model_value = model_name.selectbox("Model", ["All"] + list(models.MODEL_PATHS))
    page_size = st.selectbox("Rows per page", [10, 25, 50], key="history_page_size")
    rows, total = db.fetch_prediction_history(page=1, page_size=page_size, search=search_value, model_name=model_value)
    pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("History page", min_value=1, max_value=pages, value=1, step=1)
    if page != 1:
        rows, total = db.fetch_prediction_history(page=int(page), page_size=page_size, search=search_value, model_name=model_value)
    st.caption(f"{total:,} prediction records | Page {int(page)} of {pages}")
    if rows:
        table = pd.DataFrame(rows)
        st.dataframe(table.drop(columns=["input_features"], errors="ignore"), use_container_width=True, hide_index=True)
        selected = st.number_input("View prediction on this page", min_value=1, max_value=len(rows), value=1, step=1)
        with st.expander("Prediction details"):
            st.json(rows[int(selected) - 1])
    else:
        st.info("No prediction history is available.")


def page_about() -> None:
    st.title("About BankML")
    st.markdown(
        """
        BankML predicts whether a bank client subscribes to a term deposit using the UCI Bank Marketing dataset.
        MongoDB stores the original customer records, while Pandas and scikit-learn provide preprocessing and
        Random Forest modelling. The baseline uses balanced class weights. The separate SMOTENC experiment
        addresses the minority class using category-aware resampling without modifying the test set.

        The dataset is imbalanced, with 5,289 subscribers among 45,211 customers. Evaluation uses accuracy,
        precision, recall, F1-score, confusion matrices, and ROC-AUC with `yes` as the positive class.
        The baseline is currently recommended for subscriber detection because it has higher recall, F1-score,
        and ROC-AUC than SMOTENC. Prediction history is stored in a separate MongoDB collection.
        """
    )
    st.info("Technology stack: Streamlit, MongoDB, PyMongo, Pandas, NumPy, scikit-learn, imbalanced-learn, Plotly, and Joblib.")


PAGES = {
    "Dashboard": page_dashboard,
    "Dataset Explorer": page_explorer,
    "Data Analysis": page_analysis,
    "Model Performance": page_performance,
    "Model Comparison": page_comparison,
    "Customer Prediction": page_prediction,
    "Prediction History": page_history,
    "About Project": page_about,
}

st.sidebar.title("🏦 BankML")
st.sidebar.caption("Machine Learning Prediction System")
selection = st.sidebar.radio("Navigation", list(PAGES))
st.sidebar.divider()
st.sidebar.caption("Read-only customer data access. Predictions are stored separately.")
PAGES[selection]()
