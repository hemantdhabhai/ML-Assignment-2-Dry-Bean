from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METRICS_PATH = ROOT / "results" / "metrics.csv"

st.set_page_config(page_title="Dry Bean Classifier | ML Assignment 2", page_icon="🫘", layout="wide")

@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knearest_neighbors.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
    }
    loaded = {}
    for name, filename in model_files.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return metadata, loaded

st.title("Dry Bean Classification")
st.caption("Machine Learning Assignment 2 | Compare five classification algorithms")

try:
    metadata, models = load_artifacts()
except Exception as exc:
    st.error("Trained model files are missing. Run `python train.py` first.")
    st.exception(exc)
    st.stop()

if not models:
    st.error("No saved models were found in the models folder. Run `python train.py` first.")
    st.stop()

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Choose a classifier", list(models.keys()))
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    st.caption("CSV must include all 16 input features and the actual target column.")

if uploaded_file is None:
    st.info("Upload `test_data.csv` from this project folder to evaluate the selected model.")
    st.write("**Required target column:**", metadata["target_column"])
    st.write("**Required feature count:**", len(metadata["feature_columns"]))
    if METRICS_PATH.exists():
        st.subheader("Training-run model comparison")
        st.dataframe(pd.read_csv(METRICS_PATH), use_container_width=True)
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

target = metadata["target_column"]
features = metadata["feature_columns"]
missing = [c for c in features + [target] if c not in test_df.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if test_df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

X_test = test_df[features]
y_test = test_df[target]
model = models[selected_model]
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

scores = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC Score": roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted"),
    "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
    "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
    "MCC Score": matthews_corrcoef(y_test, predictions),
}

st.subheader(f"Evaluation: {selected_model}")
st.write(f"**Rows evaluated:** {len(test_df):,} | **Features:** {len(features)} | **Classes:** {y_test.nunique()}")
metric_cols = st.columns(6)
for col, (label, value) in zip(metric_cols, scores.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion Matrix")
    labels = sorted(pd.Series(y_test).astype(str).unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted label", ylabel="True label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Model comparison")
if METRICS_PATH.exists():
    comparison = pd.read_csv(METRICS_PATH)
    st.dataframe(comparison.round(4), use_container_width=True)
else:
    st.caption("Run training to generate the comparison table.")

with st.expander("Preview uploaded test data"):
    st.dataframe(test_df.head(20), use_container_width=True)
