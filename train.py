from pathlib import Path
import json
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATASET_DIR = ROOT / "dataset"
for folder in (MODEL_DIR, RESULTS_DIR, DATASET_DIR):
    folder.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

def main():
    print("Downloading Dry Bean dataset from UCI...")
    dry_bean = fetch_ucirepo(id=602)
    X = dry_bean.data.features.copy()
    y = dry_bean.data.targets.iloc[:, 0].copy()
    target_name = dry_bean.data.targets.columns[0]

    data = X.copy()
    data[target_name] = y
    data.to_csv(DATASET_DIR / "dry_bean.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # The uploaded evaluation file must contain both features and the true target.
    test_data = X_test.copy()
    test_data[target_name] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=None
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
    }

    rows = []
    for display_name, model in models.items():
        print(f"Training {display_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        metrics = {
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(
                y_test, probabilities, multi_class="ovr", average="weighted"
            ),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        }
        rows.append(metrics)

        filename = display_name.lower().replace(" ", "_").replace("-", "")
        joblib.dump(model, MODEL_DIR / f"{filename}.joblib")
        print("  Accuracy:", round(metrics["Accuracy"], 4), "| AUC:", round(metrics["AUC"], 4))

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    metadata = {
        "target_column": target_name,
        "feature_columns": list(X.columns),
        "class_labels": sorted(pd.Series(y).astype(str).unique().tolist()),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "dataset": "UCI Dry Bean (ID 602)"
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False))
    print("\nSaved models, metrics.csv and test_data.csv.")
    print("IMPORTANT: Keep test_data.csv with the target column for app evaluation.")

if __name__ == "__main__":
    main()
