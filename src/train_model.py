"""
train_model.py
--------------
Trains and compares three models, picks the best using cross-validation on
the TRAINING data only, and saves everything with Joblib.

Run with:  python -m src.train_model
"""

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.feature_extraction import FEATURE_NAMES, extract_features
from src.preprocessing import LABEL_NAMES, RANDOM_STATE, TEST_SIZE, load_and_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "phishing_model.joblib"   # used by the app
ALL_MODELS_PATH = MODELS_DIR / "all_models.joblib"       # used by evaluation
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"         # metadata + CV results

CV_FOLDS = 5


def get_models():
    """The three models we compare. Settings are sensible defaults, not heavily tuned."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def compare_models(models, X_train, y_train):
    """5-fold cross-validation on training data. Returns average scores per model."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",   # phishing (label 1) is the positive class
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = {}
    for name, model in models.items():
        start = time.time()
        print(f"Cross-validating {name} ...", flush=True)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        results[name] = {m: float(np.mean(scores[f"test_{m}"])) for m in scoring}
        results[name]["f1_std"] = float(np.std(scores["test_f1"]))
        print(f"  done in {time.time() - start:.0f}s | mean F1 = {results[name]['f1']:.4f}",
              flush=True)
    return results


def train():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Step 1/4: loading and preprocessing data (about 1 minute)...", flush=True)
    data = load_and_split()
    X_train, y_train = data["X_train"], data["y_train"]
    print(f"  Training rows: {len(X_train):,}", flush=True)

    print("\nStep 2/4: comparing models with cross-validation...", flush=True)
    models = get_models()
    cv_results = compare_models(models, X_train, y_train)

    best_name = max(cv_results, key=lambda n: cv_results[n]["f1"])

    print("\n=== CROSS-VALIDATION RESULTS (training data only) ===")
    print(pd.DataFrame(cv_results).T.round(4).to_string())
    print(f"\nBest model by mean F1-score: {best_name}")

    print("\nStep 3/4: training all models on the full training set...", flush=True)
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

    print("\nStep 4/4: saving files...", flush=True)
    joblib.dump(fitted, ALL_MODELS_PATH)
    joblib.dump(fitted[best_name], BEST_MODEL_PATH)

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "PhiUSIIL Phishing URL Dataset (UCI)",
        "best_model": best_name,
        "selection_metric": "mean F1-score (5-fold cross-validation on training data)",
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(data["X_test"])),
        "feature_names": FEATURE_NAMES,
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "cleaning_report": {k: int(v) for k, v in data["report"].items()},
        "cv_results": cv_results,
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  Saved: {BEST_MODEL_PATH.name}, {ALL_MODELS_PATH.name}, {MODEL_INFO_PATH.name}")

    sanity_check()


def sanity_check():
    """Load the SAVED model and predict two URLs, just to confirm the pipeline works.
    This is NOT a measure of accuracy. Real evaluation happens in Step 6."""
    model = joblib.load(BEST_MODEL_PATH)
    samples = [
        "https://www.google.com",
        "http://paypal-secure-login.example-verify.com/update?id=123&session=abc",
    ]
    print("\n=== SANITY CHECK (saved model loads and predicts) ===")
    for url in samples:
        row = pd.DataFrame([extract_features(url)], columns=FEATURE_NAMES)
        phishing_probability = float(model.predict_proba(row)[0, 1])
        label = LABEL_NAMES[int(phishing_probability >= 0.5)]
        print(f"  {url}\n    -> {label} (phishing probability {phishing_probability:.1%})")


if __name__ == "__main__":
    train()