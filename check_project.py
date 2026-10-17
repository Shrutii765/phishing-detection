"""
check_project.py
----------------
End-to-end check: confirms that feature extraction, saved models, explanations,
evaluation results and the Streamlit dashboard all work together.
It only reads files; it never changes the project.

Run with:  python check_project.py      (takes about 2 minutes)
"""

import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.evaluate_model import EVAL_RESULTS_PATH
from src.explainability import URLExplainer, total_from_explanation
from src.feature_extraction import FEATURE_NAMES, extract_features, validate_url
from src.preprocessing import DATA_PATH, load_and_split
from src.train_model import ALL_MODELS_PATH, BEST_MODEL_PATH, MODEL_INFO_PATH

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_LEGIT = ["https://www.google.com", "https://www.mygov.in", "https://www.mygov.in/"]
SAMPLE_PHISH = "http://paypal-secure-login.example-verify.com/update?id=123&session=abc"


def row_for(url):
    return pd.DataFrame([extract_features(url)], columns=FEATURE_NAMES)


def check_files():
    source_files = [
        "__init__", "preprocessing", "feature_extraction", "train_model", "evaluate_model",
        "explainability", "ui_common", "ui_home", "ui_scanner", "ui_performance", "ui_about",
    ]
    required = [
        DATA_PATH, BEST_MODEL_PATH, ALL_MODELS_PATH, MODEL_INFO_PATH, EVAL_RESULTS_PATH,
        PROJECT_ROOT / "app.py", PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md", PROJECT_ROOT / ".streamlit" / "config.toml",
    ] + [PROJECT_ROOT / "src" / f"{name}.py" for name in source_files]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.exists()]
    assert not missing, "Missing files: " + ", ".join(missing)
    return f"{len(required)} required files found"


def check_requirements():
    lines = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").split()
    pinned = dict(line.split("==") for line in lines if "==" in line)
    needed = ["pandas", "numpy", "scikit-learn", "streamlit", "plotly", "matplotlib", "joblib", "shap"]
    missing = [name for name in needed if name not in pinned]
    assert not missing, f"Not pinned in requirements.txt: {missing}"
    wrong = [f"{n} (file {pinned[n]}, installed {metadata.version(n)})"
             for n in needed if pinned[n] != metadata.version(n)]
    assert not wrong, "Version mismatch: " + "; ".join(wrong)
    return "all 8 libraries pinned and matching the installed versions"


def check_features():
    a = extract_features("https://www.mygov.in/")
    b = extract_features("https://www.mygov.in")
    assert a == b, "A trailing slash changes the features"
    assert list(a.keys()) == FEATURE_NAMES, "Feature order differs from FEATURE_NAMES"
    for bad in ["", "   ", "not a url", "http://", "a" * 3000]:
        ok, _ = validate_url(bad)
        assert not ok, f"Invalid input was accepted: {bad[:20]!r}"
    assert validate_url("example.com")[0], "URL without http:// should be accepted"
    assert extract_features("http://192.168.1.1/login")["has_ip_host"] == 1
    assert extract_features("https://bit.ly/abc")["is_shortener"] == 1
    return f"{len(FEATURE_NAMES)} features, trailing slash ignored, bad input rejected"


def check_model_files():
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    best = joblib.load(BEST_MODEL_PATH)
    all_models = joblib.load(ALL_MODELS_PATH)
    assert info["feature_names"] == FEATURE_NAMES, "Saved feature list differs; retrain"
    assert set(all_models) == {"Logistic Regression", "Decision Tree", "Random Forest"}
    assert info["best_model"] in all_models
    row = row_for("https://www.google.com")
    same = abs(best.predict_proba(row)[0, 1]
               - all_models[info["best_model"]].predict_proba(row)[0, 1])
    assert same < 1e-12, "phishing_model.joblib differs from the best model in all_models.joblib"
    return f"3 models saved; best model = {info['best_model']}"


def check_sample_predictions():
    best = joblib.load(BEST_MODEL_PATH)
    messages = []
    for url in SAMPLE_LEGIT:
        score = float(best.predict_proba(row_for(url))[0, 1])
        assert score < 0.5, f"{url} scored {score:.1%} (expected legitimate)"
        messages.append(f"{score:.0%}")
    score = float(best.predict_proba(row_for(SAMPLE_PHISH))[0, 1])
    assert score >= 0.5, f"phishing-style sample scored only {score:.1%}"
    return f"legit samples scored {', '.join(messages)}; phishing-style sample {score:.0%} (sanity check only, not accuracy)"


def check_explainer():
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    best = joblib.load(BEST_MODEL_PATH)
    explainer = URLExplainer(best, info["feature_names"])
    row = row_for(SAMPLE_PHISH)
    explanation = explainer.explain(row)
    if explainer.kind == "tree":
        gap = abs(total_from_explanation(explanation) - float(best.predict_proba(row)[0, 1]))
        assert gap < 1e-3, f"SHAP values do not add up to the prediction (gap {gap:.5f})"
        return f"{explainer.method}; explanation adds up to the prediction (gap {gap:.1e})"
    return f"{explainer.method} produced {len(explanation['items'])} contributions"


def check_evaluation_matches_test_data():
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    saved = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    assert saved["best_model"] == info["best_model"], "Evaluation belongs to a different model; rerun evaluate_model"
    assert saved["evaluated_at"] >= info["trained_at"], "Evaluation is older than the model; rerun evaluate_model"

    models = joblib.load(ALL_MODELS_PATH)
    data = load_and_split()  # same fixed split as training
    X_test, y_test = data["X_test"], data["y_test"]
    assert len(X_test) == info["test_rows"], "Test set size changed since training"

    for name, model in models.items():
        predicted = model.predict(X_test)
        s = saved["models"][name]
        assert abs(accuracy_score(y_test, predicted) - s["accuracy"]) < 1e-9, f"{name}: accuracy differs"
        assert abs(f1_score(y_test, predicted) - s["f1"]) < 1e-9, f"{name}: F1 differs"
        assert s["tn"] + s["fp"] + s["fn"] + s["tp"] == len(y_test), f"{name}: confusion matrix size wrong"
    return f"recomputed test metrics for 3 models match evaluation_results.json ({len(X_test):,} test URLs)"


def check_dashboard():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180)
    app.run()
    assert len(app.exception) == 0, f"Home page error: {app.exception[0].value}"

    for page in ["URL Scanner", "Model Performance", "About Project", "Home"]:
        app.sidebar.radio(key="nav").set_value(page).run()
        assert len(app.exception) == 0, f"{page} page error: {app.exception[0].value}"

    app.sidebar.radio(key="nav").set_value("URL Scanner").run()
    app.text_input[0].set_value("https://www.google.com")
    app.button[0].click().run()
    assert len(app.exception) == 0, f"Scanner error: {app.exception[0].value}"
    return "all 4 pages open and a test scan runs without errors"


CHECKS = [
    ("Required files exist", check_files),
    ("requirements.txt matches installed libraries", check_requirements),
    ("Feature extraction", check_features),
    ("Saved model files are consistent", check_model_files),
    ("Sample predictions (sanity)", check_sample_predictions),
    ("SHAP explanation matches the prediction", check_explainer),
    ("Saved evaluation matches the real test data", check_evaluation_matches_test_data),
    ("Streamlit dashboard", check_dashboard),
]

if __name__ == "__main__":
    outcomes = []
    for title, function in CHECKS:
        print(f"Running: {title} ...", flush=True)
        try:
            outcomes.append((title, True, function()))
        except Exception as error:  # report every problem instead of stopping at the first
            outcomes.append((title, False, f"{type(error).__name__}: {error}"))

    print("\n=== PROJECT CHECK ===")
    for title, passed, detail in outcomes:
        print(f"[{'PASS' if passed else 'FAIL'}] {title}\n        {detail}")

    failed = [title for title, passed, _ in outcomes if not passed]
    print(f"\n{len(outcomes) - len(failed)} of {len(outcomes)} checks passed.")
    sys.exit(1 if failed else 0)