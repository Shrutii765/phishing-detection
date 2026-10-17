"""
evaluate_model.py
-----------------
Evaluates all trained models on the held-out TEST set (data the models
never saw during training) and saves the real results for the dashboard.

Run with:  python -m src.evaluate_model
"""

import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files only, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import LABEL_NAMES, load_and_split
from src.train_model import ALL_MODELS_PATH, MODEL_INFO_PATH, MODELS_DIR

EVAL_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"
FIGURES_DIR = MODELS_DIR / "figures"


def get_feature_importance(model, feature_names):
    """Global feature importance for a model, sorted from highest to lowest."""
    final = model.named_steps["classifier"] if isinstance(model, Pipeline) else model

    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
        method = "Gini importance (tree-based)"
    else:
        values = np.abs(final.coef_[0])
        method = "Absolute standardized coefficient"

    ranked = sorted(zip(feature_names, values), key=lambda pair: pair[1], reverse=True)
    return {"method": method, "values": {name: float(v) for name, v in ranked}}


def evaluate_one(model, X_test, y_test):
    """Calculate all metrics for one model. Phishing (label 1) is the positive class."""
    y_pred = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    fpr, tpr, _ = roc_curve(y_test, probability)
    keep = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)  # smaller file

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_curve": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
    }


def save_confusion_matrix_figure(result, model_name, path):
    cm = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_figure(results, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, res in results.items():
        ax.plot(res["roc_curve"]["fpr"], res["roc_curve"]["tpr"],
                label=f"{name} (AUC = {res['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_importance_figure(importance, model_name, path, top_n=15):
    items = list(importance["values"].items())[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([name for name, _ in items], [value for _, value in items], color="#1f77b4")
    ax.set_xlabel(importance["method"])
    ax.set_title(f"Top feature importances - {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate():
    if not ALL_MODELS_PATH.exists() or not MODEL_INFO_PATH.exists():
        raise FileNotFoundError("Trained models not found. Run:  python -m src.train_model")

    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    models = joblib.load(ALL_MODELS_PATH)
    best_name = info["best_model"]

    print("Loading data and recreating the same train/test split (about 1 minute)...",
          flush=True)
    data = load_and_split()
    X_test, y_test = data["X_test"], data["y_test"]

    if list(X_test.columns) != info["feature_names"]:
        raise ValueError("Feature list changed since training. Please re-run train_model.")

    results = {name: evaluate_one(model, X_test, y_test) for name, model in models.items()}
    importance = {name: get_feature_importance(model, info["feature_names"])
                  for name, model in models.items()}

    # What would a "lazy" model score if it always predicted the bigger class?
    majority_share = float(max(y_test.mean(), 1 - y_test.mean()))

    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name,
        "test_rows": int(len(y_test)),
        "phishing_share_test": float(y_test.mean()),
        "baseline_accuracy_always_majority_class": majority_share,
        "models": results,
        "feature_importance": importance,
    }
    EVAL_RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    FIGURES_DIR.mkdir(exist_ok=True)
    save_confusion_matrix_figure(results[best_name], best_name,
                                 FIGURES_DIR / "confusion_matrix_best.png")
    save_roc_figure(results, FIGURES_DIR / "roc_curves.png")
    save_importance_figure(importance[best_name], best_name,
                           FIGURES_DIR / "feature_importance_best.png")

    # ---------- Print a readable report ----------
    print(f"\n=== TEST SET RESULTS ({len(y_test):,} URLs never seen in training) ===")
    table = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    print(table.round(4).to_string())
    print(f"\nFinal selected model (chosen earlier using cross-validation): {best_name}")

    print("\n=== CONFUSION MATRICES (rows = actual, columns = predicted) ===")
    for name, r in results.items():
        print(f"\n{name}")
        print(f"                     Pred Legit   Pred Phishing")
        print(f"  Actual Legit      {r['tn']:>10,}   {r['fp']:>13,}")
        print(f"  Actual Phishing   {r['fn']:>10,}   {r['tp']:>13,}")

    print("\n=== WHY ACCURACY ALONE CAN MISLEAD ===")
    print(f"  A lazy model that ALWAYS says 'Legitimate' would score "
          f"{majority_share:.1%} accuracy,")
    print("  yet it would catch 0% of phishing URLs (recall = 0).")
    print("  That is why we also check precision, recall, F1 and ROC-AUC.")

    print(f"\nTop 5 features for {best_name} ({importance[best_name]['method']}):")
    for name, value in list(importance[best_name]["values"].items())[:5]:
        print(f"  {name:24} {value:.4f}")

    print(f"\nSaved: {EVAL_RESULTS_PATH.name} and charts in models/figures/")


if __name__ == "__main__":
    evaluate()