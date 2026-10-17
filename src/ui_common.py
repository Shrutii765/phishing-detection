"""
ui_common.py
------------
Shared helpers for the Streamlit app: styling, page header, and cached
loading of the saved model, its metadata and the evaluation results.
The model is loaded once and reused, never retrained for a prediction.
"""

import json
from pathlib import Path

import joblib
import streamlit as st

from src.explainability import URLExplainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
INFO_PATH = MODELS_DIR / "model_info.json"
EVAL_PATH = MODELS_DIR / "evaluation_results.json"

CUSTOM_CSS = """
<style>
.main-header {
    background: linear-gradient(90deg, #0f2a4a 0%, #0b1220 100%);
    border: 1px solid #1f3b63;
    border-radius: 12px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.2rem;
}
.main-header h1 { margin: 0; font-size: 1.8rem; color: #00d4ff; padding: 0; }
.main-header p { margin: 0.3rem 0 0 0; color: #9fb3cc; }
.result-card {
    border-radius: 12px;
    padding: 0.9rem 1.3rem;
    margin: 0.8rem 0 0.8rem 0;
    border: 1px solid;
}
.result-card h2 { margin: 0; padding: 0; }
.result-card p { margin: 0.3rem 0 0 0; }
.result-legit { background: rgba(46, 204, 113, 0.12); border-color: #2ecc71; }
.result-phish { background: rgba(231, 76, 60, 0.12); border-color: #e74c3c; }
</style>
"""


def apply_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title, subtitle):
    st.markdown(
        f'<div class="main-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading model...")
def _load_bundle():
    model = joblib.load(MODEL_PATH)
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    explainer = URLExplainer(model, info["feature_names"])
    return {"model": model, "info": info, "explainer": explainer}


def get_bundle():
    """Returns (bundle, None) on success or (None, error_message) on failure."""
    if not MODEL_PATH.exists() or not INFO_PATH.exists():
        return None, (
            "Model files were not found. Please run  python -m src.train_model  "
            "in the terminal first, then refresh this page."
        )
    try:
        return _load_bundle(), None
    except Exception as exc:  # show a friendly message instead of crashing
        return None, f"The saved model could not be loaded: {exc}"


@st.cache_data
def load_eval_results():
    """Evaluation results saved by evaluate_model.py (None if unavailable)."""
    if not EVAL_PATH.exists():
        return None
    try:
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    

def evaluation_mismatch(info, results):
    """True if the saved evaluation does not belong to the currently saved model."""
    if results is None:
        return False
    return (
        results.get("best_model") != info.get("best_model")
        or results.get("evaluated_at", "") < info.get("trained_at", "")
    )