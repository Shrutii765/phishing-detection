"""
explainability.py
-----------------
Explains ONE prediction: which features pushed the model toward
"phishing" and which pushed toward "legitimate".

- Tree models (Random Forest, Decision Tree): SHAP TreeExplainer.
  Contributions are in probability units, so
  base_value + sum(contributions) = predicted phishing probability.
- Logistic Regression: exact contributions = coefficient x scaled value
  (log-odds units).

IMPORTANT: these explain what the MODEL relied on. They are not proof of
what actually makes a real website malicious.

Run a self-test with:  python -m src.explainability
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_extraction import FEATURE_NAMES, extract_features

FEATURE_DESCRIPTIONS = {
    "url_length": "Total number of characters in the URL",
    "hostname_length": "Length of the domain name part",
    "path_length": "Length of the path after the domain",
    "query_length": "Length of the part after '?'",
    "num_dots": "Number of dots in the URL",
    "num_subdomains": "Estimated number of subdomains (www counts as one)",
    "num_hyphens": "Number of '-' characters in the URL",
    "num_hyphens_in_host": "Number of '-' characters in the domain name",
    "num_underscores": "Number of '_' characters",
    "num_at_symbols": "Number of '@' symbols",
    "num_question_marks": "Number of '?' characters",
    "num_equals": "Number of '=' characters",
    "num_ampersands": "Number of '&' characters",
    "num_percent": "Number of '%' characters (encoded text)",
    "num_digits": "Number of digits in the URL",
    "digit_ratio": "Share of URL characters that are digits",
    "path_depth": "Number of folders in the path",
    "has_ip_host": "Domain is a raw IP address (1 = yes)",
    "is_https": "URL starts with https:// (1 = yes)",
    "has_port": "URL contains a port number (1 = yes)",
    "is_shortener": "Known URL-shortening service (1 = yes)",
    "suspicious_word_count": "Count of words like login, verify, secure, account",
    "has_punycode": "Domain uses 'xn--' encoding (1 = yes)",
}


class URLExplainer:
    """Wraps a trained model and explains single predictions."""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)

        if isinstance(model, Pipeline):
            self.kind = "linear"
            self.scaler = model.named_steps["scaler"]
            self.classifier = model.named_steps["classifier"]
            self.method = "Linear model contributions (coefficient x scaled value)"
            self.unit = "log-odds"
        else:
            import shap  # imported here because it is slow to load

            self.kind = "tree"
            self.explainer = shap.TreeExplainer(model)
            self.method = "SHAP (TreeExplainer)"
            self.unit = "probability"

    def _tree_contributions(self, row):
        values = self.explainer.shap_values(row, check_additivity=False)
        if isinstance(values, list):          # older SHAP: list of arrays per class
            sv = np.asarray(values[1])
        else:                                 # newer SHAP: (rows, features, classes)
            sv = np.asarray(values)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        expected = np.atleast_1d(self.explainer.expected_value)
        base = expected[1] if len(expected) > 1 else expected[0]
        return sv[0], float(base)             # class 1 = phishing

    def _linear_contributions(self, row):
        scaled = self.scaler.transform(row)
        contributions = self.classifier.coef_[0] * scaled[0]
        return contributions, float(self.classifier.intercept_[0])

    def explain(self, row):
        """
        row: a one-row DataFrame with the feature columns.
        Returns a dictionary with method, unit, base_value and a list of
        items sorted by importance for THIS prediction.
        """
        row = row[self.feature_names]
        if self.kind == "tree":
            contributions, base = self._tree_contributions(row)
        else:
            contributions, base = self._linear_contributions(row)

        items = []
        for name, value, contrib in zip(self.feature_names, row.iloc[0].tolist(), contributions):
            if contrib > 0:
                direction = "Toward phishing"
            elif contrib < 0:
                direction = "Toward legitimate"
            else:
                direction = "No effect"
            items.append({
                "feature": name,
                "value": float(value),
                "contribution": float(contrib),
                "direction": direction,
            })
        items.sort(key=lambda item: abs(item["contribution"]), reverse=True)

        return {
            "method": self.method,
            "unit": self.unit,
            "base_value": float(base),
            "items": items,
        }


def explanation_table(explanation, top_n=8):
    """Readable table of the top features (for the app and for printing)."""
    rows = []
    for item in explanation["items"][:top_n]:
        rows.append({
            "Feature": item["feature"],
            "Value": f"{item['value']:g}",
            "Meaning": FEATURE_DESCRIPTIONS.get(item["feature"], ""),
            "Effect": item["direction"],
            "Contribution": round(item["contribution"], 4),
        })
    return pd.DataFrame(rows)


def explanation_figure(explanation, top_n=10):
    """Plotly bar chart: red bars push toward phishing, green toward legitimate."""
    import plotly.graph_objects as go

    items = explanation["items"][:top_n][::-1]  # reversed so the biggest is on top
    colors = ["#e74c3c" if i["contribution"] > 0 else "#2ecc71" for i in items]
    labels = [f"{i['feature']} = {i['value']:g}" for i in items]

    fig = go.Figure(go.Bar(
        x=[i["contribution"] for i in items],
        y=labels,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Top factors behind this prediction",
        xaxis_title=f"<- toward legitimate | toward phishing ->  ({explanation['unit']})",
        height=60 * len(items) + 140,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def total_from_explanation(explanation):
    """base value + all contributions (used to verify the explanation)."""
    return explanation["base_value"] + sum(i["contribution"] for i in explanation["items"])


if __name__ == "__main__":
    from src.train_model import BEST_MODEL_PATH, MODEL_INFO_PATH

    if not BEST_MODEL_PATH.exists():
        raise SystemExit("Model not found. Run:  python -m src.train_model")

    model = joblib.load(BEST_MODEL_PATH)
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    explainer = URLExplainer(model, info["feature_names"])
    print(f"Model: {info['best_model']} | Method: {explainer.method}")

    samples = [
        "https://www.google.com",
        "http://paypal-secure-login.example-verify.com/update?id=123&session=abc",
    ]
    for url in samples:
        row = pd.DataFrame([extract_features(url)], columns=FEATURE_NAMES)
        probability = float(model.predict_proba(row)[0, 1])
        explanation = explainer.explain(row)

        print(f"\nURL: {url}")
        print(f"Predicted phishing probability: {probability:.1%}")
        print(explanation_table(explanation, top_n=6).to_string(index=False))

        if explainer.kind == "tree":
            total = total_from_explanation(explanation)
            print(f"Check: base value + contributions = {total:.4f} "
                  f"(model probability = {probability:.4f})")

    print("\nNote: explanations show what the MODEL relied on, not proof of real-world cause.")