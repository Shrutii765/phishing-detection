"""
ui_scanner.py
-------------
The URL Scanner page. The submitted URL is analysed ONLY as text.
It is never opened, visited, or sent to any outside service.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explainability import FEATURE_DESCRIPTIONS, explanation_figure, explanation_table
from src.feature_extraction import FEATURE_NAMES, extract_features, validate_url
from src.ui_common import get_bundle, render_header

DISCLAIMER = (
    "This result is a statistical estimate from a machine learning model, not a "
    "guarantee. A URL marked 'Likely Legitimate' can still be harmful, and a "
    "legitimate site can be flagged by mistake. No single feature, including "
    "HTTPS, proves a site is safe or malicious. Never enter passwords or personal "
    "data on a site you are unsure about."
)


def risk_level(score):
    if score < 0.30:
        return "Low"
    if score < 0.70:
        return "Medium"
    return "High"


def gauge_figure(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%"},
        title={"text": "Phishing risk score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ffffff", "thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "#2ecc71"},
                {"range": [30, 70], "color": "#f1c40f"},
                {"range": [70, 100], "color": "#e74c3c"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def analyze_url(url, bundle):
    if not url or not url.strip():
        st.warning("Please enter a URL to analyze.")
        return

    ok, message = validate_url(url)
    if not ok:
        st.error(f"Invalid input: {message}")
        return

    try:
        features = extract_features(url)
        row = pd.DataFrame([features], columns=FEATURE_NAMES)
        score = float(bundle["model"].predict_proba(row)[0, 1])
        explanation = bundle["explainer"].explain(row)
    except Exception as exc:
        st.error(f"Something went wrong while analysing this URL: {exc}")
        return

    is_phishing = score >= 0.5
    label = "Potentially Phishing" if is_phishing else "Likely Legitimate"
    css_class = "result-phish" if is_phishing else "result-legit"
    icon = "⚠️" if is_phishing else "✅"
    level = risk_level(score)

    st.code(url.strip(), language=None)
    st.markdown(
        f'<div class="result-card {css_class}"><h2>{icon} {label}</h2>'
        f"<p>Risk level: <b>{level}</b></p></div>",
        unsafe_allow_html=True,
    )

    col_gauge, col_info = st.columns([1, 1.2])
    with col_gauge:
        st.plotly_chart(gauge_figure(score))
    with col_info:
        st.metric("Phishing risk score", f"{score:.1%}")
        st.metric("Risk level", level)
        st.caption(
            f"Model: {bundle['info']['best_model']}. The score is the model's output "
            "and works as a risk indicator, not a guaranteed probability."
        )

    st.subheader("Why did the model decide this?")
    st.plotly_chart(explanation_figure(explanation, top_n=10))
    st.caption(
        "Red bars pushed the model toward phishing; green bars pushed it toward "
        "legitimate. This shows what the model relied on, not a guaranteed "
        "real-world cause."
    )
    st.dataframe(explanation_table(explanation, top_n=8), hide_index=True)

    with st.expander("Show all extracted features"):
        all_features = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Value": [f"{features[name]:g}" for name in FEATURE_NAMES],
            "Meaning": [FEATURE_DESCRIPTIONS.get(name, "") for name in FEATURE_NAMES],
        })
        st.dataframe(all_features, hide_index=True)

    st.warning(DISCLAIMER)


def show_scanner():
    render_header(
        "URL Scanner",
        "Paste a URL to estimate whether it is likely legitimate or potentially phishing.",
    )

    bundle, error = get_bundle()
    if error:
        st.error(error)
        return

    with st.form("scan_form"):
        url = st.text_input("Enter a URL", placeholder="https://example.com/login")
        submitted = st.form_submit_button("Analyze URL", type="primary")

    st.caption(
                "Privacy: the URL is analysed as text only by this app. It is never opened or "
        "visited, it is not forwarded to any other service, and it is not stored."
    )

    if submitted:
        analyze_url(url, bundle)