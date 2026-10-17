"""
ui_home.py
----------
Home dashboard: hero banner, model summary cards, how it works, key ideas
and quick-navigation buttons. All numbers are read from the saved files,
never typed by hand.
"""

import streamlit as st

from src.ui_common import evaluation_mismatch, get_bundle, load_eval_results

HOME_CSS = """
<style>
.pg-hero {
    background: radial-gradient(circle at 15% 20%, rgba(0, 212, 255, 0.18), transparent 45%),
                linear-gradient(135deg, #0f2a4a 0%, #0b1220 70%);
    border: 1px solid #1f3b63;
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.5rem;
}
.pg-hero h1 { font-size: 2.8rem; margin: 0.5rem 0 0.2rem 0; color: #ffffff; padding: 0; }
.pg-hero-badge {
    display: inline-block; font-size: 0.75rem; letter-spacing: 0.12em; color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.4); padding: 0.2rem 0.8rem; border-radius: 999px;
}
.pg-hero-text { color: #b6c7de; font-size: 1.1rem; max-width: 46rem; margin: 0.4rem 0 1rem 0; }
.pg-chips span {
    display: inline-block; margin: 0 0.5rem 0.4rem 0; padding: 0.25rem 0.8rem;
    border-radius: 999px; background: rgba(255, 255, 255, 0.07);
    color: #dbe7f7; font-size: 0.85rem;
}
.pg-stat {
    background: #131c2e; border: 1px solid #1f3b63; border-radius: 14px;
    padding: 1rem 1.2rem; min-height: 6.6rem;
    box-shadow: 0 0 18px rgba(0, 212, 255, 0.06);
}
.pg-stat-label { color: #8fa6c4; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.07em; }
.pg-stat-value { color: #00d4ff; font-size: 1.9rem; font-weight: 700; margin-top: 0.2rem; line-height: 1.2; }
.pg-stat-note { color: #8fa6c4; font-size: 0.8rem; margin-top: 0.2rem; }
.pg-card {
    background: #131c2e; border: 1px solid #1f3b63; border-radius: 14px;
    padding: 1.1rem 1.2rem; min-height: 10rem;
}
.pg-card h4 { margin: 0 0 0.4rem 0; color: #e6edf7; padding: 0; font-size: 1.05rem; }
.pg-card p { margin: 0; color: #9fb3cc; font-size: 0.92rem; }
.pg-step-num {
    width: 2rem; height: 2rem; border-radius: 50%; background: #00d4ff; color: #0b1220;
    font-weight: 700; display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.6rem;
}
</style>
"""


def hero_html():
    return (
        '<div class="pg-hero">'
        '<div class="pg-hero-badge">EXPLAINABLE AI · PHISHING URL DETECTION</div>'
        "<h1>🛡️ PhishGuard</h1>"
        '<p class="pg-hero-text">Check a link before you click it. Paste a URL to get an '
                "instant risk estimate and a clear explanation of why. URLs are analysed as text "
        "only and are never opened or stored.</p>"
                '<div class="pg-chips"><span>Never opens URLs</span><span>SHAP explanations</span>'
        "<span>Honest evaluation</span><span>Open-source tools</span></div>"
        "</div>"
    )


def stat_card(label, value, note=""):
    note_html = f'<div class="pg-stat-note">{note}</div>' if note else ""
    return (
        '<div class="pg-stat">'
        f'<div class="pg-stat-label">{label}</div>'
        f'<div class="pg-stat-value">{value}</div>'
        f"{note_html}</div>"
    )


def info_card(title, text, number=None):
    badge = f'<div class="pg-step-num">{number}</div>' if number else ""
    return f'<div class="pg-card">{badge}<h4>{title}</h4><p>{text}</p></div>'


def go_to(page):
    """Button callback: switch the sidebar navigation to another page."""
    st.session_state["nav"] = page


def show_home():
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    st.markdown(hero_html(), unsafe_allow_html=True)

    bundle, error = get_bundle()
    if error:
        st.error(error)
        return
    info = bundle["info"]
    results = load_eval_results()

    # ---------- Model at a glance ----------
    st.subheader("Model at a glance")
    if results is None:
        st.warning(
            "Evaluation results not found. Run  python -m src.evaluate_model  "
            "in the terminal, then refresh this page."
        )
    else:
        if evaluation_mismatch(info, results):
            st.warning(
                "The saved evaluation is older than the current model. Run  "
                "python -m src.evaluate_model  and refresh to see up-to-date numbers."
            )
        best = results["best_model"]
        res = results["models"][best]
        cards = [
            ("Final model", best, "chosen by cross-validation"),
            ("Test accuracy", f"{res['accuracy']:.2%}", "on unseen test URLs"),
            ("Test F1-score", f"{res['f1']:.2%}", "precision and recall combined"),
            ("Test ROC-AUC", f"{res['roc_auc']:.4f}", "1.0 would be perfect"),
        ]
        for column, (label, value, note) in zip(st.columns(4), cards):
            column.markdown(stat_card(label, value, note), unsafe_allow_html=True)

    st.write("")
    small = [
        ("Training URLs", f"{info['train_rows']:,}", "used to teach the model"),
        ("Test URLs", f"{info['test_rows']:,}", "kept unseen until evaluation"),
        ("Features", str(len(info["feature_names"])), "calculated from URL text"),
        ("Model trained on", info["trained_at"][:10], "date of the saved model"),
    ]
    for column, (label, value, note) in zip(st.columns(4), small):
        column.markdown(stat_card(label, value, note), unsafe_allow_html=True)

    # ---------- How it works ----------
    st.subheader("How it works")
    steps = [
        ("Extract features", "The URL is split into measurable features such as length, dots, hyphens and HTTPS."),
        ("Predict", f"The saved {info['best_model']} model gives a phishing risk score."),
        ("Explain", "SHAP shows which features pushed the result toward phishing or legitimate."),
        ("Decide carefully", "The result is an estimate, never a guarantee of safety."),
    ]
    for column, (index, (title, text)) in zip(st.columns(4), enumerate(steps, start=1)):
        column.markdown(info_card(title, text, number=index), unsafe_allow_html=True)

    # ---------- Key ideas ----------
    st.subheader("Why this project is different")
    ideas = [
        ("Explainable", "Every prediction comes with the main reasons behind it, so the result is not a black box."),
                ("Private by design", "URLs are studied as plain text. They are never visited, stored, or forwarded to other services."),
        ("Honestly evaluated", "Scores come from data the model never saw, and the limitations of the dataset are stated openly."),
    ]
    for column, (title, text) in zip(st.columns(3), ideas):
        column.markdown(info_card(title, text), unsafe_allow_html=True)

    # ---------- Quick navigation ----------
    st.write("")
    left, middle, _ = st.columns([1, 1, 2])
    left.button("Open URL Scanner", type="primary", on_click=go_to, args=("URL Scanner",))
    middle.button("View Model Performance", on_click=go_to, args=("Model Performance",))

    st.info(
        "These scores were measured on one public dataset (PhiUSIIL). Performance on "
        "new, real-world URLs can be lower, so treat every result as an estimate."
    )