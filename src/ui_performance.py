"""
ui_performance.py
-----------------
Model Performance page: model comparison, confusion matrix, ROC curves
and feature importance. All values come from evaluation_results.json
(created by evaluate_model.py on the held-out test set).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui_common import evaluation_mismatch, get_bundle, load_eval_results, render_header

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}


def metrics_table(results):
    rows = []
    for name, res in results["models"].items():
        row = {"Model": name}
        for metric in METRICS:
            row[DISPLAY[metric]] = res[metric]
        row["Selected"] = "Yes" if name == results["best_model"] else ""
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_figure(table):
    fig = go.Figure()
    score_columns = list(DISPLAY.values())
    for column in score_columns:
        fig.add_bar(name=column, x=table["Model"], y=table[column])
    lowest = float(table[score_columns].min().min())
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[max(0.0, lowest - 0.01), 1.0], title="Score"),
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def confusion_figure(res):
    z = [[res["tn"], res["fp"]], [res["fn"], res["tp"]]]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Legitimate", "Predicted Phishing"],
        y=["Actual Legitimate", "Actual Phishing"],
        text=[[f"{value:,}" for value in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=20, color="#ffffff"),
        colorscale=[[0, "#131c2e"], [1, "#1f6fb2"]],
        showscale=False,
        xgap=3,
        ygap=3,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def roc_figure(results):
    fig = go.Figure()
    for name, res in results["models"].items():
        fig.add_scatter(
            x=res["roc_curve"]["fpr"],
            y=res["roc_curve"]["tpr"],
            mode="lines",
            name=f"{name} (AUC {res['roc_auc']:.4f})",
        )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                    line=dict(dash="dash", color="gray"))
    fig.update_layout(
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def importance_figure(importance, top_n=15):
    items = list(importance["values"].items())[:top_n][::-1]
    fig = go.Figure(go.Bar(
        x=[value for _, value in items],
        y=[name for name, _ in items],
        orientation="h",
        marker_color="#00d4ff",
    ))
    fig.update_layout(
        xaxis_title=importance["method"],
        height=30 * len(items) + 120,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def show_performance():
    render_header(
        "Model Performance",
        "Results on the held-out test set: URLs the models never saw during training.",
    )

    bundle, error = get_bundle()
    if error:
        st.error(error)
        return
    info = bundle["info"]

    results = load_eval_results()
    if results is None:
        st.warning(
            "Evaluation results not found. Run  python -m src.evaluate_model  "
            "in the terminal, then refresh this page."
        )
        return
    if evaluation_mismatch(info, results):
        st.warning(
            "The saved evaluation is older than the current model. Run  "
            "python -m src.evaluate_model  and refresh to see up-to-date numbers."
        )

    names = list(results["models"].keys())
    chosen = st.selectbox(
        "Model to inspect (confusion matrix and feature importance)",
        names,
        index=names.index(results["best_model"]),
    )
    res = results["models"][chosen]

    tab_compare, tab_matrix, tab_roc, tab_importance = st.tabs(
        ["Model comparison", "Confusion matrix", "ROC curves", "Feature importance"]
    )

    with tab_compare:
        table = metrics_table(results)
        st.dataframe(
            table.style.format({label: "{:.4f}" for label in DISPLAY.values()}),
            hide_index=True,
        )
        st.plotly_chart(comparison_figure(table))
        st.caption(
            "The chart's vertical axis is zoomed in to make small differences visible, "
            "so differences look bigger than they are. Read the exact numbers in the table. "
            f"The final model ({results['best_model']}) was chosen using cross-validation "
            "on the training data, not by looking at these test results."
        )
        cv = pd.DataFrame(info["cv_results"]).T.rename(columns=DISPLAY)
        with st.expander("Cross-validation results (used to select the final model)"):
            st.dataframe(cv.round(4))

    with tab_matrix:
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(confusion_figure(res))
        with right:
            st.metric("Precision", f"{res['precision']:.2%}")
            st.metric("Recall", f"{res['recall']:.2%}")
            st.metric("F1-score", f"{res['f1']:.2%}")
        st.write(
            f"**{res['fn']:,}** phishing URLs were missed (predicted legitimate) and "
            f"**{res['fp']:,}** legitimate URLs were flagged by mistake."
        )
        st.caption(
            "Precision: of the URLs flagged as phishing, the share that really were. "
            "Recall: of the real phishing URLs, the share that were caught."
        )

    with tab_roc:
        st.plotly_chart(roc_figure(results))
        st.caption(
            "The closer a curve is to the top-left corner, the better the model separates "
            "phishing from legitimate URLs. ROC-AUC summarizes this in one number (1.0 is perfect)."
        )

    with tab_importance:
        importance = results["feature_importance"][chosen]
        st.plotly_chart(importance_figure(importance))
        st.caption(
            f"Method: {importance['method']}. This shows which features the model relies on "
            "most overall. It does not prove what makes a real website malicious."
        )

    with st.expander("Why accuracy alone can be misleading"):
        st.write(
            f"In the test set, {results['phishing_share_test']:.1%} of URLs are phishing. "
            "A lazy model that always answers 'Legitimate' would score "
            f"{results['baseline_accuracy_always_majority_class']:.1%} accuracy while "
            "catching no phishing at all. That is why precision, recall, F1-score and "
            "ROC-AUC are shown too."
        )

    with st.expander("Limitations of these results"):
        st.write(
            "- The scores come from one public dataset, so real-world performance may be lower.\n"
            "- URLs in this dataset have habits (for example how HTTPS and URL paths appear) "
            "that the model can pick up on.\n"
            "- Only the URL text is analysed, not the page content or the domain's age.\n"
            "- No model can guarantee that a website is safe."
        )