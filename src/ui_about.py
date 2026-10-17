"""
ui_about.py
-----------
About Project page: problem statement, objectives, technology stack,
methodology, limitations and future scope.
Dataset and model details are read from the saved model information.
"""

import streamlit as st

from src.ui_common import get_bundle, render_header


def show_about():
    render_header(
        "About Project",
        "Explainable Machine Learning for Phishing URL Detection",
    )

    bundle, _ = get_bundle()
    info = bundle["info"] if bundle else None

    tab_problem, tab_stack, tab_method, tab_future = st.tabs(
        ["Problem and objectives", "Technology stack", "Methodology", "Limitations and future scope"]
    )

    with tab_problem:
        st.subheader("Problem statement")
        st.write(
            "Phishing attacks trick people into opening fake websites that steal "
            "passwords, banking details and personal data. Attackers constantly create "
            "new fake links, and many of them look very similar to real ones, so people "
            "cannot always judge a link by eye. This project builds a quick, explainable "
            "first-check tool: a user enters a URL, and a machine learning model "
            "estimates whether it is likely legitimate or potentially phishing, and "
            "shows which features influenced that estimate."
        )
        st.subheader("Objectives")
        st.markdown(
            "- Build a complete machine learning workflow: data cleaning, feature "
            "extraction, training, evaluation and prediction.\n"
            "- Extract meaningful features from the URL text without opening the URL.\n"
            "- Compare Logistic Regression, Decision Tree and Random Forest using "
            "cross-validation, and pick the final model from validation results.\n"
            "- Evaluate honestly with accuracy, precision, recall, F1-score, ROC-AUC "
            "and a confusion matrix on unseen test data.\n"
            "- Explain each prediction with SHAP so the user can see what influenced it.\n"
            "- Present everything in a clear, easy-to-use web dashboard."
        )

    with tab_stack:
        st.markdown(
            "| Purpose | Tool |\n"
            "|---|---|\n"
            "| Programming language | Python |\n"
            "| Data processing | Pandas, NumPy |\n"
            "| Machine learning | Scikit-learn |\n"
            "| Explainable AI | SHAP |\n"
            "| Model storage | Joblib |\n"
            "| Web application | Streamlit |\n"
            "| Visualization | Plotly, Matplotlib |\n"
        )
        st.caption("All tools are free and open source. No paid APIs or external services are used.")

    with tab_method:
        if info:
            dataset = info["dataset"]
            total_rows = info["train_rows"] + info["test_rows"]
            models = ", ".join(info["cv_results"].keys())
            st.markdown(
                f"1. **Dataset:** {dataset}, a public labeled dataset of legitimate and "
                "phishing URLs. Only the URL text and the label are used.\n"
                f"2. **Preprocessing:** missing values, invalid labels, conflicting labels and "
                f"duplicate URLs are removed. {total_rows:,} URLs are used. Labels are stored "
                "as 0 = Legitimate and 1 = Phishing.\n"
                f"3. **Train/test split:** {info['train_rows']:,} URLs for training and "
                f"{info['test_rows']:,} for testing, split before any model is fitted so the "
                "test data stays unseen (no data leakage).\n"
                f"4. **Feature extraction:** {len(info['feature_names'])} features are "
                "calculated from the URL text (length, dots, hyphens, digits, IP address in "
                "the hostname, HTTPS, URL shorteners, suspicious words and more). The same "
                "function is used for training and for live predictions.\n"
                f"5. **Model selection:** {models} were compared with "
                f"{info['cv_folds']}-fold cross-validation on the training data. The final "
                f"model is **{info['best_model']}**, chosen by {info['selection_metric']}.\n"
                "6. **Evaluation:** the final scores are measured once on the held-out test set.\n"
                "7. **Explainability:** SHAP shows how much each feature pushed a single "
                "prediction toward phishing or legitimate.\n"
                                "8. **Application:** the saved model is loaded once by the Streamlit app, "
                "which analyses URLs as text and never opens them."
            )
        else:
            st.info("Model details are unavailable. Train the model first with  python -m src.train_model.")

    with tab_future:
        st.subheader("Limitations")
        st.markdown(
            "- Only the URL text is analysed. Page content, domain age and blacklists are not used.\n"
            "- The model learned from one public dataset, so real-world accuracy can be lower "
            "than the test scores. While testing, we found and corrected one dataset habit "
            "(trailing slashes), and similar habits, such as how HTTPS appears, may remain.\n"
            "- A new or unfamiliar URL can be classified wrongly, in both directions.\n"
            "- The risk score is the model's output, not a guaranteed probability.\n"
            "- URL shorteners are detected, but the real destination is not checked.\n"
            "- SHAP explains what the model relied on. It does not prove the real-world "
            "cause of a site being malicious.\n"
            "- No model can guarantee that a website is safe."
        )
        st.subheader("Future scope")
        st.markdown(
            "- Train on more varied and recent datasets, and split the data by domain.\n"
            "- Tune model settings (for example with grid search) and calibrate probabilities.\n"
            "- Add optional checks such as domain age, only with the user's consent.\n"
            "- Analyse page content in a safe, sandboxed way.\n"
            "- Add scan history, a browser extension and scheduled model retraining."
        )