# PhishGuard: Explainable Machine Learning for Phishing URL Detection

A college mini-project. A Streamlit web app that analyses a URL **as text** and estimates whether it is likely legitimate or potentially phishing, then explains the main features behind each prediction using SHAP.

> The prediction is an estimate, not a guarantee of safety. No model can promise that a website is safe.

## Features

- URL feature extraction (23 features) using only the URL text
- Comparison of Logistic Regression, Decision Tree and Random Forest with 5-fold cross-validation
- Evaluation on a held-out test set: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- SHAP explanations for every prediction
- Dashboard pages: Home, URL Scanner, Model Performance, About Project

## Project structure

```
phishing-detection/
├── .streamlit/config.toml      theme and settings
├── data/                       dataset CSV goes here
├── models/                     saved model, metadata, evaluation results, charts
├── src/
│   ├── feature_extraction.py   URL text -> numeric features (shared by training and app)
│   ├── preprocessing.py        load, clean, label conversion, train/test split
│   ├── train_model.py          compare 3 models, select and save the best
│   ├── evaluate_model.py       test-set metrics, confusion matrices, charts
│   ├── explainability.py       SHAP explanations
│   ├── ui_*.py                 Streamlit pages
├── app.py                      app entry point and navigation
├── check_project.py            end-to-end project check
├── requirements.txt
└── README.md
```

## Setup (Windows, VS Code)

Developed with Python 3.14 on Windows.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Dataset

Download the **PhiUSIIL Phishing URL Dataset** from the UCI Machine Learning Repository (https://archive.ics.uci.edu, search for the dataset name) and place the CSV at:

```
data/PhiUSIIL_Phishing_URL_Dataset.csv
```

Only the `URL` and `label` columns are used. In this dataset `1` means legitimate and `0` means phishing; the project converts this to 0 = Legitimate, 1 = Phishing. Check the dataset page for the exact citation (authors, paper and DOI).

## Run the project

```powershell
python -m src.train_model        # compare models, save the best one
python -m src.evaluate_model     # test-set evaluation, charts, results file
python -m streamlit run app.py   # start the web app (http://localhost:8501)
python check_project.py          # optional: end-to-end check
```

The app loads the saved model from `models/`; it never retrains for a prediction.
Evaluation numbers are shown on the **Model Performance** page and come from `models/evaluation_results.json`.

## How it works

1. The URL is parsed with `urllib.parse` and regular expressions; trailing slashes are ignored.
2. 23 numeric features are calculated (length, dots, hyphens, digits, IP address host, HTTPS, shortener, suspicious words, ...).
3. The saved model returns a phishing risk score.
4. SHAP shows which features pushed the score toward phishing or legitimate.

## Security and privacy

- Submitted URLs are never opened, visited or executed.
- The app does not forward URLs to any other service, and it does not store them.
- Nothing is stored; the app has no history or logging of submitted URLs.

## Limitations

- Only URL text is analysed (no page content, domain age or blacklists).
- Trained on one public dataset, so real-world accuracy can be lower than the test scores.
- Dataset habits can leak into the model; one (trailing slashes) was found by manual testing and corrected, and others, such as how HTTPS appears, may remain.
- Unfamiliar URLs can be misclassified in both directions.
- SHAP explains what the model relied on, not the real-world cause.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` or `pip` not recognized | Reinstall Python and tick "Add python.exe to PATH" |
| Script activation blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: No module named 'src...'` | Run commands from the project folder, with `python -m src.<name>` |
| "Model files were not found" in the app | Run `python -m src.train_model` |
| "Evaluation is older than the model" | Run `python -m src.evaluate_model` |
| `No such option '-m'` | Type the run command once, on a clean line |
| Port 8501 in use | `python -m streamlit run app.py --server.port 8502` |