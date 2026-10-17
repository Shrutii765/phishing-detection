"""
preprocessing.py
----------------
Loads the PhiUSIIL dataset, cleans it, converts labels to OUR format,
extracts features, and splits into training and testing sets.

Our label format (used everywhere in the project):
    0 = Legitimate
    1 = Phishing
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_extraction import extract_features_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"

URL_COLUMN = "URL"
LABEL_COLUMN = "label"

# In THIS dataset: 1 = legitimate, 0 = phishing (we checked the sample URLs).
DATASET_LEGIT_VALUE = 1
DATASET_PHISH_VALUE = 0

LABEL_NAMES = {0: "Legitimate", 1: "Phishing"}

TEST_SIZE = 0.2
RANDOM_STATE = 42  # fixed number so the split is the same every time


def load_raw_data(path=DATA_PATH):
    """Read only the URL and label columns from the CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place PhiUSIIL_Phishing_URL_Dataset.csv inside the 'data' folder."
        )
    try:
        return pd.read_csv(path, usecols=[URL_COLUMN, LABEL_COLUMN])
    except ValueError as exc:
        raise ValueError(
            f"The CSV must contain the columns '{URL_COLUMN}' and '{LABEL_COLUMN}'."
        ) from exc


def clean_data(df):
    """
    Remove bad rows and convert labels. Returns (clean_dataframe, report_dict).
    Output columns: url, label   (label: 0 = Legitimate, 1 = Phishing)
    """
    report = {"rows_loaded": len(df)}
    df = df.rename(columns={URL_COLUMN: "url", LABEL_COLUMN: "label_raw"})

    # 1. Missing values
    df = df.dropna(subset=["url", "label_raw"]).copy()
    report["missing_removed"] = report["rows_loaded"] - len(df)

    # 2. Tidy the URL text and remove empty strings
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]

    # 3. Keep only valid label values (0 or 1)
    before = len(df)
    df["label_raw"] = pd.to_numeric(df["label_raw"], errors="coerce")
    df = df[df["label_raw"].isin([DATASET_LEGIT_VALUE, DATASET_PHISH_VALUE])]
    report["invalid_labels_removed"] = before - len(df)

    # 4. Same URL with conflicting labels: we cannot trust it, so drop all copies
    labels_per_url = df.groupby("url")["label_raw"].transform("nunique")
    report["conflicting_label_rows_removed"] = int((labels_per_url > 1).sum())
    df = df[labels_per_url == 1]

    # 5. Remove exact duplicate URLs (prevents the same URL in train AND test)
    before = len(df)
    df = df.drop_duplicates(subset="url", keep="first")
    report["duplicates_removed"] = before - len(df)

    # 6. Convert to OUR label format: phishing = 1, legitimate = 0
    df["label"] = (df["label_raw"] == DATASET_PHISH_VALUE).astype(int)
    df = df[["url", "label"]].reset_index(drop=True)

    report["rows_after_cleaning"] = len(df)
    return df, report


def build_feature_table(df):
    """
    Extract features for every URL using the SAME function the app uses.
    URLs that cannot be parsed are removed.
    Returns (X, y, urls, number_of_invalid_urls_removed).
    """
    features = extract_features_batch(df["url"])
    valid = features.notna().all(axis=1)
    n_invalid = int((~valid).sum())

    X = features[valid].reset_index(drop=True).astype(float)
    y = df.loc[valid, "label"].reset_index(drop=True)
    urls = df.loc[valid, "url"].reset_index(drop=True)
    return X, y, urls, n_invalid


def load_and_split(path=DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Full pipeline: load -> clean -> features -> train/test split.
    The split happens BEFORE any model or scaler is fitted, so the test set
    stays completely unseen (no data leakage). 'stratify' keeps the same
    phishing/legitimate ratio in both sets.
    """
    raw = load_raw_data(path)
    clean, report = clean_data(raw)
    X, y, urls, n_invalid = build_feature_table(clean)
    report["unparsable_urls_removed"] = n_invalid
    report["rows_used"] = len(X)

    X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
        X, y, urls, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "urls_train": urls_train,
        "urls_test": urls_test,
        "report": report,
    }


def print_summary(data):
    """Print a readable summary of what preprocessing did."""
    print("\n=== CLEANING REPORT ===")
    for key, value in data["report"].items():
        print(f"  {key:34} {value:,}")

    y_all = pd.concat([data["y_train"], data["y_test"]])
    print("\n=== CLASS DISTRIBUTION (after cleaning) ===")
    for label, count in y_all.value_counts().sort_index().items():
        print(f"  {LABEL_NAMES[label]:12} {count:>8,}  ({count / len(y_all):.1%})")

    print("\n=== TRAIN / TEST SPLIT ===")
    print(f"  Training rows: {len(data['X_train']):,}")
    print(f"  Testing rows : {len(data['X_test']):,}")
    print(f"  Phishing share in train: {data['y_train'].mean():.1%}")
    print(f"  Phishing share in test : {data['y_test'].mean():.1%}")

    overlap = set(data["urls_train"]) & set(data["urls_test"])
    print(f"\n  URLs found in BOTH train and test (must be 0): {len(overlap)}")

    print("\n=== FIRST 3 FEATURE ROWS (training set) ===")
    print(data["X_train"].head(3).T.to_string())


if __name__ == "__main__":
    # Run with:  python -m src.preprocessing
    print("Loading and preprocessing... this may take up to a minute.")
    print_summary(load_and_split())