import pandas as pd

df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumns:", list(df.columns))

print("\nLabel counts:")
print(df["label"].value_counts(dropna=False))

print("\nMissing URLs:", df["URL"].isna().sum())
print("Duplicate URLs:", df["URL"].duplicated().sum())

for value in sorted(df["label"].dropna().unique()):
    print(f"\nSample URLs where label == {value}:")
    print(df.loc[df["label"] == value, "URL"].head(8).to_string(index=False))