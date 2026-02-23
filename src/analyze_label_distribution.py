import pandas as pd

# Load PROMISE
promise_df = pd.read_csv("data/cleaned_requirements.csv")
promise_dist = promise_df["label"].value_counts(normalize=True)

print("PROMISE Label Distribution:")
print(promise_dist)
print("\n")

# Load PURE
pure_df = pd.read_csv(
    r"C:\Users\samri\Downloads\puredataset\Pure_Annotate_Dataset.csv",
    encoding="latin1"
)

pure_df["label"] = pure_df["NFR_boolean"].apply(
    lambda x: "non-functional" if x == 1 else "functional"
)

pure_dist = pure_df["label"].value_counts(normalize=True)

print("PURE Label Distribution:")
print(pure_dist)