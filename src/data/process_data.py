from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/ai4i2020.csv")
PROCESSED_PATH = Path("data/processed/ai4i2020_model.csv")

# Load the validated raw dataset
df = pd.read_csv(RAW_PATH)

# These failure-mode columns would leak information about the target
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

df = df.drop(columns=LEAKAGE_COLUMNS)

# Convert machine Type into numeric indicator columns
df = pd.get_dummies(
    df,
    columns=["Type"],
    prefix="Type",
    dtype=int
)

# Save the processed/model-ready dataset
df.to_csv(PROCESSED_PATH, index=False)

print("\nPROCESSING COMPLETE")
print("-------------------")
print(f"Saved processed dataset to: {PROCESSED_PATH}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nProcessed columns:")
print(df.columns.tolist())