from ucimlrepo import fetch_ucirepo
from pathlib import Path
import pandas as pd

# Download the AI4I 2020 Predictive Maintenance dataset
ai4i = fetch_ucirepo(id=601)

# Combine features and targets into one table
df = pd.concat([ai4i.data.features, ai4i.data.targets], axis=1)

# Create the raw-data path
raw_path = Path("data/raw/ai4i2020.csv")

# Save an unchanged local copy
df.to_csv(raw_path, index=False)

print(f"Saved raw dataset to: {raw_path}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nColumns:")
print(df.columns.tolist())