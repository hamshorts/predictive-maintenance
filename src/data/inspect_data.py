import pandas as pd

df = pd.read_csv("data/raw/ai4i2020.csv")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nMACHINE FAILURE COUNTS")
print(df["Machine failure"].value_counts())

print("\nMACHINE FAILURE PERCENTAGES")
print(df["Machine failure"].value_counts(normalize=True) * 100)

print("\nMACHINE TYPES")
print(df["Type"].value_counts())

print("\nNUMERICAL SUMMARY")
print(df[
    [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear"
    ]
].describe())