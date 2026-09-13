from pathlib import Path
import sys
import pandas as pd


DATA_PATH = Path("data/raw/ai4i2020.csv")

EXPECTED_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

EXPECTED_MACHINE_TYPES = {"L", "M", "H"}

# Initial validation ranges based on profiling the AI4I dataset
NUMERIC_RANGES = {
    "Air temperature": (295.3, 304.5),
    "Process temperature": (305.7, 313.8),
    "Rotational speed": (1168, 2886),
    "Torque": (3.8, 76.6),
    "Tool wear": (0, 253),
}


errors = []

# Check that the file exists
if not DATA_PATH.exists():
    print(f"ERROR: Could not find {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

# Check schema
if list(df.columns) != EXPECTED_COLUMNS:
    errors.append("Dataset columns do not match the expected schema.")

# Check missing values
missing_values = df.isna().sum().sum()

if missing_values > 0:
    errors.append(f"Dataset contains {missing_values} missing values.")

# Check duplicate rows
duplicate_rows = df.duplicated().sum()

if duplicate_rows > 0:
    errors.append(f"Dataset contains {duplicate_rows} duplicate rows.")

# Check machine categories
invalid_types = set(df["Type"].unique()) - EXPECTED_MACHINE_TYPES

if invalid_types:
    errors.append(
        f"Unexpected machine types found: {invalid_types}"
    )

# Check target values
invalid_targets = set(df["Machine failure"].unique()) - {0, 1}

if invalid_targets:
    errors.append(
        f"Unexpected Machine failure values: {invalid_targets}"
    )

# Check numerical ranges
for column, (minimum, maximum) in NUMERIC_RANGES.items():

    outside_range = ~df[column].between(minimum, maximum)

    if outside_range.any():
        count = outside_range.sum()

        errors.append(
            f"{column}: {count} values outside expected "
            f"range [{minimum}, {maximum}]"
        )

# Report result
print("\nDATA VALIDATION REPORT")
print("----------------------")

if errors:

    print("VALIDATION FAILED\n")

    for error in errors:
        print(f"- {error}")

    sys.exit(1)

else:

    print("VALIDATION PASSED")
    print(f"Rows checked: {len(df)}")
    print(f"Columns checked: {len(df.columns)}")
    print(f"Missing values: {missing_values}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(
        f"Failure rate: "
        f"{df['Machine failure'].mean() * 100:.2f}%"
    )