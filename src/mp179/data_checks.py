
from pathlib import Path
import pandas as pd

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

files = [
    "train.csv",
    "later.csv",
    "holdout_features.csv"
]

for filename in files:

    print("\n" + "=" * 60)
    print(f"ANALYSING: {filename}")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_DIR / filename)

    # 1. Dataset dimensions
    print("\n1. Dataset shape:")
    print(df.shape)

    # 2. Missing values
    print("\n2. Missing values:")
    print(df.isnull().sum())

    # 3. Duplicate records
    print("\n3. Duplicate rows:")
    print(df.duplicated().sum())

    # 4. Column data types
    print("\n4. Data types:")
    print(df.dtypes)

    # 5. Numerical summary
    print("\n5. Numerical statistics:")
    print(df.describe().round(2))

    # 6. Categorical values
    for column in ["flush", "section"]:

        if column in df.columns:
            print(f"\n7. Unique values in {column}:")
            print(df[column].value_counts(dropna=False))

    print("\nAnalysis completed for", filename)
