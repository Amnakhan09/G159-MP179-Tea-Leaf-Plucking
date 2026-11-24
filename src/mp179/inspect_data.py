
from pathlib import Path
import pandas as pd

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of our original datasets
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# The three datasets we need to inspect
files = [
    "train.csv",
    "later.csv",
    "holdout_features.csv"
]

# Read and inspect each dataset
for filename in files:

    file_path = DATA_DIR / filename

    print("\n" + "=" * 50)
    print("DATASET:", filename)
    print("=" * 50)

    df = pd.read_csv(file_path)

    # Number of rows and columns
    print("Shape:", df.shape)

    # Column names
    print("\nColumns:")
    print(df.columns.tolist())

    # First three rows
    print("\nFirst three records:")
    print(df.head(3).to_string(index=False))
