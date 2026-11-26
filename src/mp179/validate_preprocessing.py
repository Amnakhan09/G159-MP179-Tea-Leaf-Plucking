
from pathlib import Path
import hashlib
import yaml
import pandas as pd
import numpy as np
import joblib

# Locate project
ROOT = Path(__file__).resolve().parents[2]

# Load configuration
with open(ROOT / "params.yaml", "r") as file:
    config = yaml.safe_load(file)

# Dataset paths
raw_dir = ROOT / "data" / "raw"
processed_dir = ROOT / "data" / "processed"

# Load datasets
raw_train = pd.read_csv(raw_dir / "train.csv")
raw_later = pd.read_csv(raw_dir / "later.csv")
raw_holdout = pd.read_csv(raw_dir / "holdout_features.csv")

train = pd.read_csv(processed_dir / "train_clean.csv")
later = pd.read_csv(processed_dir / "later_clean.csv")
holdout = pd.read_csv(processed_dir / "holdout_features_clean.csv")

print("STEP 1: DATASET SIZE VALIDATION")

assert len(raw_train) == len(train)
assert len(raw_later) == len(later)
assert len(raw_holdout) == len(holdout)

print("Train:", train.shape)
print("Later:", later.shape)
print("Holdout:", holdout.shape)
print("All row counts preserved!")


# Verify original training dataset hash
print("\nSTEP 2: RAW DATA INTEGRITY")

with open(raw_dir / "train.csv", "rb") as file:
    train_hash = hashlib.sha256(file.read()).hexdigest()

expected_hash = (
    "9530e110cd3893b3a5c9acc5040113b1ad1f6f155140dfc82344966c643b6c8c"
)

assert train_hash == expected_hash

print("Original train.csv unchanged!")
print("SHA-256:", train_hash)


# Verify cleaning corrections
print("\nSTEP 3: DATA QUALITY VALIDATION")

# Rainfall correction
lower_raw = raw_train.loc[
    raw_train["section"] == "Lower Garden",
    "rain_mm_week"
]

lower_clean = train.loc[
    train["section"] == "Lower Garden",
    "rain_mm_week"
]

assert np.isclose(
    lower_clean.median(),
    lower_raw.median() * 10
)

print("Rainfall correction verified!")

# Sentinel value
assert not (train["slope_deg"] == 9999).any()

print("Slope sentinel value removed!")

# Stuck sensor
raw_dates = pd.to_datetime(raw_train["date"])

stuck_mask = (
    raw_dates.between("2026-05-03", "2026-05-19")
    & (raw_train["bush_age_yr"] == 70)
)

assert train.loc[stuck_mask, "bush_age_yr"].isna().all()

print("Stuck sensor readings handled!")

# Categorical standardization
assert train["flush"].nunique() == 3

print("Flush categories standardized!")


# Verify target and feature columns
print("\nSTEP 4: MODEL INPUT VALIDATION")

features = (
    config["data"]["numeric_features"]
    + config["data"]["categorical_features"]
)

target = config["data"]["target"]

for dataset in [train, later, holdout]:
    missing = [c for c in features if c not in dataset.columns]
    assert not missing, f"Missing feature columns: {missing}"

assert target in train.columns
assert target in later.columns

print("Total input features:", len(features))
print("Features:", features)
print("Target:", target)


# Test exported champion model
print("\nSTEP 5: EXPORTED MODEL VALIDATION")

model_path = ROOT / config["paths"]["model"]

model = joblib.load(model_path)

predictions = model.predict(holdout[features])

assert len(predictions) == 250
assert np.isfinite(predictions).all()

print("Model loaded successfully!")
print("Predictions generated:", len(predictions))
print("All predictions are finite!")


print("\n" + "=" * 45)
print("ALL PREPROCESSING CHECKS PASSED!")
print("=" * 45)
