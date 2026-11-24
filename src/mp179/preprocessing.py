
from pathlib import Path
import pandas as pd


# 1. Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# 2. Standardize flush categories

def clean_flush(series):

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\.$", "", regex=True)
    )

    aliases = {
        "fir": "first",
        "sec": "second",
        "mon": "monsoon"
    }

    return cleaned.replace(aliases)


# 3. Common cleaning for all datasets

def clean_common(df):

    df = df.copy()

    # Convert date into proper datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Standardize category spellings
    df["flush"] = clean_flush(df["flush"])

    return df


# 4. Special cleaning for training data

def clean_training_data(df):

    df = clean_common(df)

    # FIX 1: Rainfall unit mix-up
    lower_garden = df["section"] == "Lower Garden"

    df.loc[lower_garden, "rain_mm_week"] = (
        df.loc[lower_garden, "rain_mm_week"] * 10
    )

    print("Rainfall rows corrected:", lower_garden.sum())


    # FIX 2: Sentinel value in slope
    invalid_slope = df["slope_deg"] == 9999

    df.loc[invalid_slope, "slope_deg"] = float("nan")

    print("Invalid slope values replaced:", invalid_slope.sum())


    # FIX 3: Stuck sensor
    df_dates = df["date"]

    stuck_sensor = (
        df_dates.between("2026-05-03", "2026-05-19")
        & (df["bush_age_yr"] == 70)
    )

    df.loc[stuck_sensor, "bush_age_yr"] = float("nan")

    print("Stuck sensor readings replaced:", stuck_sensor.sum())


    return df


# 5. Main execution

def main():

    # Read original training dataset
    train = pd.read_csv(RAW_DIR / "train.csv")

    # Read later dataset
    later = pd.read_csv(RAW_DIR / "later.csv")

    # Read holdout dataset
    holdout = pd.read_csv(RAW_DIR / "holdout_features.csv")


    # Clean training data
    clean_train = clean_training_data(train)

    # Standardize the other datasets
    clean_later = clean_common(later)
    clean_holdout = clean_common(holdout)


    # Save processed datasets
    clean_train.to_csv(
        PROCESSED_DIR / "train_clean.csv",
        index=False
    )

    clean_later.to_csv(
        PROCESSED_DIR / "later_clean.csv",
        index=False
    )

    clean_holdout.to_csv(
        PROCESSED_DIR / "holdout_features_clean.csv",
        index=False
    )


    print("\nAll datasets processed successfully!")

    print("\nProcessed dataset shapes:")
    print("Train:", clean_train.shape)
    print("Later:", clean_later.shape)
    print("Holdout:", clean_holdout.shape)

    print("\nRemaining missing values in training:")
    print(clean_train.isnull().sum())


if __name__ == "__main__":
    main()
