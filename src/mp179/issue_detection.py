
from pathlib import Path
import pandas as pd

# Find project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Load original datasets
train = pd.read_csv(DATA_DIR / "train.csv")
later = pd.read_csv(DATA_DIR / "later.csv")
holdout = pd.read_csv(DATA_DIR / "holdout_features.csv")

print("=" * 60)
print("MP179 - SEVEN DATA QUALITY CHECKS")
print("=" * 60)


# 1. LEAKING COLUMN

# The target is expected to be absent from holdout.
train_only = set(train.columns) - set(holdout.columns)
extra_columns = train_only - {"kg_per_worker"}

print("\n1. LEAKING COLUMN")
print("Columns missing from holdout:", sorted(train_only))
print("Extra suspicious columns:", sorted(extra_columns))


# 2. UNIT MIX-UP

print("\n2. UNIT MIX-UP")

rain_means = train.groupby("section")["rain_mm_week"].mean()

print("Average rainfall by section:")
print(rain_means.round(2))

lowest_section = rain_means.idxmin()
reference_mean = rain_means.drop(lowest_section).mean()

factor = rain_means[lowest_section] / reference_mean

print("Suspicious section:", lowest_section)
print("Wrong-to-correct factor:", round(factor, 1))


# 3. SENTINEL VALUE

print("\n3. SENTINEL VALUE")

invalid_slopes = train.loc[
    train["slope_deg"] > 90, "slope_deg"
]

print("Invalid slope values:")
print(invalid_slopes.value_counts())


# 4. DUPLICATE ROWS

print("\n4. DUPLICATE ROWS")

duplicates = train.duplicated().sum()

print("Number of duplicate rows:", duplicates)


# 5. STUCK SENSOR

print("\n5. STUCK SENSOR")

daily = (
    train.assign(date=pd.to_datetime(train["date"]))
    .groupby("date")["bush_age_yr"]
    .agg(
        first="first",
        unique="nunique",
        rows="size"
    )
    .sort_index()
)

best_run = []
current_run = []

for date, row in daily.iterrows():

    constant_day = (
        row["unique"] == 1 and row["rows"] >= 2
    )

    if constant_day:

        value = row["first"]

        consecutive = (
            current_run
            and value == current_run[-1][1]
            and date - current_run[-1][0]
            == pd.Timedelta(days=1)
        )

        if consecutive:
            current_run.append((date, value))
        else:
            current_run = [(date, value)]

        if len(current_run) > len(best_run):
            best_run = current_run.copy()

    else:
        current_run = []

if best_run:
    print("Column: bush_age_yr")
    print("Repeated value:", best_run[0][1])
    print("Start date:", best_run[0][0].date())
    print("End date:", best_run[-1][0].date())
    print("Consecutive days:", len(best_run))


# 6. TARGET TYPOS

print("\n6. TARGET TYPOS")

median_target = train["kg_per_worker"].median()

cutoff = median_target * 10

typo_count = (
    train["kg_per_worker"] > cutoff
).sum()

print("Target median:", median_target)
print("Target maximum:", train["kg_per_worker"].max())
print("Possible 10x target errors:", typo_count)


# 7. SPELLING VARIANTS

print("\n7. SPELLING VARIANTS")

raw_flush = train["flush"].astype(str)

clean_flush = (
    raw_flush
    .str.strip()
    .str.lower()
    .str.replace(r"\.$", "", regex=True)
)

# Convert known abbreviations
aliases = {
    "fir": "first",
    "sec": "second",
    "mon": "monsoon"
}

clean_flush = clean_flush.replace(aliases)

print("Original spellings:", raw_flush.nunique())
print("True categories:", clean_flush.nunique())

print("\nStandardized categories:")
print(clean_flush.value_counts())

print("\nAll seven checks completed!")
