from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    with open(PROJECT_ROOT / "params.yaml") as file:
        config = yaml.safe_load(file)

    train = pd.read_csv(
        PROJECT_ROOT / "data/processed/train_clean.csv"
    )

    later = pd.read_csv(
        PROJECT_ROOT / "data/processed/later_clean.csv"
    )

    features = (
        config["data"]["numeric_features"]
        + config["data"]["categorical_features"]
    )

    target = config["data"]["target"]

    # Sort later data chronologically
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date")

    # Split later data into 70% and 30%
    split = int(len(later) * 0.7)

    later_train = later.iloc[:split]
    later_test = later.iloc[split:]

    combined_train = pd.concat(
        [train, later_train],
        ignore_index=True
    )

    # Load existing V1 model
    model_path = PROJECT_ROOT / "models/model.joblib"
    v1_model = joblib.load(model_path)

    # Create and train V2
    v2_model = clone(v1_model)

    v2_model.fit(
        combined_train[features],
        combined_train[target]
    )

    # Evaluate both models on the same test data
    v1_predictions = v1_model.predict(
        later_test[features]
    )

    v2_predictions = v2_model.predict(
        later_test[features]
    )

    v1_mae = mean_absolute_error(
        later_test[target], v1_predictions
    )

    v2_mae = mean_absolute_error(
        later_test[target], v2_predictions
    )

    print("\nRETRAINING RESULTS")
    print("V1 MAE:", round(v1_mae, 4))
    print("V2 MAE:", round(v2_mae, 4))

    if v1_mae > 0:
        improvement = (
            (v1_mae - v2_mae) / v1_mae
        ) * 100
    else:
        improvement = 0.0

    print("Improvement:", round(improvement, 2), "%")

    joblib.dump(
        v2_model,
        PROJECT_ROOT / "models/model_v2.joblib"
    )

    if improvement >= 12:
        print("Decision: Promote V2")
    else:
        print("Decision: Retain V1")


if __name__ == "__main__":
    main()
