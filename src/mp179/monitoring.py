from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_psi(train, later, bins=10):
    train = train.dropna().astype(float)
    later = later.dropna().astype(float)

    if train.empty or later.empty:
        raise ValueError("Empty data for PSI calculation")

    edges = np.unique(
        np.quantile(train, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        edges = np.array([-np.inf, train.iloc[0], np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    expected = np.histogram(train, bins=edges)[0] / len(train)
    actual = np.histogram(later, bins=edges)[0] / len(later)

    expected = np.maximum(expected, 0.000001)
    actual = np.maximum(actual, 0.000001)

    return float(
        np.sum((actual - expected) * np.log(actual / expected))
    )


def main():
    with open(PROJECT_ROOT / "params.yaml") as file:
        config = yaml.safe_load(file)

    train = pd.read_csv(
        PROJECT_ROOT / "data/processed/train_clean.csv"
    )

    later = pd.read_csv(
        PROJECT_ROOT / "data/processed/later_clean.csv"
    )

    model = joblib.load(
        PROJECT_ROOT / "models/model.joblib"
    )

    numeric_features = config["data"]["numeric_features"]
    categorical_features = config["data"]["categorical_features"]

    features = numeric_features + categorical_features
    target = config["data"]["target"]

    psi_results = {}

    print("\nPSI DRIFT RESULTS")

    for column in numeric_features:
        psi = calculate_psi(train[column], later[column])
        psi_results[column] = psi
        print(f"{column}: {psi:.4f}")

    most_drifted = max(psi_results, key=psi_results.get)

    print("\nMost drifted feature:", most_drifted)

    later["prediction"] = model.predict(later[features])

    print("\nSECTION-WISE MAE")

    group_mae = {}

    for section, group in later.groupby("section"):

        section_mae = mean_absolute_error(
            group[target],
            group["prediction"]
        )

        group_mae[section] = float(section_mae)

        print(f"{section}: {section_mae:.4f} kg")

    # Identify the section with the highest MAE
    worst_served = max(group_mae, key=group_mae.get)

    print("\nWorst-served section:", worst_served)




if __name__ == "__main__":
    main()
