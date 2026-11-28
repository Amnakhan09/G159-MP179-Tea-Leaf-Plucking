from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    with open(PROJECT_ROOT / "params.yaml") as file:
        config = yaml.safe_load(file)

    with open(PROJECT_ROOT / "reports/findings.yaml") as file:
        findings = yaml.safe_load(file)

    # Verify the registered champion
    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )

    client = MlflowClient()
    model_name = config["mlflow"]["registered_model_name"]

    champion = client.get_model_version_by_alias(
        model_name, "champion"
    )

    expected_version = str(findings["tracking"]["champion_version"])

    if str(champion.version) != expected_version:
        raise RuntimeError("MLflow champion and findings.yaml disagree")

    # Load the selected champion pipeline
    model_path = PROJECT_ROOT / config["paths"]["model"]
    model = joblib.load(model_path)

    # Load cleaned holdout features
    holdout_path = (
        PROJECT_ROOT / config["paths"]["processed_holdout"]
    )
    holdout = pd.read_csv(holdout_path)

    features = (
        config["data"]["numeric_features"]
        + config["data"]["categorical_features"]
    )

    id_column = config["data"]["id_column"]

    required_columns = [id_column] + features
    missing = set(required_columns) - set(holdout.columns)

    if missing:
        raise ValueError(f"Missing holdout columns: {missing}")

    if len(holdout) != 250:
        raise ValueError(f"Expected 250 records, found {len(holdout)}")

    # Generate predictions
    predictions = model.predict(holdout[features])

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain invalid values")

    # Create the required submission CSV
    result = pd.DataFrame({
        "record_id": holdout[id_column],
        "prediction": predictions
    })

    if list(result.columns) != ["record_id", "prediction"]:
        raise ValueError("Incorrect prediction CSV columns")

    output_path = PROJECT_ROOT / config["paths"]["predictions"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)

    print("Champion version:", champion.version)
    print("Prediction rows:", len(result))
    print("Output columns:", list(result.columns))
    print("Saved to:", output_path)
    print("\nFirst five predictions:")
    print(result.head().to_string(index=False))
    print("\nHOLDOUT PREDICTIONS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
