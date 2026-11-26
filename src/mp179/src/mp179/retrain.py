from pathlib import Path
import hashlib

import joblib
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():

    with open(PROJECT_ROOT / "params.yaml") as file:
        config = yaml.safe_load(file)

    train = pd.read_csv(
        PROJECT_ROOT / config["paths"]["processed_train"]
    )

    later = pd.read_csv(
        PROJECT_ROOT / config["paths"]["processed_later"]
    )

    features = (
        config["data"]["numeric_features"]
        + config["data"]["categorical_features"]
    )

    target = config["data"]["target"]

    # Configure MLflow
    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    client = MlflowClient()

    model_name = config["mlflow"]["registered_model_name"]

    # Preserve original V1 model
    model_path = PROJECT_ROOT / config["paths"]["model"]
    v1_path = model_path.parent / "model_v1.joblib"

    if not v1_path.exists():
        champion = client.get_model_version_by_alias(
            model_name, "champion"
        )

        if str(champion.version) != "1":
            raise RuntimeError("Original V1 model required")

        joblib.dump(joblib.load(model_path), v1_path)

    v1_model = joblib.load(v1_path)

    # Chronological split
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date")

    split = int(len(later) * 0.70)

    later_train = later.iloc[:split]
    later_test = later.iloc[split:]

    combined_train = pd.concat(
        [train, later_train],
        ignore_index=True
    )

    # Train V2
    v2_model = clone(v1_model)

    v2_model.fit(
        combined_train[features],
        combined_train[target]
    )

    # Compare models
    v1_mae = mean_absolute_error(
        later_test[target],
        v1_model.predict(later_test[features])
    )

    v2_mae = mean_absolute_error(
        later_test[target],
        v2_model.predict(later_test[features])
    )

    improvement = (
        (v1_mae - v2_mae) / v1_mae * 100
        if v1_mae > 0 else 0.0
    )

    # Original training data fingerprint
    raw_train = PROJECT_ROOT / config["paths"]["raw_train"]

    data_sha256 = hashlib.sha256(
        raw_train.read_bytes()
    ).hexdigest()

    # Track retraining experiment
    with mlflow.start_run(run_name="retraining_v2"):

        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_name", "linear_regression_v2")
        mlflow.log_param("later_training_fraction", 0.70)

        mlflow.log_metric("v1_mae", float(v1_mae))
        mlflow.log_metric("v2_mae", float(v2_mae))
        mlflow.log_metric(
            "improvement_percent", float(improvement)
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="model",
            serialization_format="cloudpickle"
        )

    joblib.dump(
        v2_model,
        model_path.parent / "model_v2.joblib"
    )

    print("\nRETRAINING RESULTS")
    print("V1 MAE:", round(v1_mae, 4))
    print("V2 MAE:", round(v2_mae, 4))
    print("Improvement:", round(improvement, 2), "%")
    print("V2 model URI:", model_info.model_uri)

   

    if improvement >= 12:
        print("Decision: Promote V2")
        selected_model = v2_model
        selected_version = "2"
    else:
        print("Decision: Retain V1")
        selected_model = v1_model
        selected_version = "1"

  

    print("Selected version:", selected_version)


if __name__ == "__main__":
    main()
