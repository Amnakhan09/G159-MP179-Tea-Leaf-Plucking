from pathlib import Path
import hashlib

import joblib
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import numpy as np

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

        # Fingerprint the cleaned later dataset
    later_path = PROJECT_ROOT / config["paths"]["processed_later"]

    later_data_sha256 = hashlib.sha256(
        later_path.read_bytes()
    ).hexdigest()

    # Create a signature to identify this retraining dataset
    signature_text = (
        data_sha256
        + later_data_sha256
        + ",".join(features)
        + target
    )

    retraining_signature = hashlib.sha256(
        signature_text.encode("utf-8")
    ).hexdigest()

    # Track retraining experiment
    with mlflow.start_run(run_name="retraining_v2"):

        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("later_data_sha256", later_data_sha256)

        mlflow.log_param("retraining_signature", retraining_signature)
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
        v2_run_id = mlflow.active_run().info.run_id


       # Register V2 in MLflow
    model_artifact = config["mlflow"]["model_artifact_path"]
    v2_model_uri = f"runs:/{v2_run_id}/{model_artifact}"

    existing_versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    matching_version = None

    for version in existing_versions:
        version_run = client.get_run(version.run_id)

        if (
            version_run.data.params.get("retraining_signature")
            == retraining_signature
        ):
            matching_version = version
            break

    if matching_version is not None:
        v2_version = str(matching_version.version)
        print("Existing V2 registration reused:", v2_version)

    else:
        version_two_exists = any(
            str(version.version) == "2"
            for version in existing_versions
        )

        if version_two_exists:
            raise RuntimeError(
                "Version 2 already exists with a different "
                "retraining signature. Inspect MLflow first."
            )

        registered = mlflow.register_model(
            model_uri=v2_model_uri,
            name=model_name
        )

        v2_version = str(registered.version)
        print("V2 registered:", v2_version)

    if v2_version != "2":
        raise RuntimeError(
            f"Expected model version 2, found {v2_version}"
        )

    # Apply the 12% promotion rule
    if improvement >= 12:
        decision = "promote"
        selected_version = "2"
    else:
        decision = "keep"
        selected_version = "1"

    print("\nPROMOTION DECISION")
    print("V1 MAE:", v1_mae)
    print("V2 MAE:", v2_mae)
    print("Improvement:", improvement, "%")
    print("Decision:", decision)

    # Update the champion alias
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=selected_version
    )

    champion = client.get_model_version_by_alias(
        model_name, "champion"
    )

    print("Champion version:", champion.version)

    # Export the selected champion pipeline
    champion_uri = f"models:/{model_name}@champion"

    final_model = mlflow.sklearn.load_model(champion_uri)

    joblib.dump(final_model, model_path)

    # Validate exported model against the 250 holdout records
    holdout_path = (
        PROJECT_ROOT / config["paths"]["processed_holdout"]
    )

    holdout = pd.read_csv(holdout_path)
    predictions = final_model.predict(holdout[features])

    if len(predictions) != 250:
        raise ValueError("Expected exactly 250 predictions")

    if not np.isfinite(predictions).all():
        raise ValueError("Invalid holdout predictions")

    print("\nCHAMPION EXPORT SUCCESSFUL")
    print("Model path:", model_path)
    print("Holdout records tested:", len(predictions))



if __name__ == "__main__":
    main()
