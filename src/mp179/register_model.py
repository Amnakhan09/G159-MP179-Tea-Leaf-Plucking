
from pathlib import Path
import hashlib
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml", "r") as file:
    config = yaml.safe_load(file)

with open(PROJECT_ROOT / "reports" / "findings.yaml", "r") as file:
    findings = yaml.safe_load(file)


# ==========================================
# 2. CONFIGURE MLFLOW
# ==========================================

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

client = MlflowClient()

experiment = client.get_experiment_by_name(
    config["mlflow"]["experiment_name"]
)

if experiment is None:
    raise RuntimeError("MLflow experiment not found!")

print("MLflow connected successfully!")


# ==========================================
# 3. VERIFY BEST V1 RUN
# ==========================================

best_run_id = findings["tracking"]["best_run_id"]

if not best_run_id:
    raise ValueError("Best run ID missing from findings.yaml")

best_run = client.get_run(best_run_id)

if best_run.info.status != "FINISHED":
    raise RuntimeError("Selected MLflow run is not finished!")

if best_run.data.params.get("model_name") != "linear_regression":
    raise RuntimeError("The selected run is not Linear Regression!")

# Verify original training dataset hash
train_path = PROJECT_ROOT / config["paths"]["raw_train"]

with open(train_path, "rb") as file:
    expected_hash = hashlib.sha256(file.read()).hexdigest()

actual_hash = best_run.data.params.get("data_sha256")

if actual_hash != expected_hash:
    raise RuntimeError("Training dataset SHA-256 mismatch!")

if "mae_validation" not in best_run.data.metrics:
    raise RuntimeError("Validation MAE is missing!")

print("\nBEST V1 RUN VERIFIED")
print("Run ID:", best_run_id)
print("MAE:", best_run.data.metrics["mae_validation"])
print("Dataset SHA-256: Correct")


# ==========================================
# 4. LOAD THE EXISTING MODEL ARTIFACT
# ==========================================

model_artifact = config["mlflow"]["model_artifact_path"]

model_uri = f"runs:/{best_run_id}/{model_artifact}"

champion_pipeline = mlflow.sklearn.load_model(model_uri)

print("\nExisting MLflow model loaded successfully!")


# ==========================================
# 5. REGISTER MODEL VERSION 1
# ==========================================

model_name = config["mlflow"]["registered_model_name"]

versions = client.search_model_versions(
    f"name='{model_name}'"
)

# Reuse version 1 if this exact run was already registered
matching_versions = [
    version for version in versions
    if version.run_id == best_run_id
]

if matching_versions:

    registered_version = matching_versions[0].version

elif versions:

    raise RuntimeError(
        "Another model version already exists. "
        "Inspect the registry before creating a new version."
    )

else:

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    registered_version = registered.version

if str(registered_version) != "1":
    raise RuntimeError(
        f"Expected initial version 1, received {registered_version}"
    )

print("\nMODEL REGISTERED")
print("Model name:", model_name)
print("Version:", registered_version)


# ==========================================
# 6. ASSIGN CHAMPION ALIAS
# ==========================================

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=str(registered_version)
)

champion = client.get_model_version_by_alias(
    name=model_name,
    alias="champion"
)

print("\nCHAMPION ALIAS ASSIGNED")
print("Champion version:", champion.version)


# ==========================================
# 7. LOAD THE CHAMPION MODEL
# ==========================================

champion_uri = f"models:/{model_name}@champion"

final_pipeline = mlflow.sklearn.load_model(
    champion_uri
)

print("\nChampion model loaded successfully!")


# ==========================================
# 8. EXPORT COMPLETE PIPELINE
# ==========================================

model_path = PROJECT_ROOT / config["paths"]["model"]

model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(final_pipeline, model_path)

print("\nMODEL EXPORTED")
print("Location:", model_path)
print("Size:", model_path.stat().st_size, "bytes")


# ==========================================
# 9. TEST EXPORTED MODEL
# ==========================================

holdout_path = PROJECT_ROOT / config["paths"]["raw_holdout"]

holdout = pd.read_csv(holdout_path)

if len(holdout) != 250:
    raise ValueError(
        f"Expected 250 holdout records, found {len(holdout)}"
    )

feature_columns = (
    config["data"]["numeric_features"]
    + config["data"]["categorical_features"]
)

sample = holdout[feature_columns].head(5)

reloaded_pipeline = joblib.load(model_path)

predictions = reloaded_pipeline.predict(sample)

if not np.isfinite(predictions).all():
    raise ValueError("Model generated invalid predictions!")

print("\nEXPORT VERIFICATION SUCCESSFUL")
print("Test predictions:", predictions)
print("Tested records:", len(predictions))

print("\nTASK 6 V1 REGISTRATION COMPLETED!")
