
from pathlib import Path
import hashlib
import yaml
import mlflow
from mlflow.tracking import MlflowClient

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load configuration
with open(PROJECT_ROOT / "params.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configure MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

client = MlflowClient()

# Find experiment
experiment = client.get_experiment_by_name(
    config["mlflow"]["experiment_name"]
)

if experiment is None:
    raise RuntimeError("MLflow experiment not found!")

# Calculate original dataset hash
train_path = PROJECT_ROOT / config["paths"]["raw_train"]

with open(train_path, "rb") as file:
    expected_hash = hashlib.sha256(file.read()).hexdigest()

# Retrieve all runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id]
)

print("EXPERIMENT:", experiment.name)
print("NUMBER OF RUNS:", len(runs))

for run in runs:

    print("\n-------------------------------")
    print("Run ID:", run.info.run_id)
    print("Model:", run.data.params.get("model_name"))
    print("MAE:", run.data.metrics.get("mae_validation"))

    # Verify original dataset hash
    actual_hash = run.data.params.get("data_sha256")

    print("Dataset hash correct:", actual_hash == expected_hash)
    print("MAE recorded:", "mae_validation" in run.data.metrics)

print("\nOriginal dataset SHA-256:")
print(expected_hash)
