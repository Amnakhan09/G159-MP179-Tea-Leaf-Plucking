
from pathlib import Path
import yaml

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Find params.yaml
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

# Read YAML configuration
with open(PARAMS_FILE, "r") as file:
    config = yaml.safe_load(file)

# Display configuration
print("PROJECT:", config["project"]["name"])

print("\nTARGET:")
print(config["data"]["target"])

print("\nNUMERICAL FEATURES:")
print(config["data"]["numeric_features"])

print("\nCATEGORICAL FEATURES:")
print(config["data"]["categorical_features"])

print("\nMODEL CONFIGURATIONS:")
for model_name in config["models"]:
    print(model_name)

print("\nMLFLOW EXPERIMENT:")
print(config["mlflow"]["experiment_name"])

print("\nConfiguration loaded successfully!")
