from pathlib import Path
import yaml
import pandas as pd
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

import mlflow
import mlflow.sklearn

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Find params.yaml
params_path = PROJECT_ROOT / "params.yaml"

# Load configuration
with open(params_path, "r") as file:
    config = yaml.safe_load(file)

# Get training dataset path from configuration
train_path = PROJECT_ROOT / config["paths"]["processed_train"]

# Load cleaned training data
df = pd.read_csv(train_path)

# Display information
print("Dataset shape:", df.shape)
print("Target column:", config["data"]["target"])
print("Training data loaded successfully!")


# 1. Get feature and target column names
numeric_features = config["data"]["numeric_features"]
categorical_features = config["data"]["categorical_features"]
target_column = config["data"]["target"]

# 2. Combine all input features
feature_columns = numeric_features + categorical_features

# 3. Separate X and y
X = df[feature_columns]
y = df[target_column]

print("\nFEATURES (X):")
print(X.head())

print("\nTARGET (y):")
print(y.head())

# 4. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=config["validation"]["test_size"],
    random_state=config["validation"]["random_state"],
    shuffle=config["validation"]["shuffle"]
)

# 5. Display split information
print("\nDATA SPLIT:")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Number of features:", X_train.shape[1])

print("\nFeature and target separation successful!")
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=config["validation"]["test_size"],
    random_state=config["validation"]["random_state"],
    shuffle=config["validation"]["shuffle"]
)

print("\nDATA SPLIT:")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Number of features:", X_train.shape[1])

print("\nFeature and target separation successful!")

# ==========================================
# PREPROCESSING PIPELINE
# ==========================================

# 1. Numerical preprocessing
numeric_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy=config["preprocessing"]["numeric_imputation"]
            )
        )
    ]
)


# 2. Categorical preprocessing
categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy=config["preprocessing"]["categorical_imputation"]
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown=config["preprocessing"]["handle_unknown_categories"]
            )
        )
    ]
)


# 3. Combine both preprocessing branches
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocessor, numeric_features),
        ("cat", categorical_preprocessor, categorical_features)
    ]
)


print("\nPREPROCESSING PIPELINE CREATED!")

print("Numerical preprocessing: Median imputation")
print("Categorical preprocessing: Most frequent imputation")
print("Categorical encoding: One-Hot Encoding")
print("Total input features:", len(feature_columns))


# ==========================================
# EXPERIMENT 1: LINEAR REGRESSION
# ==========================================

# 1. Create the complete ML pipeline
linear_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", LinearRegression())
    ]
)

# 2. Train the model
print("\nTraining Linear Regression...")

linear_pipeline.fit(X_train, y_train)

# 3. Predict on validation data
y_pred = linear_pipeline.predict(X_val)

# 4. Calculate MAE
linear_mae = mean_absolute_error(y_val, y_pred)

# 5. Display results
print("\nLINEAR REGRESSION RESULTS")
print("-------------------------")
print("Validation MAE:", round(linear_mae, 4), "kg")
print("Model training completed!")


# ==========================================
# EXPERIMENT 2: RANDOM FOREST - 100 TREES
# ==========================================

# Get parameters from params.yaml
rf100_params = config["models"]["random_forest_100"]["params"]

# Create pipeline
rf100_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(**rf100_params))
    ]
)

# Train model
print("\nTraining Random Forest (100 trees)...")

rf100_pipeline.fit(X_train, y_train)

# Predict validation data
rf100_pred = rf100_pipeline.predict(X_val)

# Calculate MAE
rf100_mae = mean_absolute_error(y_val, rf100_pred)

# Display results
print("\nRANDOM FOREST (100 TREES) RESULTS")
print("---------------------------------")
print("Validation MAE:", round(rf100_mae, 4), "kg")
print("Model training completed!")

# ==========================================
# EXPERIMENT 3: RANDOM FOREST - 200 TREES
# ==========================================

# Get parameters from params.yaml
rf200_params = config["models"]["random_forest_200"]["params"]

# Create pipeline
rf200_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(**rf200_params))
    ]
)

# Train model
print("\nTraining Random Forest (200 trees)...")

rf200_pipeline.fit(X_train, y_train)

# Predict validation data
rf200_pred = rf200_pipeline.predict(X_val)

# Calculate MAE
rf200_mae = mean_absolute_error(y_val, rf200_pred)

# Display results
print("\nRANDOM FOREST (200 TREES) RESULTS")
print("---------------------------------")
print("Validation MAE:", round(rf200_mae, 4), "kg")
print("Model training completed!")


# ==========================================
# TASK 5: MLFLOW EXPERIMENT TRACKING
# ==========================================

# 1. Configure MLflow
mlflow.set_tracking_uri(
    config["mlflow"]["tracking_uri"]
)

mlflow.set_experiment(
    config["mlflow"]["experiment_name"]
)

print("\nMLflow tracking initialized!")


# 2. Calculate original train.csv SHA-256
raw_train_path = PROJECT_ROOT / config["paths"]["raw_train"]

with open(raw_train_path, "rb") as file:
    data_sha256 = hashlib.sha256(file.read()).hexdigest()

print("Original dataset SHA-256:", data_sha256)


# 3. Collect all three experiments
experiments = [
    {
        "name": "linear_regression",
        "estimator": "LinearRegression",
        "pipeline": linear_pipeline,
        "mae": linear_mae,
        "params": config["models"]["linear_regression"]["params"]
    },
    {
        "name": "random_forest_100",
        "estimator": "RandomForestRegressor",
        "pipeline": rf100_pipeline,
        "mae": rf100_mae,
        "params": config["models"]["random_forest_100"]["params"]
    },
    {
        "name": "random_forest_200",
        "estimator": "RandomForestRegressor",
        "pipeline": rf200_pipeline,
        "mae": rf200_mae,
        "params": config["models"]["random_forest_200"]["params"]
    }
]


# 4. Track each experiment
run_records = []

for experiment in experiments:

    with mlflow.start_run(
        run_name=experiment["name"]
    ) as run:

        # Log model information
        mlflow.log_param(
            "model_name",
            experiment["name"]
        )

        mlflow.log_param(
            "estimator",
            experiment["estimator"]
        )

        # REQUIRED: Original dataset hash
        mlflow.log_param(
            "data_sha256",
            data_sha256
        )

        # Log validation configuration
        mlflow.log_param(
            "validation_test_size",
            config["validation"]["test_size"]
        )

        mlflow.log_param(
            "validation_random_state",
            config["validation"]["random_state"]
        )

        # Log model hyperparameters
        for key, value in experiment["params"].items():

            mlflow.log_param(
                f"model_{key}",
                str(value) if value is None else value
            )

        # Log MAE
        mlflow.log_metric(
            "mae_validation",
            float(experiment["mae"])
        )

        # Log the fitted preprocessing + model pipeline
        mlflow.sklearn.log_model(
            sk_model=experiment["pipeline"],
            name="model",
            serialization_format="cloudpickle"
        )

        # Save experiment information
        run_records.append({
            "name": experiment["name"],
            "run_id": run.info.run_id,
            "mae": float(experiment["mae"])
        })

        print("\nExperiment logged:", experiment["name"])
        print("Run ID:", run.info.run_id)
        print("MAE:", round(experiment["mae"], 4))


# 5. Identify best experiment
best_run = min(
    run_records,
    key=lambda item: item["mae"]
)

print("\n" + "=" * 50)
print("BEST V1 MODEL")
print("=" * 50)

print("Model:", best_run["name"])
print("MAE:", round(best_run["mae"], 4))
print("Best Run ID:", best_run["run_id"])

print("\nAll three experiments successfully logged in MLflow!")
