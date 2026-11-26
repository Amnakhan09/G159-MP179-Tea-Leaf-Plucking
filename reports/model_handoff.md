# MP179 - Model Handoff Documentation

**Owner:** Amna Khan
**Group ID:** G159
**Problem ID:** MP179

## 1. Initial Model Details

- Model: Linear Regression
- Model version: V1
- MLflow registered name: MP179TeaProductivity
- Champion alias: champion
- Validation MAE: 1.6433 kg
- Model location: models/model.joblib
- Prediction target: kg_per_worker

## 2. Required Input Features

The model accepts exactly eight input features.

| Feature | Type |
|---|---|
| worker_exp_yr | Numerical |
| rain_mm_week | Numerical |
| slope_deg | Numerical |
| temp_c | Numerical |
| bush_age_yr | Numerical |
| hours_worked | Numerical |
| flush | Categorical |
| section | Categorical |

## 3. Preprocessing Included in Model

The saved Scikit-learn pipeline contains:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- One-Hot Encoding.
- Fitted Linear Regression model.

The API should not repeat these fitted transformations.

## 4. Data Cleaning Requirements

The training dataset required these corrections:

- Lower Garden rainfall values multiplied by 10.
- Slope sentinel value 9999 replaced with missing values.
- Identified stuck bush-age sensor readings replaced with missing values.
- Flush spelling variations standardized.

Important: These domain-specific corrections are performed by the data cleaning script. They are not all embedded in model.joblib.

The API must receive rainfall in the correct unit (mm), canonical flush categories (first, second, monsoon), and cleaned feature values. It must not multiply already-correct Lower Garden rainfall values again.

## 5. Excluded Columns

The following columns must not be passed as model features:

- record_id
- date
- kg_per_worker

The target is kg_per_worker.

## 6. API Integration

Load the exported model:

```python
import joblib

model = joblib.load("models/model.joblib")
import pandas as pd

input_df = pd.DataFrame([{
    "worker_exp_yr": 5,
    "rain_mm_week": 120,
    "slope_deg": 20,
    "temp_c": 25,
    "bush_age_yr": 15,
    "hours_worked": 8,
    "flush": "first",
    "section": "Lower Garden"
}])

prediction = model.predict(input_df)
print(prediction[0])