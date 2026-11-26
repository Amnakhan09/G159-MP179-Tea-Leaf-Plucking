from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from fastapi import HTTPException


app = FastAPI(title="Tea Leaf Productivity API")

class PredictionInput(BaseModel):
    worker_exp_yr: float
    rain_mm_week: float = Field(..., ge=0)
    slope_deg: float
    temp_c: float
    bush_age_yr: float
    hours_worked: float = Field(..., le=20)
    flush: str
    section: str

@app.get("/health")
def health():
    MODEL_PATH = Path("models/model.joblib")

@app.post("/predict")
def predict(data: PredictionInput):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )

    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]

    return {"prediction": float(prediction)}
    return {"status": "ok"}
