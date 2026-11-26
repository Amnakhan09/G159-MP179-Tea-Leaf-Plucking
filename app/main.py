import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


app = FastAPI(title="Tea Leaf Productivity API")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model.joblib"
)

# Configure logging
logger = logging.getLogger("mp179_api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))

logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


# Input validation
class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )

    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([data.model_dump()])

    prediction = float(model.predict(input_df)[0])

    logger.info(json.dumps({
        "event": "prediction",
        "prediction": prediction,
        "status": "success"
    }))

    return {"prediction": prediction}
