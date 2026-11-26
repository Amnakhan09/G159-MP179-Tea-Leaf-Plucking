from fastapi import FastAPI
from pydantic import BaseModel, Field

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
    return {"status": "ok"}
