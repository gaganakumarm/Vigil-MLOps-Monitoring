import os
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from db_models import PredictionLog, get_db  # Import models and DB utility

# --- FastAPI Initialization ---
app = FastAPI(
    title="MLOps Prediction API",
    description="Serves predictions from the deployed ML model and logs requests.",
    version="v1.0.0"
)

# --- Model Loading ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/model.joblib")  # Docker-safe path
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    else:
        print(f"Model file not found at {MODEL_PATH}. Using placeholder predictions.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Pydantic Schemas ---

class PredictionRequest(BaseModel):
    feature_1: float = Field(..., description="First continuous feature value.", example=0.5)
    feature_2: float = Field(..., description="Second continuous feature value.", example=12.3)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="The model's predicted class (e.g., 0 or 1).", example=1)
    model_version: str = Field(MODEL_VERSION, description="Version of the model used.")

# --- API Endpoints ---

@app.get("/status")
async def get_status():
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "service": "MLOps Prediction API",
        "model_loaded": model is not None,
        "version": MODEL_VERSION
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_and_log(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    # 1. Prepare input data
    data_df = pd.DataFrame([request.model_dump()])

    # 2. Generate prediction
    if model:
        try:
            prediction_result = int(model.predict(data_df)[0])
        except Exception as e:
            print(f"Error during model prediction: {e}")
            raise HTTPException(status_code=500, detail="Internal Model Prediction Error.")
    else:
        print("Using placeholder prediction (model not loaded).")
        prediction_result = 0

    # 3. Log prediction to DB
    try:
        new_log = PredictionLog(
            feature_1=request.feature_1,
            feature_2=request.feature_2,
            prediction=prediction_result,
            model_version=MODEL_VERSION
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)
        print(f"Prediction logged to DB: ID {new_log.id}, Result {prediction_result}")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Failed to log prediction to database: {e}")

    # 4. Return response
    return PredictionResponse(
        prediction=prediction_result,
        model_version=MODEL_VERSION
    )
