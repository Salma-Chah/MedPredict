import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="API for predicting 30-day hospital readmission risk",
    version="1.0.0",
)

MODEL_PATH = os.path.join("models", "best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
FEATURES_PATH = os.path.join("models", "feature_names.pkl")

model = None
scaler = None
feature_names = None


def load_artifacts():
    global model, scaler, feature_names
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "rb") as f:
            feature_names = pickle.load(f)


@app.on_event("startup")
def startup_event():
    load_artifacts()


class PatientData(BaseModel):
    age: Optional[str] = "[70-80)"
    time_in_hospital: Optional[int] = 3
    num_lab_procedures: Optional[int] = 40
    num_procedures: Optional[int] = 1
    num_medications: Optional[int] = 15
    number_outpatient: Optional[int] = 0
    number_emergency: Optional[int] = 0
    number_inpatient: Optional[int] = 0
    number_diagnoses: Optional[int] = 9
    gender: Optional[str] = "Female"
    race: Optional[str] = "Caucasian"
    admission_type_id: Optional[int] = 1
    discharge_disposition_id: Optional[int] = 1
    admission_source_id: Optional[int] = 7
    insulin: Optional[str] = "No"
    diabetesMed: Optional[str] = "Yes"


class PredictionResponse(BaseModel):
    readmitted_probability: float
    readmitted_prediction: int
    risk_level: str


@app.get("/")
def root():
    return {"message": "Hospital Readmission Prediction API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    data = patient.dict()
    df = pd.DataFrame([data])

    # Encode categorical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Align with training features
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    X_scaled = scaler.transform(df)
    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(prob >= 0.5)

    if prob < 0.3:
        risk = "Low"
    elif prob < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        readmitted_probability=round(float(prob), 4),
        readmitted_prediction=pred,
        risk_level=risk,
    )


@app.get("/model/info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": type(model).__name__,
        "features": feature_names if feature_names else [],
        "n_features": len(feature_names) if feature_names else 0,
    }
