"""
Add ML model prediction endpoints to the API
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional
import pandas as pd
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Load models
MODELS = {}

def load_ml_models():
    """Load trained ML models"""
    model_dir = Path("models")
    
    if (model_dir / "diabetes_model.pkl").exists():
        MODELS['diabetes'] = joblib.load(model_dir / "diabetes_model.pkl")
        logger.info("Loaded diabetes model")
    
    if (model_dir / "cvd_model.pkl").exists():
        MODELS['cvd'] = joblib.load(model_dir / "cvd_model.pkl")
        logger.info("Loaded CVD model")
    
    if (model_dir / "hypertension_model.pkl").exists():
        MODELS['hypertension'] = joblib.load(model_dir / "hypertension_model.pkl")
        logger.info("Loaded hypertension model")
    
    # Try loading full-scale models if available
    full_scale_dir = model_dir / "full_scale"
    if full_scale_dir.exists():
        if (full_scale_dir / "diabetes_model_5M.pkl").exists():
            MODELS['diabetes_5M'] = joblib.load(full_scale_dir / "diabetes_model_5M.pkl")
            logger.info("Loaded full-scale diabetes model (5M)")
        
        if (full_scale_dir / "cvd_model_5M.pkl").exists():
            MODELS['cvd_5M'] = joblib.load(full_scale_dir / "cvd_model_5M.pkl")
            logger.info("Loaded full-scale CVD model (5M)")
        
        if (full_scale_dir / "hypertension_model_5M.pkl").exists():
            MODELS['hypertension_5M'] = joblib.load(full_scale_dir / "hypertension_model_5M.pkl")
            logger.info("Loaded full-scale hypertension model (5M)")


class MLPredictionRequest(BaseModel):
    patient_id: str
    age: int
    gender: str
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    glucose_mmol_l: float
    hba1c_percent: float
    total_cholesterol_mmol_l: float
    ldl_cholesterol_mmol_l: float
    hdl_cholesterol_mmol_l: float
    exercise_hours_per_week: float
    sleep_hours_per_night: float
    alcohol_units_per_week: float
    diet_quality_score: int
    stress_level: int
    smoking_status: str


def prepare_features(patient: MLPredictionRequest) -> pd.DataFrame:
    """Prepare features for ML prediction"""
    features = {
        'age': patient.age,
        'bmi': patient.bmi,
        'systolic_bp': patient.systolic_bp,
        'diastolic_bp': patient.diastolic_bp,
        'heart_rate': patient.heart_rate,
        'glucose_mmol_l': patient.glucose_mmol_l,
        'hba1c_percent': patient.hba1c_percent,
        'total_cholesterol_mmol_l': patient.total_cholesterol_mmol_l,
        'ldl_cholesterol_mmol_l': patient.ldl_cholesterol_mmol_l,
        'hdl_cholesterol_mmol_l': patient.hdl_cholesterol_mmol_l,
        'exercise_hours_per_week': patient.exercise_hours_per_week,
        'sleep_hours_per_night': patient.sleep_hours_per_night,
        'alcohol_units_per_week': patient.alcohol_units_per_week,
        'diet_quality_score': patient.diet_quality_score,
        'stress_level': patient.stress_level,
        'gender_male': 1 if patient.gender == 'male' else 0,
        'smoking_current': 1 if patient.smoking_status == 'current' else 0,
        'smoking_former': 1 if patient.smoking_status == 'former' else 0
    }
    
    return pd.DataFrame([features])


@router.post("/ml_predict_diabetes")
def ml_predict_diabetes(patient: MLPredictionRequest):
    """Predict diabetes using trained ML model"""
    if 'diabetes' not in MODELS and 'diabetes_5M' not in MODELS:
        return {"error": "Diabetes model not loaded. Train model first."}
    
    # Use full-scale model if available, otherwise use standard model
    model = MODELS.get('diabetes_5M', MODELS.get('diabetes'))
    
    X = prepare_features(patient)
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    return {
        "patient_id": patient.patient_id,
        "disease": "diabetes",
        "prediction": "positive" if prediction == 1 else "negative",
        "probability": float(probability),
        "risk_level": "high" if probability > 0.7 else "moderate" if probability > 0.4 else "low",
        "model_type": "full_scale_5M" if 'diabetes_5M' in MODELS else "standard_50K"
    }


@router.post("/ml_predict_cvd")
def ml_predict_cvd(patient: MLPredictionRequest):
    """Predict cardiovascular disease using trained ML model"""
    if 'cvd' not in MODELS and 'cvd_5M' not in MODELS:
        return {"error": "CVD model not loaded. Train model first."}
    
    model = MODELS.get('cvd_5M', MODELS.get('cvd'))
    
    X = prepare_features(patient)
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    return {
        "patient_id": patient.patient_id,
        "disease": "cardiovascular_disease",
        "prediction": "positive" if prediction == 1 else "negative",
        "probability": float(probability),
        "risk_level": "high" if probability > 0.7 else "moderate" if probability > 0.4 else "low",
        "model_type": "full_scale_5M" if 'cvd_5M' in MODELS else "standard_50K"
    }


@router.post("/ml_predict_hypertension")
def ml_predict_hypertension(patient: MLPredictionRequest):
    """Predict hypertension using trained ML model"""
    if 'hypertension' not in MODELS and 'hypertension_5M' not in MODELS:
        return {"error": "Hypertension model not loaded. Train model first."}
    
    model = MODELS.get('hypertension_5M', MODELS.get('hypertension'))
    
    X = prepare_features(patient)
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    return {
        "patient_id": patient.patient_id,
        "disease": "hypertension",
        "prediction": "positive" if prediction == 1 else "negative",
        "probability": float(probability),
        "risk_level": "high" if probability > 0.7 else "moderate" if probability > 0.4 else "low",
        "model_type": "full_scale_5M" if 'hypertension_5M' in MODELS else "standard_50K"
    }


@router.post("/ml_predict_all")
def ml_predict_all(patient: MLPredictionRequest):
    """Predict all diseases using trained ML models"""
    results = {}
    
    if 'diabetes' in MODELS or 'diabetes_5M' in MODELS:
        results['diabetes'] = ml_predict_diabetes(patient)
    
    if 'cvd' in MODELS or 'cvd_5M' in MODELS:
        results['cardiovascular_disease'] = ml_predict_cvd(patient)
    
    if 'hypertension' in MODELS or 'hypertension_5M' in MODELS:
        results['hypertension'] = ml_predict_hypertension(patient)
    
    return {
        "patient_id": patient.patient_id,
        "predictions": results
    }


# Load models when module is imported
load_ml_models()
