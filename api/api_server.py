from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import MultiAgentSystem
from agents.cardiology_agent import CardiologyAgent
from agents.metabolic_agent import MetabolicAgent
from agents.lifestyle_agent import LifestyleAgent
from prediction_engine.risk_predictor import RiskPredictor
from simulation_engine.intervention_simulator import InterventionSimulator

# Import ML prediction endpoints
try:
    from api.ml_prediction_endpoint import router as ml_router
    ml_endpoints_available = True
except Exception as e:
    ml_endpoints_available = False
    print(f"ML endpoints not available: {e}")

app = FastAPI(title="Health Digital Twin API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

multi_agent_system = MultiAgentSystem()
multi_agent_system.register_agent(CardiologyAgent())
multi_agent_system.register_agent(MetabolicAgent())
multi_agent_system.register_agent(LifestyleAgent())

risk_predictor = RiskPredictor()
intervention_simulator = InterventionSimulator()

# Include ML prediction endpoints if available
if ml_endpoints_available:
    app.include_router(ml_router, prefix="/ml", tags=["Machine Learning Predictions"])


class PatientData(BaseModel):
    patient_id: str
    age: int
    gender: str
    bmi: float
    systolic_bp: Optional[float] = 120
    diastolic_bp: Optional[float] = 80
    heart_rate: Optional[float] = 70
    total_cholesterol_mmol_l: Optional[float] = 5.0
    hdl_cholesterol_mmol_l: Optional[float] = 1.4
    ldl_cholesterol_mmol_l: Optional[float] = 3.0
    glucose_mmol_l: Optional[float] = 5.5
    hba1c_percent: Optional[float] = 5.5
    smoking_status: Optional[str] = "never"
    exercise_hours_per_week: Optional[float] = 3.0
    diet_quality_score: Optional[int] = 5
    alcohol_units_per_week: Optional[float] = 5.0
    stress_level: Optional[int] = 5
    sleep_hours_per_night: Optional[float] = 7.0
    diabetes: Optional[bool] = False
    hypertension: Optional[bool] = False
    heart_disease: Optional[bool] = False


class InterventionRequest(BaseModel):
    patient_data: PatientData
    interventions: List[str]


@app.get("/")
def root():
    return {
        "message": "Health Digital Twin Prediction Platform API",
        "version": "1.0.0",
        "endpoints": [
            "/evaluate",
            "/predict_risk",
            "/simulate_intervention",
            "/rank_interventions"
        ]
    }


@app.post("/evaluate")
def evaluate_patient(patient: PatientData):
    patient_series = pd.Series(patient.dict())
    
    evaluation = multi_agent_system.evaluate_patient(patient_series)
    
    return {
        "patient_id": patient.patient_id,
        "evaluation": evaluation
    }


@app.post("/predict_risk")
def predict_risk(patient: PatientData, time_horizon_years: int = 10):
    patient_series = pd.Series(patient.dict())
    
    predictions = risk_predictor.predict_all_risks(patient_series, time_horizon_years)
    
    return predictions


@app.post("/simulate_intervention")
def simulate_intervention(request: InterventionRequest):
    patient_series = pd.Series(request.patient_data.dict())
    
    results = intervention_simulator.simulate_multiple_interventions(
        patient_series,
        request.interventions
    )
    
    return {
        "patient_id": request.patient_data.patient_id,
        "interventions": results
    }


@app.post("/rank_interventions")
def rank_interventions(patient: PatientData):
    patient_series = pd.Series(patient.dict())
    
    ranked = intervention_simulator.rank_interventions(patient_series)
    
    return {
        "patient_id": patient.patient_id,
        "ranked_interventions": ranked
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
