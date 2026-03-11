import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
        if model_path:
            self.load_models(model_path)
        else:
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        logger.info("Initializing default risk prediction models...")
        
        self.models['cvd'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.models['diabetes'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.models['cancer'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.scalers['cvd'] = StandardScaler()
        self.scalers['diabetes'] = StandardScaler()
        self.scalers['cancer'] = StandardScaler()
        
        self.feature_names['cvd'] = [
            'age', 'gender_male', 'bmi', 'systolic_bp', 'diastolic_bp',
            'total_cholesterol_mmol_l', 'hdl_cholesterol_mmol_l', 'ldl_cholesterol_mmol_l',
            'smoking_current', 'smoking_former', 'diabetes', 'exercise_hours_per_week'
        ]
        
        self.feature_names['diabetes'] = [
            'age', 'bmi', 'glucose_mmol_l', 'hba1c_percent',
            'systolic_bp', 'exercise_hours_per_week', 'diet_quality_score',
            'smoking_current', 'hypertension'
        ]
        
        self.feature_names['cancer'] = [
            'age', 'gender_male', 'bmi', 'smoking_current', 'smoking_former',
            'alcohol_units_per_week', 'exercise_hours_per_week', 'diet_quality_score'
        ]
    
    def prepare_features(self, patient: pd.Series, disease: str) -> np.ndarray:
        features = []
        
        for feature_name in self.feature_names[disease]:
            if feature_name == 'gender_male':
                features.append(1 if patient.get('gender') == 'male' else 0)
            elif feature_name == 'smoking_current':
                features.append(1 if patient.get('smoking_status') == 'current' else 0)
            elif feature_name == 'smoking_former':
                features.append(1 if patient.get('smoking_status') == 'former' else 0)
            else:
                default_values = {
                    'age': 40, 'bmi': 25, 'systolic_bp': 120, 'diastolic_bp': 80,
                    'total_cholesterol_mmol_l': 5.0, 'hdl_cholesterol_mmol_l': 1.4,
                    'ldl_cholesterol_mmol_l': 3.0, 'glucose_mmol_l': 5.5,
                    'hba1c_percent': 5.5, 'exercise_hours_per_week': 3,
                    'diet_quality_score': 5, 'alcohol_units_per_week': 5
                }
                features.append(patient.get(feature_name, default_values.get(feature_name, 0)))
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, patient: pd.Series, disease: str, 
                    time_horizon_years: int = 10) -> Dict:
        if disease not in self.models:
            logger.error(f"No model available for disease: {disease}")
            return {'error': f'Unknown disease: {disease}'}
        
        features = self.prepare_features(patient, disease)
        
        risk_score = self._calculate_rule_based_risk(patient, disease)
        
        return {
            'disease': disease,
            'time_horizon_years': time_horizon_years,
            'risk_score': risk_score,
            'risk_percentage': risk_score * 100,
            'risk_level': self._get_risk_level(risk_score),
            'confidence': 0.75
        }
    
    def _calculate_rule_based_risk(self, patient: pd.Series, disease: str) -> float:
        if disease == 'cvd':
            return self._cvd_risk(patient)
        elif disease == 'diabetes':
            return self._diabetes_risk(patient)
        elif disease == 'cancer':
            return self._cancer_risk(patient)
        else:
            return 0.0
    
    def _cvd_risk(self, patient: pd.Series) -> float:
        risk = 0.05
        risk += (patient.get('age', 40) - 40) * 0.012
        if patient.get('gender') == 'male':
            risk += 0.08
        if patient.get('systolic_bp', 120) > 140:
            risk += 0.15
        if patient.get('total_cholesterol_mmol_l', 5.0) > 6.0:
            risk += 0.10
        if patient.get('smoking_status') == 'current':
            risk += 0.20
        if patient.get('diabetes', False):
            risk += 0.15
        return min(risk, 0.95)
    
    def _diabetes_risk(self, patient: pd.Series) -> float:
        risk = 0.05
        risk += (patient.get('age', 40) - 40) * 0.01
        if patient.get('bmi', 25) > 30:
            risk += (patient.get('bmi', 25) - 30) * 0.05
        if patient.get('hba1c_percent', 5.0) > 5.7:
            risk += 0.25
        if patient.get('hypertension', False):
            risk += 0.12
        return min(risk, 0.95)
    
    def _cancer_risk(self, patient: pd.Series) -> float:
        risk = 0.02
        risk += (patient.get('age', 40) - 40) * 0.008
        if patient.get('smoking_status') == 'current':
            risk += 0.25
        if patient.get('alcohol_units_per_week', 0) > 14:
            risk += 0.08
        return min(risk, 0.80)
    
    def _get_risk_level(self, risk_score: float) -> str:
        if risk_score < 0.1:
            return "very_low"
        elif risk_score < 0.2:
            return "low"
        elif risk_score < 0.4:
            return "moderate"
        elif risk_score < 0.6:
            return "high"
        else:
            return "very_high"
    
    def predict_all_risks(self, patient: pd.Series, 
                         time_horizon_years: int = 10) -> Dict:
        diseases = ['cvd', 'diabetes', 'cancer']
        
        results = {}
        for disease in diseases:
            results[disease] = self.predict_risk(patient, disease, time_horizon_years)
        
        overall_risk = np.mean([results[d]['risk_score'] for d in diseases])
        
        return {
            'patient_id': patient.get('patient_id', 'Unknown'),
            'time_horizon_years': time_horizon_years,
            'individual_risks': results,
            'overall_risk_score': overall_risk,
            'overall_risk_level': self._get_risk_level(overall_risk)
        }
    
    def save_models(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for disease, model in self.models.items():
            joblib.dump(model, output_path / f'{disease}_model.pkl')
            joblib.dump(self.scalers[disease], output_path / f'{disease}_scaler.pkl')
        
        logger.info(f"Models saved to {output_path}")
    
    def load_models(self, model_dir: str):
        model_path = Path(model_dir)
        
        for disease in ['cvd', 'diabetes', 'cancer']:
            model_file = model_path / f'{disease}_model.pkl'
            scaler_file = model_path / f'{disease}_scaler.pkl'
            
            if model_file.exists():
                self.models[disease] = joblib.load(model_file)
                self.scalers[disease] = joblib.load(scaler_file)
                logger.info(f"Loaded {disease} model")


if __name__ == "__main__":
    test_patient = pd.Series({
        'patient_id': 'P00000001',
        'age': 55,
        'gender': 'male',
        'bmi': 32,
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'total_cholesterol_mmol_l': 6.5,
        'hdl_cholesterol_mmol_l': 0.9,
        'ldl_cholesterol_mmol_l': 4.3,
        'glucose_mmol_l': 6.8,
        'hba1c_percent': 6.2,
        'smoking_status': 'current',
        'diabetes': False,
        'hypertension': True,
        'exercise_hours_per_week': 1,
        'diet_quality_score': 4,
        'alcohol_units_per_week': 18
    })
    
    predictor = RiskPredictor()
    
    results = predictor.predict_all_risks(test_patient, time_horizon_years=10)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Risk Prediction Results for Patient {results['patient_id']}")
    logger.info(f"Time Horizon: {results['time_horizon_years']} years")
    logger.info(f"{'='*60}")
    
    for disease, risk_data in results['individual_risks'].items():
        logger.info(f"\n{disease.upper()}:")
        logger.info(f"  Risk Score: {risk_data['risk_percentage']:.1f}%")
        logger.info(f"  Risk Level: {risk_data['risk_level']}")
    
    logger.info(f"\nOVERALL RISK:")
    logger.info(f"  Score: {results['overall_risk_score']:.2%}")
    logger.info(f"  Level: {results['overall_risk_level']}")
