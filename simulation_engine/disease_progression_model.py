import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy.stats import weibull_min

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseProgressionModel:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
    def simulate_diabetes_progression(self, patient: pd.Series, years: int = 10) -> pd.DataFrame:
        trajectory = []
        
        current_hba1c = patient.get('hba1c_percent', 5.5)
        current_glucose = patient.get('glucose_mmol_l', 5.5)
        current_bmi = patient.get('bmi', 25)
        
        has_diabetes = patient.get('diabetes', False)
        
        for year in range(years + 1):
            if not has_diabetes:
                progression_rate = 0.05 + (current_bmi - 25) * 0.01
                current_hba1c += self.rng.normal(progression_rate, 0.1)
                current_glucose += self.rng.normal(progression_rate * 0.5, 0.2)
                
                if current_hba1c >= 6.5:
                    has_diabetes = True
            else:
                current_hba1c += self.rng.normal(0.15, 0.2)
                current_glucose += self.rng.normal(0.3, 0.5)
            
            current_bmi += self.rng.normal(0.3, 0.5)
            
            trajectory.append({
                'year': year,
                'age': patient['age'] + year,
                'hba1c_percent': np.clip(current_hba1c, 4.0, 14.0),
                'glucose_mmol_l': np.clip(current_glucose, 3.5, 20.0),
                'bmi': np.clip(current_bmi, 18, 50),
                'has_diabetes': has_diabetes,
                'diabetes_complications': has_diabetes and year > 5 and self.rng.random() < 0.3
            })
        
        return pd.DataFrame(trajectory)
    
    def simulate_cvd_progression(self, patient: pd.Series, years: int = 10) -> pd.DataFrame:
        trajectory = []
        
        current_bp_sys = patient.get('systolic_bp', 120)
        current_bp_dia = patient.get('diastolic_bp', 80)
        current_chol = patient.get('total_cholesterol_mmol_l', 5.0)
        
        has_cvd = patient.get('heart_disease', False)
        
        for year in range(years + 1):
            age_factor = (patient['age'] + year - 40) / 50
            
            current_bp_sys += self.rng.normal(1.0 + age_factor, 3)
            current_bp_dia += self.rng.normal(0.5 + age_factor * 0.5, 2)
            current_chol += self.rng.normal(0.1, 0.2)
            
            if not has_cvd:
                cvd_risk = 0.02 + age_factor * 0.05
                if current_bp_sys > 140:
                    cvd_risk += 0.05
                if current_chol > 6.0:
                    cvd_risk += 0.03
                
                if self.rng.random() < cvd_risk:
                    has_cvd = True
            
            trajectory.append({
                'year': year,
                'age': patient['age'] + year,
                'systolic_bp': np.clip(current_bp_sys, 90, 220),
                'diastolic_bp': np.clip(current_bp_dia, 60, 140),
                'total_cholesterol': np.clip(current_chol, 3.0, 12.0),
                'has_cvd': has_cvd,
                'major_cardiac_event': has_cvd and self.rng.random() < 0.05
            })
        
        return pd.DataFrame(trajectory)
    
    def simulate_cancer_progression(self, patient: pd.Series, years: int = 10) -> pd.DataFrame:
        trajectory = []
        
        has_cancer = patient.get('cancer', False)
        cancer_stage = 0 if not has_cancer else 1
        
        base_risk = 0.01
        if patient.get('smoking_status') == 'current':
            base_risk *= 5
        if patient.get('age', 40) > 60:
            base_risk *= 2
        
        for year in range(years + 1):
            if not has_cancer:
                if self.rng.random() < base_risk:
                    has_cancer = True
                    cancer_stage = 1
            else:
                if cancer_stage < 4 and self.rng.random() < 0.2:
                    cancer_stage += 1
            
            trajectory.append({
                'year': year,
                'age': patient['age'] + year,
                'has_cancer': has_cancer,
                'cancer_stage': cancer_stage,
                'in_remission': has_cancer and self.rng.random() < 0.3
            })
        
        return pd.DataFrame(trajectory)
    
    def simulate_multi_disease_progression(self, patient: pd.Series, 
                                          years: int = 10) -> Dict[str, pd.DataFrame]:
        diabetes_traj = self.simulate_diabetes_progression(patient, years)
        cvd_traj = self.simulate_cvd_progression(patient, years)
        cancer_traj = self.simulate_cancer_progression(patient, years)
        
        combined = diabetes_traj.merge(cvd_traj, on=['year', 'age'], suffixes=('_diabetes', '_cvd'))
        combined = combined.merge(cancer_traj, on=['year', 'age'])
        
        return {
            'diabetes': diabetes_traj,
            'cardiovascular': cvd_traj,
            'cancer': cancer_traj,
            'combined': combined
        }


if __name__ == "__main__":
    test_patient = pd.Series({
        'patient_id': 'P00000001',
        'age': 50,
        'bmi': 32,
        'hba1c_percent': 6.0,
        'glucose_mmol_l': 6.5,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'total_cholesterol_mmol_l': 6.2,
        'smoking_status': 'current',
        'diabetes': False,
        'heart_disease': False,
        'cancer': False
    })
    
    model = DiseaseProgressionModel()
    trajectories = model.simulate_multi_disease_progression(test_patient, years=10)
    
    logger.info("\nDiabetes Progression:")
    logger.info(trajectories['diabetes'][['year', 'age', 'hba1c_percent', 'has_diabetes']])
    
    logger.info("\nCardiovascular Progression:")
    logger.info(trajectories['cardiovascular'][['year', 'age', 'systolic_bp', 'has_cvd']])
    
    logger.info("\nCombined Multi-Disease Trajectory:")
    logger.info(trajectories['combined'].head(10))
