import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseProgressionGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
    def calculate_diabetes_risk(self, patient: pd.Series) -> float:
        risk = 0.05
        
        if patient['age'] > 45:
            risk += (patient['age'] - 45) * 0.01
        
        if patient['bmi'] > 25:
            risk += (patient['bmi'] - 25) * 0.03
        elif patient['bmi'] > 30:
            risk += (patient['bmi'] - 30) * 0.05
        
        if patient.get('hba1c_percent', 5.0) > 5.7:
            risk += (patient['hba1c_percent'] - 5.7) * 0.15
        
        if patient.get('glucose_mmol_l', 5.0) > 6.1:
            risk += (patient['glucose_mmol_l'] - 6.1) * 0.08
        
        if patient.get('exercise_hours_per_week', 5) < 2:
            risk += 0.15
        
        if patient.get('smoking_status') == 'current':
            risk += 0.10
        
        if patient.get('hypertension', False):
            risk += 0.12
        
        family_history_prob = 0.3
        if self.rng.random() < family_history_prob:
            risk += 0.20
        
        return min(risk, 0.95)
    
    def calculate_cardiovascular_risk(self, patient: pd.Series) -> float:
        risk = 0.03
        
        if patient['age'] > 40:
            risk += (patient['age'] - 40) * 0.012
        
        if patient['gender'] == 'male':
            risk += 0.08
        
        if patient.get('systolic_bp', 120) > 140:
            risk += (patient['systolic_bp'] - 140) * 0.01
        
        if patient.get('total_cholesterol_mmol_l', 5.0) > 6.0:
            risk += (patient['total_cholesterol_mmol_l'] - 6.0) * 0.08
        
        if patient.get('ldl_cholesterol_mmol_l', 3.0) > 4.0:
            risk += (patient['ldl_cholesterol_mmol_l'] - 4.0) * 0.10
        
        if patient.get('hdl_cholesterol_mmol_l', 1.4) < 1.0:
            risk += 0.12
        
        if patient.get('smoking_status') == 'current':
            risk += 0.20
        elif patient.get('smoking_status') == 'former':
            risk += 0.08
        
        if patient.get('diabetes', False):
            risk += 0.15
        
        if patient.get('hypertension', False):
            risk += 0.12
        
        if patient['bmi'] > 30:
            risk += 0.10
        
        return min(risk, 0.95)
    
    def calculate_cancer_risk(self, patient: pd.Series) -> float:
        risk = 0.01
        
        if patient['age'] > 50:
            risk += (patient['age'] - 50) * 0.008
        
        if patient.get('smoking_status') == 'current':
            risk += 0.25
        elif patient.get('smoking_status') == 'former':
            risk += 0.10
        
        if patient.get('alcohol_units_per_week', 0) > 14:
            risk += (patient['alcohol_units_per_week'] - 14) * 0.005
        
        if patient['bmi'] > 30:
            risk += 0.08
        
        family_history_prob = 0.15
        if self.rng.random() < family_history_prob:
            risk += 0.30
        
        return min(risk, 0.80)
    
    def calculate_kidney_disease_risk(self, patient: pd.Series) -> float:
        risk = 0.02
        
        if patient['age'] > 60:
            risk += (patient['age'] - 60) * 0.01
        
        if patient.get('diabetes', False):
            risk += 0.25
        
        if patient.get('hypertension', False):
            risk += 0.18
        
        if patient.get('creatinine_umol_l', 80) > 110:
            risk += (patient['creatinine_umol_l'] - 110) * 0.005
        
        return min(risk, 0.85)
    
    def simulate_disease_trajectory(self, patient: pd.Series, years: int = 10) -> pd.DataFrame:
        trajectory = []
        
        current_state = patient.copy()
        
        diabetes_risk = self.calculate_diabetes_risk(current_state)
        cvd_risk = self.calculate_cardiovascular_risk(current_state)
        cancer_risk = self.calculate_cancer_risk(current_state)
        kidney_risk = self.calculate_kidney_disease_risk(current_state)
        
        has_diabetes = current_state.get('diabetes', False)
        has_cvd = current_state.get('heart_disease', False)
        has_cancer = current_state.get('cancer', False)
        has_kidney = current_state.get('kidney_disease', False)
        
        for year in range(years + 1):
            if not has_diabetes and self.rng.random() < diabetes_risk * 0.1:
                has_diabetes = True
                current_state['hba1c_percent'] = min(current_state.get('hba1c_percent', 5.5) + 1.5, 12.0)
            
            if not has_cvd and self.rng.random() < cvd_risk * 0.08:
                has_cvd = True
            
            if not has_cancer and self.rng.random() < cancer_risk * 0.05:
                has_cancer = True
            
            if not has_kidney and self.rng.random() < kidney_risk * 0.06:
                has_kidney = True
            
            current_state['age'] = patient['age'] + year
            current_state['bmi'] = current_state['bmi'] + self.rng.normal(0.2, 0.5)
            current_state['systolic_bp'] = current_state.get('systolic_bp', 120) + self.rng.normal(0.5, 2)
            
            trajectory.append({
                'patient_id': patient['patient_id'],
                'year': year,
                'age': current_state['age'],
                'bmi': current_state['bmi'],
                'diabetes': has_diabetes,
                'cardiovascular_disease': has_cvd,
                'cancer': has_cancer,
                'kidney_disease': has_kidney,
                'diabetes_risk': diabetes_risk,
                'cvd_risk': cvd_risk,
                'cancer_risk': cancer_risk,
                'kidney_risk': kidney_risk,
                'alive': True
            })
            
            mortality_risk = 0.01 * (1.5 ** (current_state['age'] - 40) / 10)
            if has_diabetes:
                mortality_risk *= 1.5
            if has_cvd:
                mortality_risk *= 2.0
            if has_cancer:
                mortality_risk *= 3.0
            
            if self.rng.random() < mortality_risk:
                trajectory[-1]['alive'] = False
                break
        
        return pd.DataFrame(trajectory)
    
    def generate_population_trajectories(self, patients: pd.DataFrame, 
                                        years: int = 10, 
                                        output_dir: str = "data/synthetic") -> pd.DataFrame:
        logger.info(f"Simulating disease trajectories for {len(patients)} patients over {years} years...")
        
        all_trajectories = []
        
        for idx, patient in patients.iterrows():
            trajectory = self.simulate_disease_trajectory(patient, years)
            all_trajectories.append(trajectory)
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(patients)} patients")
        
        combined_trajectories = pd.concat(all_trajectories, ignore_index=True)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        combined_trajectories.to_csv(output_path / 'disease_trajectories.csv', index=False)
        
        logger.info(f"Trajectories saved to {output_path / 'disease_trajectories.csv'}")
        
        return combined_trajectories
    
    def calculate_intervention_effect(self, patient: pd.Series, 
                                     intervention: str) -> Dict[str, float]:
        effects = {
            'diabetes_risk_reduction': 0.0,
            'cvd_risk_reduction': 0.0,
            'cancer_risk_reduction': 0.0,
            'life_expectancy_gain_years': 0.0
        }
        
        if intervention == 'exercise_increase':
            effects['diabetes_risk_reduction'] = 0.25
            effects['cvd_risk_reduction'] = 0.30
            effects['cancer_risk_reduction'] = 0.15
            effects['life_expectancy_gain_years'] = 2.5
            
        elif intervention == 'smoking_cessation':
            effects['cvd_risk_reduction'] = 0.40
            effects['cancer_risk_reduction'] = 0.50
            effects['life_expectancy_gain_years'] = 5.0
            
        elif intervention == 'weight_loss':
            effects['diabetes_risk_reduction'] = 0.35
            effects['cvd_risk_reduction'] = 0.25
            effects['cancer_risk_reduction'] = 0.20
            effects['life_expectancy_gain_years'] = 3.0
            
        elif intervention == 'medication_adherence':
            if patient.get('diabetes', False):
                effects['diabetes_risk_reduction'] = 0.15
            if patient.get('hypertension', False):
                effects['cvd_risk_reduction'] = 0.35
            effects['life_expectancy_gain_years'] = 2.0
            
        elif intervention == 'diet_improvement':
            effects['diabetes_risk_reduction'] = 0.20
            effects['cvd_risk_reduction'] = 0.20
            effects['cancer_risk_reduction'] = 0.10
            effects['life_expectancy_gain_years'] = 1.5
        
        return effects


if __name__ == "__main__":
    from patient_population_generator import PatientPopulationGenerator
    
    pop_gen = PatientPopulationGenerator()
    patients_data = pop_gen.generate_complete_population(n=1000)
    patients = patients_data['complete']
    
    prog_gen = DiseaseProgressionGenerator()
    
    trajectories = prog_gen.generate_population_trajectories(patients.head(100), years=10)
    
    logger.info(f"\nGenerated {len(trajectories)} trajectory records")
    logger.info(f"\nSample trajectory:\n{trajectories[trajectories['patient_id'] == trajectories['patient_id'].iloc[0]]}")
    
    sample_patient = patients.iloc[0]
    intervention_effect = prog_gen.calculate_intervention_effect(sample_patient, 'exercise_increase')
    logger.info(f"\nIntervention effects:\n{intervention_effect}")
