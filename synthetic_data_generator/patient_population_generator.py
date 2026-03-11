import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientPopulationGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
    def generate_demographics(self, n: int) -> pd.DataFrame:
        data = {
            'patient_id': [f"P{i:08d}" for i in range(n)],
            'age': self.rng.integers(18, 90, n),
            'gender': self.rng.choice(['male', 'female'], n),
            'ethnicity': self.rng.choice([
                'caucasian', 'african_american', 'hispanic', 
                'asian', 'other'
            ], n, p=[0.6, 0.13, 0.18, 0.06, 0.03]),
            'height_cm': self.rng.normal(170, 10, n),
            'weight_kg': self.rng.normal(75, 15, n),
        }
        
        df = pd.DataFrame(data)
        df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
        
        return df
    
    def generate_vital_signs(self, demographics: pd.DataFrame) -> pd.DataFrame:
        n = len(demographics)
        
        age_factor = (demographics['age'] - 40) / 50
        bmi_factor = (demographics['bmi'] - 25) / 10
        
        vitals = pd.DataFrame({
            'patient_id': demographics['patient_id'],
            'systolic_bp': np.clip(
                120 + age_factor * 20 + bmi_factor * 10 + self.rng.normal(0, 10, n),
                90, 200
            ),
            'diastolic_bp': np.clip(
                80 + age_factor * 10 + bmi_factor * 5 + self.rng.normal(0, 8, n),
                60, 130
            ),
            'heart_rate': np.clip(
                70 + self.rng.normal(0, 10, n) - age_factor * 5,
                50, 120
            ),
            'respiratory_rate': np.clip(
                16 + self.rng.normal(0, 2, n),
                12, 25
            ),
            'temperature_c': np.clip(
                36.8 + self.rng.normal(0, 0.3, n),
                36.0, 38.5
            ),
            'oxygen_saturation': np.clip(
                98 + self.rng.normal(0, 1.5, n),
                85, 100
            )
        })
        
        return vitals
    
    def generate_lab_results(self, demographics: pd.DataFrame) -> pd.DataFrame:
        n = len(demographics)
        
        age_factor = (demographics['age'] - 40) / 50
        bmi_factor = (demographics['bmi'] - 25) / 10
        
        labs = pd.DataFrame({
            'patient_id': demographics['patient_id'],
            'glucose_mmol_l': np.clip(
                5.5 + bmi_factor * 1.5 + age_factor * 0.8 + self.rng.normal(0, 1, n),
                3.5, 15.0
            ),
            'hba1c_percent': np.clip(
                5.5 + bmi_factor * 0.8 + age_factor * 0.5 + self.rng.normal(0, 0.5, n),
                4.0, 12.0
            ),
            'total_cholesterol_mmol_l': np.clip(
                5.0 + age_factor * 1.0 + bmi_factor * 0.5 + self.rng.normal(0, 0.8, n),
                3.0, 10.0
            ),
            'ldl_cholesterol_mmol_l': np.clip(
                3.0 + age_factor * 0.8 + bmi_factor * 0.4 + self.rng.normal(0, 0.6, n),
                1.5, 7.0
            ),
            'hdl_cholesterol_mmol_l': np.clip(
                1.4 - bmi_factor * 0.2 + self.rng.normal(0, 0.3, n),
                0.8, 2.5
            ),
            'triglycerides_mmol_l': np.clip(
                1.5 + bmi_factor * 0.5 + self.rng.normal(0, 0.5, n),
                0.5, 5.0
            ),
            'creatinine_umol_l': np.clip(
                80 + age_factor * 20 + self.rng.normal(0, 15, n),
                50, 200
            ),
            'alt_u_l': np.clip(
                25 + bmi_factor * 10 + self.rng.exponential(10, n),
                10, 150
            ),
            'ast_u_l': np.clip(
                25 + bmi_factor * 8 + self.rng.exponential(8, n),
                10, 120
            ),
            'hemoglobin_g_l': np.clip(
                140 + (demographics['gender'] == 'male') * 20 + self.rng.normal(0, 10, n),
                100, 180
            ),
            'white_blood_cells_10_9_l': np.clip(
                7.0 + self.rng.normal(0, 2, n),
                3.0, 15.0
            ),
            'platelets_10_9_l': np.clip(
                250 + self.rng.normal(0, 50, n),
                150, 450
            )
        })
        
        return labs
    
    def generate_lifestyle_data(self, demographics: pd.DataFrame) -> pd.DataFrame:
        n = len(demographics)
        
        age_factor = (demographics['age'] - 40) / 50
        
        lifestyle = pd.DataFrame({
            'patient_id': demographics['patient_id'],
            'smoking_status': self.rng.choice(
                ['never', 'former', 'current'], 
                n, 
                p=[0.6, 0.25, 0.15]
            ),
            'alcohol_units_per_week': np.clip(
                self.rng.exponential(5, n),
                0, 50
            ),
            'exercise_hours_per_week': np.clip(
                5 - age_factor * 2 + self.rng.exponential(3, n),
                0, 20
            ),
            'sleep_hours_per_night': np.clip(
                7 + self.rng.normal(0, 1, n),
                4, 10
            ),
            'daily_steps': np.clip(
                7000 - age_factor * 2000 + self.rng.normal(0, 2000, n),
                1000, 20000
            ).astype(int),
            'stress_level': self.rng.integers(1, 11, n),
            'diet_quality_score': self.rng.integers(1, 11, n)
        })
        
        return lifestyle
    
    def generate_medical_history(self, demographics: pd.DataFrame) -> pd.DataFrame:
        n = len(demographics)
        
        age = demographics['age'].values
        bmi = demographics['bmi'].values
        
        hypertension_prob = np.clip(0.05 + (age - 40) * 0.01 + (bmi - 25) * 0.02, 0, 0.8)
        diabetes_prob = np.clip(0.03 + (age - 40) * 0.008 + (bmi - 25) * 0.025, 0, 0.6)
        heart_disease_prob = np.clip(0.02 + (age - 50) * 0.01 + (bmi - 25) * 0.01, 0, 0.5)
        
        history = pd.DataFrame({
            'patient_id': demographics['patient_id'],
            'hypertension': self.rng.random(n) < hypertension_prob,
            'diabetes': self.rng.random(n) < diabetes_prob,
            'heart_disease': self.rng.random(n) < heart_disease_prob,
            'stroke': self.rng.random(n) < np.clip((age - 60) * 0.005, 0, 0.3),
            'cancer': self.rng.random(n) < np.clip((age - 50) * 0.005, 0, 0.2),
            'asthma': self.rng.random(n) < 0.08,
            'copd': self.rng.random(n) < np.clip((age - 50) * 0.008, 0, 0.25),
            'kidney_disease': self.rng.random(n) < np.clip((age - 60) * 0.006, 0, 0.15),
            'liver_disease': self.rng.random(n) < 0.05,
            'depression': self.rng.random(n) < 0.15,
            'anxiety': self.rng.random(n) < 0.18
        })
        
        return history
    
    def generate_medications(self, medical_history: pd.DataFrame) -> pd.DataFrame:
        medications_list = []
        
        for idx, row in medical_history.iterrows():
            patient_id = row['patient_id']
            meds = []
            
            if row['hypertension']:
                meds.extend(self.rng.choice([
                    'lisinopril', 'amlodipine', 'losartan', 'metoprolol'
                ], size=self.rng.integers(1, 3), replace=False).tolist())
            
            if row['diabetes']:
                meds.extend(self.rng.choice([
                    'metformin', 'insulin', 'glipizide', 'sitagliptin'
                ], size=self.rng.integers(1, 3), replace=False).tolist())
            
            if row['heart_disease']:
                meds.extend(self.rng.choice([
                    'aspirin', 'atorvastatin', 'clopidogrel', 'warfarin'
                ], size=self.rng.integers(1, 2), replace=False).tolist())
            
            if row['depression']:
                meds.append(self.rng.choice(['sertraline', 'fluoxetine', 'escitalopram']))
            
            medications_list.append({
                'patient_id': patient_id,
                'medications': ','.join(meds) if meds else 'none',
                'medication_count': len(meds)
            })
        
        return pd.DataFrame(medications_list)
    
    def generate_complete_population(self, n: int, output_dir: str = "data/synthetic") -> Dict[str, pd.DataFrame]:
        logger.info(f"Generating synthetic population of {n} patients...")
        
        demographics = self.generate_demographics(n)
        vitals = self.generate_vital_signs(demographics)
        labs = self.generate_lab_results(demographics)
        lifestyle = self.generate_lifestyle_data(demographics)
        medical_history = self.generate_medical_history(demographics)
        medications = self.generate_medications(medical_history)
        
        complete_data = demographics.merge(vitals, on='patient_id') \
                                    .merge(labs, on='patient_id') \
                                    .merge(lifestyle, on='patient_id') \
                                    .merge(medical_history, on='patient_id') \
                                    .merge(medications, on='patient_id')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        complete_data.to_csv(output_path / 'complete_patient_data.csv', index=False)
        demographics.to_csv(output_path / 'demographics.csv', index=False)
        vitals.to_csv(output_path / 'vital_signs.csv', index=False)
        labs.to_csv(output_path / 'lab_results.csv', index=False)
        lifestyle.to_csv(output_path / 'lifestyle.csv', index=False)
        medical_history.to_csv(output_path / 'medical_history.csv', index=False)
        medications.to_csv(output_path / 'medications.csv', index=False)
        
        logger.info(f"Synthetic data saved to {output_path}")
        
        return {
            'complete': complete_data,
            'demographics': demographics,
            'vitals': vitals,
            'labs': labs,
            'lifestyle': lifestyle,
            'medical_history': medical_history,
            'medications': medications
        }


if __name__ == "__main__":
    generator = PatientPopulationGenerator()
    
    data = generator.generate_complete_population(n=10000)
    
    logger.info(f"Generated {len(data['complete'])} patient records")
    logger.info(f"\nSample patient data:\n{data['complete'].head()}")
    logger.info(f"\nData statistics:\n{data['complete'].describe()}")
