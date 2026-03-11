import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LifestyleGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
    def generate_daily_activity_pattern(self, patient: pd.Series, days: int = 30) -> pd.DataFrame:
        baseline_steps = patient.get('daily_steps', 7000)
        baseline_exercise = patient.get('exercise_hours_per_week', 3)
        baseline_sleep = patient.get('sleep_hours_per_night', 7)
        
        activity_data = []
        
        for day in range(days):
            is_weekend = day % 7 in [5, 6]
            
            steps = baseline_steps + self.rng.normal(0, baseline_steps * 0.2)
            if is_weekend:
                steps *= self.rng.uniform(0.8, 1.3)
            
            exercise_minutes = (baseline_exercise * 60 / 7) + self.rng.normal(0, 15)
            if is_weekend:
                exercise_minutes *= self.rng.uniform(1.0, 1.5)
            
            sleep_hours = baseline_sleep + self.rng.normal(0, 0.8)
            if is_weekend:
                sleep_hours += self.rng.uniform(0, 1.5)
            
            activity_data.append({
                'patient_id': patient['patient_id'],
                'day': day,
                'is_weekend': is_weekend,
                'steps': max(0, int(steps)),
                'exercise_minutes': max(0, exercise_minutes),
                'sleep_hours': np.clip(sleep_hours, 4, 12),
                'sedentary_hours': np.clip(
                    self.rng.uniform(6, 12) - (exercise_minutes / 60),
                    2, 16
                ),
                'screen_time_hours': np.clip(
                    self.rng.uniform(2, 8),
                    1, 14
                )
            })
        
        return pd.DataFrame(activity_data)
    
    def generate_dietary_intake(self, patient: pd.Series, days: int = 30) -> pd.DataFrame:
        diet_quality = patient.get('diet_quality_score', 5)
        bmi = patient.get('bmi', 25)
        
        base_calories = 2000 + (bmi - 25) * 50
        
        dietary_data = []
        
        for day in range(days):
            is_weekend = day % 7 in [5, 6]
            
            calories = base_calories + self.rng.normal(0, 300)
            if is_weekend:
                calories *= self.rng.uniform(1.0, 1.3)
            
            protein_ratio = 0.15 + (diet_quality / 50)
            carb_ratio = 0.50 - (diet_quality / 100)
            fat_ratio = 1 - protein_ratio - carb_ratio
            
            dietary_data.append({
                'patient_id': patient['patient_id'],
                'day': day,
                'total_calories': calories,
                'protein_g': (calories * protein_ratio) / 4,
                'carbohydrates_g': (calories * carb_ratio) / 4,
                'fat_g': (calories * fat_ratio) / 9,
                'fiber_g': 15 + diet_quality * 2 + self.rng.normal(0, 5),
                'sugar_g': 50 - diet_quality * 3 + self.rng.normal(0, 10),
                'sodium_mg': 2500 - diet_quality * 100 + self.rng.normal(0, 500),
                'fruits_servings': max(0, diet_quality / 2 + self.rng.normal(0, 1)),
                'vegetables_servings': max(0, diet_quality / 2 + self.rng.normal(0, 1)),
                'water_liters': np.clip(2 + self.rng.normal(0, 0.5), 0.5, 5)
            })
        
        return pd.DataFrame(dietary_data)
    
    def generate_stress_patterns(self, patient: pd.Series, days: int = 30) -> pd.DataFrame:
        baseline_stress = patient.get('stress_level', 5)
        
        stress_data = []
        
        for day in range(days):
            is_weekend = day % 7 in [5, 6]
            is_monday = day % 7 == 0
            
            stress_level = baseline_stress + self.rng.normal(0, 2)
            
            if is_monday:
                stress_level += self.rng.uniform(1, 3)
            elif is_weekend:
                stress_level -= self.rng.uniform(1, 2)
            
            stress_data.append({
                'patient_id': patient['patient_id'],
                'day': day,
                'stress_level': np.clip(stress_level, 1, 10),
                'work_hours': 0 if is_weekend else self.rng.uniform(6, 10),
                'relaxation_minutes': self.rng.uniform(30, 180) * (1.5 if is_weekend else 1.0),
                'social_interaction_hours': self.rng.uniform(1, 6) * (1.5 if is_weekend else 1.0)
            })
        
        return pd.DataFrame(stress_data)
    
    def generate_substance_use(self, patient: pd.Series, days: int = 30) -> pd.DataFrame:
        smoking_status = patient.get('smoking_status', 'never')
        baseline_alcohol = patient.get('alcohol_units_per_week', 5)
        
        substance_data = []
        
        for day in range(days):
            is_weekend = day % 7 in [5, 6]
            
            if smoking_status == 'current':
                cigarettes = self.rng.poisson(10)
            elif smoking_status == 'former':
                cigarettes = 0
            else:
                cigarettes = 0
            
            alcohol_units = (baseline_alcohol / 7) + self.rng.exponential(1)
            if is_weekend:
                alcohol_units *= self.rng.uniform(1.5, 3.0)
            
            substance_data.append({
                'patient_id': patient['patient_id'],
                'day': day,
                'cigarettes': cigarettes,
                'alcohol_units': max(0, alcohol_units),
                'caffeine_mg': self.rng.uniform(100, 400)
            })
        
        return pd.DataFrame(substance_data)
    
    def generate_complete_lifestyle_timeline(self, patient: pd.Series, 
                                            days: int = 30) -> Dict[str, pd.DataFrame]:
        activity = self.generate_daily_activity_pattern(patient, days)
        diet = self.generate_dietary_intake(patient, days)
        stress = self.generate_stress_patterns(patient, days)
        substance = self.generate_substance_use(patient, days)
        
        combined = activity.merge(diet, on=['patient_id', 'day']) \
                          .merge(stress, on=['patient_id', 'day']) \
                          .merge(substance, on=['patient_id', 'day'])
        
        return {
            'activity': activity,
            'diet': diet,
            'stress': stress,
            'substance': substance,
            'combined': combined
        }


if __name__ == "__main__":
    from patient_population_generator import PatientPopulationGenerator
    
    pop_gen = PatientPopulationGenerator()
    patients_data = pop_gen.generate_complete_population(n=10)
    sample_patient = patients_data['complete'].iloc[0]
    
    lifestyle_gen = LifestyleGenerator()
    
    lifestyle_timeline = lifestyle_gen.generate_complete_lifestyle_timeline(sample_patient, days=30)
    
    logger.info(f"\nGenerated lifestyle timeline for patient {sample_patient['patient_id']}")
    logger.info(f"\nSample daily activity:\n{lifestyle_timeline['activity'].head()}")
    logger.info(f"\nSample dietary intake:\n{lifestyle_timeline['diet'].head()}")
    logger.info(f"\nCombined lifestyle data shape: {lifestyle_timeline['combined'].shape}")
