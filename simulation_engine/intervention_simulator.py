import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterventionSimulator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        self.intervention_effects = {
            'smoking_cessation': {
                'cvd_risk_reduction': 0.40,
                'cancer_risk_reduction': 0.50,
                'diabetes_risk_reduction': 0.10,
                'life_expectancy_gain': 5.0,
                'adherence_rate': 0.25
            },
            'exercise_program': {
                'cvd_risk_reduction': 0.30,
                'diabetes_risk_reduction': 0.25,
                'cancer_risk_reduction': 0.15,
                'life_expectancy_gain': 2.5,
                'adherence_rate': 0.50
            },
            'weight_loss_10_percent': {
                'cvd_risk_reduction': 0.25,
                'diabetes_risk_reduction': 0.35,
                'cancer_risk_reduction': 0.20,
                'life_expectancy_gain': 3.0,
                'adherence_rate': 0.30
            },
            'mediterranean_diet': {
                'cvd_risk_reduction': 0.30,
                'diabetes_risk_reduction': 0.20,
                'cancer_risk_reduction': 0.15,
                'life_expectancy_gain': 2.0,
                'adherence_rate': 0.60
            },
            'statin_therapy': {
                'cvd_risk_reduction': 0.35,
                'diabetes_risk_reduction': -0.05,
                'life_expectancy_gain': 2.5,
                'adherence_rate': 0.70
            },
            'blood_pressure_medication': {
                'cvd_risk_reduction': 0.40,
                'diabetes_risk_reduction': 0.05,
                'life_expectancy_gain': 3.0,
                'adherence_rate': 0.75
            },
            'metformin': {
                'diabetes_risk_reduction': 0.30,
                'cvd_risk_reduction': 0.10,
                'life_expectancy_gain': 2.0,
                'adherence_rate': 0.80
            },
            'stress_management': {
                'cvd_risk_reduction': 0.15,
                'diabetes_risk_reduction': 0.10,
                'life_expectancy_gain': 1.0,
                'adherence_rate': 0.45
            }
        }
    
    def simulate_intervention(self, patient: pd.Series, intervention: str, 
                            years: int = 10) -> Dict:
        if intervention not in self.intervention_effects:
            logger.warning(f"Unknown intervention: {intervention}")
            return {}
        
        effects = self.intervention_effects[intervention]
        
        baseline_cvd_risk = self._calculate_cvd_risk(patient)
        baseline_diabetes_risk = self._calculate_diabetes_risk(patient)
        baseline_cancer_risk = self._calculate_cancer_risk(patient)
        
        adherence = self.rng.random() < effects['adherence_rate']
        
        if adherence:
            new_cvd_risk = baseline_cvd_risk * (1 - effects.get('cvd_risk_reduction', 0))
            new_diabetes_risk = baseline_diabetes_risk * (1 - effects.get('diabetes_risk_reduction', 0))
            new_cancer_risk = baseline_cancer_risk * (1 - effects.get('cancer_risk_reduction', 0))
            life_expectancy_gain = effects.get('life_expectancy_gain', 0)
        else:
            new_cvd_risk = baseline_cvd_risk
            new_diabetes_risk = baseline_diabetes_risk
            new_cancer_risk = baseline_cancer_risk
            life_expectancy_gain = 0
        
        return {
            'intervention': intervention,
            'adherence': adherence,
            'baseline_risks': {
                'cvd': baseline_cvd_risk,
                'diabetes': baseline_diabetes_risk,
                'cancer': baseline_cancer_risk
            },
            'post_intervention_risks': {
                'cvd': new_cvd_risk,
                'diabetes': new_diabetes_risk,
                'cancer': new_cancer_risk
            },
            'risk_reductions': {
                'cvd': baseline_cvd_risk - new_cvd_risk,
                'diabetes': baseline_diabetes_risk - new_diabetes_risk,
                'cancer': baseline_cancer_risk - new_cancer_risk
            },
            'life_expectancy_gain_years': life_expectancy_gain
        }
    
    def simulate_multiple_interventions(self, patient: pd.Series, 
                                       interventions: List[str],
                                       years: int = 10) -> Dict:
        results = {}
        
        for intervention in interventions:
            result = self.simulate_intervention(patient, intervention, years)
            results[intervention] = result
        
        return results
    
    def rank_interventions(self, patient: pd.Series, 
                          interventions: List[str] = None) -> List[Dict]:
        if interventions is None:
            interventions = list(self.intervention_effects.keys())
        
        results = []
        
        for intervention in interventions:
            sim_result = self.simulate_intervention(patient, intervention)
            
            if sim_result:
                total_risk_reduction = (
                    sim_result['risk_reductions']['cvd'] +
                    sim_result['risk_reductions']['diabetes'] +
                    sim_result['risk_reductions']['cancer']
                )
                
                benefit_score = (
                    total_risk_reduction * 100 +
                    sim_result['life_expectancy_gain_years'] * 10
                )
                
                results.append({
                    'intervention': intervention,
                    'benefit_score': benefit_score,
                    'life_expectancy_gain': sim_result['life_expectancy_gain_years'],
                    'total_risk_reduction': total_risk_reduction,
                    'adherence_rate': self.intervention_effects[intervention]['adherence_rate']
                })
        
        return sorted(results, key=lambda x: x['benefit_score'], reverse=True)
    
    def _calculate_cvd_risk(self, patient: pd.Series) -> float:
        risk = 0.05
        risk += (patient.get('age', 40) - 40) * 0.01
        if patient.get('systolic_bp', 120) > 140:
            risk += 0.15
        if patient.get('total_cholesterol_mmol_l', 5.0) > 6.0:
            risk += 0.10
        if patient.get('smoking_status') == 'current':
            risk += 0.20
        if patient.get('diabetes', False):
            risk += 0.15
        if patient.get('bmi', 25) > 30:
            risk += 0.10
        return min(risk, 0.95)
    
    def _calculate_diabetes_risk(self, patient: pd.Series) -> float:
        risk = 0.05
        risk += (patient.get('age', 40) - 40) * 0.008
        if patient.get('bmi', 25) > 30:
            risk += (patient.get('bmi', 25) - 30) * 0.05
        if patient.get('hba1c_percent', 5.0) > 5.7:
            risk += 0.25
        return min(risk, 0.95)
    
    def _calculate_cancer_risk(self, patient: pd.Series) -> float:
        risk = 0.02
        risk += (patient.get('age', 40) - 40) * 0.005
        if patient.get('smoking_status') == 'current':
            risk += 0.25
        if patient.get('alcohol_units_per_week', 0) > 14:
            risk += 0.08
        return min(risk, 0.80)


if __name__ == "__main__":
    test_patient = pd.Series({
        'patient_id': 'P00000001',
        'age': 55,
        'bmi': 32,
        'systolic_bp': 145,
        'total_cholesterol_mmol_l': 6.5,
        'hba1c_percent': 6.0,
        'smoking_status': 'current',
        'diabetes': False,
        'alcohol_units_per_week': 18
    })
    
    simulator = InterventionSimulator()
    
    logger.info("\n" + "="*60)
    logger.info("Intervention Simulation Results")
    logger.info("="*60)
    
    smoking_result = simulator.simulate_intervention(test_patient, 'smoking_cessation')
    logger.info(f"\nSmoking Cessation:")
    logger.info(f"  Adherence: {smoking_result['adherence']}")
    logger.info(f"  CVD risk reduction: {smoking_result['risk_reductions']['cvd']:.2%}")
    logger.info(f"  Life expectancy gain: {smoking_result['life_expectancy_gain_years']:.1f} years")
    
    ranked = simulator.rank_interventions(test_patient)
    logger.info(f"\n\nRanked Interventions (by benefit score):")
    for i, intervention in enumerate(ranked[:5], 1):
        logger.info(f"\n{i}. {intervention['intervention']}")
        logger.info(f"   Benefit Score: {intervention['benefit_score']:.1f}")
        logger.info(f"   Life Expectancy Gain: {intervention['life_expectancy_gain']:.1f} years")
        logger.info(f"   Total Risk Reduction: {intervention['total_risk_reduction']:.2%}")
        logger.info(f"   Adherence Rate: {intervention['adherence_rate']:.0%}")
