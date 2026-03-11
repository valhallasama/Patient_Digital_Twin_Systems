import pandas as pd
import numpy as np
from typing import Dict, List
from agents.base_agent import BaseAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CardiologyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Cardiology Agent", specialty="cardiology")
        
    def get_required_fields(self) -> List[str]:
        return [
            'age', 'gender', 'systolic_bp', 'diastolic_bp',
            'total_cholesterol_mmol_l', 'hdl_cholesterol_mmol_l',
            'smoking_status', 'diabetes'
        ]
    
    def evaluate_patient(self, patient_data: pd.Series) -> Dict:
        risk_score = self.calculate_cardiovascular_risk(patient_data)
        risk_level = self.get_risk_level(risk_score)
        
        findings = self.analyze_cardiovascular_markers(patient_data)
        recommendations = self.generate_recommendations(patient_data, {'risk_score': risk_score})
        
        confidence = self.calculate_confidence(patient_data.to_dict())
        
        summary = f"Cardiovascular risk: {risk_level} ({risk_score:.2%})"
        self.log_observation(summary)
        
        return {
            'specialty': self.specialty,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': confidence,
            'findings': findings,
            'recommendations': recommendations,
            'summary': summary
        }
    
    def calculate_cardiovascular_risk(self, patient: pd.Series) -> float:
        risk = 0.05
        
        age = patient.get('age', 40)
        if age > 40:
            risk += (age - 40) * 0.01
        
        if patient.get('gender') == 'male':
            risk += 0.08
        
        systolic_bp = patient.get('systolic_bp', 120)
        if systolic_bp > 140:
            risk += (systolic_bp - 140) * 0.01
        elif systolic_bp > 130:
            risk += (systolic_bp - 130) * 0.005
        
        total_chol = patient.get('total_cholesterol_mmol_l', 5.0)
        if total_chol > 6.0:
            risk += (total_chol - 6.0) * 0.08
        
        hdl = patient.get('hdl_cholesterol_mmol_l', 1.4)
        if hdl < 1.0:
            risk += 0.12
        
        smoking = patient.get('smoking_status', 'never')
        if smoking == 'current':
            risk += 0.20
        elif smoking == 'former':
            risk += 0.08
        
        if patient.get('diabetes', False):
            risk += 0.15
        
        if patient.get('hypertension', False):
            risk += 0.12
        
        bmi = patient.get('bmi', 25)
        if bmi > 30:
            risk += 0.10
        
        exercise = patient.get('exercise_hours_per_week', 3)
        if exercise < 2:
            risk += 0.10
        
        return min(risk, 0.95)
    
    def analyze_cardiovascular_markers(self, patient: pd.Series) -> List[str]:
        findings = []
        
        systolic_bp = patient.get('systolic_bp', 120)
        diastolic_bp = patient.get('diastolic_bp', 80)
        
        if systolic_bp >= 140 or diastolic_bp >= 90:
            findings.append(f"Hypertension detected (BP: {systolic_bp:.0f}/{diastolic_bp:.0f})")
        elif systolic_bp >= 130 or diastolic_bp >= 85:
            findings.append(f"Elevated blood pressure (BP: {systolic_bp:.0f}/{diastolic_bp:.0f})")
        
        total_chol = patient.get('total_cholesterol_mmol_l', 5.0)
        if total_chol > 6.2:
            findings.append(f"High cholesterol ({total_chol:.1f} mmol/L)")
        elif total_chol > 5.2:
            findings.append(f"Borderline high cholesterol ({total_chol:.1f} mmol/L)")
        
        ldl = patient.get('ldl_cholesterol_mmol_l', 3.0)
        if ldl > 4.1:
            findings.append(f"High LDL cholesterol ({ldl:.1f} mmol/L)")
        
        hdl = patient.get('hdl_cholesterol_mmol_l', 1.4)
        if hdl < 1.0:
            findings.append(f"Low HDL cholesterol ({hdl:.1f} mmol/L)")
        
        heart_rate = patient.get('heart_rate', 70)
        if heart_rate > 100:
            findings.append(f"Tachycardia (HR: {heart_rate:.0f} bpm)")
        elif heart_rate < 60:
            findings.append(f"Bradycardia (HR: {heart_rate:.0f} bpm)")
        
        if not findings:
            findings.append("Cardiovascular markers within normal range")
        
        return findings
    
    def generate_recommendations(self, patient_data: pd.Series, 
                                evaluation: Dict) -> List[str]:
        recommendations = []
        risk_score = evaluation.get('risk_score', 0)
        
        if patient_data.get('smoking_status') == 'current':
            recommendations.append("CRITICAL: Smoking cessation program (reduces CVD risk by 40%)")
        
        if patient_data.get('systolic_bp', 120) > 140:
            recommendations.append("Start antihypertensive medication (consult physician)")
            recommendations.append("Reduce sodium intake to <2000mg/day")
        
        if patient_data.get('total_cholesterol_mmol_l', 5.0) > 6.0:
            recommendations.append("Consider statin therapy (consult physician)")
            recommendations.append("Adopt heart-healthy diet (Mediterranean or DASH)")
        
        if patient_data.get('exercise_hours_per_week', 3) < 2.5:
            recommendations.append("Increase aerobic exercise to 150 min/week minimum")
        
        if patient_data.get('bmi', 25) > 30:
            recommendations.append("Weight loss program (target 5-10% reduction)")
        
        if risk_score > 0.5:
            recommendations.append("Schedule comprehensive cardiovascular screening")
            recommendations.append("Consider aspirin therapy (consult physician)")
        
        if patient_data.get('stress_level', 5) > 7:
            recommendations.append("Stress management techniques (meditation, yoga)")
        
        return recommendations


if __name__ == "__main__":
    test_patient = pd.Series({
        'patient_id': 'P00000001',
        'age': 55,
        'gender': 'male',
        'bmi': 32,
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'heart_rate': 78,
        'total_cholesterol_mmol_l': 6.5,
        'ldl_cholesterol_mmol_l': 4.3,
        'hdl_cholesterol_mmol_l': 0.9,
        'smoking_status': 'current',
        'diabetes': False,
        'hypertension': True,
        'exercise_hours_per_week': 1,
        'stress_level': 8
    })
    
    agent = CardiologyAgent()
    evaluation = agent.evaluate_patient(test_patient)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Cardiology Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Risk Score: {evaluation['risk_score']:.2%}")
    logger.info(f"Risk Level: {evaluation['risk_level']}")
    logger.info(f"Confidence: {evaluation['confidence']:.2%}")
    logger.info(f"\nFindings:")
    for finding in evaluation['findings']:
        logger.info(f"  - {finding}")
    logger.info(f"\nRecommendations:")
    for rec in evaluation['recommendations']:
        logger.info(f"  - {rec}")
