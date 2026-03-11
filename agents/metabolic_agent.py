import pandas as pd
import numpy as np
from typing import Dict, List
from agents.base_agent import BaseAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetabolicAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Metabolic Agent", specialty="metabolic")
        
    def get_required_fields(self) -> List[str]:
        return [
            'age', 'bmi', 'glucose_mmol_l', 'hba1c_percent',
            'exercise_hours_per_week', 'diet_quality_score'
        ]
    
    def evaluate_patient(self, patient_data: pd.Series) -> Dict:
        diabetes_risk = self.calculate_diabetes_risk(patient_data)
        metabolic_syndrome = self.assess_metabolic_syndrome(patient_data)
        
        findings = self.analyze_metabolic_markers(patient_data)
        recommendations = self.generate_recommendations(patient_data, {
            'diabetes_risk': diabetes_risk,
            'metabolic_syndrome': metabolic_syndrome
        })
        
        confidence = self.calculate_confidence(patient_data.to_dict())
        
        risk_level = self.get_risk_level(diabetes_risk)
        summary = f"Diabetes risk: {risk_level} ({diabetes_risk:.2%})"
        if metabolic_syndrome:
            summary += " | Metabolic syndrome detected"
        
        self.log_observation(summary)
        
        return {
            'specialty': self.specialty,
            'risk_score': diabetes_risk,
            'risk_level': risk_level,
            'metabolic_syndrome': metabolic_syndrome,
            'confidence': confidence,
            'findings': findings,
            'recommendations': recommendations,
            'summary': summary
        }
    
    def calculate_diabetes_risk(self, patient: pd.Series) -> float:
        risk = 0.05
        
        age = patient.get('age', 40)
        if age > 45:
            risk += (age - 45) * 0.01
        
        bmi = patient.get('bmi', 25)
        if bmi > 30:
            risk += (bmi - 30) * 0.05
        elif bmi > 25:
            risk += (bmi - 25) * 0.03
        
        hba1c = patient.get('hba1c_percent', 5.0)
        if hba1c >= 6.5:
            risk = 0.95
        elif hba1c >= 5.7:
            risk += (hba1c - 5.7) * 0.20
        
        glucose = patient.get('glucose_mmol_l', 5.0)
        if glucose >= 7.0:
            risk += 0.30
        elif glucose >= 6.1:
            risk += (glucose - 6.1) * 0.15
        
        exercise = patient.get('exercise_hours_per_week', 3)
        if exercise < 2:
            risk += 0.15
        
        if patient.get('hypertension', False):
            risk += 0.12
        
        if patient.get('smoking_status') == 'current':
            risk += 0.10
        
        diet_quality = patient.get('diet_quality_score', 5)
        if diet_quality < 4:
            risk += 0.10
        
        return min(risk, 0.95)
    
    def assess_metabolic_syndrome(self, patient: pd.Series) -> bool:
        criteria_met = 0
        
        waist_circumference_high = patient.get('bmi', 25) > 30
        if waist_circumference_high:
            criteria_met += 1
        
        triglycerides = patient.get('triglycerides_mmol_l', 1.5)
        if triglycerides >= 1.7:
            criteria_met += 1
        
        hdl = patient.get('hdl_cholesterol_mmol_l', 1.4)
        gender = patient.get('gender', 'male')
        if (gender == 'male' and hdl < 1.0) or (gender == 'female' and hdl < 1.3):
            criteria_met += 1
        
        systolic_bp = patient.get('systolic_bp', 120)
        diastolic_bp = patient.get('diastolic_bp', 80)
        if systolic_bp >= 130 or diastolic_bp >= 85 or patient.get('hypertension', False):
            criteria_met += 1
        
        glucose = patient.get('glucose_mmol_l', 5.0)
        if glucose >= 5.6 or patient.get('diabetes', False):
            criteria_met += 1
        
        return criteria_met >= 3
    
    def analyze_metabolic_markers(self, patient: pd.Series) -> List[str]:
        findings = []
        
        hba1c = patient.get('hba1c_percent', 5.0)
        if hba1c >= 6.5:
            findings.append(f"Diabetes confirmed (HbA1c: {hba1c:.1f}%)")
        elif hba1c >= 5.7:
            findings.append(f"Prediabetes (HbA1c: {hba1c:.1f}%)")
        
        glucose = patient.get('glucose_mmol_l', 5.0)
        if glucose >= 7.0:
            findings.append(f"Fasting glucose elevated ({glucose:.1f} mmol/L)")
        elif glucose >= 6.1:
            findings.append(f"Impaired fasting glucose ({glucose:.1f} mmol/L)")
        
        bmi = patient.get('bmi', 25)
        if bmi >= 30:
            findings.append(f"Obesity (BMI: {bmi:.1f})")
        elif bmi >= 25:
            findings.append(f"Overweight (BMI: {bmi:.1f})")
        
        triglycerides = patient.get('triglycerides_mmol_l', 1.5)
        if triglycerides >= 2.3:
            findings.append(f"High triglycerides ({triglycerides:.1f} mmol/L)")
        
        alt = patient.get('alt_u_l', 25)
        if alt > 40:
            findings.append(f"Elevated liver enzymes (ALT: {alt:.0f} U/L) - possible fatty liver")
        
        if not findings:
            findings.append("Metabolic markers within normal range")
        
        return findings
    
    def generate_recommendations(self, patient_data: pd.Series, 
                                evaluation: Dict) -> List[str]:
        recommendations = []
        
        diabetes_risk = evaluation.get('diabetes_risk', 0)
        metabolic_syndrome = evaluation.get('metabolic_syndrome', False)
        
        if patient_data.get('hba1c_percent', 5.0) >= 5.7:
            recommendations.append("Diabetes screening every 3-6 months")
            recommendations.append("Consider metformin therapy (consult physician)")
        
        if patient_data.get('bmi', 25) > 30:
            recommendations.append("PRIORITY: Weight loss program (target 7-10% reduction)")
            recommendations.append("Caloric restriction: reduce intake by 500-750 kcal/day")
        
        if patient_data.get('exercise_hours_per_week', 3) < 2.5:
            recommendations.append("Increase physical activity to 150 min/week minimum")
            recommendations.append("Add resistance training 2-3 times/week")
        
        diet_quality = patient_data.get('diet_quality_score', 5)
        if diet_quality < 6:
            recommendations.append("Dietary intervention: low glycemic index diet")
            recommendations.append("Reduce refined carbohydrates and added sugars")
            recommendations.append("Increase fiber intake to 25-30g/day")
        
        if metabolic_syndrome:
            recommendations.append("CRITICAL: Metabolic syndrome detected - comprehensive intervention needed")
            recommendations.append("Monitor blood pressure, lipids, and glucose regularly")
        
        if diabetes_risk > 0.5:
            recommendations.append("Diabetes prevention program enrollment recommended")
            recommendations.append("Consider continuous glucose monitoring")
        
        if patient_data.get('alt_u_l', 25) > 40:
            recommendations.append("Liver ultrasound to assess for fatty liver disease")
        
        return recommendations


if __name__ == "__main__":
    test_patient = pd.Series({
        'patient_id': 'P00000002',
        'age': 52,
        'gender': 'female',
        'bmi': 33,
        'glucose_mmol_l': 6.8,
        'hba1c_percent': 6.2,
        'triglycerides_mmol_l': 2.5,
        'hdl_cholesterol_mmol_l': 1.1,
        'systolic_bp': 135,
        'diastolic_bp': 88,
        'alt_u_l': 45,
        'exercise_hours_per_week': 1.5,
        'diet_quality_score': 4,
        'hypertension': True,
        'diabetes': False
    })
    
    agent = MetabolicAgent()
    evaluation = agent.evaluate_patient(test_patient)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Metabolic Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Diabetes Risk: {evaluation['risk_score']:.2%}")
    logger.info(f"Risk Level: {evaluation['risk_level']}")
    logger.info(f"Metabolic Syndrome: {evaluation['metabolic_syndrome']}")
    logger.info(f"Confidence: {evaluation['confidence']:.2%}")
    logger.info(f"\nFindings:")
    for finding in evaluation['findings']:
        logger.info(f"  - {finding}")
    logger.info(f"\nRecommendations:")
    for rec in evaluation['recommendations']:
        logger.info(f"  - {rec}")
