import pandas as pd
import numpy as np
from typing import Dict, List
from agents.base_agent import BaseAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LifestyleAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Lifestyle Agent", specialty="lifestyle")
        
    def get_required_fields(self) -> List[str]:
        return [
            'exercise_hours_per_week', 'sleep_hours_per_night',
            'smoking_status', 'alcohol_units_per_week',
            'diet_quality_score', 'stress_level'
        ]
    
    def evaluate_patient(self, patient_data: pd.Series) -> Dict:
        lifestyle_score = self.calculate_lifestyle_score(patient_data)
        risk_score = 1.0 - lifestyle_score
        
        findings = self.analyze_lifestyle_factors(patient_data)
        recommendations = self.generate_recommendations(patient_data, {
            'lifestyle_score': lifestyle_score
        })
        
        confidence = self.calculate_confidence(patient_data.to_dict())
        
        risk_level = self.get_risk_level(risk_score)
        summary = f"Lifestyle health score: {lifestyle_score:.2%} (risk: {risk_level})"
        
        self.log_observation(summary)
        
        return {
            'specialty': self.specialty,
            'lifestyle_score': lifestyle_score,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': confidence,
            'findings': findings,
            'recommendations': recommendations,
            'summary': summary
        }
    
    def calculate_lifestyle_score(self, patient: pd.Series) -> float:
        score = 0.0
        max_score = 7.0
        
        exercise = patient.get('exercise_hours_per_week', 0)
        if exercise >= 5:
            score += 1.5
        elif exercise >= 2.5:
            score += 1.0
        elif exercise >= 1:
            score += 0.5
        
        sleep = patient.get('sleep_hours_per_night', 7)
        if 7 <= sleep <= 9:
            score += 1.0
        elif 6 <= sleep <= 10:
            score += 0.5
        
        smoking = patient.get('smoking_status', 'never')
        if smoking == 'never':
            score += 1.5
        elif smoking == 'former':
            score += 0.75
        
        alcohol = patient.get('alcohol_units_per_week', 0)
        if alcohol == 0:
            score += 1.0
        elif alcohol <= 7:
            score += 0.75
        elif alcohol <= 14:
            score += 0.5
        
        diet = patient.get('diet_quality_score', 5)
        score += (diet / 10) * 1.0
        
        stress = patient.get('stress_level', 5)
        if stress <= 3:
            score += 1.0
        elif stress <= 5:
            score += 0.75
        elif stress <= 7:
            score += 0.5
        
        steps = patient.get('daily_steps', 0)
        if steps >= 10000:
            score += 1.0
        elif steps >= 7000:
            score += 0.75
        elif steps >= 5000:
            score += 0.5
        
        return min(score / max_score, 1.0)
    
    def analyze_lifestyle_factors(self, patient: pd.Series) -> List[str]:
        findings = []
        
        exercise = patient.get('exercise_hours_per_week', 0)
        if exercise < 1:
            findings.append("Sedentary lifestyle - critically low physical activity")
        elif exercise < 2.5:
            findings.append(f"Insufficient exercise ({exercise:.1f} hrs/week, need 2.5+)")
        elif exercise >= 5:
            findings.append(f"Excellent exercise habits ({exercise:.1f} hrs/week)")
        
        sleep = patient.get('sleep_hours_per_night', 7)
        if sleep < 6:
            findings.append(f"Sleep deprivation ({sleep:.1f} hrs/night)")
        elif sleep > 9:
            findings.append(f"Excessive sleep ({sleep:.1f} hrs/night)")
        else:
            findings.append(f"Adequate sleep ({sleep:.1f} hrs/night)")
        
        smoking = patient.get('smoking_status', 'never')
        if smoking == 'current':
            findings.append("CRITICAL: Active smoker")
        elif smoking == 'former':
            findings.append("Former smoker (good progress)")
        
        alcohol = patient.get('alcohol_units_per_week', 0)
        if alcohol > 14:
            findings.append(f"Excessive alcohol consumption ({alcohol:.1f} units/week)")
        elif alcohol > 7:
            findings.append(f"Moderate alcohol consumption ({alcohol:.1f} units/week)")
        
        diet = patient.get('diet_quality_score', 5)
        if diet < 4:
            findings.append(f"Poor diet quality (score: {diet}/10)")
        elif diet < 6:
            findings.append(f"Fair diet quality (score: {diet}/10)")
        elif diet >= 8:
            findings.append(f"Excellent diet quality (score: {diet}/10)")
        
        stress = patient.get('stress_level', 5)
        if stress >= 8:
            findings.append(f"High stress levels ({stress}/10)")
        elif stress >= 6:
            findings.append(f"Moderate stress levels ({stress}/10)")
        
        steps = patient.get('daily_steps', 0)
        if steps < 5000:
            findings.append(f"Low daily activity ({steps} steps/day)")
        elif steps >= 10000:
            findings.append(f"Excellent daily activity ({steps} steps/day)")
        
        return findings
    
    def generate_recommendations(self, patient_data: pd.Series, 
                                evaluation: Dict) -> List[str]:
        recommendations = []
        
        exercise = patient_data.get('exercise_hours_per_week', 0)
        if exercise < 2.5:
            recommendations.append("Increase aerobic exercise to 150 min/week (brisk walking, cycling, swimming)")
            recommendations.append("Start with 10-minute sessions and gradually increase")
        
        if exercise < 5:
            recommendations.append("Add resistance training 2-3 times per week")
        
        sleep = patient_data.get('sleep_hours_per_night', 7)
        if sleep < 7:
            recommendations.append("Improve sleep hygiene: consistent bedtime, dark room, no screens 1hr before bed")
            recommendations.append("Target 7-9 hours of sleep per night")
        
        smoking = patient_data.get('smoking_status', 'never')
        if smoking == 'current':
            recommendations.append("URGENT: Enroll in smoking cessation program")
            recommendations.append("Consider nicotine replacement therapy or prescription medications")
            recommendations.append("Join support group for smoking cessation")
        
        alcohol = patient_data.get('alcohol_units_per_week', 0)
        if alcohol > 14:
            recommendations.append("Reduce alcohol consumption to ≤14 units/week")
            recommendations.append("Have at least 2-3 alcohol-free days per week")
        
        diet = patient_data.get('diet_quality_score', 5)
        if diet < 6:
            recommendations.append("Adopt Mediterranean or DASH diet pattern")
            recommendations.append("Increase fruits and vegetables to 5+ servings/day")
            recommendations.append("Reduce processed foods and added sugars")
            recommendations.append("Consider nutrition counseling")
        
        stress = patient_data.get('stress_level', 5)
        if stress >= 7:
            recommendations.append("Stress management program: meditation, mindfulness, or yoga")
            recommendations.append("Consider cognitive behavioral therapy (CBT)")
            recommendations.append("Ensure regular breaks and work-life balance")
        
        steps = patient_data.get('daily_steps', 0)
        if steps < 7000:
            recommendations.append(f"Increase daily steps from {steps} to 10,000+ (use pedometer/fitness tracker)")
            recommendations.append("Take walking breaks every hour during work")
        
        return recommendations


if __name__ == "__main__":
    test_patient = pd.Series({
        'patient_id': 'P00000003',
        'exercise_hours_per_week': 1.5,
        'sleep_hours_per_night': 5.5,
        'smoking_status': 'current',
        'alcohol_units_per_week': 18,
        'diet_quality_score': 3,
        'stress_level': 8,
        'daily_steps': 4000
    })
    
    agent = LifestyleAgent()
    evaluation = agent.evaluate_patient(test_patient)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Lifestyle Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Lifestyle Score: {evaluation['lifestyle_score']:.2%}")
    logger.info(f"Risk Level: {evaluation['risk_level']}")
    logger.info(f"Confidence: {evaluation['confidence']:.2%}")
    logger.info(f"\nFindings:")
    for finding in evaluation['findings']:
        logger.info(f"  - {finding}")
    logger.info(f"\nRecommendations:")
    for rec in evaluation['recommendations']:
        logger.info(f"  - {rec}")
