#!/usr/bin/env python3
"""
Single Patient Digital Twin Simulator - DEMO VERSION

YOUR VISION IMPLEMENTED:
1. Upload patient health data (text + test results)
2. Create digital twin from this information  
3. Simulate 5-10 years of organ changes
4. Predict disease risks with probabilities
5. Recommend interventions to reduce risks

This is a standalone demo that works WITHOUT the full trained model.
"""

import numpy as np
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PatientData:
    """Structured patient data"""
    # Demographics
    age: int
    sex: str  # 'male' or 'female'
    
    # Lifestyle
    occupation: str = "unknown"
    exercise_hours_per_week: float = 0.0
    sleep_hours_per_night: float = 7.0
    smoking: bool = False
    alcohol_drinks_per_week: float = 0.0
    stress_level: float = 0.5  # 0-1 scale
    diet_quality: str = "moderate"
    
    # Test results
    bmi: float = 25.0
    systolic_bp: float = 120.0
    diastolic_bp: float = 80.0
    fasting_glucose: float = 100.0
    hba1c: float = 5.5
    total_cholesterol: float = 200.0
    ldl: float = 100.0
    hdl: float = 50.0
    triglycerides: float = 150.0
    creatinine: float = 1.0
    egfr: float = 90.0
    alt: float = 25.0
    ast: float = 25.0
    crp: float = 1.0


class NaturalLanguageParser:
    """Parse natural language patient descriptions"""
    
    @staticmethod
    def parse_lifestyle_description(text: str) -> Dict:
        """Parse lifestyle from natural language"""
        text = text.lower()
        
        lifestyle = {
            'occupation': 'unknown',
            'exercise_hours_per_week': 0.0,
            'sleep_hours_per_night': 7.0,
            'smoking': False,
            'alcohol_drinks_per_week': 0.0,
            'stress_level': 0.5,
            'diet_quality': 'moderate'
        }
        
        # Occupation
        if 'office' in text or 'desk' in text or 'sedentary' in text:
            lifestyle['occupation'] = 'office_worker'
        elif 'manual' in text or 'labor' in text:
            lifestyle['occupation'] = 'manual_labor'
        
        # Exercise
        if 'sedentary' in text or 'no exercise' in text:
            lifestyle['exercise_hours_per_week'] = 0.0
        elif 'active' in text or 'exercise' in text:
            lifestyle['exercise_hours_per_week'] = 3.0
        
        # Sleep
        sleep_match = re.search(r'sleep[s]?\s*(\d+)\s*(hour|hr)', text)
        if sleep_match:
            lifestyle['sleep_hours_per_night'] = float(sleep_match.group(1))
        elif 'poor sleep' in text:
            lifestyle['sleep_hours_per_night'] = 5.0
        
        # Smoking
        if 'smok' in text or 'cigarette' in text:
            lifestyle['smoking'] = True
        
        # Alcohol
        if 'drink' in text or 'alcohol' in text or 'beer' in text:
            alcohol_match = re.search(r'(\d+)\s*(drink|beer|glass)', text)
            if alcohol_match:
                lifestyle['alcohol_drinks_per_week'] = float(alcohol_match.group(1)) * 7
            else:
                lifestyle['alcohol_drinks_per_week'] = 7.0
        
        # Stress
        if 'high stress' in text or 'stressed' in text:
            lifestyle['stress_level'] = 0.8
        elif 'low stress' in text:
            lifestyle['stress_level'] = 0.2
        
        # Diet
        if 'fast food' in text or 'poor diet' in text:
            lifestyle['diet_quality'] = 'poor'
        elif 'healthy' in text or 'vegetables' in text:
            lifestyle['diet_quality'] = 'good'
        
        return lifestyle


class OrganSimulator:
    """Simulate individual organ evolution"""
    
    def __init__(self, organ_name: str, initial_values: np.ndarray):
        self.organ_name = organ_name
        self.values = initial_values.copy()
        self.history = [initial_values.copy()]
    
    def step(self, time_delta: float, external_factors: Dict):
        """Simulate one timestep"""
        # Add biological drift
        drift = np.random.randn(len(self.values)) * 0.02 * time_delta
        
        # Add aging effect
        aging_effect = np.ones(len(self.values)) * 0.001 * time_delta
        
        # Add lifestyle effects
        lifestyle_effect = np.zeros(len(self.values))
        
        if self.organ_name == 'metabolic':
            # Glucose, HbA1c, BMI, Waist
            if external_factors.get('poor_diet', False):
                lifestyle_effect[0] += 0.5 * time_delta  # Glucose increases
                lifestyle_effect[1] += 0.02 * time_delta  # HbA1c increases
            if external_factors.get('no_exercise', False):
                lifestyle_effect[2] += 0.1 * time_delta  # BMI increases
        
        elif self.organ_name == 'cardiovascular':
            # BP_sys, BP_dia, LDL, HDL, Triglycerides
            if external_factors.get('high_stress', False):
                lifestyle_effect[0] += 0.5 * time_delta  # BP increases
                lifestyle_effect[1] += 0.3 * time_delta
            if external_factors.get('smoking', False):
                lifestyle_effect[2] += 0.3 * time_delta  # LDL increases
                lifestyle_effect[3] -= 0.1 * time_delta  # HDL decreases
        
        elif self.organ_name == 'liver':
            # ALT, AST
            if external_factors.get('alcohol', False):
                lifestyle_effect += 0.5 * time_delta
        
        elif self.organ_name == 'kidney':
            # Creatinine, eGFR
            if external_factors.get('high_bp', False):
                lifestyle_effect[0] += 0.01 * time_delta
                lifestyle_effect[1] -= 0.5 * time_delta
        
        # Update values
        self.values += drift + aging_effect + lifestyle_effect
        
        # Apply constraints (values can't go negative, etc.)
        self.values = np.maximum(self.values, 0.1)
        
        self.history.append(self.values.copy())


class PatientDigitalTwin:
    """Digital Twin for a single patient - DEMO VERSION"""
    
    DISEASES = [
        'diabetes', 'prediabetes', 'hypertension', 'cvd', 'ckd_stage_3',
        'nafld', 'metabolic_syndrome', 'obesity', 'dyslipidemia'
    ]
    
    def __init__(self, patient_data: PatientData):
        self.patient = patient_data
        
        # Initialize organ simulators
        self.organs = {
            'metabolic': OrganSimulator('metabolic', np.array([
                patient_data.fasting_glucose,
                patient_data.hba1c,
                patient_data.bmi,
                patient_data.bmi * 2.5  # Waist approx
            ])),
            
            'cardiovascular': OrganSimulator('cardiovascular', np.array([
                patient_data.systolic_bp,
                patient_data.diastolic_bp,
                patient_data.ldl,
                patient_data.hdl,
                patient_data.triglycerides
            ])),
            
            'liver': OrganSimulator('liver', np.array([
                patient_data.alt,
                patient_data.ast
            ])),
            
            'kidney': OrganSimulator('kidney', np.array([
                patient_data.creatinine,
                patient_data.egfr
            ])),
        }
    
    def simulate(self, years: int = 10) -> Dict:
        """Simulate patient's future health trajectory"""
        num_months = years * 12
        
        # Extract lifestyle factors
        external_factors = {
            'poor_diet': self.patient.diet_quality == 'poor',
            'no_exercise': self.patient.exercise_hours_per_week < 2,
            'smoking': self.patient.smoking,
            'alcohol': self.patient.alcohol_drinks_per_week > 14,
            'high_stress': self.patient.stress_level > 0.6,
            'high_bp': self.patient.systolic_bp > 140
        }
        
        # Simulate forward
        for month in range(num_months):
            # Update external factors based on current state
            metabolic_values = self.organs['metabolic'].values
            cardio_values = self.organs['cardiovascular'].values
            
            external_factors['high_bp'] = cardio_values[0] > 140
            external_factors['high_glucose'] = metabolic_values[0] > 126
            
            # Each organ evolves
            for organ in self.organs.values():
                organ.step(time_delta=1.0, external_factors=external_factors)
        
        # Predict diseases from final state
        disease_risks = self._predict_diseases()
        
        # Generate interventions
        interventions = self._recommend_interventions(disease_risks)
        
        return {
            'disease_risks': disease_risks,
            'interventions': interventions,
            'trajectory': {
                name: organ.history for name, organ in self.organs.items()
            }
        }
    
    def _predict_diseases(self) -> Dict[str, float]:
        """Predict disease risks from current organ states"""
        metabolic = self.organs['metabolic'].values
        cardio = self.organs['cardiovascular'].values
        liver = self.organs['liver'].values
        kidney = self.organs['kidney'].values
        
        glucose = metabolic[0]
        hba1c = metabolic[1]
        bmi = metabolic[2]
        
        bp_sys = cardio[0]
        bp_dia = cardio[1]
        ldl = cardio[2]
        hdl = cardio[3]
        trig = cardio[4]
        
        alt = liver[0]
        creat = kidney[0]
        egfr = kidney[1]
        
        risks = {}
        
        # Diabetes
        diabetes_risk = 0.0
        if glucose > 100: diabetes_risk += 0.15
        if glucose > 126: diabetes_risk += 0.35
        if hba1c > 5.7: diabetes_risk += 0.15
        if hba1c > 6.5: diabetes_risk += 0.35
        if bmi > 30: diabetes_risk += 0.20
        if self.patient.age > 45: diabetes_risk += 0.10
        risks['diabetes'] = min(diabetes_risk, 1.0)
        
        # Prediabetes
        if glucose >= 100 and glucose < 126:
            risks['prediabetes'] = 0.7
        else:
            risks['prediabetes'] = 0.1
        
        # Hypertension
        hypertension_risk = 0.0
        if bp_sys > 130: hypertension_risk += 0.25
        if bp_sys > 140: hypertension_risk += 0.40
        if bp_dia > 90: hypertension_risk += 0.20
        if bmi > 30: hypertension_risk += 0.15
        risks['hypertension'] = min(hypertension_risk, 1.0)
        
        # CVD
        cvd_risk = 0.0
        if bp_sys > 140: cvd_risk += 0.20
        if ldl > 130: cvd_risk += 0.20
        if hdl < 40: cvd_risk += 0.15
        if self.patient.smoking: cvd_risk += 0.30
        if diabetes_risk > 0.5: cvd_risk += 0.20
        if self.patient.age > 55: cvd_risk += 0.15
        risks['cvd'] = min(cvd_risk, 1.0)
        
        # CKD
        ckd_risk = 0.0
        if egfr < 60: ckd_risk += 0.50
        if creat > 1.3: ckd_risk += 0.20
        if bp_sys > 140: ckd_risk += 0.15
        if diabetes_risk > 0.5: ckd_risk += 0.20
        risks['ckd_stage_3'] = min(ckd_risk, 1.0)
        
        # NAFLD
        nafld_risk = 0.0
        if bmi > 30: nafld_risk += 0.30
        if alt > 40: nafld_risk += 0.25
        if diabetes_risk > 0.5: nafld_risk += 0.20
        if trig > 150: nafld_risk += 0.15
        risks['nafld'] = min(nafld_risk, 1.0)
        
        # Metabolic syndrome
        met_syn_count = 0
        if bmi > 30: met_syn_count += 1
        if bp_sys > 130: met_syn_count += 1
        if glucose > 100: met_syn_count += 1
        if trig > 150: met_syn_count += 1
        if hdl < 40: met_syn_count += 1
        risks['metabolic_syndrome'] = min(met_syn_count / 3.0, 1.0)
        
        # Obesity
        risks['obesity'] = 1.0 if bmi > 30 else 0.3 if bmi > 25 else 0.1
        
        # Dyslipidemia
        dyslip_risk = 0.0
        if ldl > 130: dyslip_risk += 0.40
        if hdl < 40: dyslip_risk += 0.30
        if trig > 150: dyslip_risk += 0.30
        risks['dyslipidemia'] = min(dyslip_risk, 1.0)
        
        return risks
    
    def _recommend_interventions(self, disease_risks: Dict[str, float]) -> List[Dict]:
        """Recommend interventions to reduce risks"""
        interventions = []
        
        high_risk_diseases = [d for d, r in disease_risks.items() if r > 0.3]
        
        # Weight loss
        if self.patient.bmi > 30:
            impact_diseases = [d for d in ['diabetes', 'hypertension', 'cvd', 'nafld'] 
                             if d in high_risk_diseases]
            if impact_diseases:
                interventions.append({
                    'action': 'Lose 10% body weight through diet and exercise',
                    'diseases_affected': impact_diseases,
                    'risk_reduction': '20-30%',
                    'timeframe': '6-12 months',
                    'priority': 'HIGH'
                })
        
        # Smoking cessation
        if self.patient.smoking:
            impact_diseases = [d for d in ['cvd', 'hypertension'] 
                             if d in high_risk_diseases]
            if impact_diseases:
                interventions.append({
                    'action': 'Quit smoking',
                    'diseases_affected': impact_diseases,
                    'risk_reduction': '30-40%',
                    'timeframe': 'Immediate',
                    'priority': 'CRITICAL'
                })
        
        # Exercise
        if self.patient.exercise_hours_per_week < 2.5:
            impact_diseases = [d for d in ['diabetes', 'cvd', 'hypertension'] 
                             if d in high_risk_diseases]
            if impact_diseases:
                interventions.append({
                    'action': 'Exercise 150 minutes/week (moderate intensity)',
                    'diseases_affected': impact_diseases,
                    'risk_reduction': '15-25%',
                    'timeframe': '3-6 months',
                    'priority': 'HIGH'
                })
        
        # Alcohol reduction
        if self.patient.alcohol_drinks_per_week > 14:
            interventions.append({
                'action': 'Reduce alcohol to <7 drinks/week',
                'diseases_affected': ['nafld', 'hypertension'],
                'risk_reduction': '25-35%',
                'timeframe': '1-3 months',
                'priority': 'MEDIUM'
                })
        
        # Sleep improvement
        if self.patient.sleep_hours_per_night < 6:
            interventions.append({
                'action': 'Improve sleep to 7-8 hours/night',
                'diseases_affected': ['hypertension', 'diabetes', 'cvd'],
                'risk_reduction': '10-15%',
                'timeframe': '1-2 months',
                'priority': 'MEDIUM'
            })
        
        # Medical interventions
        if disease_risks.get('hypertension', 0) > 0.6:
            interventions.append({
                'action': 'Consider antihypertensive medication',
                'diseases_affected': ['hypertension', 'cvd', 'ckd_stage_3'],
                'risk_reduction': '40-50%',
                'timeframe': 'Consult physician',
                'priority': 'HIGH'
            })
        
        if disease_risks.get('diabetes', 0) > 0.5:
            interventions.append({
                'action': 'Consider metformin for prediabetes',
                'diseases_affected': ['diabetes'],
                'risk_reduction': '30-40%',
                'timeframe': 'Consult physician',
                'priority': 'HIGH'
            })
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        interventions.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return interventions


def print_simulation_results(patient: PatientData, results: Dict):
    """Print formatted simulation results"""
    print("\n" + "=" * 80)
    print("DIGITAL TWIN SIMULATION RESULTS (10-YEAR OUTLOOK)")
    print("=" * 80)
    
    print("\n📋 PATIENT PROFILE:")
    print(f"  Age: {patient.age}, Sex: {patient.sex}")
    print(f"  BMI: {patient.bmi:.1f}, BP: {patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f}")
    print(f"  Glucose: {patient.fasting_glucose:.0f} mg/dL, HbA1c: {patient.hba1c:.1f}%")
    print(f"  Lifestyle: {'Smoker' if patient.smoking else 'Non-smoker'}, "
          f"Exercise: {patient.exercise_hours_per_week:.1f} hrs/week")
    
    print("\n🎯 DISEASE RISK PREDICTIONS:")
    risks = results['disease_risks']
    sorted_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)
    
    for disease, risk in sorted_risks:
        if risk > 0.1:
            risk_level = "🔴 HIGH" if risk > 0.5 else "🟡 MODERATE" if risk > 0.3 else "🟢 LOW"
            disease_name = disease.replace('_', ' ').title()
            print(f"  {risk_level:12s} {disease_name:25s} {risk*100:5.1f}% risk")
    
    print("\n💡 RECOMMENDED INTERVENTIONS:")
    for i, intervention in enumerate(results['interventions'][:6], 1):
        priority_icon = "🚨" if intervention['priority'] == 'CRITICAL' else "⚠️" if intervention['priority'] == 'HIGH' else "ℹ️"
        print(f"\n  {i}. {priority_icon} {intervention['action']}")
        print(f"     Affects: {', '.join(intervention['diseases_affected'])}")
        print(f"     Risk reduction: {intervention['risk_reduction']}")
        print(f"     Timeframe: {intervention['timeframe']}")


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("PATIENT DIGITAL TWIN SIMULATOR - DEMO")
    print("YOUR VISION: Natural language input → Digital twin → 10-year simulation")
    print("=" * 80)
    
    # Example: Patient described in natural language
    patient_description = """
    Sedentary office worker, smoker (1 pack/day), drinks 3 beers daily,
    sleeps only 5 hours per night, high stress job, eats fast food regularly,
    no exercise, chaotic lifestyle
    """
    
    test_results = {
        'bmi': 32.0,
        'systolic_bp': 145.0,
        'diastolic_bp': 92.0,
        'fasting_glucose': 115.0,
        'hba1c': 5.9,
        'total_cholesterol': 245.0,
        'ldl': 165.0,
        'hdl': 38.0,
        'triglycerides': 210.0,
        'creatinine': 1.1,
        'egfr': 82.0,
        'alt': 42.0,
        'ast': 38.0,
        'crp': 3.2
    }
    
    # Parse lifestyle
    lifestyle = NaturalLanguageParser.parse_lifestyle_description(patient_description)
    
    # Create patient
    patient = PatientData(
        age=45,
        sex='male',
        **lifestyle,
        **test_results
    )
    
    # Create digital twin
    print("\n🔬 Creating digital twin from patient data...")
    twin = PatientDigitalTwin(patient)
    
    # Simulate 10 years
    print("⏳ Simulating 10-year health trajectory...")
    results = twin.simulate(years=10)
    
    # Print results
    print_simulation_results(patient, results)
    
    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE!")
    print("\nThis demonstrates YOUR VISION:")
    print("  ✓ Natural language patient description")
    print("  ✓ Digital twin creation from health data")
    print("  ✓ 10-year organ simulation with interactions")
    print("  ✓ Disease risk prediction with probabilities")
    print("  ✓ Personalized intervention recommendations")
    print("\nOnce training completes, this will use the learned GNN+Transformer model")
    print("for even more accurate predictions!")
    print("=" * 80)
