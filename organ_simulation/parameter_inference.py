#!/usr/bin/env python3
"""
Missing Parameter Inference System

When patient data is incomplete, infer missing parameters using:
1. Cross-sectional patterns (age, gender, BMI correlations)
2. Medical correlations (biomarker relationships)
3. Population baselines (age/gender-matched means)
4. Reverse inference (infer lifestyle from biomarkers)
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
import pickle


class ParameterInferenceEngine:
    """
    Infer missing patient parameters from available data
    
    Uses multiple strategies:
    - Age/gender/BMI-based cross-sectional patterns
    - Biomarker correlations (e.g., glucose → insulin)
    - Reverse inference (e.g., high ALT → high alcohol)
    - Population baselines with uncertainty
    """
    
    def __init__(self, cross_sectional_patterns_path: str = './models/cross_sectional_patterns.pkl'):
        # Load cross-sectional patterns
        try:
            with open(cross_sectional_patterns_path, 'rb') as f:
                self.cs_patterns = pickle.load(f)
        except:
            self.cs_patterns = {}
        
        # Population baselines (age/gender-matched)
        self.population_baselines = self._initialize_baselines()
        
        # Biomarker correlations from medical literature
        self.medical_correlations = self._initialize_correlations()
    
    def _initialize_baselines(self) -> Dict:
        """Initialize population baseline values"""
        return {
            # Metabolic
            'glucose': {'mean': 95, 'std': 10, 'unit': 'mg/dL'},
            'HbA1c': {'mean': 5.4, 'std': 0.3, 'unit': '%'},
            'insulin': {'mean': 12, 'std': 5, 'unit': 'μU/mL'},
            'triglycerides': {'mean': 120, 'std': 40, 'unit': 'mg/dL'},
            
            # Cardiovascular
            'systolic_bp': {'mean': 120, 'std': 10, 'unit': 'mmHg', 'age_slope': 0.19},
            'diastolic_bp': {'mean': 80, 'std': 8, 'unit': 'mmHg'},
            'total_cholesterol': {'mean': 190, 'std': 30, 'unit': 'mg/dL'},
            'HDL': {'mean': 50, 'std': 12, 'unit': 'mg/dL'},
            'LDL': {'mean': 115, 'std': 30, 'unit': 'mg/dL'},
            
            # Liver
            'ALT': {'mean': 25, 'std': 10, 'unit': 'U/L'},
            'AST': {'mean': 22, 'std': 8, 'unit': 'U/L'},
            
            # Kidney
            'creatinine': {'mean': 1.0, 'std': 0.2, 'unit': 'mg/dL'},
            'BUN': {'mean': 15, 'std': 5, 'unit': 'mg/dL'},
            
            # Immune
            'WBC': {'mean': 7.0, 'std': 2.0, 'unit': 'K/μL'},
            
            # Neural
            'cognitive_score': {'mean': 0.85, 'std': 0.1, 'unit': 'normalized'},
            
            # Lifestyle
            'exercise_frequency': {'mean': 0.4, 'std': 0.2, 'unit': 'normalized'},
            'alcohol_consumption': {'mean': 0.3, 'std': 0.2, 'unit': 'normalized'},
            'diet_quality': {'mean': 0.5, 'std': 0.15, 'unit': 'normalized'},
            'sleep_hours': {'mean': 7.0, 'std': 1.0, 'unit': 'hours'},
        }
    
    def _initialize_correlations(self) -> Dict:
        """Initialize medical correlations between biomarkers"""
        return {
            # Glucose → Insulin (insulin resistance)
            'glucose_insulin': {
                'slope': 0.5,  # Insulin increases 0.5 μU/mL per mg/dL glucose above 100
                'threshold': 100
            },
            
            # Glucose → Triglycerides
            'glucose_triglycerides': {
                'slope': 1.5,  # Triglycerides increase 1.5 mg/dL per mg/dL glucose above 100
                'threshold': 100
            },
            
            # BMI → Glucose
            'bmi_glucose': {
                'slope': 2.0,  # Glucose increases 2 mg/dL per BMI point above 25
                'threshold': 25
            },
            
            # BMI → Systolic BP
            'bmi_bp': {
                'slope': 1.5,  # BP increases 1.5 mmHg per BMI point above 25
                'threshold': 25
            },
            
            # Age → Systolic BP
            'age_bp': {
                'slope': 0.19,  # From cross-sectional analysis
                'baseline_age': 40
            },
            
            # Alcohol → ALT (reverse inference)
            'alt_alcohol': {
                'threshold': 40,  # ALT >40 suggests alcohol
                'slope': 0.01  # Alcohol score increases 0.01 per U/L above 40
            },
            
            # Glucose → Diet Quality (reverse inference)
            'glucose_diet': {
                'threshold': 100,
                'slope': -0.02  # Diet quality decreases 0.02 per mg/dL above 100
            },
            
            # BP → Exercise (reverse inference)
            'bp_exercise': {
                'threshold': 120,
                'slope': -0.01  # Exercise decreases 0.01 per mmHg above 120
            }
        }
    
    def infer_complete_profile(
        self,
        partial_profile: Dict,
        demographics: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Infer missing parameters from partial patient data
        
        Args:
            partial_profile: Available biomarkers (may be incomplete)
            demographics: Age, gender, BMI, etc.
        
        Returns:
            complete_profile: Full biomarker profile
            inference_notes: Explanation of what was inferred
        """
        complete_profile = {}
        inference_notes = {}
        
        age = demographics.get('age', 40)
        gender = demographics.get('gender', 'male')
        bmi = demographics.get('bmi', 25)
        
        # 1. Copy available parameters
        for organ, biomarkers in partial_profile.items():
            if organ not in complete_profile:
                complete_profile[organ] = {}
            for param, value in biomarkers.items():
                complete_profile[organ][param] = value
                inference_notes[f"{organ}.{param}"] = "Provided by user"
        
        # 2. Infer metabolic parameters
        complete_profile['metabolic'] = self._infer_metabolic(
            partial_profile.get('metabolic', {}),
            demographics,
            inference_notes
        )
        
        # 3. Infer cardiovascular parameters
        complete_profile['cardiovascular'] = self._infer_cardiovascular(
            partial_profile.get('cardiovascular', {}),
            demographics,
            inference_notes
        )
        
        # 4. Infer liver parameters
        complete_profile['liver'] = self._infer_liver(
            partial_profile.get('liver', {}),
            partial_profile.get('lifestyle', {}),
            demographics,
            inference_notes
        )
        
        # 5. Infer kidney parameters
        complete_profile['kidney'] = self._infer_kidney(
            partial_profile.get('kidney', {}),
            demographics,
            complete_profile.get('cardiovascular', {}),
            inference_notes
        )
        
        # 6. Infer immune parameters
        complete_profile['immune'] = self._infer_immune(
            partial_profile.get('immune', {}),
            demographics,
            inference_notes
        )
        
        # 7. Infer neural parameters
        complete_profile['neural'] = self._infer_neural(
            partial_profile.get('neural', {}),
            demographics,
            complete_profile.get('cardiovascular', {}),
            inference_notes
        )
        
        # 8. Infer lifestyle (reverse inference from biomarkers)
        complete_profile['lifestyle'] = self._infer_lifestyle(
            partial_profile.get('lifestyle', {}),
            complete_profile,
            demographics,
            inference_notes
        )
        
        return complete_profile, inference_notes
    
    def _infer_metabolic(self, metabolic: Dict, demographics: Dict, notes: Dict) -> Dict:
        """Infer missing metabolic parameters"""
        result = metabolic.copy()
        age = demographics.get('age', 40)
        bmi = demographics.get('bmi', 25)
        
        # Glucose
        if 'glucose' not in result:
            # Age effect + BMI effect
            base_glucose = self.population_baselines['glucose']['mean']
            age_effect = (age - 40) * 0.18  # From cross-sectional analysis
            bmi_effect = max(0, (bmi - 25) * 2.0)  # BMI correlation
            result['glucose'] = base_glucose + age_effect + bmi_effect
            notes['metabolic.glucose'] = f"Inferred from age ({age}) and BMI ({bmi:.1f})"
        
        # HbA1c (follows glucose)
        if 'HbA1c' not in result:
            glucose = result.get('glucose', 95)
            # HbA1c ≈ (glucose + 46.7) / 28.7 (rough conversion)
            result['HbA1c'] = (glucose + 46.7) / 28.7
            notes['metabolic.HbA1c'] = f"Inferred from glucose ({glucose:.1f} mg/dL)"
        
        # Insulin (from glucose + BMI)
        if 'insulin' not in result:
            glucose = result.get('glucose', 95)
            base_insulin = self.population_baselines['insulin']['mean']
            if glucose > 100:
                insulin_resistance = (glucose - 100) * 0.5
                result['insulin'] = base_insulin + insulin_resistance
            else:
                result['insulin'] = base_insulin
            notes['metabolic.insulin'] = f"Inferred from glucose and insulin resistance"
        
        # Triglycerides (from glucose + BMI)
        if 'triglycerides' not in result:
            glucose = result.get('glucose', 95)
            base_trig = self.population_baselines['triglycerides']['mean']
            if glucose > 100:
                trig_increase = (glucose - 100) * 1.5
                result['triglycerides'] = base_trig + trig_increase
            else:
                result['triglycerides'] = base_trig
            notes['metabolic.triglycerides'] = f"Inferred from glucose and metabolic status"
        
        return result
    
    def _infer_cardiovascular(self, cv: Dict, demographics: Dict, notes: Dict) -> Dict:
        """Infer missing cardiovascular parameters"""
        result = cv.copy()
        age = demographics.get('age', 40)
        bmi = demographics.get('bmi', 25)
        
        # Systolic BP
        if 'systolic_bp' not in result:
            base_bp = self.population_baselines['systolic_bp']['mean']
            age_effect = (age - 40) * 0.19
            bmi_effect = max(0, (bmi - 25) * 1.5)
            result['systolic_bp'] = base_bp + age_effect + bmi_effect
            notes['cardiovascular.systolic_bp'] = f"Inferred from age ({age}) and BMI ({bmi:.1f})"
        
        # Diastolic BP (follows systolic)
        if 'diastolic_bp' not in result:
            systolic = result.get('systolic_bp', 120)
            result['diastolic_bp'] = systolic * 0.67  # Typical ratio
            notes['cardiovascular.diastolic_bp'] = f"Inferred from systolic BP ({systolic:.1f} mmHg)"
        
        # Total Cholesterol
        if 'total_cholesterol' not in result:
            base_chol = self.population_baselines['total_cholesterol']['mean']
            age_effect = (age - 40) * 0.5
            result['total_cholesterol'] = base_chol + age_effect
            notes['cardiovascular.total_cholesterol'] = f"Inferred from age-matched population baseline"
        
        # HDL
        if 'HDL' not in result:
            result['HDL'] = self.population_baselines['HDL']['mean']
            notes['cardiovascular.HDL'] = "Inferred from population baseline (data quality issue)"
        
        # LDL
        if 'LDL' not in result:
            total_chol = result.get('total_cholesterol', 190)
            hdl = result.get('HDL', 50)
            # LDL ≈ Total - HDL - (Triglycerides/5)
            result['LDL'] = total_chol - hdl - 20  # Assuming normal triglycerides
            notes['cardiovascular.LDL'] = f"Calculated from total cholesterol and HDL"
        
        return result
    
    def _infer_liver(self, liver: Dict, lifestyle: Dict, demographics: Dict, notes: Dict) -> Dict:
        """Infer missing liver parameters"""
        result = liver.copy()
        age = demographics.get('age', 40)
        bmi = demographics.get('bmi', 25)
        
        # ALT (from alcohol + BMI)
        if 'ALT' not in result:
            base_alt = self.population_baselines['ALT']['mean']
            
            # Alcohol effect
            alcohol = lifestyle.get('alcohol_consumption', 0.3)
            if alcohol > 0.5:
                alcohol_effect = (alcohol - 0.5) * 40  # Heavy drinking increases ALT
            else:
                alcohol_effect = 0
            
            # BMI effect (fatty liver)
            if bmi > 25:
                bmi_effect = (bmi - 25) * 2
            else:
                bmi_effect = 0
            
            result['ALT'] = base_alt + alcohol_effect + bmi_effect
            notes['liver.ALT'] = f"Inferred from alcohol ({alcohol:.1f}), BMI ({bmi:.1f})"
        
        # AST (follows ALT)
        if 'AST' not in result:
            alt = result.get('ALT', 25)
            result['AST'] = alt * 0.7  # Typical AST/ALT ratio
            notes['liver.AST'] = f"Inferred from ALT ({alt:.1f} U/L)"
        
        return result
    
    def _infer_kidney(self, kidney: Dict, demographics: Dict, cv: Dict, notes: Dict) -> Dict:
        """Infer missing kidney parameters"""
        result = kidney.copy()
        age = demographics.get('age', 40)
        gender = demographics.get('gender', 'male')
        
        # Creatinine (depends on muscle mass, age, gender)
        if 'creatinine' not in result:
            if gender == 'male':
                base_creat = 1.0
            else:
                base_creat = 0.8
            
            # Age effect (kidney function declines)
            if age > 50:
                age_effect = (age - 50) * 0.01
            else:
                age_effect = 0
            
            # BP effect (hypertension damages kidneys)
            systolic = cv.get('systolic_bp', 120)
            if systolic > 140:
                bp_effect = (systolic - 140) * 0.005
            else:
                bp_effect = 0
            
            result['creatinine'] = base_creat + age_effect + bp_effect
            notes['kidney.creatinine'] = f"Inferred from age ({age}), gender ({gender}), BP"
        
        # BUN
        if 'BUN' not in result:
            result['BUN'] = self.population_baselines['BUN']['mean']
            notes['kidney.BUN'] = "Inferred from population baseline"
        
        return result
    
    def _infer_immune(self, immune: Dict, demographics: Dict, notes: Dict) -> Dict:
        """Infer missing immune parameters"""
        result = immune.copy()
        
        if 'WBC' not in result:
            result['WBC'] = self.population_baselines['WBC']['mean']
            notes['immune.WBC'] = "Inferred from population baseline (data quality issue)"
        
        return result
    
    def _infer_neural(self, neural: Dict, demographics: Dict, cv: Dict, notes: Dict) -> Dict:
        """Infer missing neural parameters"""
        result = neural.copy()
        age = demographics.get('age', 40)
        
        if 'cognitive_score' not in result:
            base_score = 0.95
            
            # Age effect
            if age > 60:
                age_effect = (age - 60) * -0.01
            else:
                age_effect = 0
            
            # Vascular health effect
            systolic = cv.get('systolic_bp', 120)
            if systolic > 140:
                bp_effect = (systolic - 140) * -0.002
            else:
                bp_effect = 0
            
            result['cognitive_score'] = max(0.4, base_score + age_effect + bp_effect)
            notes['neural.cognitive_score'] = f"Inferred from age ({age}) and vascular health"
        
        return result
    
    def _infer_lifestyle(self, lifestyle: Dict, biomarkers: Dict, demographics: Dict, notes: Dict) -> Dict:
        """Infer lifestyle from biomarkers (reverse inference)"""
        result = lifestyle.copy()
        
        # Alcohol (from ALT)
        if 'alcohol_consumption' not in result:
            alt = biomarkers.get('liver', {}).get('ALT', 25)
            if alt > 40:
                # High ALT suggests alcohol consumption
                result['alcohol_consumption'] = min(1.0, 0.3 + (alt - 40) * 0.01)
                notes['lifestyle.alcohol_consumption'] = f"Reverse-inferred from elevated ALT ({alt:.1f} U/L)"
            else:
                result['alcohol_consumption'] = 0.2
                notes['lifestyle.alcohol_consumption'] = "Inferred from normal liver enzymes"
        
        # Diet quality (from glucose + triglycerides)
        if 'diet_quality' not in result:
            glucose = biomarkers.get('metabolic', {}).get('glucose', 95)
            trig = biomarkers.get('metabolic', {}).get('triglycerides', 120)
            
            if glucose > 100 or trig > 150:
                result['diet_quality'] = max(0.2, 0.7 - (glucose - 100) * 0.02)
                notes['lifestyle.diet_quality'] = f"Reverse-inferred from glucose ({glucose:.1f}) and triglycerides"
            else:
                result['diet_quality'] = 0.6
                notes['lifestyle.diet_quality'] = "Inferred from normal metabolic markers"
        
        # Exercise (from BP + BMI)
        if 'exercise_frequency' not in result:
            bp = biomarkers.get('cardiovascular', {}).get('systolic_bp', 120)
            bmi = demographics.get('bmi', 25)
            
            if bp > 130 or bmi > 27:
                result['exercise_frequency'] = max(0.1, 0.5 - (bp - 120) * 0.01)
                notes['lifestyle.exercise_frequency'] = f"Reverse-inferred from BP ({bp:.1f}) and BMI ({bmi:.1f})"
            else:
                result['exercise_frequency'] = 0.5
                notes['lifestyle.exercise_frequency'] = "Inferred from cardiovascular health"
        
        # Sleep
        if 'sleep_hours' not in result:
            result['sleep_hours'] = 7.0
            notes['lifestyle.sleep_hours'] = "Assumed population average (7 hours)"
        
        return result


def test_inference():
    """Test parameter inference with incomplete patient data"""
    print("="*80)
    print("PARAMETER INFERENCE TEST")
    print("="*80)
    
    engine = ParameterInferenceEngine()
    
    # Test case: Minimal patient data
    partial_profile = {
        'metabolic': {
            'glucose': 115  # Only glucose provided
        },
        'cardiovascular': {
            'systolic_bp': 145  # Only BP provided
        },
        'liver': {
            'ALT': 65  # Elevated ALT
        }
        # No kidney, immune, neural, lifestyle data
    }
    
    demographics = {
        'age': 45,
        'gender': 'male',
        'bmi': 29.5
    }
    
    print("\nINPUT (Incomplete):")
    print(f"Demographics: Age={demographics['age']}, Gender={demographics['gender']}, BMI={demographics['bmi']}")
    print(f"Metabolic: glucose={partial_profile['metabolic']['glucose']} mg/dL")
    print(f"Cardiovascular: systolic_bp={partial_profile['cardiovascular']['systolic_bp']} mmHg")
    print(f"Liver: ALT={partial_profile['liver']['ALT']} U/L")
    print("(All other parameters missing)")
    
    # Infer complete profile
    complete_profile, inference_notes = engine.infer_complete_profile(partial_profile, demographics)
    
    print("\n" + "="*80)
    print("OUTPUT (Complete Profile with Inferences):")
    print("="*80)
    
    for organ, biomarkers in complete_profile.items():
        print(f"\n{organ.upper()}:")
        for param, value in biomarkers.items():
            note = inference_notes.get(f"{organ}.{param}", "Unknown")
            if isinstance(value, float):
                print(f"  {param}: {value:.2f} - {note}")
            else:
                print(f"  {param}: {value} - {note}")


if __name__ == '__main__':
    test_inference()
