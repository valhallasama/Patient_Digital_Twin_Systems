"""
Comprehensive Disease Label Extraction
Extracts labels for ALL diseases that can be predicted from NHANES data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class ComprehensiveDiseaseLabeler:
    """Extract disease labels for multi-disease prediction"""
    
    def __init__(self):
        self.disease_definitions = self._create_disease_definitions()
    
    def _create_disease_definitions(self) -> Dict:
        """Define clinical criteria for all diseases"""
        return {
            # METABOLIC DISEASES
            'diabetes': {
                'criteria': [
                    ('hba1c', '>=', 6.5),
                    ('fasting_glucose', '>=', 126),
                    ('doctor_told_diabetes', '==', True),
                    ('taking_insulin', '==', True),
                    ('taking_diabetes_pills', '==', True)
                ],
                'logic': 'any'
            },
            'prediabetes': {
                'criteria': [
                    ('hba1c', 'between', (5.7, 6.4)),
                    ('fasting_glucose', 'between', (100, 125))
                ],
                'logic': 'any',
                'exclude_if': ['diabetes']
            },
            'metabolic_syndrome': {
                'criteria': [
                    ('waist_circumference', '>', {'M': 102, 'F': 88}),
                    ('triglycerides', '>=', 150),
                    ('hdl', '<', {'M': 40, 'F': 50}),
                    ('systolic_bp', '>=', 130),
                    ('fasting_glucose', '>=', 100)
                ],
                'logic': 'count >= 3'
            },
            'obesity': {
                'criteria': [('bmi', '>=', 30)],
                'logic': 'any'
            },
            
            # CARDIOVASCULAR DISEASES
            'hypertension': {
                'criteria': [
                    ('systolic_bp', '>=', 140),
                    ('diastolic_bp', '>=', 90),
                    ('doctor_told_hypertension', '==', True),
                    ('taking_bp_medication', '==', True)
                ],
                'logic': 'any'
            },
            'prehypertension': {
                'criteria': [
                    ('systolic_bp', 'between', (120, 139)),
                    ('diastolic_bp', 'between', (80, 89))
                ],
                'logic': 'any',
                'exclude_if': ['hypertension']
            },
            'coronary_heart_disease': {
                'criteria': [
                    ('doctor_told_chd', '==', True),
                    ('doctor_told_heart_attack', '==', True),
                    ('doctor_told_angina', '==', True)
                ],
                'logic': 'any'
            },
            'heart_failure': {
                'criteria': [('doctor_told_chf', '==', True)],
                'logic': 'any'
            },
            'stroke': {
                'criteria': [('doctor_told_stroke', '==', True)],
                'logic': 'any'
            },
            'dyslipidemia': {
                'criteria': [
                    ('total_cholesterol', '>=', 240),
                    ('ldl', '>=', 160),
                    ('hdl', '<', 40),
                    ('triglycerides', '>=', 200)
                ],
                'logic': 'any'
            },
            
            # KIDNEY DISEASES
            'chronic_kidney_disease': {
                'criteria': [
                    ('egfr', '<', 60),
                    ('albumin_creatinine_ratio', '>=', 30),
                    ('doctor_told_kidney_disease', '==', True)
                ],
                'logic': 'any'
            },
            'ckd_stage_3': {
                'criteria': [('egfr', 'between', (30, 59))],
                'logic': 'any'
            },
            'ckd_stage_4': {
                'criteria': [('egfr', 'between', (15, 29))],
                'logic': 'any'
            },
            'ckd_stage_5': {
                'criteria': [('egfr', '<', 15)],
                'logic': 'any'
            },
            
            # LIVER DISEASES
            'nafld': {
                'criteria': [
                    ('alt', '>', 40),
                    ('ast', '>', 40),
                    ('bmi', '>=', 25)
                ],
                'logic': 'all',
                'exclude_if': ['heavy_drinker']
            },
            'elevated_liver_enzymes': {
                'criteria': [
                    ('alt', '>', 40),
                    ('ast', '>', 40)
                ],
                'logic': 'any'
            },
            'liver_disease': {
                'criteria': [('doctor_told_liver_condition', '==', True)],
                'logic': 'any'
            },
            
            # CANCER
            'cancer_any': {
                'criteria': [('doctor_told_cancer', '==', True)],
                'logic': 'any'
            },
            
            # RESPIRATORY
            'copd': {
                'criteria': [
                    ('doctor_told_copd', '==', True),
                    ('doctor_told_emphysema', '==', True),
                    ('doctor_told_chronic_bronchitis', '==', True)
                ],
                'logic': 'any'
            },
            'asthma': {
                'criteria': [('doctor_told_asthma', '==', True)],
                'logic': 'any'
            },
            
            # ANEMIA
            'anemia': {
                'criteria': [
                    ('hemoglobin', '<', {'M': 13.5, 'F': 12.0}),
                    ('hematocrit', '<', {'M': 39, 'F': 36})
                ],
                'logic': 'any'
            },
            
            # THYROID
            'hypothyroidism': {
                'criteria': [('doctor_told_thyroid_problem', '==', True)],
                'logic': 'any'
            },
            
            # INFLAMMATION
            'chronic_inflammation': {
                'criteria': [('crp', '>', 3.0)],
                'logic': 'any'
            },
            
            # COMPOSITE CVD RISK
            'high_cvd_risk': {
                'criteria': [
                    ('age', '>=', 65),
                    ('smoking', '==', True),
                    ('diabetes', '==', True),
                    ('hypertension', '==', True),
                    ('dyslipidemia', '==', True)
                ],
                'logic': 'count >= 2'
            }
        }
    
    def extract_disease_label(self, patient_data: Dict, disease: str) -> bool:
        """
        Extract disease label for a single disease
        
        Args:
            patient_data: Patient feature dictionary
            disease: Disease name
            
        Returns:
            True if patient has disease, False otherwise
        """
        if disease not in self.disease_definitions:
            return False
        
        definition = self.disease_definitions[disease]
        
        # Check exclusions first
        if 'exclude_if' in definition:
            for exclude_disease in definition['exclude_if']:
                if self.extract_disease_label(patient_data, exclude_disease):
                    return False
        
        criteria = definition['criteria']
        logic = definition['logic']
        
        matches = []
        
        for criterion in criteria:
            if len(criterion) == 3:
                feature, operator, threshold = criterion
                
                if feature not in patient_data or patient_data[feature] is None:
                    matches.append(False)
                    continue
                
                value = patient_data[feature]
                
                # Handle sex-specific thresholds
                if isinstance(threshold, dict):
                    sex = patient_data.get('sex', 'M')
                    threshold = threshold.get(sex, threshold.get('M'))
                
                # Evaluate criterion
                if operator == '>=':
                    matches.append(value >= threshold)
                elif operator == '>':
                    matches.append(value > threshold)
                elif operator == '<=':
                    matches.append(value <= threshold)
                elif operator == '<':
                    matches.append(value < threshold)
                elif operator == '==':
                    matches.append(value == threshold)
                elif operator == 'between':
                    matches.append(threshold[0] <= value <= threshold[1])
        
        # Apply logic
        if logic == 'any':
            return any(matches) if matches else False
        elif logic == 'all':
            return all(matches) if matches else False
        elif logic.startswith('count >='):
            count_threshold = int(logic.split('>=')[1].strip())
            return sum(matches) >= count_threshold
        
        return False
    
    def extract_all_disease_labels(self, patient_data: Dict) -> Dict[str, bool]:
        """
        Extract all disease labels for a patient
        
        Args:
            patient_data: Patient feature dictionary
            
        Returns:
            Dictionary of disease labels
        """
        labels = {}
        
        for disease in self.disease_definitions.keys():
            labels[disease] = self.extract_disease_label(patient_data, disease)
        
        return labels
    
    def get_disease_count(self, patient_data: Dict) -> int:
        """Count number of diseases patient has"""
        labels = self.extract_all_disease_labels(patient_data)
        return sum(labels.values())
    
    def get_disease_list(self, patient_data: Dict) -> List[str]:
        """Get list of diseases patient has"""
        labels = self.extract_all_disease_labels(patient_data)
        return [disease for disease, has_disease in labels.items() if has_disease]
    
    def calculate_disease_prevalence(self, cohort: List[Dict]) -> Dict[str, float]:
        """
        Calculate disease prevalence in a cohort
        
        Args:
            cohort: List of patient dictionaries
            
        Returns:
            Dictionary of disease prevalence (proportion)
        """
        disease_counts = {disease: 0 for disease in self.disease_definitions.keys()}
        
        for patient in cohort:
            labels = self.extract_all_disease_labels(patient)
            for disease, has_disease in labels.items():
                if has_disease:
                    disease_counts[disease] += 1
        
        n = len(cohort)
        prevalence = {disease: count / n for disease, count in disease_counts.items()}
        
        return prevalence
    
    def get_disease_categories(self) -> Dict[str, List[str]]:
        """Group diseases by category"""
        return {
            'Metabolic': ['diabetes', 'prediabetes', 'metabolic_syndrome', 'obesity'],
            'Cardiovascular': ['hypertension', 'prehypertension', 'coronary_heart_disease', 
                              'heart_failure', 'stroke', 'dyslipidemia', 'high_cvd_risk'],
            'Kidney': ['chronic_kidney_disease', 'ckd_stage_3', 'ckd_stage_4', 'ckd_stage_5'],
            'Liver': ['nafld', 'elevated_liver_enzymes', 'liver_disease'],
            'Respiratory': ['copd', 'asthma'],
            'Hematologic': ['anemia'],
            'Endocrine': ['hypothyroidism'],
            'Inflammatory': ['chronic_inflammation'],
            'Oncologic': ['cancer_any']
        }
    
    def get_all_diseases(self) -> List[str]:
        """Get list of all diseases"""
        return list(self.disease_definitions.keys())


if __name__ == '__main__':
    labeler = ComprehensiveDiseaseLabeler()
    
    print("Comprehensive Disease Labeling System")
    print("=" * 80)
    
    categories = labeler.get_disease_categories()
    
    print(f"\nTotal diseases: {len(labeler.get_all_diseases())}")
    print(f"\nDisease categories:")
    
    for category, diseases in categories.items():
        print(f"\n{category} ({len(diseases)} diseases):")
        for disease in diseases:
            print(f"  - {disease}")
    
    # Test example
    print("\n" + "=" * 80)
    print("Example Patient:")
    test_patient = {
        'age': 55,
        'sex': 'M',
        'bmi': 32,
        'hba1c': 6.8,
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'ldl': 165,
        'egfr': 55,
        'alt': 45,
        'smoking': True
    }
    
    labels = labeler.extract_all_disease_labels(test_patient)
    disease_list = labeler.get_disease_list(test_patient)
    
    print(f"\nPatient has {len(disease_list)} diseases:")
    for disease in disease_list:
        print(f"  ✓ {disease}")
