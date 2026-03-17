#!/usr/bin/env python3
"""
Data Harmonizer
Standardizes patient data from different sources (MIMIC-IV, NHANES, Synthea, manual input)
into a unified format for the digital twin system
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataHarmonizer:
    """Harmonize patient data from multiple sources into standard format"""
    
    # Standard feature names and units
    STANDARD_FEATURES = {
        # Demographics
        'patient_id': str,
        'age': int,
        'sex': str,  # 'M' or 'F'
        'race': str,
        
        # Anthropometrics
        'height': float,  # cm
        'weight': float,  # kg
        'bmi': float,  # kg/m²
        'waist_circumference': float,  # cm
        
        # Metabolic
        'fasting_glucose': float,  # mg/dL
        'hba1c': float,  # %
        'insulin': float,  # μU/mL
        
        # Cardiovascular
        'systolic_bp': float,  # mmHg
        'diastolic_bp': float,  # mmHg
        'heart_rate': float,  # bpm
        
        # Lipids
        'total_cholesterol': float,  # mg/dL
        'ldl': float,  # mg/dL
        'hdl': float,  # mg/dL
        'triglycerides': float,  # mg/dL
        
        # Liver
        'alt': float,  # U/L
        'ast': float,  # U/L
        'bilirubin': float,  # mg/dL
        
        # Kidney
        'creatinine': float,  # mg/dL
        'egfr': float,  # mL/min/1.73m²
        'bun': float,  # mg/dL
        
        # Inflammation
        'crp': float,  # mg/L
        
        # Lifestyle
        'physical_activity': str,  # 'sedentary', 'moderate', 'vigorous'
        'diet_quality': str,  # 'poor', 'fair', 'good', 'excellent'
        'smoking': bool,
        'alcohol_per_week': float,  # drinks
        'sleep_hours': float,  # hours
        'stress_level': str,  # 'low', 'moderate', 'high'
        
        # Medical history
        'has_diabetes': bool,
        'has_hypertension': bool,
        'has_cvd': bool,
        'has_ckd': bool,
        'family_history_diabetes': bool,
        'family_history_cvd': bool,
    }
    
    def harmonize(self, data: Dict, source: str = 'unknown') -> Dict:
        """
        Harmonize patient data to standard format
        
        Args:
            data: Raw patient data dictionary
            source: Data source ('mimic', 'nhanes', 'synthea', 'manual')
        
        Returns:
            Standardized patient data dictionary
        """
        if source == 'mimic':
            return self._harmonize_mimic(data)
        elif source == 'nhanes':
            return self._harmonize_nhanes(data)
        elif source == 'synthea':
            return self._harmonize_synthea(data)
        else:
            return self._harmonize_manual(data)
    
    def _harmonize_mimic(self, data: Dict) -> Dict:
        """Harmonize MIMIC-IV data"""
        harmonized = {}
        
        # Direct mappings
        harmonized['patient_id'] = f"MIMIC_{data.get('subject_id', 'UNKNOWN')}"
        harmonized['age'] = data.get('age', 0)
        harmonized['sex'] = 'M' if data.get('sex') == 'M' else 'F'
        
        # Labs (MIMIC uses different units sometimes)
        harmonized['fasting_glucose'] = data.get('glucose')
        harmonized['hba1c'] = data.get('hba1c')
        harmonized['creatinine'] = data.get('creatinine')
        harmonized['alt'] = data.get('alt')
        harmonized['ast'] = data.get('ast')
        harmonized['ldl'] = data.get('ldl')
        harmonized['hdl'] = data.get('hdl')
        harmonized['triglycerides'] = data.get('triglycerides')
        harmonized['crp'] = data.get('crp')
        
        # Vitals
        harmonized['systolic_bp'] = data.get('systolic_bp')
        harmonized['diastolic_bp'] = data.get('diastolic_bp')
        harmonized['heart_rate'] = data.get('heart_rate')
        
        # Calculate derived features
        if 'height' in data and 'weight' in data:
            harmonized['height'] = data['height']
            harmonized['weight'] = data['weight']
            harmonized['bmi'] = self._calculate_bmi(data['weight'], data['height'])
        
        # Calculate eGFR if creatinine available
        if harmonized.get('creatinine'):
            harmonized['egfr'] = self._calculate_egfr(
                harmonized['creatinine'],
                harmonized['age'],
                harmonized['sex']
            )
        
        # Diagnoses to boolean flags
        diagnoses = data.get('diagnoses', [])
        harmonized['has_diabetes'] = any('diabetes' in d.lower() for d in diagnoses)
        harmonized['has_hypertension'] = any('hypertension' in d.lower() for d in diagnoses)
        harmonized['has_cvd'] = any(
            any(term in d.lower() for term in ['coronary', 'myocardial', 'heart disease'])
            for d in diagnoses
        )
        harmonized['has_ckd'] = any(
            any(term in d.lower() for term in ['kidney', 'renal'])
            for d in diagnoses
        )
        
        return self._fill_defaults(harmonized)
    
    def _harmonize_nhanes(self, data: Dict) -> Dict:
        """Harmonize NHANES data"""
        harmonized = {}
        
        # Direct mappings (NHANES already uses standard units mostly)
        harmonized['patient_id'] = data.get('patient_id', 'NHANES_UNKNOWN')
        harmonized['age'] = data.get('age', 0)
        harmonized['sex'] = data.get('sex', 'M')
        harmonized['race'] = data.get('race', 'Unknown')
        
        # Anthropometrics
        harmonized['height'] = data.get('height')
        harmonized['weight'] = data.get('weight')
        harmonized['bmi'] = data.get('bmi')
        harmonized['waist_circumference'] = data.get('waist_circumference')
        
        # Labs
        harmonized['fasting_glucose'] = data.get('fasting_glucose')
        harmonized['hba1c'] = data.get('hba1c')
        harmonized['creatinine'] = data.get('creatinine')
        harmonized['alt'] = data.get('alt')
        harmonized['ast'] = data.get('ast')
        harmonized['total_cholesterol'] = data.get('total_cholesterol')
        harmonized['ldl'] = data.get('ldl')
        harmonized['hdl'] = data.get('hdl')
        harmonized['triglycerides'] = data.get('triglycerides')
        harmonized['crp'] = data.get('crp')
        
        # Vitals
        harmonized['systolic_bp'] = data.get('systolic_bp')
        harmonized['diastolic_bp'] = data.get('diastolic_bp')
        
        # Lifestyle
        harmonized['physical_activity'] = data.get('physical_activity', 'sedentary')
        harmonized['smoking'] = data.get('smoking', False)
        harmonized['alcohol_per_week'] = data.get('alcohol_per_week', 0)
        
        # Calculate eGFR
        if harmonized.get('creatinine'):
            harmonized['egfr'] = self._calculate_egfr(
                harmonized['creatinine'],
                harmonized['age'],
                harmonized['sex']
            )
        
        # Infer disease status from lab values
        harmonized['has_diabetes'] = (
            harmonized.get('hba1c', 0) >= 6.5 or
            harmonized.get('fasting_glucose', 0) >= 126
        )
        harmonized['has_hypertension'] = (
            harmonized.get('systolic_bp', 0) >= 140 or
            harmonized.get('diastolic_bp', 0) >= 90
        )
        
        return self._fill_defaults(harmonized)
    
    def _harmonize_synthea(self, data: Dict) -> Dict:
        """Harmonize Synthea synthetic data"""
        # Synthea typically follows FHIR format
        harmonized = {}
        
        # Extract from FHIR-like structure
        harmonized['patient_id'] = data.get('id', 'SYNTHEA_UNKNOWN')
        
        # Demographics
        if 'birthDate' in data:
            from datetime import datetime
            birth_year = int(data['birthDate'][:4])
            harmonized['age'] = datetime.now().year - birth_year
        
        harmonized['sex'] = 'M' if data.get('gender') == 'male' else 'F'
        
        # Observations (labs and vitals)
        observations = data.get('observations', [])
        for obs in observations:
            code = obs.get('code', {}).get('coding', [{}])[0].get('code', '')
            value = obs.get('valueQuantity', {}).get('value')
            
            # Map LOINC codes to features
            if code == '2339-0':  # Glucose
                harmonized['fasting_glucose'] = value
            elif code == '4548-4':  # HbA1c
                harmonized['hba1c'] = value
            elif code == '2160-0':  # Creatinine
                harmonized['creatinine'] = value
            elif code == '8480-6':  # Systolic BP
                harmonized['systolic_bp'] = value
            elif code == '8462-4':  # Diastolic BP
                harmonized['diastolic_bp'] = value
            # Add more LOINC mappings as needed
        
        return self._fill_defaults(harmonized)
    
    def _harmonize_manual(self, data: Dict) -> Dict:
        """Harmonize manually entered data"""
        # Assume data is already in reasonable format, just validate
        harmonized = {}
        
        for key, value in data.items():
            if key in self.STANDARD_FEATURES:
                harmonized[key] = value
        
        # Calculate derived features if possible
        if 'height' in harmonized and 'weight' in harmonized and 'bmi' not in harmonized:
            harmonized['bmi'] = self._calculate_bmi(
                harmonized['weight'],
                harmonized['height']
            )
        
        if 'creatinine' in harmonized and 'egfr' not in harmonized:
            harmonized['egfr'] = self._calculate_egfr(
                harmonized['creatinine'],
                harmonized.get('age', 50),
                harmonized.get('sex', 'M')
            )
        
        return self._fill_defaults(harmonized)
    
    def _fill_defaults(self, data: Dict) -> Dict:
        """Fill missing values with reasonable defaults or None"""
        for feature, dtype in self.STANDARD_FEATURES.items():
            if feature not in data or pd.isna(data[feature]):
                if dtype == bool:
                    data[feature] = False
                elif dtype == str:
                    data[feature] = 'unknown'
                else:
                    data[feature] = None
        
        return data
    
    def _calculate_bmi(self, weight_kg: float, height_cm: float) -> float:
        """Calculate BMI from weight and height"""
        if not weight_kg or not height_cm or height_cm == 0:
            return None
        height_m = height_cm / 100.0
        return weight_kg / (height_m ** 2)
    
    def _calculate_egfr(self, creatinine: float, age: int, sex: str) -> float:
        """
        Calculate eGFR using CKD-EPI equation
        
        Args:
            creatinine: Serum creatinine (mg/dL)
            age: Age in years
            sex: 'M' or 'F'
        
        Returns:
            eGFR in mL/min/1.73m²
        """
        if not creatinine or not age:
            return None
        
        # CKD-EPI equation
        kappa = 0.7 if sex == 'F' else 0.9
        alpha = -0.329 if sex == 'F' else -0.411
        sex_factor = 1.018 if sex == 'F' else 1.0
        
        egfr = 141 * min(creatinine / kappa, 1) ** alpha
        egfr *= max(creatinine / kappa, 1) ** -1.209
        egfr *= 0.993 ** age
        egfr *= sex_factor
        
        return round(egfr, 1)
    
    def validate(self, data: Dict) -> tuple[bool, List[str]]:
        """
        Validate harmonized data
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required = ['patient_id', 'age', 'sex']
        for field in required:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Value ranges
        if data.get('age') and (data['age'] < 0 or data['age'] > 120):
            errors.append(f"Invalid age: {data['age']}")
        
        if data.get('bmi') and (data['bmi'] < 10 or data['bmi'] > 80):
            errors.append(f"Invalid BMI: {data['bmi']}")
        
        if data.get('systolic_bp') and (data['systolic_bp'] < 60 or data['systolic_bp'] > 250):
            errors.append(f"Invalid systolic BP: {data['systolic_bp']}")
        
        if data.get('hba1c') and (data['hba1c'] < 3 or data['hba1c'] > 20):
            errors.append(f"Invalid HbA1c: {data['hba1c']}")
        
        return (len(errors) == 0, errors)
    
    def to_patient_state_format(self, data: Dict) -> Dict:
        """
        Convert harmonized data to PatientState initialization format
        
        Returns:
            Dictionary suitable for PatientState(data)
        """
        return {
            'patient_id': data.get('patient_id'),
            'age': data.get('age'),
            'sex': data.get('sex'),
            'race': data.get('race'),
            'height': data.get('height'),
            'weight': data.get('weight'),
            'bmi': data.get('bmi'),
            'hba1c': data.get('hba1c'),
            'fasting_glucose': data.get('fasting_glucose'),
            'blood_pressure': {
                'systolic': data.get('systolic_bp'),
                'diastolic': data.get('diastolic_bp')
            },
            'cholesterol': {
                'total': data.get('total_cholesterol'),
                'ldl': data.get('ldl'),
                'hdl': data.get('hdl'),
                'triglycerides': data.get('triglycerides')
            },
            'liver_enzymes': {
                'alt': data.get('alt'),
                'ast': data.get('ast')
            },
            'kidney_function': {
                'creatinine': data.get('creatinine'),
                'egfr': data.get('egfr')
            },
            'inflammation': {
                'crp': data.get('crp')
            },
            'lifestyle': {
                'physical_activity': data.get('physical_activity'),
                'diet_quality': data.get('diet_quality'),
                'smoking': data.get('smoking'),
                'alcohol': data.get('alcohol_per_week'),
                'sleep_hours': data.get('sleep_hours'),
                'stress_level': data.get('stress_level')
            },
            'medical_history': {
                'diabetes': data.get('has_diabetes'),
                'hypertension': data.get('has_hypertension'),
                'cvd': data.get('has_cvd'),
                'ckd': data.get('has_ckd'),
                'family_history': {
                    'diabetes': data.get('family_history_diabetes'),
                    'cvd': data.get('family_history_cvd')
                }
            }
        }


# Example usage
if __name__ == '__main__':
    harmonizer = DataHarmonizer()
    
    # Example NHANES data
    nhanes_data = {
        'patient_id': 'NHANES_12345',
        'age': 45,
        'sex': 'M',
        'bmi': 30.5,
        'hba1c': 5.9,
        'systolic_bp': 135,
        'diastolic_bp': 85,
        'physical_activity': 'sedentary'
    }
    
    harmonized = harmonizer.harmonize(nhanes_data, source='nhanes')
    is_valid, errors = harmonizer.validate(harmonized)
    
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    patient_state_data = harmonizer.to_patient_state_format(harmonized)
    print(f"Patient state format: {patient_state_data}")
