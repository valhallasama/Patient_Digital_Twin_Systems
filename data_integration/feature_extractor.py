#!/usr/bin/env python3
"""
Feature Extractor
Extracts and engineers features for ML models and graph neural networks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and engineer features from harmonized patient data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_all_features(self, patient_data: Dict) -> np.ndarray:
        """
        Extract all features for ML/GNN
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        self.feature_names = []
        
        # Demographics
        features.extend(self._extract_demographic_features(patient_data))
        
        # Anthropometric
        features.extend(self._extract_anthropometric_features(patient_data))
        
        # Metabolic
        features.extend(self._extract_metabolic_features(patient_data))
        
        # Cardiovascular
        features.extend(self._extract_cardiovascular_features(patient_data))
        
        # Liver
        features.extend(self._extract_liver_features(patient_data))
        
        # Kidney
        features.extend(self._extract_kidney_features(patient_data))
        
        # Inflammation
        features.extend(self._extract_inflammation_features(patient_data))
        
        # Lifestyle
        features.extend(self._extract_lifestyle_features(patient_data))
        
        # Derived/engineered features
        features.extend(self._extract_derived_features(patient_data))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_demographic_features(self, data: Dict) -> List[float]:
        """Extract demographic features"""
        features = []
        
        # Age (continuous)
        age = data.get('age', 50)
        features.append(age)
        self.feature_names.append('age')
        
        # Age squared (non-linear effects)
        features.append(age ** 2)
        self.feature_names.append('age_squared')
        
        # Sex (binary)
        sex = 1.0 if data.get('sex') == 'M' else 0.0
        features.append(sex)
        self.feature_names.append('sex_male')
        
        return features
    
    def _extract_anthropometric_features(self, data: Dict) -> List[float]:
        """Extract body measurement features"""
        features = []
        
        # BMI
        bmi = data.get('bmi', 25.0)
        features.append(bmi)
        self.feature_names.append('bmi')
        
        # BMI categories (one-hot-ish)
        features.append(1.0 if bmi >= 30 else 0.0)  # Obese
        self.feature_names.append('is_obese')
        
        features.append(1.0 if 25 <= bmi < 30 else 0.0)  # Overweight
        self.feature_names.append('is_overweight')
        
        # Waist circumference (if available)
        waist = data.get('waist_circumference')
        if waist:
            features.append(waist)
            self.feature_names.append('waist_circumference')
            
            # Waist-to-height ratio (if height available)
            height = data.get('height')
            if height:
                features.append(waist / height)
                self.feature_names.append('waist_height_ratio')
            else:
                features.append(0.0)
                self.feature_names.append('waist_height_ratio')
        else:
            features.extend([0.0, 0.0])
            self.feature_names.extend(['waist_circumference', 'waist_height_ratio'])
        
        return features
    
    def _extract_metabolic_features(self, data: Dict) -> List[float]:
        """Extract metabolic features"""
        features = []
        
        # Glucose
        glucose = data.get('fasting_glucose', 100.0)
        features.append(glucose)
        self.feature_names.append('fasting_glucose')
        
        # HbA1c
        hba1c = data.get('hba1c', 5.5)
        features.append(hba1c)
        self.feature_names.append('hba1c')
        
        # Diabetes status (categorical)
        features.append(1.0 if hba1c >= 6.5 else 0.0)  # Diabetic
        self.feature_names.append('is_diabetic')
        
        features.append(1.0 if 5.7 <= hba1c < 6.5 else 0.0)  # Prediabetic
        self.feature_names.append('is_prediabetic')
        
        # Estimated insulin resistance (HOMA-IR approximation)
        # HOMA-IR ≈ (glucose * insulin) / 405
        # Without insulin, use proxy: (glucose * BMI) / 1000
        bmi = data.get('bmi', 25.0)
        insulin_resistance_proxy = (glucose * bmi) / 1000.0
        features.append(insulin_resistance_proxy)
        self.feature_names.append('insulin_resistance_proxy')
        
        return features
    
    def _extract_cardiovascular_features(self, data: Dict) -> List[float]:
        """Extract cardiovascular features"""
        features = []
        
        # Blood pressure
        systolic = data.get('systolic_bp', 120.0)
        diastolic = data.get('diastolic_bp', 80.0)
        
        features.append(systolic)
        self.feature_names.append('systolic_bp')
        
        features.append(diastolic)
        self.feature_names.append('diastolic_bp')
        
        # Pulse pressure (systolic - diastolic)
        features.append(systolic - diastolic)
        self.feature_names.append('pulse_pressure')
        
        # Mean arterial pressure
        map_value = diastolic + (systolic - diastolic) / 3.0
        features.append(map_value)
        self.feature_names.append('mean_arterial_pressure')
        
        # Hypertension status
        features.append(1.0 if systolic >= 140 or diastolic >= 90 else 0.0)
        self.feature_names.append('is_hypertensive')
        
        # Lipids
        ldl = data.get('ldl', 100.0)
        hdl = data.get('hdl', 50.0)
        tg = data.get('triglycerides', 150.0)
        
        features.append(ldl)
        self.feature_names.append('ldl')
        
        features.append(hdl)
        self.feature_names.append('hdl')
        
        features.append(tg)
        self.feature_names.append('triglycerides')
        
        # LDL/HDL ratio (atherogenic index)
        features.append(ldl / hdl if hdl > 0 else 0.0)
        self.feature_names.append('ldl_hdl_ratio')
        
        # Total cholesterol (if available, else estimate)
        total_chol = data.get('total_cholesterol', ldl + hdl + tg / 5.0)
        features.append(total_chol)
        self.feature_names.append('total_cholesterol')
        
        return features
    
    def _extract_liver_features(self, data: Dict) -> List[float]:
        """Extract liver function features"""
        features = []
        
        # Liver enzymes
        alt = data.get('alt', 25.0)
        ast = data.get('ast', 25.0)
        
        features.append(alt)
        self.feature_names.append('alt')
        
        features.append(ast)
        self.feature_names.append('ast')
        
        # AST/ALT ratio (De Ritis ratio)
        features.append(ast / alt if alt > 0 else 1.0)
        self.feature_names.append('ast_alt_ratio')
        
        # Elevated liver enzymes flag
        features.append(1.0 if alt > 40 or ast > 40 else 0.0)
        self.feature_names.append('elevated_liver_enzymes')
        
        return features
    
    def _extract_kidney_features(self, data: Dict) -> List[float]:
        """Extract kidney function features"""
        features = []
        
        # Creatinine
        creatinine = data.get('creatinine', 1.0)
        features.append(creatinine)
        self.feature_names.append('creatinine')
        
        # eGFR
        egfr = data.get('egfr', 90.0)
        features.append(egfr)
        self.feature_names.append('egfr')
        
        # CKD stages
        features.append(1.0 if egfr < 60 else 0.0)  # CKD stage 3+
        self.feature_names.append('has_ckd')
        
        features.append(1.0 if egfr < 30 else 0.0)  # CKD stage 4+
        self.feature_names.append('has_severe_ckd')
        
        return features
    
    def _extract_inflammation_features(self, data: Dict) -> List[float]:
        """Extract inflammation markers"""
        features = []
        
        # CRP
        crp = data.get('crp', 1.0)
        features.append(crp)
        self.feature_names.append('crp')
        
        # High inflammation flag
        features.append(1.0 if crp > 3.0 else 0.0)
        self.feature_names.append('high_inflammation')
        
        return features
    
    def _extract_lifestyle_features(self, data: Dict) -> List[float]:
        """Extract lifestyle features"""
        features = []
        
        # Physical activity (ordinal encoding)
        activity = data.get('physical_activity', 'sedentary')
        activity_map = {'sedentary': 0.0, 'moderate': 0.5, 'vigorous': 1.0}
        features.append(activity_map.get(activity, 0.0))
        self.feature_names.append('physical_activity_level')
        
        # Smoking (binary)
        features.append(1.0 if data.get('smoking', False) else 0.0)
        self.feature_names.append('is_smoker')
        
        # Alcohol (continuous)
        alcohol = data.get('alcohol_per_week') or 0.0
        features.append(alcohol)
        self.feature_names.append('alcohol_per_week')
        
        # Heavy drinking flag
        features.append(1.0 if (alcohol and alcohol > 14) else 0.0)
        self.feature_names.append('heavy_drinker')
        
        # Sleep (if available)
        sleep = data.get('sleep_hours') or 7.0
        features.append(sleep)
        self.feature_names.append('sleep_hours')
        
        # Poor sleep flag
        features.append(1.0 if (sleep and (sleep < 6 or sleep > 9)) else 0.0)
        self.feature_names.append('poor_sleep')
        
        return features
    
    def _extract_derived_features(self, data: Dict) -> List[float]:
        """Extract engineered/derived features"""
        features = []
        
        # Metabolic syndrome score (0-5 components)
        ms_score = 0
        
        # 1. Abdominal obesity
        waist = data.get('waist_circumference') or 0
        sex = data.get('sex', 'M')
        if waist and ((sex == 'M' and waist > 102) or (sex == 'F' and waist > 88)):
            ms_score += 1
        
        # 2. High triglycerides
        tg = data.get('triglycerides') or 0
        if tg and tg >= 150:
            ms_score += 1
        
        # 3. Low HDL
        hdl = data.get('hdl') or 100
        if hdl and ((sex == 'M' and hdl < 40) or (sex == 'F' and hdl < 50)):
            ms_score += 1
        
        # 4. High blood pressure
        systolic = data.get('systolic_bp') or 0
        diastolic = data.get('diastolic_bp') or 0
        if (systolic and systolic >= 130) or (diastolic and diastolic >= 85):
            ms_score += 1
        
        # 5. High fasting glucose
        glucose = data.get('fasting_glucose') or 0
        if glucose and glucose >= 100:
            ms_score += 1
        
        features.append(float(ms_score))
        self.feature_names.append('metabolic_syndrome_score')
        
        # Has metabolic syndrome (3+ components)
        features.append(1.0 if ms_score >= 3 else 0.0)
        self.feature_names.append('has_metabolic_syndrome')
        
        # Cardiovascular risk score (simplified Framingham-like)
        cv_risk = 0.0
        age = data.get('age') or 50
        if age:
            if age >= 65:
                cv_risk += 3.0
            elif age >= 55:
                cv_risk += 2.0
            elif age >= 45:
                cv_risk += 1.0
        
        if data.get('smoking', False):
            cv_risk += 2.0
        
        if data.get('has_diabetes', False):
            cv_risk += 2.0
        
        systolic = data.get('systolic_bp') or 0
        if systolic and systolic >= 140:
            cv_risk += 1.5
        
        ldl = data.get('ldl') or 0
        if ldl and ldl >= 160:
            cv_risk += 1.5
        
        features.append(cv_risk)
        self.feature_names.append('cv_risk_score')
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features using StandardScaler
        
        Args:
            features: Feature array (n_samples, n_features)
            fit: Whether to fit the scaler (True for training, False for inference)
        
        Returns:
            Normalized features
        """
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)
    
    def extract_graph_features(self, patient_data: Dict) -> Dict[str, np.ndarray]:
        """
        Extract features organized by organ system (for graph neural network)
        
        Returns:
            Dictionary mapping organ system to feature vector
        """
        graph_features = {}
        
        # Metabolic node
        graph_features['metabolic'] = np.array([
            patient_data.get('fasting_glucose', 100.0),
            patient_data.get('hba1c', 5.5),
            patient_data.get('bmi', 25.0),
            patient_data.get('waist_circumference', 90.0) or 90.0
        ], dtype=np.float32)
        
        # Cardiovascular node
        graph_features['cardiovascular'] = np.array([
            patient_data.get('systolic_bp', 120.0),
            patient_data.get('diastolic_bp', 80.0),
            patient_data.get('ldl', 100.0),
            patient_data.get('hdl', 50.0),
            patient_data.get('triglycerides', 150.0)
        ], dtype=np.float32)
        
        # Liver node
        graph_features['liver'] = np.array([
            patient_data.get('alt', 25.0),
            patient_data.get('ast', 25.0)
        ], dtype=np.float32)
        
        # Kidney node
        graph_features['kidney'] = np.array([
            patient_data.get('creatinine', 1.0),
            patient_data.get('egfr', 90.0)
        ], dtype=np.float32)
        
        # Inflammation node
        graph_features['inflammation'] = np.array([
            patient_data.get('crp', 1.0)
        ], dtype=np.float32)
        
        # Lifestyle node
        activity_map = {'sedentary': 0.0, 'moderate': 0.5, 'vigorous': 1.0}
        graph_features['lifestyle'] = np.array([
            activity_map.get(patient_data.get('physical_activity', 'sedentary'), 0.0),
            1.0 if patient_data.get('smoking', False) else 0.0,
            patient_data.get('alcohol_per_week', 0.0),
            patient_data.get('sleep_hours', 7.0)
        ], dtype=np.float32)
        
        return graph_features


# Example usage
if __name__ == '__main__':
    extractor = FeatureExtractor()
    
    # Example patient data
    patient = {
        'age': 45,
        'sex': 'M',
        'bmi': 30.5,
        'hba1c': 5.9,
        'fasting_glucose': 110,
        'systolic_bp': 135,
        'diastolic_bp': 85,
        'ldl': 140,
        'hdl': 38,
        'triglycerides': 180,
        'alt': 42,
        'ast': 38,
        'creatinine': 1.1,
        'egfr': 85,
        'crp': 4.2,
        'physical_activity': 'sedentary',
        'smoking': True,
        'alcohol_per_week': 10
    }
    
    # Extract all features
    features = extractor.extract_all_features(patient)
    print(f"Extracted {len(features)} features")
    print(f"Feature names: {extractor.get_feature_names()}")
    
    # Extract graph features
    graph_features = extractor.extract_graph_features(patient)
    print(f"\nGraph features:")
    for organ, feat in graph_features.items():
        print(f"  {organ}: {feat.shape}")
