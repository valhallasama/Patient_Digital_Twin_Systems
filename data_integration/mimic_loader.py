#!/usr/bin/env python3
"""
MIMIC-IV Data Loader
Loads and processes ICU clinical data from MIMIC-IV database

Access Requirements:
- Complete CITI training at https://physionet.org/
- Sign data use agreement
- Download MIMIC-IV from https://physionet.org/content/mimiciv/

MIMIC-IV Structure:
- hosp/ - Hospital data (labs, diagnoses, procedures)
- icu/ - ICU data (vitals, medications, charts)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MIMICLoader:
    """Load and process MIMIC-IV clinical data"""
    
    def __init__(self, mimic_path: str):
        """
        Initialize MIMIC-IV loader
        
        Args:
            mimic_path: Path to MIMIC-IV root directory
        """
        self.mimic_path = Path(mimic_path)
        self.hosp_path = self.mimic_path / 'hosp'
        self.icu_path = self.mimic_path / 'icu'
        
        if not self.mimic_path.exists():
            logger.warning(f"MIMIC-IV path not found: {mimic_path}")
            logger.info("To use MIMIC-IV:")
            logger.info("1. Complete CITI training: https://physionet.org/")
            logger.info("2. Download MIMIC-IV: https://physionet.org/content/mimiciv/")
    
    def load_patient_demographics(self, subject_id: int) -> Dict:
        """
        Load patient demographics
        
        Returns:
            {age, sex, race, admission_type, ...}
        """
        try:
            # Load patients table
            patients = pd.read_csv(self.hosp_path / 'patients.csv.gz')
            patient = patients[patients['subject_id'] == subject_id].iloc[0]
            
            # Load admissions
            admissions = pd.read_csv(self.hosp_path / 'admissions.csv.gz')
            admission = admissions[admissions['subject_id'] == subject_id].iloc[0]
            
            return {
                'subject_id': subject_id,
                'sex': patient['gender'],
                'age': self._calculate_age(patient, admission),
                'race': admission.get('race', 'Unknown'),
                'admission_type': admission.get('admission_type', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error loading demographics for {subject_id}: {e}")
            return {}
    
    def load_lab_values(self, subject_id: int, hadm_id: int = None) -> pd.DataFrame:
        """
        Load laboratory test results
        
        Key labs for digital twin:
        - Glucose, HbA1c
        - Creatinine, BUN
        - ALT, AST
        - Cholesterol (LDL, HDL, triglycerides)
        - CRP
        
        Returns:
            DataFrame with columns: [charttime, itemid, label, value, valuenum, valueuom]
        """
        try:
            labevents = pd.read_csv(
                self.hosp_path / 'labevents.csv.gz',
                usecols=['subject_id', 'hadm_id', 'charttime', 'itemid', 
                        'value', 'valuenum', 'valueuom']
            )
            
            # Filter by subject
            labs = labevents[labevents['subject_id'] == subject_id]
            
            if hadm_id:
                labs = labs[labs['hadm_id'] == hadm_id]
            
            # Load lab item definitions
            d_labitems = pd.read_csv(self.hosp_path / 'd_labitems.csv.gz')
            
            # Merge to get lab names
            labs = labs.merge(d_labitems[['itemid', 'label']], on='itemid', how='left')
            
            return labs.sort_values('charttime')
        except Exception as e:
            logger.error(f"Error loading labs for {subject_id}: {e}")
            return pd.DataFrame()
    
    def load_vital_signs(self, subject_id: int, stay_id: int = None) -> pd.DataFrame:
        """
        Load vital signs from ICU
        
        Key vitals:
        - Blood pressure (systolic, diastolic)
        - Heart rate
        - Respiratory rate
        - Temperature
        - SpO2
        
        Returns:
            DataFrame with vital signs over time
        """
        try:
            chartevents = pd.read_csv(
                self.icu_path / 'chartevents.csv.gz',
                usecols=['subject_id', 'stay_id', 'charttime', 'itemid', 
                        'value', 'valuenum', 'valueuom']
            )
            
            vitals = chartevents[chartevents['subject_id'] == subject_id]
            
            if stay_id:
                vitals = vitals[vitals['stay_id'] == stay_id]
            
            # Load chart item definitions
            d_items = pd.read_csv(self.icu_path / 'd_items.csv.gz')
            
            # Merge to get vital names
            vitals = vitals.merge(d_items[['itemid', 'label']], on='itemid', how='left')
            
            return vitals.sort_values('charttime')
        except Exception as e:
            logger.error(f"Error loading vitals for {subject_id}: {e}")
            return pd.DataFrame()
    
    def load_diagnoses(self, subject_id: int, hadm_id: int = None) -> List[str]:
        """
        Load ICD diagnosis codes
        
        Returns:
            List of diagnosis descriptions
        """
        try:
            diagnoses = pd.read_csv(self.hosp_path / 'diagnoses_icd.csv.gz')
            diag = diagnoses[diagnoses['subject_id'] == subject_id]
            
            if hadm_id:
                diag = diag[diag['hadm_id'] == hadm_id]
            
            # Load ICD definitions
            d_icd = pd.read_csv(self.hosp_path / 'd_icd_diagnoses.csv.gz')
            
            # Merge to get descriptions
            diag = diag.merge(d_icd[['icd_code', 'long_title']], 
                            left_on='icd_code', right_on='icd_code', how='left')
            
            return diag['long_title'].tolist()
        except Exception as e:
            logger.error(f"Error loading diagnoses for {subject_id}: {e}")
            return []
    
    def extract_patient_features(self, subject_id: int, hadm_id: int = None) -> Dict:
        """
        Extract structured features for digital twin
        
        Returns:
            Dictionary with all relevant patient features
        """
        features = {}
        
        # Demographics
        demographics = self.load_patient_demographics(subject_id)
        features.update(demographics)
        
        # Lab values (most recent or average)
        labs = self.load_lab_values(subject_id, hadm_id)
        if not labs.empty:
            features.update(self._extract_lab_features(labs))
        
        # Vital signs (average)
        vitals = self.load_vital_signs(subject_id)
        if not vitals.empty:
            features.update(self._extract_vital_features(vitals))
        
        # Diagnoses
        diagnoses = self.load_diagnoses(subject_id, hadm_id)
        features['diagnoses'] = diagnoses
        features['has_diabetes'] = any('diabetes' in d.lower() for d in diagnoses)
        features['has_hypertension'] = any('hypertension' in d.lower() for d in diagnoses)
        features['has_ckd'] = any('kidney' in d.lower() or 'renal' in d.lower() for d in diagnoses)
        
        return features
    
    def _extract_lab_features(self, labs: pd.DataFrame) -> Dict:
        """Extract key lab values"""
        features = {}
        
        # Key lab mappings (MIMIC itemid → feature name)
        lab_mappings = {
            'glucose': ['50809', '50931'],  # Glucose
            'hba1c': ['50852'],  # HbA1c
            'creatinine': ['50912'],  # Creatinine
            'alt': ['50861'],  # ALT
            'ast': ['50878'],  # AST
            'ldl': ['50901'],  # LDL
            'hdl': ['50907'],  # HDL
            'triglycerides': ['51000'],  # Triglycerides
            'crp': ['50889']  # CRP
        }
        
        for feature, itemids in lab_mappings.items():
            values = labs[labs['itemid'].isin([int(i) for i in itemids])]['valuenum']
            if not values.empty:
                features[feature] = values.mean()  # Average if multiple measurements
        
        return features
    
    def _extract_vital_features(self, vitals: pd.DataFrame) -> Dict:
        """Extract key vital signs"""
        features = {}
        
        # Vital sign mappings
        vital_mappings = {
            'systolic_bp': ['220050', '220179'],  # Systolic BP
            'diastolic_bp': ['220051', '220180'],  # Diastolic BP
            'heart_rate': ['220045'],  # Heart rate
            'respiratory_rate': ['220210'],  # Respiratory rate
            'temperature': ['223761'],  # Temperature
            'spo2': ['220277']  # SpO2
        }
        
        for feature, itemids in vital_mappings.items():
            values = vitals[vitals['itemid'].isin([int(i) for i in itemids])]['valuenum']
            if not values.empty:
                features[feature] = values.mean()
        
        return features
    
    def _calculate_age(self, patient: pd.Series, admission: pd.Series) -> int:
        """Calculate patient age at admission"""
        try:
            # MIMIC-IV provides anchor_age
            return int(patient.get('anchor_age', 0))
        except:
            return 0
    
    def get_cohort(self, 
                   min_age: int = 18,
                   max_age: int = 90,
                   has_labs: bool = True,
                   limit: int = 1000) -> List[int]:
        """
        Get a cohort of patient IDs matching criteria
        
        Args:
            min_age: Minimum age
            max_age: Maximum age
            has_labs: Require lab data
            limit: Maximum number of patients
        
        Returns:
            List of subject_ids
        """
        try:
            patients = pd.read_csv(self.hosp_path / 'patients.csv.gz')
            
            # Filter by age (using anchor_age as proxy)
            cohort = patients[
                (patients['anchor_age'] >= min_age) & 
                (patients['anchor_age'] <= max_age)
            ]
            
            if has_labs:
                # Check which patients have lab data
                labevents = pd.read_csv(
                    self.hosp_path / 'labevents.csv.gz',
                    usecols=['subject_id']
                )
                patients_with_labs = labevents['subject_id'].unique()
                cohort = cohort[cohort['subject_id'].isin(patients_with_labs)]
            
            return cohort['subject_id'].head(limit).tolist()
        except Exception as e:
            logger.error(f"Error getting cohort: {e}")
            return []


# Example usage
if __name__ == '__main__':
    # Initialize loader
    loader = MIMICLoader('/path/to/mimic-iv/')
    
    # Get a cohort
    cohort = loader.get_cohort(min_age=40, max_age=70, limit=100)
    print(f"Found {len(cohort)} patients")
    
    # Extract features for first patient
    if cohort:
        features = loader.extract_patient_features(cohort[0])
        print(f"Patient features: {features}")
