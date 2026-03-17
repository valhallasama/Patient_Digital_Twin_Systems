#!/usr/bin/env python3
"""
Synthea Synthetic Patient Generator
Generates realistic synthetic patient data for training

Synthea is an open-source synthetic patient generator that creates
realistic medical records following clinical guidelines.
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SyntheaGenerator:
    """Generate synthetic patients using Synthea"""
    
    def __init__(self, synthea_path: str = None, output_path: str = './data/synthea_output'):
        """
        Initialize Synthea generator
        
        Args:
            synthea_path: Path to Synthea jar file (will download if not provided)
            output_path: Where to save generated data
        """
        self.synthea_path = Path(synthea_path) if synthea_path else None
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def download_synthea(self):
        """Download Synthea if not already present"""
        if self.synthea_path and self.synthea_path.exists():
            logger.info(f"Synthea found at {self.synthea_path}")
            return
        
        logger.info("Downloading Synthea...")
        synthea_dir = Path('./synthea')
        synthea_dir.mkdir(exist_ok=True)
        
        # Download latest Synthea release
        download_url = "https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"
        
        try:
            subprocess.run([
                'wget', download_url,
                '-O', str(synthea_dir / 'synthea.jar')
            ], check=True)
            
            self.synthea_path = synthea_dir / 'synthea.jar'
            logger.info(f"Synthea downloaded to {self.synthea_path}")
        except Exception as e:
            logger.error(f"Failed to download Synthea: {e}")
            logger.info("Please download manually from: https://github.com/synthetichealth/synthea/releases")
    
    def generate_patients(
        self,
        num_patients: int = 1000,
        state: str = "Massachusetts",
        city: str = None,
        age_range: tuple = (18, 90),
        seed: int = 42
    ) -> Path:
        """
        Generate synthetic patients
        
        Args:
            num_patients: Number of patients to generate
            state: US state for demographics
            city: Specific city (optional)
            age_range: (min_age, max_age)
            seed: Random seed for reproducibility
        
        Returns:
            Path to output directory
        """
        if not self.synthea_path or not self.synthea_path.exists():
            self.download_synthea()
        
        logger.info(f"Generating {num_patients} synthetic patients...")
        
        # Synthea command
        cmd = [
            'java', '-jar', str(self.synthea_path),
            '-p', str(num_patients),
            '-s', str(seed),
            '--exporter.baseDirectory', str(self.output_path),
            state
        ]
        
        if city:
            cmd.extend(['--exporter.city', city])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Generated {num_patients} patients in {self.output_path}")
            return self.output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Synthea generation failed: {e.stderr}")
            raise
    
    def load_generated_data(self) -> pd.DataFrame:
        """
        Load generated Synthea data
        
        Returns:
            DataFrame with patient data
        """
        # Synthea outputs FHIR bundles and CSV files
        csv_path = self.output_path / 'csv'
        
        if not csv_path.exists():
            logger.warning(f"No CSV data found in {csv_path}")
            return pd.DataFrame()
        
        # Load key files
        patients = pd.read_csv(csv_path / 'patients.csv')
        observations = pd.read_csv(csv_path / 'observations.csv')
        conditions = pd.read_csv(csv_path / 'conditions.csv')
        
        logger.info(f"Loaded {len(patients)} patients with {len(observations)} observations")
        
        return patients, observations, conditions
    
    def extract_patient_features(
        self,
        patients: pd.DataFrame,
        observations: pd.DataFrame,
        conditions: pd.DataFrame
    ) -> List[Dict]:
        """
        Extract features from Synthea data
        
        Returns:
            List of patient feature dictionaries
        """
        patient_features = []
        
        for _, patient in patients.iterrows():
            patient_id = patient['Id']
            
            # Demographics
            features = {
                'patient_id': f'SYNTHEA_{patient_id}',
                'age': self._calculate_age(patient['BIRTHDATE']),
                'sex': 'M' if patient['GENDER'] == 'M' else 'F',
                'race': patient.get('RACE', 'Unknown'),
            }
            
            # Get observations for this patient
            patient_obs = observations[observations['PATIENT'] == patient_id]
            
            # Extract lab values using LOINC codes
            features.update(self._extract_labs(patient_obs))
            
            # Get conditions
            patient_cond = conditions[conditions['PATIENT'] == patient_id]
            features.update(self._extract_conditions(patient_cond))
            
            patient_features.append(features)
        
        return patient_features
    
    def _calculate_age(self, birthdate: str) -> int:
        """Calculate age from birthdate"""
        from datetime import datetime
        birth = pd.to_datetime(birthdate)
        today = datetime.now()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    
    def _extract_labs(self, observations: pd.DataFrame) -> Dict:
        """Extract lab values from observations"""
        labs = {}
        
        # LOINC code mappings
        loinc_map = {
            '2339-0': 'fasting_glucose',
            '4548-4': 'hba1c',
            '2160-0': 'creatinine',
            '1742-6': 'alt',
            '1920-8': 'ast',
            '2093-3': 'total_cholesterol',
            '2085-9': 'hdl',
            '2571-8': 'triglycerides',
            '1988-5': 'crp',
            '8480-6': 'systolic_bp',
            '8462-4': 'diastolic_bp',
            '8867-4': 'heart_rate',
            '39156-5': 'bmi',
            '29463-7': 'weight',
            '8302-2': 'height'
        }
        
        for loinc, feature in loinc_map.items():
            obs = observations[observations['CODE'] == loinc]
            if not obs.empty:
                # Get most recent value
                latest = obs.sort_values('DATE').iloc[-1]
                labs[feature] = float(latest['VALUE'])
        
        return labs
    
    def _extract_conditions(self, conditions: pd.DataFrame) -> Dict:
        """Extract disease conditions"""
        cond_dict = {}
        
        # Check for key conditions
        condition_map = {
            'diabetes': ['diabetes', 'type 2 diabetes', 'prediabetes'],
            'hypertension': ['hypertension', 'essential hypertension'],
            'cvd': ['coronary', 'myocardial', 'heart disease', 'atherosclerosis'],
            'ckd': ['chronic kidney disease', 'renal']
        }
        
        for disease, keywords in condition_map.items():
            has_condition = False
            for _, cond in conditions.iterrows():
                desc = str(cond.get('DESCRIPTION', '')).lower()
                if any(kw in desc for kw in keywords):
                    has_condition = True
                    break
            cond_dict[f'has_{disease}'] = has_condition
        
        return cond_dict


def generate_training_dataset(
    num_patients: int = 10000,
    output_file: str = './data/training_data.csv'
) -> pd.DataFrame:
    """
    Generate complete training dataset
    
    Args:
        num_patients: Number of synthetic patients
        output_file: Where to save CSV
    
    Returns:
        DataFrame with all patient features
    """
    logger.info(f"Generating {num_patients} synthetic patients for training...")
    
    # For now, generate synthetic data directly without Synthea
    # (Synthea requires Java and is slow for large datasets)
    
    np.random.seed(42)
    
    patients = []
    for i in range(num_patients):
        # Generate realistic patient
        age = np.random.randint(18, 90)
        sex = np.random.choice(['M', 'F'])
        
        # BMI distribution (realistic)
        bmi = np.random.gamma(shape=5, scale=5) + 18
        bmi = np.clip(bmi, 15, 50)
        
        # Glucose and HbA1c (correlated)
        base_glucose = 90 + (bmi - 25) * 2 + np.random.normal(0, 10)
        glucose = np.clip(base_glucose, 70, 400)
        hba1c = (glucose + 46.7) / 28.7 + np.random.normal(0, 0.3)
        hba1c = np.clip(hba1c, 4.0, 15.0)
        
        # Blood pressure (age and BMI dependent)
        systolic = 110 + age * 0.5 + (bmi - 25) * 0.8 + np.random.normal(0, 10)
        systolic = np.clip(systolic, 90, 220)
        diastolic = 70 + age * 0.2 + (bmi - 25) * 0.3 + np.random.normal(0, 5)
        diastolic = np.clip(diastolic, 60, 130)
        
        # Lipids
        ldl = 100 + (bmi - 25) * 2 + np.random.normal(0, 20)
        ldl = np.clip(ldl, 50, 250)
        hdl = 60 - (bmi - 25) * 0.5 + np.random.normal(0, 10)
        hdl = np.clip(hdl, 20, 100)
        tg = 120 + (bmi - 25) * 3 + np.random.normal(0, 30)
        tg = np.clip(tg, 50, 500)
        
        # Liver enzymes
        alt = 25 + (bmi - 25) * 0.8 + np.random.gamma(2, 5)
        alt = np.clip(alt, 10, 200)
        ast = alt * (0.8 + np.random.normal(0, 0.1))
        ast = np.clip(ast, 10, 200)
        
        # Kidney function
        creatinine = 0.9 + age * 0.003 + np.random.normal(0, 0.2)
        creatinine = np.clip(creatinine, 0.5, 3.0)
        
        # eGFR (CKD-EPI)
        kappa = 0.7 if sex == 'F' else 0.9
        alpha = -0.329 if sex == 'F' else -0.411
        sex_factor = 1.018 if sex == 'F' else 1.0
        egfr = 141 * min(creatinine / kappa, 1) ** alpha
        egfr *= max(creatinine / kappa, 1) ** -1.209
        egfr *= 0.993 ** age
        egfr *= sex_factor
        egfr = np.clip(egfr, 10, 150)
        
        # CRP (inflammation)
        crp = 1.0 + (bmi - 25) * 0.2 + np.random.gamma(1, 1)
        crp = np.clip(crp, 0.1, 20)
        
        # Lifestyle
        activity_levels = ['sedentary', 'moderate', 'vigorous']
        activity_probs = [0.5, 0.35, 0.15]  # Most people sedentary
        physical_activity = np.random.choice(activity_levels, p=activity_probs)
        
        smoking = np.random.random() < 0.15  # 15% smokers
        alcohol_per_week = np.random.gamma(2, 2) if np.random.random() < 0.6 else 0
        
        # Disease status
        has_diabetes = hba1c >= 6.5
        has_hypertension = systolic >= 140 or diastolic >= 90
        has_ckd = egfr < 60
        
        patient = {
            'patient_id': f'SYNTH_{i:05d}',
            'age': age,
            'sex': sex,
            'bmi': round(bmi, 1),
            'fasting_glucose': round(glucose, 1),
            'hba1c': round(hba1c, 2),
            'systolic_bp': round(systolic, 1),
            'diastolic_bp': round(diastolic, 1),
            'ldl': round(ldl, 1),
            'hdl': round(hdl, 1),
            'triglycerides': round(tg, 1),
            'alt': round(alt, 1),
            'ast': round(ast, 1),
            'creatinine': round(creatinine, 2),
            'egfr': round(egfr, 1),
            'crp': round(crp, 2),
            'physical_activity': physical_activity,
            'smoking': smoking,
            'alcohol_per_week': round(alcohol_per_week, 1),
            'has_diabetes': has_diabetes,
            'has_hypertension': has_hypertension,
            'has_ckd': has_ckd
        }
        
        patients.append(patient)
    
    df = pd.DataFrame(patients)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Generated {len(df)} patients")
    logger.info(f"Saved to {output_path}")
    logger.info(f"Diabetes prevalence: {df['has_diabetes'].mean():.1%}")
    logger.info(f"Hypertension prevalence: {df['has_hypertension'].mean():.1%}")
    logger.info(f"CKD prevalence: {df['has_ckd'].mean():.1%}")
    
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Generate training dataset
    df = generate_training_dataset(num_patients=10000)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few patients:")
    print(df.head())
    print(f"\nSummary statistics:")
    print(df.describe())
