#!/usr/bin/env python3
"""
NHANES Data Loader
National Health and Nutrition Examination Survey

Public dataset from CDC with:
- Demographics
- Laboratory tests
- Physical examinations
- Lifestyle questionnaires
- Disease outcomes

Download from: https://wwwn.cdc.gov/nchs/nhanes/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NHANESLoader:
    """Load and process NHANES survey data"""
    
    def __init__(self, data_path: str, cycle: str = '2017-2018'):
        """
        Initialize NHANES loader
        
        Args:
            data_path: Path to NHANES data directory
            cycle: NHANES cycle (e.g., '2017-2018')
        """
        self.data_path = Path(data_path)
        self.cycle = cycle
        self.cycle_path = self.data_path / cycle
        
        # Map cycle years to file suffix
        # NHANES uses letters: I=2015-2016, J=2017-2018, K=2019-2020, etc.
        cycle_map = {
            '2015-2016': 'I',
            '2017-2018': 'J',
            '2019-2020': 'K',
            '2021-2022': 'L'
        }
        self.file_suffix = cycle_map.get(cycle, 'J')
        
        if not self.cycle_path.exists():
            logger.warning(f"NHANES path not found: {self.cycle_path}")
            logger.info("Download NHANES data from: https://wwwn.cdc.gov/nchs/nhanes/")
    
    def load_demographics(self) -> pd.DataFrame:
        """
        Load demographics file
        
        Key variables:
        - SEQN: Respondent sequence number (ID)
        - RIAGENDR: Gender
        - RIDAGEYR: Age in years
        - RIDRETH3: Race/ethnicity
        - DMDEDUC2: Education level
        """
        try:
            # Try with cycle suffix first (e.g., DEMO_J.XPT for 2017-2018)
            demo_file = self.cycle_path / f'DEMO_{self.file_suffix}.XPT'
            
            if not demo_file.exists():
                # Try without suffix
                demo_file = self.cycle_path / 'DEMO.XPT'
            
            if demo_file.exists():
                return pd.read_sas(demo_file)
            else:
                logger.warning(f"Demographics file not found: {demo_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading demographics: {e}")
            return pd.DataFrame()
    
    def load_laboratory(self) -> pd.DataFrame:
        """
        Load laboratory data
        
        Key files:
        - GLU: Glucose
        - GHB: Glycohemoglobin (HbA1c)
        - BIOPRO: Biochemistry profile (ALT, AST, creatinine)
        - TRIGLY: Cholesterol (LDL, HDL, triglycerides)
        - CRP: C-reactive protein
        """
        lab_data = {}
        
        lab_files = {
            'glucose': 'GLU',
            'hba1c': 'GHB',
            'biochem': 'BIOPRO',
            'lipids': 'TRIGLY',
            'crp': 'CRP'
        }
        
        for name, file_prefix in lab_files.items():
            try:
                # Try with suffix first
                file_path = self.cycle_path / f'{file_prefix}_{self.file_suffix}.XPT'
                if not file_path.exists():
                    # Try without suffix
                    file_path = self.cycle_path / f'{file_prefix}.XPT'
                
                if file_path.exists():
                    lab_data[name] = pd.read_sas(file_path)
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
        
        # Merge all lab data on SEQN
        if lab_data:
            merged = lab_data[list(lab_data.keys())[0]]
            for name, df in list(lab_data.items())[1:]:
                merged = merged.merge(df, on='SEQN', how='outer')
            return merged
        
        return pd.DataFrame()
    
    def load_body_measures(self) -> pd.DataFrame:
        """
        Load body measurements
        
        Key variables:
        - BMXBMI: BMI
        - BMXWT: Weight (kg)
        - BMXHT: Height (cm)
        - BMXWAIST: Waist circumference
        """
        try:
            bmx_file = self.cycle_path / f'BMX_{self.file_suffix}.XPT'
            if not bmx_file.exists():
                bmx_file = self.cycle_path / 'BMX.XPT'
            
            if bmx_file.exists():
                return pd.read_sas(bmx_file)
            else:
                logger.warning(f"Body measures file not found: {bmx_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading body measures: {e}")
            return pd.DataFrame()
    
    def load_blood_pressure(self) -> pd.DataFrame:
        """
        Load blood pressure measurements
        
        Key variables:
        - BPXSY1, BPXSY2, BPXSY3: Systolic BP (3 readings)
        - BPXDI1, BPXDI2, BPXDI3: Diastolic BP (3 readings)
        """
        try:
            bpx_file = self.cycle_path / f'BPX_{self.file_suffix}.XPT'
            if not bpx_file.exists():
                bpx_file = self.cycle_path / 'BPX.XPT'
            
            if bpx_file.exists():
                return pd.read_sas(bpx_file)
            else:
                logger.warning(f"Blood pressure file not found: {bpx_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading blood pressure: {e}")
            return pd.DataFrame()
    
    def load_questionnaire(self, questionnaire: str) -> pd.DataFrame:
        """
        Load questionnaire data
        
        Common questionnaires:
        - PAQ: Physical activity
        - DBQ: Diet behavior
        - SMQ: Smoking
        - ALQ: Alcohol use
        - SLQ: Sleep disorders
        - MCQ: Medical conditions
        """
        try:
            q_file = self.cycle_path / f'{questionnaire}_{self.file_suffix}.XPT'
            if not q_file.exists():
                q_file = self.cycle_path / f'{questionnaire}.XPT'
            
            if q_file.exists():
                return pd.read_sas(q_file)
            else:
                logger.warning(f"Questionnaire file not found: {q_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading questionnaire {questionnaire}: {e}")
            return pd.DataFrame()
    
    def extract_patient_features(self, seqn: int) -> Dict:
        """
        Extract all features for a single patient
        
        Returns:
            Dictionary with standardized feature names
        """
        features = {'patient_id': f'NHANES_{seqn}'}
        
        # Load all data
        demo = self.load_demographics()
        labs = self.load_laboratory()
        body = self.load_body_measures()
        bp = self.load_blood_pressure()
        
        # Extract demographics
        if not demo.empty and seqn in demo['SEQN'].values:
            patient_demo = demo[demo['SEQN'] == seqn].iloc[0]
            features['age'] = int(patient_demo.get('RIDAGEYR', 0))
            features['sex'] = 'M' if patient_demo.get('RIAGENDR') == 1 else 'F'
            features['race'] = self._map_race(patient_demo.get('RIDRETH3'))
        
        # Extract labs
        if not labs.empty and seqn in labs['SEQN'].values:
            patient_labs = labs[labs['SEQN'] == seqn].iloc[0]
            
            # Glucose (LBXGLU)
            features['fasting_glucose'] = patient_labs.get('LBXGLU')
            
            # HbA1c (LBXGH)
            features['hba1c'] = patient_labs.get('LBXGH')
            
            # Creatinine (LBXSCR)
            features['creatinine'] = patient_labs.get('LBXSCR')
            
            # ALT (LBXSATSI)
            features['alt'] = patient_labs.get('LBXSATSI')
            
            # AST (LBXSASSI)
            features['ast'] = patient_labs.get('LBXSASSI')
            
            # Lipids
            features['total_cholesterol'] = patient_labs.get('LBXTC')
            features['ldl'] = patient_labs.get('LBDLDL')
            features['hdl'] = patient_labs.get('LBDHDD')
            features['triglycerides'] = patient_labs.get('LBXTR')
            
            # CRP (LBXCRP)
            features['crp'] = patient_labs.get('LBXCRP')
        
        # Extract body measures
        if not body.empty and seqn in body['SEQN'].values:
            patient_body = body[body['SEQN'] == seqn].iloc[0]
            features['bmi'] = patient_body.get('BMXBMI')
            features['weight'] = patient_body.get('BMXWT')
            features['height'] = patient_body.get('BMXHT')
            features['waist_circumference'] = patient_body.get('BMXWAIST')
        
        # Extract blood pressure (average of 3 readings)
        if not bp.empty and seqn in bp['SEQN'].values:
            patient_bp = bp[bp['SEQN'] == seqn].iloc[0]
            sys_readings = [patient_bp.get(f'BPXSY{i}') for i in [1, 2, 3]]
            dia_readings = [patient_bp.get(f'BPXDI{i}') for i in [1, 2, 3]]
            
            sys_valid = [x for x in sys_readings if pd.notna(x)]
            dia_valid = [x for x in dia_readings if pd.notna(x)]
            
            if sys_valid:
                features['systolic_bp'] = np.mean(sys_valid)
            if dia_valid:
                features['diastolic_bp'] = np.mean(dia_valid)
        
        # Extract lifestyle from questionnaires
        features.update(self._extract_lifestyle(seqn))
        
        return features
    
    def _extract_lifestyle(self, seqn: int) -> Dict:
        """Extract lifestyle factors from questionnaires"""
        lifestyle = {}
        
        # Physical activity
        paq = self.load_questionnaire('PAQ')
        if not paq.empty and seqn in paq['SEQN'].values:
            patient_paq = paq[paq['SEQN'] == seqn].iloc[0]
            # PAQ605: Vigorous activity
            # PAQ620: Moderate activity
            vigorous = patient_paq.get('PAQ605', 2)  # 1=Yes, 2=No
            moderate = patient_paq.get('PAQ620', 2)
            
            if vigorous == 1:
                lifestyle['physical_activity'] = 'vigorous'
            elif moderate == 1:
                lifestyle['physical_activity'] = 'moderate'
            else:
                lifestyle['physical_activity'] = 'sedentary'
        
        # Smoking
        smq = self.load_questionnaire('SMQ')
        if not smq.empty and seqn in smq['SEQN'].values:
            patient_smq = smq[smq['SEQN'] == seqn].iloc[0]
            # SMQ040: Current smoker
            lifestyle['smoking'] = patient_smq.get('SMQ040') == 1
        
        # Alcohol
        alq = self.load_questionnaire('ALQ')
        if not alq.empty and seqn in alq['SEQN'].values:
            patient_alq = alq[alq['SEQN'] == seqn].iloc[0]
            # ALQ120Q: How often drink alcohol
            drinks_per_week = patient_alq.get('ALQ120Q', 0)
            lifestyle['alcohol_per_week'] = drinks_per_week
        
        return lifestyle
    
    def _map_race(self, race_code: int) -> str:
        """Map NHANES race codes to readable strings"""
        race_map = {
            1: 'Mexican American',
            2: 'Other Hispanic',
            3: 'Non-Hispanic White',
            4: 'Non-Hispanic Black',
            6: 'Non-Hispanic Asian',
            7: 'Other/Multiracial'
        }
        return race_map.get(race_code, 'Unknown')
    
    def get_cohort(self,
                   min_age: int = 18,
                   max_age: int = 90,
                   has_labs: bool = True,
                   limit: int = 1000) -> List[int]:
        """
        Get a cohort of patient IDs matching criteria
        
        Returns:
            List of SEQN identifiers
        """
        demo = self.load_demographics()
        
        if demo.empty:
            return []
        
        # Filter by age
        cohort = demo[
            (demo['RIDAGEYR'] >= min_age) & 
            (demo['RIDAGEYR'] <= max_age)
        ]
        
        if has_labs:
            labs = self.load_laboratory()
            if not labs.empty:
                cohort = cohort[cohort['SEQN'].isin(labs['SEQN'])]
        
        return cohort['SEQN'].head(limit).tolist()


# Example usage
if __name__ == '__main__':
    # Initialize loader
    loader = NHANESLoader('/path/to/nhanes/', cycle='2017-2018')
    
    # Get a cohort
    cohort = loader.get_cohort(min_age=40, max_age=70, limit=100)
    print(f"Found {len(cohort)} patients")
    
    # Extract features for first patient
    if cohort:
        features = loader.extract_patient_features(cohort[0])
        print(f"Patient features: {features}")
